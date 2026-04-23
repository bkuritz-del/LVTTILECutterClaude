"""
LVT Tile Cutter (v5)
─────────────────────────────────────────────────────────────────────────────
Changes over v4:

  ENCODING / NAMING ROBUSTNESS
  • stdout and stderr are forced to UTF-8 at startup — no more cp1252 crashes
    from non-ASCII characters anywhere in a path or filename
  • sanitize_path() normalises every path before use: strips leading/trailing
    whitespace, replaces non-breaking spaces and other Unicode space variants
    with regular spaces, and resolves the path to its canonical form
  • sanitize_filename() strips or replaces every character that is illegal on
    Windows (< > : " / \ | ? *) plus control characters and private-use glyphs
    (U+E000–U+F8FF, the source of the original \uf028 crash)
  • All print/log calls encode to ASCII with 'replace' so a bad character
    becomes '?' in the Gooey console instead of an unhandled exception
  • A pre-flight scan runs before processing begins: it lists every file whose
    name contains a problematic character, prints a clear warning for each,
    and renames the file to a safe version (with a backup copy of the original
    name written to a sidecar .txt file)

  GOOEY PROGRESS
  • Gooey progress_regex now matches a single integer on its own line
    ("progress: N") which Gooey maps directly to the progress bar percentage
  • Progress is emitted at three levels:
      - Per-image  : "Starting image N/TOTAL – <name>"
      - Per-size   : "  Size <label>: <cols>x<rows> = <n> tiles"
      - Per-row    : progress percentage updated every tile row
  • The Gooey header is updated live via "progress: N" lines so the bar moves
    smoothly rather than jumping at the end of each image

  OTHER
  • Workers default kept at 1 (serial) — ProcessPoolExecutor is available but
    off by default because Gooey's progress bar only works reliably in the
    main process; set --workers > 1 only if you don't need live progress
  • Removed the --workers 0 auto-detect option (it caused confusion)
  • Output folder created up front with a clear error if it can't be made
  • Dry-run output is clearly prefixed "[DRY RUN]" in the Gooey console

Requires: Pillow, Gooey
"""

# ── Force UTF-8 I/O before anything else ─────────────────────────────────────
import io as _io
import sys

if hasattr(sys.stdout, "buffer"):
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ── Standard library ──────────────────────────────────────────────────────────
import csv
import datetime
import logging
import os
import re
import shutil
import time
import traceback
import unicodedata
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import List, Optional, Tuple

# ── Third-party ───────────────────────────────────────────────────────────────
from PIL import Image, ImageCms
from gooey import Gooey, GooeyParser

# ── Logging (UTF-8, errors=replace so bad chars never crash the logger) ───────
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter(
    "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
))
log = logging.getLogger(__name__)
log.addHandler(_handler)
log.setLevel(logging.INFO)
log.propagate = False   # don't double-print via root logger

# Allow very large images
Image.MAX_IMAGE_PIXELS = None

# ── Windows-illegal filename characters + private-use Unicode block ────────────
_WIN_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f\x7f]')
_PRIVATE_USE = re.compile(r'[\ue000-\uf8ff]')   # catches \uf028 (the inch glyph)


# ─────────────────────────────────────────────────────────────────────────────
# SINGLE SOURCE OF TRUTH: all supported tile sizes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TileSpec:
    flag: str        # argparse dest, e.g. "t12x12"
    label: str       # GUI label,    e.g. '12"x12"'
    short_in: float  # shorter physical dimension (inches)
    long_in: float   # longer physical dimension (inches)

    @property
    def token(self) -> str:
        """Filesystem-safe size token, e.g. '6x36' or '7.25x48'."""
        return re.sub(r'[^A-Za-z0-9x.]', '', self.label)


ALL_TILE_SPECS: List[TileSpec] = [
    TileSpec("t12x12",  '12"x12"',    12,   12),
    TileSpec("t12x18",  '12"x18"',    12,   18),
    TileSpec("t12x24",  '12"x24"',    12,   24),
    TileSpec("t18x18",  '18"x18"',    18,   18),
    TileSpec("t18x36",  '18"x36"',    18,   36),
    TileSpec("t18x48",  '18"x48"',    18,   48),
    TileSpec("t4x36",    '4"x36"',     4,   36),
    TileSpec("t6x36",    '6"x36"',     6,   36),
    TileSpec("t9x36",    '9"x36"',     9,   36),
    TileSpec("t6x48",    '6"x48"',     6,   48),
    TileSpec("t9x48",    '9"x48"',     9,   48),
    TileSpec("t7x48",  '7.25"x48"',  7.25,  48),
    TileSpec("t3x36",    '3"x36"',     3,   36),
]

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}

# Regex to extract an inch value from the filename, e.g. "48in" → 48.0
_INCH_RE = re.compile(r'(\d+(?:\.\d+)?)in', re.IGNORECASE)


# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class PatternMeta:
    color_code: str
    collection: str
    colorway: str
    slab_inches: Optional[float]

    @property
    def prefix(self) -> str:
        return f"{self.color_code}-{self.collection}-{self.colorway}".strip("-")


@dataclass
class TileResult:
    source: str
    size_label: str
    tiles_written: int = 0
    tiles_skipped: int = 0
    error: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Encoding / path safety helpers
# ─────────────────────────────────────────────────────────────────────────────
def safe_str(text: str) -> str:
    """Return text encoded to ASCII with '?' replacing unencodable characters.
    Use this for all print/log calls so nothing can crash the Gooey console."""
    return text.encode("ascii", errors="replace").decode("ascii")


def sanitize_path(raw: str) -> str:
    """Normalise a folder path: collapse Unicode spaces, strip whitespace,
    resolve to absolute path."""
    # Replace all Unicode space variants (non-breaking space, em space, etc.)
    # with a regular ASCII space, then strip
    normalised = "".join(
        " " if unicodedata.category(ch) == "Zs" else ch for ch in raw
    ).strip()
    return os.path.realpath(normalised)


def sanitize_filename(name: str) -> str:
    """Return a Windows-safe version of a filename (no extension).

    Replaces:
      - Private-use Unicode glyphs (e.g. \uf028, the inch glyph) with 'in'
      - Windows-illegal characters with '-'
      - Runs of whitespace with a single space
      - Leading/trailing dots and spaces (Windows reserved)
    """
    # Private-use block → 'in' (handles the \uf028 inch glyph specifically)
    name = _PRIVATE_USE.sub("in", name)
    # NFC normalisation to collapse composed characters
    name = unicodedata.normalize("NFC", name)
    # Windows-illegal characters → hyphen
    name = _WIN_ILLEGAL.sub("-", name)
    # Collapse runs of whitespace
    name = re.sub(r"\s+", " ", name).strip(". ")
    return name or "UNNAMED"


def has_unsafe_chars(name: str) -> bool:
    """Return True if the filename contains anything sanitize_filename would change."""
    return sanitize_filename(name) != name


def sanitize(text: str) -> str:
    """Strip non-alphanumeric characters for use in output filename components."""
    text = text.strip()
    text = re.sub(r"\s+", "", text)
    return re.sub(r"[^A-Za-z0-9\-_.]", "", text)


# ─────────────────────────────────────────────────────────────────────────────
# Pre-flight: scan input folder and rename any file with unsafe characters
# ─────────────────────────────────────────────────────────────────────────────
def preflight_rename(input_folder: str) -> List[Tuple[str, str]]:
    """Scan input_folder for files with problematic names and rename them.

    Returns list of (original_name, safe_name) pairs that were renamed.
    For each renamed file a sidecar <safename>.original.txt is written
    recording what the original name was.
    """
    renames: List[Tuple[str, str]] = []

    for filename in sorted(os.listdir(input_folder)):
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            continue

        stem = os.path.splitext(filename)[0]
        safe_stem = sanitize_filename(stem)
        safe_filename = safe_stem + ext

        if safe_filename == filename:
            continue  # already clean

        src = os.path.join(input_folder, filename)
        dst = os.path.join(input_folder, safe_filename)

        # Avoid clobbering an existing file
        if os.path.exists(dst):
            safe_stem = safe_stem + "_renamed"
            safe_filename = safe_stem + ext
            dst = os.path.join(input_folder, safe_filename)

        try:
            shutil.move(src, dst)
            # Write sidecar with original name so nothing is lost
            sidecar = os.path.join(input_folder, safe_stem + ".original.txt")
            with open(sidecar, "w", encoding="utf-8") as fh:
                fh.write(f"Original filename: {filename}\n")
                fh.write(f"Renamed to:        {safe_filename}\n")
                fh.write(f"Renamed at:        {datetime.datetime.now()}\n")
            renames.append((filename, safe_filename))
        except OSError as exc:
            log.warning("Could not rename %s: %s", safe_str(filename), exc)

    return renames


# ─────────────────────────────────────────────────────────────────────────────
# Image helpers
# ─────────────────────────────────────────────────────────────────────────────
def parse_pattern_metadata(base_name: str) -> PatternMeta:
    """Extract color_code, collection, colorway, slab_inches from a base filename."""
    color = collection = colorway = ""

    if "_" in base_name:
        parts = base_name.split("_")
        color = parts[0] if parts else ""
        name_combo = parts[1] if len(parts) > 1 else ""
        if name_combo:
            sub = name_combo.split("-")
            collection = sub[0]
            colorway = "-".join(sub[1:]) if len(sub) > 1 else ""
    else:
        tokens = base_name.split("-")
        if len(tokens) >= 3:
            color = tokens[0]
            collection = tokens[1]
            colorway = tokens[2]

    inch_match = _INCH_RE.search(base_name)
    slab_inches = float(inch_match.group(1)) if inch_match else None

    return PatternMeta(
        color_code=sanitize(color) or "UNKNOWN",
        collection=sanitize(collection) or "NA",
        colorway=sanitize(colorway) or "NA",
        slab_inches=slab_inches,
    )


def compute_scale(im: Image.Image, meta: PatternMeta, fallback_inches: float) -> float:
    """Return pixels-per-inch, derived from filename or fallback."""
    slab = meta.slab_inches if meta.slab_inches else fallback_inches
    if slab <= 0:
        raise ValueError(f"Invalid slab size: {slab}")
    short_px = min(im.width, im.height)
    factor = short_px / slab
    source = "filename" if meta.slab_inches else "fallback default"
    log.info(
        "  Scale: short side %dpx / %.2fin (%s) = %.4f px/in",
        short_px, slab, source, factor,
    )
    return factor


def ensure_srgb(img: Image.Image) -> Tuple[Image.Image, dict]:
    """Convert img to sRGB, embed ICC profile. Returns (img, save_kwargs)."""
    try:
        dst_profile = ImageCms.createProfile("sRGB")
        src_profile = None
        if img.info.get("icc_profile"):
            try:
                src_profile = ImageCms.ImageCmsProfile(
                    _io.BytesIO(img.info["icc_profile"])
                )
            except Exception as exc:
                log.warning("  Could not parse embedded ICC profile: %s", exc)

        if src_profile:
            out_mode = "RGBA" if img.mode in ("RGBA", "LA") else "RGB"
            img = ImageCms.profileToProfile(
                img, src_profile, dst_profile, outputMode=out_mode
            )
        else:
            if img.mode not in ("RGB", "RGBA"):
                img = img.convert("RGB")

      icc_kwargs: dict = {}
        try:
            # Pillow 9+: profile bytes live on the profile's profile attribute
            profile_obj = dst_profile.profile
            icc_bytes = (
                profile_obj.tobytes()      # littlecms2 object method
                if hasattr(profile_obj, "tobytes")
                else bytes(profile_obj)    # fallback
            )
            if icc_bytes:
                icc_kwargs["icc_profile"] = icc_bytes
        except Exception as exc:
            log.warning("  Could not serialise sRGB profile: %s", exc)

        return img, icc_kwargs

    except Exception as exc:
        log.warning("  ensure_srgb failed (%s); falling back to plain RGB.", exc)
        fallback = img.convert("RGB") if img.mode not in ("RGB", "RGBA") else img
        return fallback, {}


def make_tile(im: Image.Image, col: int, row: int, tile_w: int, tile_h: int) -> Image.Image:
    """Crop one tile; rotate to landscape if non-square and portrait-oriented."""
    left = col * tile_w
    upper = row * tile_h
    tile = im.crop((left, upper, left + tile_w, upper + tile_h))
    if tile.width != tile.height and tile.width < tile.height:
        tile = tile.rotate(90, expand=True)
    return tile


# ─────────────────────────────────────────────────────────────────────────────
# Core: one image × one tile size
# ─────────────────────────────────────────────────────────────────────────────
def process_one_size(
    im: Image.Image,
    meta: PatternMeta,
    factor: float,
    orientation: str,
    spec: TileSpec,
    size_folder: str,
    force: bool,
    dry_run: bool,
    image_index: int,
    image_total: int,
    size_index: int,
    size_total: int,
) -> TileResult:
    result = TileResult(source=meta.prefix, size_label=spec.label)

    if orientation == "portrait":
        tile_w = int(round(spec.short_in * factor))
        tile_h = int(round(spec.long_in * factor))
    else:
        tile_w = int(round(spec.long_in * factor))
        tile_h = int(round(spec.short_in * factor))

    num_cols = im.width // tile_w
    num_rows = im.height // tile_h
    total = num_cols * num_rows

    if total == 0:
        log.warning("  [%s] Tile too large for image; skipping.", spec.label)
        return result

    rem_x = im.width - num_cols * tile_w
    rem_y = im.height - num_rows * tile_h
    if rem_x or rem_y:
        log.info(
            "  [%s] Grid %dx%d = %d tiles  (trimming %dpx right, %dpx bottom)",
            spec.label, num_cols, num_rows, total, rem_x, rem_y,
        )
    else:
        log.info(
            "  [%s] Grid %dx%d = %d tiles",
            spec.label, num_cols, num_rows, total,
        )

    # Emit size description for Gooey console
    print(f"  Size {spec.label}: {num_cols}x{num_rows} = {total} tiles")
    sys.stdout.flush()

    if not dry_run:
        os.makedirs(size_folder, exist_ok=True)

    idx = 0
    for row in range(num_rows):
        for col in range(num_cols):
            idx += 1
            out_name = f"{meta.prefix}-{spec.token}-{idx}.png"
            out_path = os.path.join(size_folder, out_name)

            if not force and os.path.exists(out_path):
                result.tiles_skipped += 1
                continue

            if dry_run:
                print(f"  [DRY RUN] would write: {safe_str(out_name)}")
                sys.stdout.flush()
                result.tiles_skipped += 1
                continue

            tile = make_tile(im, col, row, tile_w, tile_h)
            tile, icc_kwargs = ensure_srgb(tile)
            try:
                tile.save(out_path, "PNG", dpi=(300, 300), **icc_kwargs)
            except TypeError:
                tile.save(out_path, "PNG", dpi=(300, 300))
            result.tiles_written += 1

        # ── Emit Gooey progress ───────────────────────────────────────────────
        # Overall percentage = (image progress) + (size progress within image)
        # image_base: how far through all images we are at the START of this image
        # size fraction: progress within the current image across all sizes
        image_base_pct = int((image_index - 1) / image_total * 100)
        image_span_pct = int(1 / image_total * 100)
        size_base_frac = (size_index - 1) / size_total
        size_span_frac = 1 / size_total
        row_frac = (row + 1) / num_rows
        within_image_pct = (size_base_frac + size_span_frac * row_frac) * image_span_pct
        pct = min(99, int(image_base_pct + within_image_pct))
        print(f"progress: {pct}")
        sys.stdout.flush()

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Per-image entry point
# ─────────────────────────────────────────────────────────────────────────────
def process_image(
    image_path: str,
    output_folder: str,
    selected_specs: List[TileSpec],
    fallback_inches: float,
    force: bool,
    dry_run: bool,
    image_index: int,
    image_total: int,
) -> List[TileResult]:
    results: List[TileResult] = []
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    safe_base = safe_str(base_name)

    print(f"\n{'='*60}")
    print(f"Image {image_index}/{image_total}: {safe_base}")
    sys.stdout.flush()

    try:
        im = Image.open(image_path)
    except Exception as exc:
        log.error("  Cannot open file: %s", exc)
        return [TileResult(source=safe_base, size_label="*", error=str(exc))]

    meta = parse_pattern_metadata(base_name)
    log.info("  Output prefix: %s", meta.prefix)

    try:
        factor = compute_scale(im, meta, fallback_inches)
    except ValueError as exc:
        log.error("  Scale error: %s", exc)
        return [TileResult(source=safe_base, size_label="*", error=str(exc))]

    orientation = "portrait" if im.width <= im.height else "landscape"
    log.info("  Orientation: %s  (%dx%d px)", orientation, im.width, im.height)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_folder = os.path.join(output_folder, f"{meta.prefix}_{timestamp}")

    size_total = len(selected_specs)
    for size_index, spec in enumerate(selected_specs, start=1):
        size_folder = os.path.join(base_folder, spec.token)
        r = process_one_size(
            im, meta, factor, orientation, spec,
            size_folder, force, dry_run,
            image_index, image_total,
            size_index, size_total,
        )
        results.append(r)
        log.info(
            "  [%s] done — wrote %d  skipped %d",
            spec.label, r.tiles_written, r.tiles_skipped,
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Batch processor
# ─────────────────────────────────────────────────────────────────────────────
def process_folder(
    input_folder: str,
    output_folder: str,
    selected_specs: List[TileSpec],
    fallback_inches: float,
    force: bool,
    dry_run: bool,
    workers: int,
) -> None:

    # ── Validate folders ──────────────────────────────────────────────────────
    input_folder = sanitize_path(input_folder)
    output_folder = sanitize_path(output_folder)

    if not os.path.isdir(input_folder):
        log.error("Input folder does not exist: %s", safe_str(input_folder))
        return

    try:
        os.makedirs(output_folder, exist_ok=True)
    except OSError as exc:
        log.error("Cannot create output folder: %s", exc)
        return

    # ── Pre-flight: fix bad filenames ─────────────────────────────────────────
    print("\nPre-flight: scanning for problematic filenames...")
    sys.stdout.flush()
    renames = preflight_rename(input_folder)
    if renames:
        print(f"  Renamed {len(renames)} file(s) with unsafe characters:")
        for orig, safe in renames:
            print(f"    {safe_str(orig)}  ->  {safe_str(safe)}")
    else:
        print("  All filenames are clean.")
    sys.stdout.flush()

    # ── Collect images ────────────────────────────────────────────────────────
    image_paths = [
        os.path.join(input_folder, f)
        for f in sorted(os.listdir(input_folder))
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]

    if not image_paths:
        log.warning("No supported image files found in: %s", safe_str(input_folder))
        return

    image_total = len(image_paths)
    size_total = len(selected_specs)
    total_tiles_estimate = "unknown"  # we don't know yet without opening images

    print(f"\nFound {image_total} image(s)  x  {size_total} size(s) selected")
    if dry_run:
        print("[DRY RUN MODE — no files will be written]")
    sys.stdout.flush()

    # ── Process ───────────────────────────────────────────────────────────────
    all_results: List[TileResult] = []
    start = time.time()

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    process_image,
                    p, output_folder, selected_specs,
                    fallback_inches, force, dry_run,
                    idx, image_total,
                ): p
                for idx, p in enumerate(image_paths, start=1)
            }
            for fut in as_completed(futures):
                try:
                    all_results.extend(fut.result())
                except Exception as exc:
                    log.error("Worker error: %s", exc)
                    log.debug(traceback.format_exc())
    else:
        for idx, p in enumerate(image_paths, start=1):
            all_results.extend(
                process_image(
                    p, output_folder, selected_specs,
                    fallback_inches, force, dry_run,
                    idx, image_total,
                )
            )

    elapsed = time.time() - start

    # ── Final progress = 100 ─────────────────────────────────────────────────
    print("progress: 100")
    sys.stdout.flush()

    # ── Summary ───────────────────────────────────────────────────────────────
    total_written = sum(r.tiles_written for r in all_results)
    total_skipped = sum(r.tiles_skipped for r in all_results)
    errors = [r for r in all_results if r.error]

    print(f"\n{'='*60}")
    print(f"COMPLETE in {elapsed:.1f}s")
    print(f"  Tiles written : {total_written}")
    print(f"  Tiles skipped : {total_skipped}")
    print(f"  Errors        : {len(errors)}")
    if errors:
        print("\nERRORS:")
        for r in errors:
            print(f"  {safe_str(r.source)} [{r.size_label}]: {safe_str(r.error)}")
    sys.stdout.flush()

    # ── CSV report ────────────────────────────────────────────────────────────
    if not dry_run:
        report_path = os.path.join(
            output_folder,
            f"run_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        )
        try:
            with open(report_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["source", "size_label", "tiles_written", "tiles_skipped", "error"],
                )
                writer.writeheader()
                for r in all_results:
                    writer.writerow({
                        "source": r.source,
                        "size_label": r.size_label,
                        "tiles_written": r.tiles_written,
                        "tiles_skipped": r.tiles_skipped,
                        "error": r.error,
                    })
            print(f"\nRun report: {safe_str(report_path)}")
        except OSError as exc:
            log.warning("Could not write run report: %s", exc)
        sys.stdout.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Gooey GUI
# ─────────────────────────────────────────────────────────────────────────────
@Gooey(
    program_name="LVT Tile Cutter v5",
    show_success_modal=True,
    clear_before_run=True,
    default_size=(1000, 860),
    # Match bare integer on its own line: "progress: 42"
    progress_regex=r"^progress:\s*(\d+)$",
    progress_expr="x",
    # Keep the Gooey console visible during the run
    hide_progress_msg=False,
    timing_options={"show_time_remaining": True, "hide_time_remaining_on_complete": True},
)
def main() -> None:
    parser = GooeyParser(
        description=(
            "Cut LVT slab images into tile-sized PNGs (sRGB, 300 DPI).\n"
            "Filenames with special characters are automatically renamed before processing."
        )
    )

    # ── I/O ──────────────────────────────────────────────────────────────────
    io_group = parser.add_argument_group("Input / Output")
    io_group.add_argument(
        "input_folder", widget="DirChooser",
        help="Folder containing slab images to process",
    )
    io_group.add_argument(
        "output_folder", widget="DirChooser",
        help="Root folder where tile subfolders will be created",
    )

    # ── Scale ─────────────────────────────────────────────────────────────────
    scale_group = parser.add_argument_group(
        "Scale",
        "Slab size is read from the filename (e.g. '48in-Slab'). "
        "The fallback is used only when no inch value is found in the filename.",
    )
    scale_group.add_argument(
        "--fallback_slab_inches",
        type=float, default=48.0, metavar="inches",
        help="Fallback short-side slab size in inches (default: 48)",
    )

    # ── Options ───────────────────────────────────────────────────────────────
    opt_group = parser.add_argument_group("Options")
    opt_group.add_argument(
        "--workers", type=int, default=1, metavar="N",
        help="Parallel worker processes. Keep at 1 for live progress bar. "
             "Increase only for unattended batch runs.",
    )
    opt_group.add_argument(
        "--force", action="store_true", default=False,
        help="Overwrite tiles that already exist (default: skip existing)",
    )
    opt_group.add_argument(
        "--dry_run", action="store_true", default=False,
        help="Preview mode: plan tiles and print counts without writing any files",
    )

    # ── Tile sizes ────────────────────────────────────────────────────────────
    tile_group = parser.add_argument_group(
        "Tile Sizes",
        "Check the sizes you want to generate",
        gooey_options={"columns": 3},
    )
    for spec in ALL_TILE_SPECS:
        tile_group.add_argument(
            f"--{spec.flag}",
            action="store_true", default=True,
            help="",
            gooey_options={"label": spec.label},
        )

    args = parser.parse_args()

    selected_specs = [
        spec for spec in ALL_TILE_SPECS if getattr(args, spec.flag, False)
    ]
    if not selected_specs:
        print("ERROR: No tile sizes selected. Please check at least one size and try again.")
        sys.exit(1)

    process_folder(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        selected_specs=selected_specs,
        fallback_inches=args.fallback_slab_inches,
        force=args.force,
        dry_run=args.dry_run,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
