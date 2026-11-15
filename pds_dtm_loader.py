#!/usr/bin/env python3
"""Memory-Mapped PDS3 HiRISE DTM Loader and Equirectangular Coordinate Converter.

Reads raw 32-bit little-endian floating-point elevation rasters (`PC_REAL`) from
NASA PDS3 `.IMG` files without loading the full 360 MB volume into RAM, masks
PDS missing constants to `np.nan`, and converts pixel coordinates into Mars
planetocentric latitude/longitude and equirectangular meters.

Target Product Metadata (from embedded PDS3 label):
    - Product ID: DTEEC_001488_1750_001752_1750_U02
    - Dimensions: 13,375 rows x 6,736 cols (26,944 bytes/record; data at record 2)
    - Resolution: 1.0117467945795 m/pixel (Mars radius = 3,396,190.0 m)

Example:
    Run standalone verification on the default Mount Sharp 500x500 clip:
        $ python3 pds_dtm_loader.py
"""

import argparse
import logging
import math
from pathlib import Path
import sys
from typing import Tuple
import numpy as np


# ==============================================================================
# PDS3 LABEL CONSTANTS FOR DTEEC_001488_1750_001752_1750_U02
# ==============================================================================
EXPECTED_FILE_SIZE_BYTES: int = 360_402_944
RECORD_BYTES: int = 26_944
LABEL_RECORDS: int = 1  # ^IMAGE = 2 -> skip 1 header record (26,944 bytes)
LINES: int = 13_375
LINE_SAMPLES: int = 6_736
PDS_DTYPE: str = "<f4"  # PC_REAL / 32-bit IEEE little-endian float
MISSING_THRESHOLD: float = -1.0e30  # 16#FF7FFFFB# (~ -3.4028e38)
MAX_NAN_FRACTION: float = 0.5  # Reject clips that are more than half missing data

# Equirectangular map projection parameters from PDS3 IMAGE_MAP_PROJECTION
MAP_SCALE_M_PER_PX: float = 1.0117467945795
MAP_RESOLUTION_PX_PER_DEG: float = 58586.494013
LINE_PROJECTION_OFFSET: float = -276559.50000001
SAMPLE_PROJECTION_OFFSET: float = 3577.5000000001
CENTER_LONGITUDE_DEG: float = 137.3
MARS_RADIUS_M: float = 3_396_190.0

# Recommended artifact-free clip on the north flank of Mount Sharp
DEFAULT_CLIP_ROW: int = 4550
DEFAULT_CLIP_COL: int = 4850
DEFAULT_CLIP_SIZE: int = 500
WEST_SEAM_COL_LIMIT: int = 1600


# ==============================================================================
# LOGGING CONFIGURATION (MOBILE-FRIENDLY TERMINAL & FILE TELEMETRY)
# ==============================================================================
def setup_loader_logger(level: int = logging.INFO) -> logging.Logger:
    """Configures compact, line-numbered stdout logging for mobile SSH review.

    Args:
        level: Logging severity threshold.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger("PDSDTMLoader")
    logger.setLevel(level)
    # Print here only; don't also hand messages up to the root logger (avoids duplicate lines)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-7s | %(funcName)s:%(lineno)d | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)
    return logger


# ==============================================================================
# PDS3 LABEL READER, CLIP LOADER & COORDINATE CONVERSIONS
# ==============================================================================
def read_pds_label(img_path: str | Path) -> str:
    """Extracts the ASCII/Latin-1 PDS3 header block preceding the `END` statement.

    Args:
        img_path: Path to the PDS3 `.IMG` file.

    Returns:
        Decoded PDS3 label text.
    """
    with open(img_path, "rb") as f:
        head = f.read(60_000).decode("latin1", errors="replace")
    end_idx = head.find("\nEND")
    return head[:end_idx] if end_idx != -1 else head


def pixel_to_latlon(row: float, col: float) -> Tuple[float, float]:
    """Converts global PDS3 grid `(row, col)` indices to `(latitude, longitude)` in degrees.

    Uses the standard PDS3 Equirectangular projection equations:
        lat = (LINE_PROJECTION_OFFSET - row) / MAP_RESOLUTION
        lon = CENTER_LONGITUDE + (col - SAMPLE_PROJECTION_OFFSET) / MAP_RESOLUTION

    Args:
        row: Zero-based global line index (`0` to `LINES - 1`).
        col: Zero-based global sample index (`0` to `LINE_SAMPLES - 1`).

    Returns:
        Tuple of `(latitude_deg, longitude_deg)` in planetocentric coordinates.
    """
    lat_deg = (LINE_PROJECTION_OFFSET - float(row)) / MAP_RESOLUTION_PX_PER_DEG
    lon_deg = CENTER_LONGITUDE_DEG + (float(col) - SAMPLE_PROJECTION_OFFSET) / MAP_RESOLUTION_PX_PER_DEG
    return lat_deg, lon_deg


def latlon_to_meters_eqc(lat_deg: float, lon_deg: float) -> Tuple[float, float]:
    """Projects `(latitude, longitude)` degrees into Mars equirectangular meters.

    Args:
        lat_deg: Planetocentric latitude in degrees.
        lon_deg: East longitude in degrees.

    Returns:
        Tuple of `(easting_m, northing_m)` using `MARS_RADIUS_M = 3,396,190.0 m`.
    """
    easting_m = math.radians(lon_deg) * MARS_RADIUS_M
    northing_m = math.radians(lat_deg) * MARS_RADIUS_M
    return easting_m, northing_m


def load_clip(
    img_path: str | Path,
    r0: int = DEFAULT_CLIP_ROW,
    c0: int = DEFAULT_CLIP_COL,
    size: int = DEFAULT_CLIP_SIZE,
    cols: int | None = None,
    verify_size: bool = True,
) -> np.ndarray:
    """Memory-maps a PDS3 HiRISE `.IMG` file and extracts a float64 elevation clip.

    Args:
        img_path: Path to `DTEEC_001488_1750_001752_1750_U02.IMG`.
        r0: Starting global row index (default: 4550).
        c0: Starting global column index (default: 4850).
        size: Number of rows to extract (default: 500).
        cols: Number of columns to extract (defaults to `size` for a square clip).
        verify_size: If True, checks that the `.IMG` file matches 360,402,944 bytes.

    Returns:
        2D `np.float64` elevation array of shape `(size, n_cols)` with missing
        PDS3 cells converted to `np.nan`.

    Raises:
        FileNotFoundError: If `img_path` does not exist.
        ValueError: If file size is truncated or clip indices exceed raster bounds.
    """
    logger = setup_loader_logger()
    path = Path(img_path)
    if not path.exists():
        raise FileNotFoundError(
            f"HiRISE DTM file not found at '{path}'. Download it into 'realdata/' first."
        )

    actual_bytes = path.stat().st_size
    if verify_size and actual_bytes != EXPECTED_FILE_SIZE_BYTES:
        raise ValueError(
            f"Incomplete PDS3 file '{path}': expected {EXPECTED_FILE_SIZE_BYTES:,} bytes, "
            f"got {actual_bytes:,} bytes. Re-run curl to complete the download."
        )

    n_rows = int(size)
    n_cols = int(cols) if cols is not None else n_rows

    if r0 < 0 or c0 < 0 or (r0 + n_rows) > LINES or (c0 + n_cols) > LINE_SAMPLES:
        raise ValueError(
            f"Clip [r={r0}:{r0 + n_rows}, c={c0}:{c0 + n_cols}] exceeds DTM bounds "
            f"({LINES} rows x {LINE_SAMPLES} cols)."
        )

    if c0 < WEST_SEAM_COL_LIMIT:
        logger.warning(
            "Clip starts at col %d (< %d): far-west strip contains vertical CCD stitching seams.",
            c0,
            WEST_SEAM_COL_LIMIT,
        )

    # Open memory-mapped view skipping the 1-record (26,944-byte) ASCII label
    dem_memmap = np.memmap(
        path,
        dtype=PDS_DTYPE,
        mode="r",
        offset=RECORD_BYTES * LABEL_RECORDS,
        shape=(LINES, LINE_SAMPLES),
    )

    # Materialize only the requested sub-window into a standard float64 array
    sub = np.array(dem_memmap[r0 : r0 + n_rows, c0 : c0 + n_cols], dtype=np.float64)
    del dem_memmap  # Close mmap handle immediately

    # Replace PDS3 MISSING_CONSTANT (-3.4e38) with IEEE NaN
    sub[sub < MISSING_THRESHOLD] = np.nan

    # Fail early if the clip falls mostly outside the DTM's tilted data footprint
    nan_fraction = float(np.isnan(sub).mean())
    if nan_fraction > MAX_NAN_FRACTION:
        raise ValueError(
            f"Clip [r={r0}:{r0 + n_rows}, c={c0}:{c0 + n_cols}] is {nan_fraction:.0%} missing data "
            "(outside the DTM footprint). Pick a clip further toward the middle of the strip."
        )

    nan_count = int(np.isnan(sub).sum())
    lat_tl, lon_tl = pixel_to_latlon(r0, c0)
    logger.info(
        "Loaded HiRISE clip r%d-%d c%d-%d | Shape: %s | NaNs: %d | Elev: %.2f to %.2f m | TL: %.4f°S, %.4f°E",
        r0,
        r0 + n_rows - 1,
        c0,
        c0 + n_cols - 1,
        sub.shape,
        nan_count,
        float(np.nanmin(sub)),
        float(np.nanmax(sub)),
        abs(lat_tl),
        lon_tl,
    )
    return sub


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify PDS3 HiRISE DTM clip loading and projection.")
    parser.add_argument(
        "--img",
        type=str,
        default="realdata/DTEEC_001488_1750_001752_1750_U02.IMG",
        help="Path to downloaded PDS3 .IMG file.",
    )
    parser.add_argument("--row", type=int, default=DEFAULT_CLIP_ROW, help="Top-left global row.")
    parser.add_argument("--col", type=int, default=DEFAULT_CLIP_COL, help="Top-left global col.")
    parser.add_argument("--size", type=int, default=DEFAULT_CLIP_SIZE, help="Square clip size (px).")
    args = parser.parse_args()

    clip = load_clip(args.img, r0=args.row, c0=args.col, size=args.size)
    lat0, lon0 = pixel_to_latlon(0, 0)
    lat_clip, lon_clip = pixel_to_latlon(args.row, args.col)
    print(f"Grid (0,0) check   : lat={lat0:.4f}°, lon={lon0:.4f}°")
    print(f"Clip TL ({args.row},{args.col}): lat={lat_clip:.4f}°, lon={lon_clip:.4f}°")
    print(
        f"Clip stats         : shape={clip.shape}, nans={np.isnan(clip).sum()}, "
        f"min={np.nanmin(clip):.2f} m, max={np.nanmax(clip):.2f} m"
    )