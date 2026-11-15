#!/usr/bin/env python3
"""Runs `TerrainHazardMapper` on the Calibrated North Mount Sharp HiRISE Clip.

Extracts rows 4550-5049, columns 4850-5349 (500 x 500 px at 1.0117 m/px) from
`DTEEC_001488_1750_001752_1750_U02.IMG`, computes exact Mars equirectangular
coordinates (`4.7982°S, 137.3217°E`), builds the single-source-of-truth hazard
archive (`hazard_layers.npz`), and writes `terrain_hazard_map.png`.

Example:
    $ python3 run_mapper_real.py
    $ python3 run_mapper_real.py --despike-seams --output-dir ./real_run
"""

import argparse
import logging
from pathlib import Path
import sys

from pds_dtm_loader import (
    DEFAULT_CLIP_COL,
    DEFAULT_CLIP_ROW,
    DEFAULT_CLIP_SIZE,
    MAP_SCALE_M_PER_PX,
    latlon_to_meters_eqc,
    load_clip,
    pixel_to_latlon,
)
from terrain_hazard_mapper import (
    HazardThresholds,
    TerrainHazardMapper,
    save_hazard_artifacts,
    save_hazard_preview,
)


def main() -> None:
    """CLI entry point for generating real HiRISE hazard layers."""
    parser = argparse.ArgumentParser(
        description="Build single-source-of-truth hazard layers from real HiRISE PDS3 DTM."
    )
    parser.add_argument(
        "--img",
        type=str,
        default="realdata/DTEEC_001488_1750_001752_1750_U02.IMG",
        help="Path to DTEEC_001488_1750_001752_1750_U02.IMG.",
    )
    parser.add_argument("--row", type=int, default=DEFAULT_CLIP_ROW, help="Top-left global row.")
    parser.add_argument("--col", type=int, default=DEFAULT_CLIP_COL, help="Top-left global col.")
    parser.add_argument("--size", type=int, default=DEFAULT_CLIP_SIZE, help="Square clip size (px).")
    parser.add_argument(
        "--despike-seams",
        action="store_true",
        help="Apply 3x3 median filter to suppress HiRISE stereo box/seam artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for hazard_layers.npz, manifest, and preview PNG (default: '.').",
    )
    args = parser.parse_args()

    # Give this script its own printer instead of logging.basicConfig(). basicConfig
    # adds a printer to the root logger, which made the loader's and mapper's
    # messages print twice (once by their own handler, once by the root).
    logger = logging.getLogger("RunMapperReal")
    logger.setLevel(logging.INFO)
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

    try:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load calibrated 500x500 elevation clip via memory-mapped PDS3 loader
        sub_dem = load_clip(args.img, r0=args.row, c0=args.col, size=args.size)

        # 2. Compute planetocentric lat/lon & equirectangular meter origin
        lat_deg, lon_deg = pixel_to_latlon(args.row, args.col)
        origin_xy = latlon_to_meters_eqc(lat_deg, lon_deg)
        source_label = (
            f"DTEEC_001488_1750_001752_1750_U02 clip "
            f"r{args.row}-{args.row + args.size - 1} c{args.col}-{args.col + args.size - 1}"
        )

        logger.info(
            "Clip Top-Left Coordinate: %.4f°S, %.4f°E | Origin XY: (%.2f m, %.2f m)",
            abs(lat_deg),
            lon_deg,
            origin_xy[0],
            origin_xy[1],
        )

        # 3. Configure thresholds using the true 1.0117467945795 m/px MAP_SCALE
        thresholds = HazardThresholds(
            resolution_m=MAP_SCALE_M_PER_PX,
            despike_hirise_seams=args.despike_seams,
        )
        mapper = TerrainHazardMapper(thresholds=thresholds)

        product = mapper.build_from_dem(
            dem=sub_dem,
            is_synthetic_layout=False,
            source_label=source_label,
            crs_wkt="Mars equirectangular (eqc), R=3396190 m",
            geo_origin_xy=origin_xy,
        )

        # 4. Save single-source-of-truth artifacts & diagnostic map
        save_hazard_artifacts(
            product=product,
            thresholds=thresholds,
            npz_path=str(out_dir / "hazard_layers.npz"),
            manifest_path=str(out_dir / "hazard_manifest.json"),
        )
        save_hazard_preview(
            product=product,
            output_png=str(out_dir / "terrain_hazard_map.png"),
        )
        logger.info("Real HiRISE hazard mapping complete in %s", out_dir.resolve())

    except Exception as exc:
        logger.exception("Real HiRISE hazard mapping failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()