"""Martian Terrain & Hazard Map Generator (Single Source of Truth).

Processes elevation rasters (1.0 m/pixel HiRISE DTM GeoTIFFs or a clearly labeled
synthetic test layout) into terrain and hazard rasters consumed directly by
`martian_eva_router.py`:
    1. Slope raster (degrees) via central finite differences.
    2. Surface micro-roughness raster via 3x3 Terrain Ruggedness Index (TRI).
    3. Local obstacle prominence via 3x3 Rock Protrusion Index (positive TPI,
       measuring elevation above the immediate 8-neighbor mean in meters).
    4. Deterministic 1-pixel safety buffer around missing (NaN/NoData) cells
       without artificially blocking valid map border pixels.
    5. 4-Tier Hazard Classification map (0=Nominal, 1=Caution, 2=High-Cost,
       3=No-Go) and continuous traversal cost multiplier grid (`np.inf` = No-Go).

Outputs:
    - `hazard_layers.npz`: Self-contained compressed archive read by the router.
    - `hazard_manifest.json`: Audit summary of assumptions, bounds, and tier coverage.
    - `terrain_hazard_map.png`: 3-panel visual diagnostic (Hillshade DEM, Slope,
      and 4-Tier Hazard & Obstacle Map).

Example:
    Generate hazard layers before running the route engine:
        $ python3 terrain_hazard_mapper.py
"""

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, Optional, Tuple
import numpy as np

# Optional GIS and visualization libraries
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

try:
    from matplotlib.colors import BoundaryNorm, LightSource, ListedColormap
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ==============================================================================
# LOGGING CONFIGURATION (MOBILE-FRIENDLY TERMINAL & FILE TELEMETRY)
# ==============================================================================
def setup_logger(log_file: str = "hazard_mapper.log", level: int = logging.INFO) -> logging.Logger:
    """Configures a dual-output logger for terminal and mobile log inspection.

    Args:
        log_file: Destination file path for persistent execution logs.
        level: Logging severity threshold.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger("MartianHazardMapper")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(funcName)s:%(lineno)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


logger = setup_logger()


# ==============================================================================
# HAZARD THRESHOLDS & MISSION ASSUMPTIONS
# ==============================================================================
TIER_NOMINAL: int = 0
TIER_CAUTION: int = 1
TIER_HIGH_COST: int = 2
TIER_NO_GO: int = 3


@dataclass
class HazardThresholds:
    """Physical terrain thresholds separating Nominal, Caution, High-Cost, and No-Go zones.

    Note:
        All slope, roughness, and rock protrusion index limits below are
        engineering assumptions calibrated for a suited astronaut on Mars and
        must be cited as assumptions in mission documentation.

    Attributes:
        resolution_m: Spatial pixel size in meters/pixel (default 1.0 m for HiRISE DTMs).
        caution_slope_deg: [ASSUMPTION] Slope (deg) where Tier 1 Caution begins.
        high_cost_slope_deg: [ASSUMPTION] Slope (deg) where Tier 2 High-Cost begins.
        no_go_slope_deg: [ASSUMPTION] Hard slope cutoff (deg) for Tier 3 No-Go cliffs.
        caution_roughness_m: [ASSUMPTION] 3x3 TRI (m) where Tier 1 Caution begins.
        high_cost_roughness_m: [ASSUMPTION] 3x3 TRI (m) where Tier 2 High-Cost begins.
        no_go_roughness_m: [ASSUMPTION] Hard 3x3 TRI cutoff (m) for Tier 3 No-Go terrain.
        high_cost_protrusion_m: [ASSUMPTION] Rock Protrusion Index (positive 3x3 TPI
            in meters above 8-neighbor mean) flagged as a Tier 2 climbable obstacle.
        no_go_protrusion_m: [ASSUMPTION] Rock Protrusion Index (positive 3x3 TPI in
            meters above 8-neighbor mean) flagged as a Tier 3 impassable rock/ledge.
        despike_hirise_seams: If True, runs a 3x3 median filter before computing
            gradients to suppress ~0.5-1.0 m rectangular stereo correlation artifacts
            known to occur in some HiRISE DTMs.
    """
    resolution_m: float = 1.0
    caution_slope_deg: float = 10.0
    high_cost_slope_deg: float = 15.0
    no_go_slope_deg: float = 22.0
    caution_roughness_m: float = 0.25
    high_cost_roughness_m: float = 0.50
    no_go_roughness_m: float = 0.85
    high_cost_protrusion_m: float = 0.30
    no_go_protrusion_m: float = 0.65
    despike_hirise_seams: bool = False


@dataclass
class HazardMapProduct:
    """Container holding all derived terrain and hazard rasters for the route engine.

    Attributes:
        dem: 2D float64 elevation array (meters).
        slope_deg: 2D float64 terrain slope array (degrees).
        roughness_tri_m: 2D float64 3x3 Terrain Ruggedness Index array (meters).
        rock_protrusion_index_m: 2D float64 positive 3x3 Topographic Position Index
            (meters above 8-neighbor mean), highlighting local rock protrusions.
        hazard_tier: 2D uint8 array with values {0: Nominal, 1: Caution,
            2: High-Cost, 3: No-Go}.
        cost_multiplier: 2D float64 traversal cost multiplier grid (>= 1.0 for
            passable terrain, `np.inf` for Tier 3 No-Go cells).
        no_go_mask: 2D boolean mask (True where `hazard_tier == 3`).
        resolution_m: Pixel spacing in meters.
        no_go_slope_deg: Hard slope cutoff (deg) used when building the hazard map.
        is_synthetic_layout: True if generated from synthetic terrain.
        source_label: Dataset or synthetic generator identifier.
        crs_wkt: Coordinate Reference System string.
        geo_origin_xy: Top-left (Easting, Northing) map coordinate in meters.
    """
    dem: np.ndarray
    slope_deg: np.ndarray
    roughness_tri_m: np.ndarray
    rock_protrusion_index_m: np.ndarray
    hazard_tier: np.ndarray
    cost_multiplier: np.ndarray
    no_go_mask: np.ndarray
    resolution_m: float
    no_go_slope_deg: float
    is_synthetic_layout: bool
    source_label: str
    crs_wkt: str
    geo_origin_xy: Tuple[float, float]


# ==============================================================================
# TERRAIN & HAZARD MAP BUILDER
# ==============================================================================
class TerrainHazardMapper:
    """Transforms raw elevation data into classified slope, obstacle, and cost maps."""

    def __init__(self, thresholds: Optional[HazardThresholds] = None) -> None:
        """Initializes the mapper with physical hazard thresholds.

        Args:
            thresholds: Optional HazardThresholds; uses calibrated defaults if None.
        """
        self.thresholds = thresholds or HazardThresholds()

    def build_from_dem(
        self,
        dem: np.ndarray,
        is_synthetic_layout: bool = False,
        source_label: str = "UNVERIFIED_DTM",
        crs_wkt: str = "LOCAL_MARS_METRIC",
        geo_origin_xy: Tuple[float, float] = (0.0, 0.0),
    ) -> HazardMapProduct:
        """Computes slope, rock protrusion index, hazard tiers, and cost multipliers.

        Args:
            dem: 2D NumPy elevation array in meters.
            is_synthetic_layout: Flag marking whether the DEM is synthetic.
            source_label: Name of the source file or synthetic scenario.
            crs_wkt: Coordinate Reference System identifier.
            geo_origin_xy: Map origin (Easting, Northing) in meters.

        Returns:
            HazardMapProduct containing all derived rasters and metadata.

        Raises:
            ValueError: If `dem` is not a non-empty 2D array.
        """
        if dem.ndim != 2 or dem.size == 0:
            logger.error("Invalid DEM shape passed to mapper: %s", getattr(dem, "shape", None))
            raise ValueError("DEM must be a non-empty 2D NumPy array.")

        raw_dem = dem.astype(np.float64)
        rows, cols = raw_dem.shape

        # Optional 3x3 median despike filter for HiRISE stereo patch seam artifacts
        if self.thresholds.despike_hirise_seams:
            logger.info("Applying 3x3 median filter to suppress HiRISE stereo seam boxes.")
            clean_dem = self._median_filter_3x3(raw_dem)
        else:
            clean_dem = raw_dem

        logger.info(
            "Building hazard layers [%s] | SyntheticLayout=%s | Shape=%dx%d px (%.2f m/px)",
            source_label,
            is_synthetic_layout,
            rows,
            cols,
            self.thresholds.resolution_m,
        )

        # 1. Deterministic 1-pixel buffer around missing/NaN cells (edges remain valid)
        nodata_buffer = self._compute_nodata_buffer(clean_dem)

        # 2. Compute continuous terrain derivatives
        slope_deg = self._compute_slope_deg(clean_dem)
        roughness_tri_m = self._compute_roughness_tri(clean_dem)
        rock_protrusion_index_m = self._compute_rock_protrusion_index(clean_dem)

        # 3. Classify into discrete Hazard Tiers (0..3) and continuous Cost Multipliers
        hazard_tier, cost_multiplier = self._classify_hazards(
            slope_deg=slope_deg,
            roughness_tri_m=roughness_tri_m,
            rock_protrusion_index_m=rock_protrusion_index_m,
            nodata_buffer=nodata_buffer,
        )

        no_go_mask = hazard_tier == TIER_NO_GO
        self._log_coverage_stats(hazard_tier, rock_protrusion_index_m)

        return HazardMapProduct(
            dem=clean_dem,
            slope_deg=slope_deg,
            roughness_tri_m=roughness_tri_m,
            rock_protrusion_index_m=rock_protrusion_index_m,
            hazard_tier=hazard_tier,
            cost_multiplier=cost_multiplier,
            no_go_mask=no_go_mask,
            resolution_m=self.thresholds.resolution_m,
            no_go_slope_deg=self.thresholds.no_go_slope_deg,
            is_synthetic_layout=is_synthetic_layout,
            source_label=source_label,
            crs_wkt=crs_wkt,
            geo_origin_xy=geo_origin_xy,
        )

    def build_from_geotiff(self, filepath: str) -> HazardMapProduct:
        """Loads a HiRISE DTM GeoTIFF (e.g., PDS 'DTEEC_*' product) and builds hazard maps.

        Args:
            filepath: Path to the metric GeoTIFF elevation raster.

        Returns:
            Populated HazardMapProduct.

        Raises:
            ImportError: If `rasterio` is not installed.
            FileNotFoundError: If `filepath` does not exist.
            ValueError: If raster elevation units are not meters.
        """
        if not HAS_RASTERIO:
            logger.critical("rasterio is required to load HiRISE GeoTIFF rasters.")
            raise ImportError("Install 'rasterio' to ingest HiRISE GeoTIFF files.")

        path = Path(filepath)
        if not path.exists():
            logger.error("GeoTIFF not found: %s", filepath)
            raise FileNotFoundError(f"GeoTIFF not found: {filepath}")

        logger.info("Loading HiRISE DTM GeoTIFF: %s", filepath)
        with rasterio.open(path) as ds:
            dem = ds.read(1).astype(np.float64)
            if ds.nodata is not None:
                dem = np.where(dem == ds.nodata, np.nan, dem)

            units = ds.units[0] if ds.units else None
            if units and units.lower() not in ("m", "meter", "meters", "metre", "metres"):
                logger.error("Non-metric DTM vertical units detected: %s", units)
                raise ValueError(f"Expected vertical units in meters, got: {units}")

            self.thresholds.resolution_m = abs(float(ds.transform.a))
            crs_str = ds.crs.to_string() if ds.crs else "UNKNOWN_CRS"
            origin_xy = (float(ds.transform.c), float(ds.transform.f))

        return self.build_from_dem(
            dem=dem,
            is_synthetic_layout=False,
            source_label=path.name,
            crs_wkt=crs_str,
            geo_origin_xy=origin_xy,
        )

    def build_synthetic_mount_sharp(self, rows: int = 350, cols: int = 450) -> HazardMapProduct:
        """Generates a clearly labeled synthetic Mount Sharp layout for pipeline testing.

        Includes:
            - Gentle regional Mount Sharp flank slope (~3.4 deg).
            - Graded flat lander apron (< 2 deg) around (row=45, col=60).
            - Fractured boxwork ridge network (~0.9-1.3 m tall) with eroded gaps
              and ~10 m sandy hollows.
            - ~65 discrete boulder mounds (0.45-1.25 m peak amplitude) to exercise
              the Rock Protrusion Index obstacle detector.
            - Steep-walled Gediz Vallis canyon (> 28 deg walls) on the eastern flank.

        Args:
            rows: Grid height in meters (pixels).
            cols: Grid width in meters (pixels).

        Returns:
            HazardMapProduct marked `is_synthetic_layout=True`.
        """
        logger.info("Generating SYNTHETIC LAYOUT DEM (%dx%d px)...", rows, cols)
        rng = np.random.default_rng(seed=42)
        y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)

        # 1. Baseline regional northward downslope
        dem = 0.06 * y

        # 2. Fractured polygonal boxwork ridge network (cols 55..275)
        ridge_wave_1 = np.abs(np.sin(0.19 * x + 0.12 * y))
        ridge_wave_2 = np.abs(np.cos(0.14 * x - 0.21 * y))
        raw_ridges = 1.15 * np.exp(-8.5 * np.minimum(ridge_wave_1, ridge_wave_2) ** 2)

        # Erosional breaches cutting traversable gaps through ridges
        breach_field = 0.5 * (
            1.0 + np.sin(0.065 * x - 0.045 * y) * np.cos(0.055 * x + 0.075 * y)
        )
        erosion_multiplier = np.where(breach_field < 0.35, 0.20, 1.0)
        boxwork_envelope = np.exp(-((x - 165.0) / 95.0) ** 4)
        dem += raw_ridges * erosion_multiplier * boxwork_envelope

        # 3. Steep-walled Gediz Vallis canyon cutting north-south around col ~365
        channel_center = 365.0 + 14.0 * np.sin(y / 40.0)
        dist_from_channel = np.abs(x - channel_center)
        channel_incision = -16.0 * np.exp(-(dist_from_channel / 20.0) ** 2)
        dem += channel_incision

        # 4. Isolated boulder formations (0.45 to 1.25 m Gaussian mounds)
        rock_rows = rng.integers(25, rows - 25, size=65)
        rock_cols = rng.integers(85, 325, size=65)
        rock_heights = rng.uniform(0.45, 1.25, size=65)
        for rr, cc, hh in zip(rock_rows, rock_cols, rock_heights):
            # Keep clear of the landing pad center
            if np.hypot(rr - 45.0, cc - 60.0) < 28.0:
                continue
            r_min, r_max = max(0, rr - 2), min(rows, rr + 3)
            c_min, c_max = max(0, cc - 2), min(cols, cc + 3)
            sub_y, sub_x = np.mgrid[r_min:r_max, c_min:c_max]
            rock_mound = hh * np.exp(-((sub_y - rr) ** 2 + (sub_x - cc) ** 2) / 1.1)
            dem[r_min:r_max, c_min:c_max] += rock_mound

        # 5. Subtle background sand ripples
        dem += 0.025 * np.sin(0.6 * x) + rng.normal(0.0, 0.012, size=(rows, cols))

        # 6. Grade a flat lander apron (< 2 deg) around Base (row=45, col=60)
        dist_from_base = np.hypot(y - 45.0, x - 60.0)
        pad_blend = np.exp(-(dist_from_base / 18.0) ** 4)
        base_elev = float(dem[45, 60])
        dem = dem * (1.0 - pad_blend) + base_elev * pad_blend

        return self.build_from_dem(
            dem=dem,
            is_synthetic_layout=True,
            source_label="SYNTHETIC_LAYOUT_BOXWORK_GEDIZ_VALLIS",
            crs_wkt="PROJCS[\"Synthetic_Mars_Local_Meters_Unverified_Scale\"]",
            geo_origin_xy=(8136500.0, -283500.0),
        )

    @staticmethod
    def _median_filter_3x3(arr: np.ndarray) -> np.ndarray:
        """Applies a pure-NumPy 3x3 nanmedian filter to remove stereo seam artifacts."""
        padded = np.pad(arr, pad_width=1, mode="edge")
        neighbors = [
            padded[1 + dy : 1 + dy + arr.shape[0], 1 + dx : 1 + dx + arr.shape[1]]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
        return np.nanmedian(np.stack(neighbors, axis=0), axis=0)

    @staticmethod
    def _compute_nodata_buffer(dem: np.ndarray) -> np.ndarray:
        """Builds a 1-pixel (3x3) buffer mask around actual NaN cells (border cells stay valid)."""
        nan_mask = np.isnan(dem)
        rows, cols = nan_mask.shape
        padded = np.pad(nan_mask, pad_width=1, mode="constant", constant_values=False)
        buffered = np.zeros_like(nan_mask, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                buffered |= padded[1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + cols]
        return buffered

    def _compute_slope_deg(self, dem: np.ndarray) -> np.ndarray:
        """Computes local slope angle in degrees using central finite differences."""
        dz_dy, dz_dx = np.gradient(dem, self.thresholds.resolution_m)
        return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))

    @staticmethod
    def _compute_roughness_tri(dem: np.ndarray) -> np.ndarray:
        """Computes 3x3 Terrain Ruggedness Index (RMS elevation difference to 8 neighbors)."""
        rows, cols = dem.shape
        padded = np.pad(dem, pad_width=1, mode="edge")
        sq_diffs = np.zeros_like(dem, dtype=np.float64)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = padded[1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + cols]
                sq_diffs += (dem - neighbor) ** 2

        return np.sqrt(sq_diffs / 8.0)

    @staticmethod
    def _compute_rock_protrusion_index(dem: np.ndarray) -> np.ndarray:
        """Computes the Rock Protrusion Index (positive 3x3 Topographic Position Index).

        Measures each cell's elevation above the mean of its 8 immediate neighbors
        (clamped at >= 0.0 m). Because 1-2 m wide boulders also elevate adjacent
        ring pixels, this index measures local crest prominence rather than true
        base-to-peak rock height.
        """
        rows, cols = dem.shape
        padded = np.pad(dem, pad_width=1, mode="edge")
        neighbor_sum = np.zeros_like(dem, dtype=np.float64)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor_sum += padded[1 + dy : 1 + dy + rows, 1 + dx : 1 + dx + cols]

        neighbor_mean = neighbor_sum / 8.0
        return np.maximum(0.0, dem - neighbor_mean)

    def _classify_hazards(
        self,
        slope_deg: np.ndarray,
        roughness_tri_m: np.ndarray,
        rock_protrusion_index_m: np.ndarray,
        nodata_buffer: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Assigns each cell a discrete Hazard Tier (0-3) and continuous cost multiplier."""
        t = self.thresholds
        rows, cols = slope_deg.shape

        tier = np.full((rows, cols), TIER_NOMINAL, dtype=np.uint8)
        cost = np.ones((rows, cols), dtype=np.float64)

        slope_ratio = np.nan_to_num(slope_deg / t.no_go_slope_deg, nan=1.0)
        rough_ratio = np.nan_to_num(roughness_tri_m / t.no_go_roughness_m, nan=1.0)
        protrusion_ratio = np.nan_to_num(
            rock_protrusion_index_m / t.no_go_protrusion_m, nan=1.0
        )

        # Tier 1: Caution (moderate slopes or undulating ground)
        caution_mask = (slope_deg > t.caution_slope_deg) | (
            roughness_tri_m > t.caution_roughness_m
        )
        tier = np.where(caution_mask, TIER_CAUTION, tier)
        cost = np.where(
            caution_mask,
            1.25 + 1.5 * (slope_ratio ** 2) + 1.2 * rough_ratio,
            cost,
        )

        # Tier 2: High-Cost (steep ridge gaps or climbable rock protrusions)
        high_cost_mask = (
            (slope_deg > t.high_cost_slope_deg)
            | (roughness_tri_m > t.high_cost_roughness_m)
            | (rock_protrusion_index_m > t.high_cost_protrusion_m)
        )
        tier = np.where(high_cost_mask, TIER_HIGH_COST, tier)
        cost = np.where(
            high_cost_mask,
            3.0 + 4.0 * (slope_ratio ** 2) + 3.0 * np.maximum(rough_ratio, protrusion_ratio),
            cost,
        )

        # Tier 3: No-Go (cliffs, canyon walls, high rock protrusions, and NoData buffer)
        no_go_mask = (
            nodata_buffer
            | np.isnan(slope_deg)
            | (slope_deg > t.no_go_slope_deg)
            | (roughness_tri_m > t.no_go_roughness_m)
            | (rock_protrusion_index_m > t.no_go_protrusion_m)
        )
        tier = np.where(no_go_mask, TIER_NO_GO, tier)
        cost = np.where(no_go_mask, np.inf, cost)

        return tier, cost

    def _log_coverage_stats(
        self, hazard_tier: np.ndarray, rock_protrusion_index_m: np.ndarray
    ) -> None:
        """Logs percentage breakdown across hazard tiers and rock protrusion detections."""
        total = float(hazard_tier.size)
        pct_0 = 100.0 * float(np.sum(hazard_tier == TIER_NOMINAL)) / total
        pct_1 = 100.0 * float(np.sum(hazard_tier == TIER_CAUTION)) / total
        pct_2 = 100.0 * float(np.sum(hazard_tier == TIER_HIGH_COST)) / total
        pct_3 = 100.0 * float(np.sum(hazard_tier == TIER_NO_GO)) / total
        no_go_rocks = int(np.sum(rock_protrusion_index_m > self.thresholds.no_go_protrusion_m))

        logger.info(
            "Hazard Classification Complete | Nominal: %.1f%% | Caution: %.1f%% | High-Cost: %.1f%% | No-Go: %.1f%% (%d No-Go rock protrusion cells, max index=%.2f m)",
            pct_0,
            pct_1,
            pct_2,
            pct_3,
            no_go_rocks,
            float(np.nanmax(rock_protrusion_index_m)),
        )


# ==============================================================================
# EXPORT & DIAGNOSTIC VISUALIZATION
# ==============================================================================
def save_hazard_artifacts(
    product: HazardMapProduct,
    thresholds: HazardThresholds,
    npz_path: str = "hazard_layers.npz",
    manifest_path: str = "hazard_manifest.json",
) -> None:
    """Saves self-contained hazard rasters (.npz) and audit metadata (.json) for the router.

    Args:
        product: Computed HazardMapProduct.
        thresholds: HazardThresholds used to generate the layers.
        npz_path: Output path for compressed NumPy layers.
        manifest_path: Output path for JSON metadata and tier statistics.
    """
    np.savez_compressed(
        npz_path,
        dem=product.dem,
        slope_deg=product.slope_deg,
        roughness_tri_m=product.roughness_tri_m,
        rock_protrusion_index_m=product.rock_protrusion_index_m,
        hazard_tier=product.hazard_tier,
        cost_multiplier=product.cost_multiplier,
        no_go_mask=product.no_go_mask,
        resolution_m=np.array([product.resolution_m], dtype=np.float64),
        no_go_slope_deg=np.array([product.no_go_slope_deg], dtype=np.float64),
        geo_origin_xy=np.array(product.geo_origin_xy, dtype=np.float64),
        is_synthetic_layout=np.array([product.is_synthetic_layout], dtype=bool),
        source_label=np.array([product.source_label]),
        crs_wkt=np.array([product.crs_wkt]),
    )
    logger.info("Saved pre-routing hazard layers to %s", npz_path)

    total_px = int(product.hazard_tier.size)
    manifest: Dict[str, Any] = {
        "source_label": product.source_label,
        "is_synthetic_layout": product.is_synthetic_layout,
        "crs": product.crs_wkt,
        "grid_shape_rows_cols": [int(product.dem.shape[0]), int(product.dem.shape[1])],
        "resolution_m": product.resolution_m,
        "geo_origin_xy": list(product.geo_origin_xy),
        "elevation_range_m": [
            round(float(np.nanmin(product.dem)), 3),
            round(float(np.nanmax(product.dem)), 3),
        ],
        "tier_coverage_percent": {
            "tier_0_nominal": round(100.0 * float(np.sum(product.hazard_tier == TIER_NOMINAL)) / total_px, 2),
            "tier_1_caution": round(100.0 * float(np.sum(product.hazard_tier == TIER_CAUTION)) / total_px, 2),
            "tier_2_high_cost": round(100.0 * float(np.sum(product.hazard_tier == TIER_HIGH_COST)) / total_px, 2),
            "tier_3_no_go": round(100.0 * float(np.sum(product.hazard_tier == TIER_NO_GO)) / total_px, 2),
        },
        "thresholds_assumptions": asdict(thresholds),
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved hazard audit manifest to %s", manifest_path)


def save_hazard_preview(
    product: HazardMapProduct, output_png: str = "terrain_hazard_map.png"
) -> None:
    """Renders a 3-panel visual diagnostic of the terrain and classified hazard zones.

    Args:
        product: Computed HazardMapProduct.
        output_png: Destination PNG file path.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed; skipping PNG preview.")
        return

    tag = "[SYNTHETIC LAYOUT] " if product.is_synthetic_layout else ""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), dpi=150)

    # Panel 1: Hillshaded DEM
    ls = LightSource(azdeg=315, altdeg=35)
    dem_filled = np.nan_to_num(product.dem, nan=float(np.nanmin(product.dem)))
    rgb_hillshade = ls.shade(
        dem_filled,
        cmap=plt.cm.terrain,
        vert_exag=1.5,
        blend_mode="overlay",
    )
    axes[0].imshow(rgb_hillshade, origin="upper")
    axes[0].set_title(f"{tag}Shaded Relief DEM (m)")
    axes[0].set_xlabel("East-West (m)")
    axes[0].set_ylabel("North-South (m)")

    # Panel 2: Slope Angle (deg)
    im1 = axes[1].imshow(product.slope_deg, cmap="magma", vmin=0, vmax=25, origin="upper")
    axes[1].set_title(f"{tag}Terrain Slope (deg)")
    axes[1].set_xlabel("East-West (m)")
    fig.colorbar(im1, ax=axes[1], shrink=0.78, label="Slope (deg)")

    # Panel 3: Discrete 4-Tier Hazard Classification
    tier_cmap = ListedColormap(["#1b4332", "#e9c46a", "#f4a261", "#d62828"])
    tier_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], tier_cmap.N)
    im2 = axes[2].imshow(product.hazard_tier, cmap=tier_cmap, norm=tier_norm, origin="upper")
    axes[2].set_title(f"{tag}4-Tier Hazard & Obstacle Map")
    axes[2].set_xlabel("East-West (m)")

    cbar = fig.colorbar(im2, ax=axes[2], ticks=[0, 1, 2, 3], shrink=0.78)
    cbar.ax.set_yticklabels(
        [
            "0: Nominal (Safe)",
            "1: Caution (10-15°)",
            "2: High-Cost (15-22° / Protrusion)",
            "3: No-Go (>22° / Rock Outcrop)",
        ],
        fontsize=8,
    )

    plt.tight_layout()
    plt.savefig(output_png)
    plt.close(fig)
    logger.info("Saved 3-panel hazard visualization to %s", output_png)


# ==============================================================================
# ENTRY POINT (RUN BEFORE MARTIAN_EVA_ROUTER.PY)
# ==============================================================================
if __name__ == "__main__":
    try:
        thresholds = HazardThresholds(
            resolution_m=1.0,
            caution_slope_deg=10.0,
            high_cost_slope_deg=15.0,
            no_go_slope_deg=22.0,
            caution_roughness_m=0.25,
            high_cost_roughness_m=0.50,
            no_go_roughness_m=0.85,
            high_cost_protrusion_m=0.30,
            no_go_protrusion_m=0.65,
            despike_hirise_seams=False,
        )

        mapper = TerrainHazardMapper(thresholds=thresholds)

        # Swap with `mapper.build_from_geotiff("<verified_DTEEC_product>.tif")`
        # once your HiRISE DTM GeoTIFF is downloaded.
        hazard_product = mapper.build_synthetic_mount_sharp(rows=350, cols=450)

        save_hazard_artifacts(
            product=hazard_product,
            thresholds=thresholds,
            npz_path="hazard_layers.npz",
            manifest_path="hazard_manifest.json",
        )
        save_hazard_preview(product=hazard_product, output_png="terrain_hazard_map.png")

    except Exception as exc:
        logger.exception("Fatal error building terrain + hazard map: %s", exc)
        sys.exit(1)