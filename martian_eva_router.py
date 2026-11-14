"""Martian EVA Route Planning Engine for Gale Crater (Mount Sharp / Gediz Vallis).

This module implements a terrain-aware 8-connected A* pathfinding and hazard-modeling
engine for astronaut Extravehicular Activity (EVA) planning in the Mount Sharp
Boxwork and Gediz Vallis corridor.

Data & Scale Disclaimer:
    - Real-terrain mode ingests 1.0 m/pixel HiRISE Digital Terrain Model (DTM)
      GeoTIFF rasters (PDS 'DTEEC_*' products). Exact stereo DTM product IDs
      covering the Mount Sharp boxwork and Gediz Vallis must be verified in the
      University of Arizona HiRISE DTM catalog prior to operational runs.
    - Standalone demonstration mode generates a clearly labeled SYNTHETIC LAYOUT
      (~250 m synthetic separation between boxwork and Gediz Vallis) for
      algorithmic verification only.

Example:
    Run standalone with the synthetic Mount Sharp layout:
        $ python3 martian_eva_router.py
"""

from dataclasses import dataclass, field
import heapq
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

# Optional GIS and plotting dependencies; engine degrades gracefully if absent.
try:
    import rasterio
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False


try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ==============================================================================
# LOGGING CONFIGURATION (MOBILE-FRIENDLY TERMINAL & FILE TELEMETRY)
# ==============================================================================
def setup_logger(log_file: str = "eva_router.log", level: int = logging.INFO) -> logging.Logger:
    """Configures a dual-output logger for terminal and mobile log inspection.

    Formats log lines with compact timestamps, severity levels, function names,
    and line numbers so failures can be diagnosed quickly on a mobile SSH view.

    Args:
        log_file: Destination path for persistent execution logs.
        level: Logging threshold (e.g., logging.INFO, logging.DEBUG).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger("MartianEVARouter")
    logger.setLevel(level)

    # Avoid attaching duplicate handlers if module is re-imported
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
# DATA STRUCTURES & MISSION ASSUMPTIONS
# ==============================================================================
MARS_GRAVITY_MPS2: float = 3.721  # Mars surface gravitational acceleration (m/s^2)


@dataclass
class EVACostConfig:
    """Biomechanical and safety constraints for an astronaut surface EVA.

    Note:
        All physiological and suit parameters below are baseline mission
        planning assumptions and must be documented as such in mission reports.

    Attributes:
        resolution_m: Spatial resolution of the elevation grid in meters/pixel.
        astronaut_total_mass_kg: [ASSUMPTION] Suited astronaut mass (82 kg crew +
            98 kg next-gen Mars EVA suit = 180 kg).
        max_passable_slope_deg: [ASSUMPTION] Hard slope cutoff (deg); cells or
            pixel transitions steeper than this are blocked as cliff/scarp hazards.
        caution_slope_deg: [ASSUMPTION] Slope threshold (deg) above which an
            extra steep-grade penalty discourages A* from using near-limit slopes.
        max_roughness_m: [ASSUMPTION] Hard 3x3 Terrain Ruggedness Index (TRI)
            cutoff in meters representing severe boulder/step trip hazards.
        nominal_walk_speed_mps: [ASSUMPTION] Flat-ground suited walking speed (m/s).
        max_walkback_distance_m: [ASSUMPTION] Maximum safe return path distance (m)
            from any point on the EVA back to the HAB/Lander under suit contingency.
        slope_penalty_weight: [ASSUMPTION] Nonlinear cost weight penalizing steep
            ridge climbs so A* prefers winding through flat sandy hollows.
        roughness_penalty_weight: [ASSUMPTION] Cost weight penalizing uneven ground.
        despike_hirise_seams: If True, applies a 3x3 median filter to suppress
            ~0.5-1.0 m rectangular stereo correlation seam artifacts found in
            some HiRISE DTMs.
    """
    resolution_m: float = 1.0
    astronaut_total_mass_kg: float = 180.0
    max_passable_slope_deg: float = 22.0
    caution_slope_deg: float = 15.0
    max_roughness_m: float = 0.85
    nominal_walk_speed_mps: float = 0.85
    max_walkback_distance_m: float = 1500.0
    slope_penalty_weight: float = 4.5
    roughness_penalty_weight: float = 4.0
    despike_hirise_seams: bool = False


@dataclass
class Waypoint:
    """Represents a named mission coordinate on the DEM grid.

    Attributes:
        name: Human-readable target identifier (e.g., 'Boxwork Vein Node').
        row: Grid row index (Y-axis).
        col: Grid column index (X-axis).
        science_priority: Mission priority tag ('BASE', 'PRIMARY', 'ANOMALY').
    """
    name: str
    row: int
    col: int
    science_priority: str = "PRIMARY"


@dataclass
class LegTelemetry:
    """Computed engineering and safety metrics for a single traverse leg.

    Note on Metabolic Energy Model:
        `estimated_energy_kj` uses a first-order mechanical work model combining
        horizontal suited locomotion (~2.8 J/kg/m) and positive gravitational
        climb work (`m * g_mars * max(0, dz) / 0.25` muscle efficiency). Downhill
        segments (`dz < 0`) contribute zero gravitational climb work, making
        return legs lower energy than outbound climbs. In real pressurized suits,
        eccentric braking on steep descents adds moderate metabolic effort.

    Attributes:
        start_name: Origin waypoint name.
        end_name: Destination waypoint name.
        path_coords: Ordered list of (row, col) grid coordinates.
        distance_m: Total 3D surface path length in meters.
        elevation_gain_m: Cumulative positive vertical climb in meters.
        elevation_loss_m: Cumulative vertical descent in meters.
        max_slope_deg: Steepest single pixel transition encountered on the leg.
        mean_slope_deg: Distance-weighted average slope along the leg.
        steps_over_15deg: Count of grid steps exceeding 15.0 degrees.
        steps_over_20deg: Count of grid steps exceeding 20.0 degrees (ridge gaps).
        steep_distance_over_20deg_m: Total path distance (m) spent above 20.0 deg.
        mean_roughness_m: Average Terrain Ruggedness Index along the leg.
        estimated_energy_kj: Estimated astronaut metabolic energy expenditure (kJ).
        estimated_duration_min: Step-by-step integrated traverse duration (minutes).
    """
    start_name: str
    end_name: str
    path_coords: List[Tuple[int, int]] = field(default_factory=list)
    distance_m: float = 0.0
    elevation_gain_m: float = 0.0
    elevation_loss_m: float = 0.0
    max_slope_deg: float = 0.0
    mean_slope_deg: float = 0.0
    steps_over_15deg: int = 0
    steps_over_20deg: int = 0
    steep_distance_over_20deg_m: float = 0.0
    mean_roughness_m: float = 0.0
    estimated_energy_kj: float = 0.0
    estimated_duration_min: float = 0.0


# ==============================================================================
# TERRAIN INGESTION & HAZARD ANALYSIS
# ==============================================================================
class TerrainAnalyzer:
    """Loads elevation rasters and derives slope, roughness, and hazard masks."""

    def __init__(
        self,
        dem: np.ndarray,
        config: EVACostConfig,
        is_synthetic: bool = False,
        source_label: str = "UNVERIFIED_HIRISE_DTM",
        geo_origin_xy: Tuple[float, float] = (0.0, 0.0),
        crs_wkt: str = "LOCAL_MARS_METRIC",
        raster_transform: Optional[Any] = None,
    ) -> None:
        """Initializes the analyzer with a 2D elevation array and spatial metadata.

        Args:
            dem: 2D NumPy float array of surface elevations in meters.
            config: EVACostConfig containing spatial resolution and thresholds.
            is_synthetic: Flag explicitly marking whether the DEM is synthetic.
            source_label: Human-readable dataset identifier for logs/exports.
            geo_origin_xy: (Easting, Northing) origin in meters for local projection.
            crs_wkt: Coordinate Reference System string or WKT from GeoTIFF.
            raster_transform: Optional rasterio Affine transform for exact mapping.

        Raises:
            ValueError: If the DEM is not a valid non-empty 2D array.
        """
        if dem.ndim != 2 or dem.size == 0:
            logger.error("Invalid DEM dimensions: shape=%s", getattr(dem, "shape", None))
            raise ValueError("DEM must be a non-empty 2D NumPy array.")

        self.config: EVACostConfig = config
        self.is_synthetic: bool = is_synthetic
        self.source_label: str = source_label
        self.geo_origin_xy: Tuple[float, float] = geo_origin_xy
        self.crs_wkt: str = crs_wkt
        self.raster_transform: Optional[Any] = raster_transform

        raw_dem = dem.astype(np.float64)
        self.rows, self.cols = raw_dem.shape

        # Optional 3x3 median filter to suppress HiRISE stereo correlation patch seams
        if self.config.despike_hirise_seams:
            logger.info("Applying 3x3 median despike filter for HiRISE stereo patch seams.")
            self.dem = self._median_filter_3x3(raw_dem)
        else:
            self.dem = raw_dem

        logger.info(
            "TerrainAnalyzer ready [%s] | SyntheticLayout=%s | Grid: %dx%d px (%.2f m/px) | Elev: %.2f to %.2f m",
            self.source_label,
            self.is_synthetic,
            self.rows,
            self.cols,
            self.config.resolution_m,
            float(np.nanmin(self.dem)),
            float(np.nanmax(self.dem)),
        )

        self.slope_deg: np.ndarray = self._compute_slope_deg()
        self.roughness_m: np.ndarray = self._compute_roughness_tri()
        self.impassable_mask: np.ndarray = self._build_impassable_mask()

    def pixel_to_map_xy(self, row: int, col: int) -> Tuple[float, float]:
        """Converts grid (row, col) indices into geospatial (X, Y) coordinates.

        Args:
            row: Grid row index (Y-axis).
            col: Grid column index (X-axis).

        Returns:
            Tuple of (X_easting, Y_northing) coordinates in map units (meters).
        """
        if HAS_RASTERIO and self.raster_transform is not None:
            x_geo, y_geo = rasterio.transform.xy(self.raster_transform, row, col)
            return float(x_geo), float(y_geo)

        x_m = self.geo_origin_xy[0] + col * self.config.resolution_m
        y_m = self.geo_origin_xy[1] - row * self.config.resolution_m
        return float(x_m), float(y_m)

    @classmethod
    def from_geotiff(cls, filepath: str, config: EVACostConfig) -> "TerrainAnalyzer":
        """Loads a HiRISE DTM GeoTIFF raster (e.g., PDS 'DTEEC_*' product) from disk.

        Args:
            filepath: Path to the GeoTIFF elevation raster.
            config: EVACostConfig instance.

        Returns:
            Initialized TerrainAnalyzer instance with real geospatial metadata.

        Raises:
            ImportError: If rasterio is not installed.
            FileNotFoundError: If the GeoTIFF path does not exist.
            ValueError: If raster vertical units appear non-metric.
        """
        if not HAS_RASTERIO:
            logger.critical("Attempted GeoTIFF load without rasterio installed.")
            raise ImportError("Install 'rasterio' to read HiRISE GeoTIFF files.")

        path = Path(filepath)
        if not path.exists():
            logger.error("GeoTIFF file not found at path: %s", filepath)
            raise FileNotFoundError(f"GeoTIFF not found: {filepath}")

        logger.info("Reading HiRISE DTM GeoTIFF: %s", filepath)
        with rasterio.open(path) as dataset:
            dem = dataset.read(1).astype(np.float64)
            if dataset.nodata is not None:
                dem = np.where(dem == dataset.nodata, np.nan, dem)

            # Verify linear units are meters (standard for HiRISE DTEEC DTMs)
            units = dataset.units[0] if dataset.units else None
            if units and units.lower() not in ("m", "meter", "meters", "metre", "metres"):
                logger.error("Unsupported DTM vertical units: %s (expected meters)", units)
                raise ValueError(f"DTM vertical units must be meters, got: {units}")

            config.resolution_m = abs(float(dataset.transform.a))
            crs_str = dataset.crs.to_string() if dataset.crs else "UNKNOWN_CRS"

            return cls(
                dem=dem,
                config=config,
                is_synthetic=False,
                source_label=path.name,
                crs_wkt=crs_str,
                raster_transform=dataset.transform,
            )

    @classmethod
    def from_synthetic_mount_sharp(
        cls, rows: int = 350, cols: int = 450, config: Optional[EVACostConfig] = None
    ) -> "TerrainAnalyzer":
        """Generates a synthetic layout of the Boxwork + Gediz Vallis corridor.

        Note:
            This is a SYNTHETIC LAYOUT for route-engine verification only. On
            this synthetic grid, the boxwork zone and Gediz Vallis rim sit ~250 m
            apart; actual separation must be measured from a verified HiRISE DTM.

        Args:
            rows: Grid height in meters (pixels).
            cols: Grid width in meters (pixels).
            config: Optional EVACostConfig; defaults to 1.0 m/px standard config.

        Returns:
            Initialized TerrainAnalyzer explicitly marked `is_synthetic=True`.
        """
        cfg = config or EVACostConfig(resolution_m=1.0)
        logger.info(
            "Generating SYNTHETIC LAYOUT (Mount Sharp Boxwork + Gediz Vallis, %dx%d)...",
            rows,
            cols,
        )

        rng = np.random.default_rng(seed=42)
        y, x = np.mgrid[0:rows, 0:cols].astype(np.float64)

        # 1. Gentle regional slope rising toward the south (Mount Sharp flank)
        dem = 0.06 * y

        # 2. Fractured polygonal boxwork ridge network (cols 55..275)
        # Erosional breaches cut gaps through the ridges so hollows remain connected.
        ridge_wave_1 = np.abs(np.sin(0.19 * x + 0.12 * y))
        ridge_wave_2 = np.abs(np.cos(0.14 * x - 0.21 * y))
        raw_ridges = 1.15 * np.exp(-8.5 * np.minimum(ridge_wave_1, ridge_wave_2) ** 2)

        breach_field = 0.5 * (
            1.0 + np.sin(0.065 * x - 0.045 * y) * np.cos(0.055 * x + 0.075 * y)
        )
        erosion_multiplier = np.where(breach_field < 0.35, 0.20, 1.0)

        boxwork_envelope = np.exp(-((x - 165.0) / 95.0) ** 4)
        dem += raw_ridges * erosion_multiplier * boxwork_envelope

        # 3. Gediz Vallis Canyon cutting north-south around col ~365
        channel_center = 365.0 + 14.0 * np.sin(y / 40.0)
        dist_from_channel = np.abs(x - channel_center)
        channel_incision = -16.0 * np.exp(-(dist_from_channel / 20.0) ** 2)
        dem += channel_incision

        # 4. Controlled boulder/talus micro-roughness along the Gediz Vallis rim
        rim_zone = np.exp(-((dist_from_channel - 28.0) / 10.0) ** 2)
        boulder_noise = rng.normal(0.0, 0.14, size=(rows, cols)) * rim_zone
        micro_ripples = 0.03 * np.sin(0.6 * x) + rng.normal(0.0, 0.015, size=(rows, cols))
        dem += boulder_noise + micro_ripples

        # 5. Grade a realistic flat lander pad (< 2 deg) around Base (row=45, col=60)
        dist_from_base = np.hypot(y - 45.0, x - 60.0)
        pad_blend = np.exp(-(dist_from_base / 18.0) ** 4)
        base_target_elev = float(dem[45, 60])
        dem = dem * (1.0 - pad_blend) + base_target_elev * pad_blend

        return cls(
            dem=dem,
            config=cfg,
            is_synthetic=True,
            source_label="SYNTHETIC_LAYOUT_BOXWORK_GEDIZ_VALLIS",
            geo_origin_xy=(8136500.0, -283500.0),
            crs_wkt="PROJCS[\"Synthetic_Mars_Local_Meters_Unverified_Scale\"]",
        )

    @staticmethod
    def _median_filter_3x3(arr: np.ndarray) -> np.ndarray:
        """Applies a pure-NumPy 3x3 nanmedian filter to remove isolated seam spikes.

        Args:
            arr: 2D elevation array.

        Returns:
            Filtered 2D elevation array.
        """
        padded = np.pad(arr, pad_width=1, mode="edge")
        neighbors = [
            padded[1 + dy : 1 + dy + arr.shape[0], 1 + dx : 1 + dx + arr.shape[1]]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
        stacked = np.stack(neighbors, axis=0)
        return np.nanmedian(stacked, axis=0)

    def _compute_slope_deg(self) -> np.ndarray:
        """Computes terrain slope in degrees using central finite differences.

        Returns:
            2D array of terrain slopes in degrees.
        """
        dz_dy, dz_dx = np.gradient(self.dem, self.config.resolution_m)
        slope_rad = np.arctan(np.hypot(dz_dx, dz_dy))
        return np.degrees(slope_rad)

    def _compute_roughness_tri(self) -> np.ndarray:
        """Computes a 3x3 Terrain Ruggedness Index (TRI) in pure NumPy.

        Returns:
            2D array of root-mean-square neighbor elevation differences in meters.
        """
        padded = np.pad(self.dem, pad_width=1, mode="edge")
        sq_diffs = np.zeros_like(self.dem, dtype=np.float64)

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                neighbor = padded[1 + dy : 1 + dy + self.rows, 1 + dx : 1 + dx + self.cols]
                sq_diffs += (self.dem - neighbor) ** 2

        return np.sqrt(sq_diffs / 8.0)

    def _build_impassable_mask(self) -> np.ndarray:
        """Identifies grid cells violating safety rules, including a 1-px NoData buffer.

        Returns:
            Boolean 2D mask where True indicates impassable/no-go terrain.
        """
        nan_mask = np.isnan(self.dem) | np.isnan(self.slope_deg) | np.isnan(self.roughness_m)

        # Explicitly dilate NoData gaps by 1 pixel (3x3) so routes never graze missing data
        padded_nan = np.pad(nan_mask, pad_width=1, mode="constant", constant_values=True)
        nodata_buffer = np.zeros_like(nan_mask, dtype=bool)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                nodata_buffer |= padded_nan[1 + dy : 1 + dy + self.rows, 1 + dx : 1 + dx + self.cols]

        steep_mask = self.slope_deg > self.config.max_passable_slope_deg
        rough_mask = self.roughness_m > self.config.max_roughness_m
        impassable = nodata_buffer | steep_mask | rough_mask

        blocked_pct = 100.0 * float(np.mean(impassable))
        logger.info("Hazard mask built | Impassable terrain: %.1f%% of map area", blocked_pct)
        return impassable


# ==============================================================================
# A* ROUTE PLANNING ENGINE
# ==============================================================================
class MartianEVARouter:
    """Terrain-aware 8-connected A* route planner for Martian surface EVAs."""

    _NEIGHBORS: Tuple[Tuple[int, int, float], ...] = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, 1.41421356237),
        (-1, 1, 1.41421356237),
        (1, -1, 1.41421356237),
        (1, 1, 1.41421356237),
    )

    def __init__(self, terrain: TerrainAnalyzer) -> None:
        """Initializes the router with an analyzed terrain model.

        Args:
            terrain: TerrainAnalyzer containing DEM, slope, roughness, and hazard mask.
        """
        self.terrain = terrain
        self.config = terrain.config

    def snap_to_safe_cell(self, wp: Waypoint, search_radius_px: int = 8) -> Waypoint:
        """Returns the waypoint if safe, or snaps to the nearest passable cell within radius.

        Args:
            wp: Candidate Waypoint.
            search_radius_px: Maximum pixel radius to search for safe ground.

        Returns:
            Validated (and possibly snapped) Waypoint.

        Raises:
            ValueError: If out of bounds or no safe ground exists within radius.
        """
        if not (0 <= wp.row < self.terrain.rows and 0 <= wp.col < self.terrain.cols):
            msg = f"Waypoint '{wp.name}' ({wp.row}, {wp.col}) is outside DEM bounds."
            logger.error(msg)
            raise ValueError(msg)

        if not self.terrain.impassable_mask[wp.row, wp.col]:
            return wp

        best_cell: Optional[Tuple[int, int]] = None
        best_score = np.inf

        for dr in range(-search_radius_px, search_radius_px + 1):
            for dc in range(-search_radius_px, search_radius_px + 1):
                nr, nc = wp.row + dr, wp.col + dc
                if not (0 <= nr < self.terrain.rows and 0 <= nc < self.terrain.cols):
                    continue
                if self.terrain.impassable_mask[nr, nc]:
                    continue
                dist = np.hypot(dr, dc)
                score = dist + 0.15 * self.terrain.slope_deg[nr, nc]
                if score < best_score:
                    best_score = score
                    best_cell = (nr, nc)

        if best_cell is None:
            msg = (
                f"Waypoint '{wp.name}' ({wp.row}, {wp.col}) sits on impassable ground and "
                f"no safe cell was found within {search_radius_px} px."
            )
            logger.error(msg)
            raise ValueError(msg)

        logger.warning(
            "Snapped waypoint '%s' from (%d,%d) to safer ground (%d,%d) [slope=%.1f deg]",
            wp.name,
            wp.row,
            wp.col,
            best_cell[0],
            best_cell[1],
            self.terrain.slope_deg[best_cell[0], best_cell[1]],
        )
        return Waypoint(
            name=wp.name,
            row=best_cell[0],
            col=best_cell[1],
            science_priority=wp.science_priority,
        )

    def _heuristic(self, r1: int, c1: int, r2: int, c2: int) -> float:
        """Admissible octile-distance heuristic scaled to minimum flat-ground cost.

        Args:
            r1: Current row index.
            c1: Current column index.
            r2: Goal row index.
            c2: Goal column index.

        Returns:
            Lower-bound remaining traversal cost in meters.
        """
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        octile_steps = (dr + dc) + (1.41421356237 - 2.0) * min(dr, dc)
        return octile_steps * self.config.resolution_m

    def _step_walk_speed_mps(self, signed_slope_deg: float, roughness_m: float) -> float:
        """Computes local astronaut walking speed (m/s) for a single terrain step.

        Args:
            signed_slope_deg: Signed slope in degrees (+ uphill, - downhill).
            roughness_m: Local 3x3 Terrain Ruggedness Index in meters.

        Returns:
            Effective walking speed in meters per second.
        """
        slope_penalty = np.exp(-2.8 * abs(np.radians(signed_slope_deg + 3.0)))
        rough_penalty = max(0.45, 1.0 - 0.5 * (roughness_m / self.config.max_roughness_m))
        speed = self.config.nominal_walk_speed_mps * slope_penalty * rough_penalty
        return float(max(0.25, speed))

    def _step_cost(self, r1: int, c1: int, r2: int, c2: int, dist_mult: float) -> float:
        """Computes traversal cost between adjacent cells incorporating slope & roughness.

        Applies an additional steep-grade surcharge above `caution_slope_deg` (15 deg)
        so A* detours around near-limit slopes unless required to cross a ridge gap.

        Args:
            r1: Origin row index.
            c1: Origin column index.
            r2: Destination row index.
            c2: Destination column index.
            dist_mult: Planar step multiplier (1.0 cardinal, sqrt(2) diagonal).

        Returns:
            Scalar traversal cost, or np.inf if the step violates safety thresholds.
        """
        if self.terrain.impassable_mask[r2, c2]:
            return np.inf

        planar_dist = dist_mult * self.config.resolution_m
        z1 = self.terrain.dem[r1, c1]
        z2 = self.terrain.dem[r2, c2]
        if np.isnan(z1) or np.isnan(z2):
            return np.inf

        dz = z2 - z1
        step_slope_deg = np.degrees(np.arctan2(abs(dz), planar_dist))
        if step_slope_deg > self.config.max_passable_slope_deg:
            return np.inf

        surface_dist = np.hypot(planar_dist, dz)
        norm_slope = step_slope_deg / self.config.max_passable_slope_deg
        slope_factor = 1.0 + self.config.slope_penalty_weight * (norm_slope ** 2)

        # Extra caution surcharge above 15 deg so near-limit steps are used only in gaps
        if step_slope_deg > self.config.caution_slope_deg:
            excess_ratio = (step_slope_deg - self.config.caution_slope_deg) / (
                self.config.max_passable_slope_deg - self.config.caution_slope_deg
            )
            slope_factor += 3.0 * (excess_ratio ** 2)

        uphill_penalty = 1.6 * max(0.0, dz)
        dest_roughness = self.terrain.roughness_m[r2, c2]
        roughness_factor = 1.0 + self.config.roughness_penalty_weight * (
            dest_roughness / self.config.max_roughness_m
        )

        return float(surface_dist * slope_factor * roughness_factor + uphill_penalty)

    def plan_leg(self, start: Waypoint, goal: Waypoint) -> LegTelemetry:
        """Finds the optimal safe EVA path between two waypoints using A*.

        Args:
            start: Origin Waypoint.
            goal: Target Waypoint.

        Returns:
            LegTelemetry populated with path coordinates and engineering metrics.

        Raises:
            RuntimeError: If no safe path exists under current slope/roughness limits.
        """
        safe_start = self.snap_to_safe_cell(start)
        safe_goal = self.snap_to_safe_cell(goal)

        logger.info(
            "Planning EVA leg: '%s' (%d,%d) -> '%s' (%d,%d)",
            safe_start.name,
            safe_start.row,
            safe_start.col,
            safe_goal.name,
            safe_goal.row,
            safe_goal.col,
        )

        start_node = (safe_start.row, safe_start.col)
        goal_node = (safe_goal.row, safe_goal.col)

        open_heap: List[Tuple[float, Tuple[int, int]]] = [(0.0, start_node)]
        came_from: Dict[Tuple[int, int], Tuple[int, int]] = {}
        g_score: Dict[Tuple[int, int], float] = {start_node: 0.0}
        closed_set = set()
        nodes_expanded = 0

        while open_heap:
            _, current = heapq.heappop(open_heap)
            if current in closed_set:
                continue
            closed_set.add(current)
            nodes_expanded += 1

            if current == goal_node:
                path = self._reconstruct_path(came_from, current)
                telemetry = self._compute_leg_telemetry(safe_start.name, safe_goal.name, path)
                logger.info(
                    "Leg solved (%d nodes) | Dist: %.1f m | Gain: +%.1f m | Mean/Max Slope: %.1f/%.1f deg "
                    "(%d steps >20 deg, %.1f m) | Time: %.1f min",
                    nodes_expanded,
                    telemetry.distance_m,
                    telemetry.elevation_gain_m,
                    telemetry.mean_slope_deg,
                    telemetry.max_slope_deg,
                    telemetry.steps_over_20deg,
                    telemetry.steep_distance_over_20deg_m,
                    telemetry.estimated_duration_min,
                )
                return telemetry

            r, c = current
            current_g = g_score[current]

            for dr, dc, dist_mult in self._NEIGHBORS:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < self.terrain.rows and 0 <= nc < self.terrain.cols):
                    continue

                neighbor = (nr, nc)
                if neighbor in closed_set:
                    continue

                edge_cost = self._step_cost(r, c, nr, nc, dist_mult)
                if np.isinf(edge_cost):
                    continue

                tentative_g = current_g + edge_cost
                if tentative_g < g_score.get(neighbor, np.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(nr, nc, safe_goal.row, safe_goal.col)
                    heapq.heappush(open_heap, (f_score, neighbor))

        err_msg = f"No passable EVA route found from '{safe_start.name}' to '{safe_goal.name}'."
        logger.error(err_msg)
        raise RuntimeError(err_msg)

    def plan_mission(self, waypoints: List[Waypoint]) -> List[LegTelemetry]:
        """Plans a sequential multi-waypoint EVA mission with leg caching.

        Caches solved `(start_node, goal_node)` legs so that walkback contingency
        checks reuse already-planned return legs (such as the final leg back to
        Base) without duplicate A* computation.

        Args:
            waypoints: Ordered list of Waypoint objects.

        Returns:
            List of LegTelemetry objects for each sequential traverse leg.

        Raises:
            ValueError: If fewer than 2 waypoints are provided.
        """
        if len(waypoints) < 2:
            raise ValueError("Mission requires at least a start and end waypoint.")

        safe_waypoints = [self.snap_to_safe_cell(wp) for wp in waypoints]
        base_wp = safe_waypoints[0]
        base_node = (base_wp.row, base_wp.col)

        # Cache keyed by ((start_row, start_col), (goal_row, goal_col)) to avoid duplicate A* runs
        leg_cache: Dict[Tuple[Tuple[int, int], Tuple[int, int]], LegTelemetry] = {}
        legs: List[LegTelemetry] = []

        for idx in range(len(safe_waypoints) - 1):
            wp_a = safe_waypoints[idx]
            wp_b = safe_waypoints[idx + 1]
            cache_key = ((wp_a.row, wp_a.col), (wp_b.row, wp_b.col))

            if cache_key in leg_cache:
                logger.info("Reusing cached leg: '%s' -> '%s'", wp_a.name, wp_b.name)
                leg = leg_cache[cache_key]
            else:
                leg = self.plan_leg(wp_a, wp_b)
                leg_cache[cache_key] = leg

            legs.append(leg)

        # True walkback check: evaluate safe return distance to Base from each stop,
        # reusing cached return legs whenever already computed in the route.
        max_walkback_m = 0.0
        furthest_stop = base_wp.name
        checked_stops = {base_node}

        for wp in safe_waypoints[1:]:
            stop_node = (wp.row, wp.col)
            if stop_node in checked_stops:
                continue
            checked_stops.add(stop_node)

            return_key = (stop_node, base_node)
            if return_key in leg_cache:
                return_leg = leg_cache[return_key]
                logger.info(
                    "Walkback check for '%s' reused cached return leg (%.1f m).",
                    wp.name,
                    return_leg.distance_m,
                )
            else:
                logger.info("Walkback check computing contingency return from '%s' to Base...", wp.name)
                return_leg = self.plan_leg(wp, base_wp)
                leg_cache[return_key] = return_leg

            if return_leg.distance_m > max_walkback_m:
                max_walkback_m = return_leg.distance_m
                furthest_stop = wp.name

        total_dist = sum(leg.distance_m for leg in legs)
        total_energy = sum(leg.estimated_energy_kj for leg in legs)
        total_time = sum(leg.estimated_duration_min for leg in legs)

        logger.info(
            "MISSION SUMMARY [%s] | Legs: %d | Loop Dist: %.1f m | Max Walkback: %.1f m / %.1f m limit (from '%s') | Energy: %.1f kJ | Time: %.1f min",
            "SYNTHETIC_LAYOUT" if self.terrain.is_synthetic else "REAL_DTM",
            len(legs),
            total_dist,
            max_walkback_m,
            self.config.max_walkback_distance_m,
            furthest_stop,
            total_energy,
            total_time,
        )

        if max_walkback_m > self.config.max_walkback_distance_m:
            logger.warning(
                "WALKBACK LIMIT EXCEEDED: Return path from '%s' (%.1f m) exceeds suit walkback limit (%.1f m)!",
                furthest_stop,
                max_walkback_m,
                self.config.max_walkback_distance_m,
            )

        return legs

    @staticmethod
    def _reconstruct_path(
        came_from: Dict[Tuple[int, int], Tuple[int, int]], current: Tuple[int, int]
    ) -> List[Tuple[int, int]]:
        """Reconstructs the ordered coordinate path from the A* parent map.

        Args:
            came_from: Mapping of child node to parent node.
            current: Goal node coordinate.

        Returns:
            Ordered list of (row, col) tuples from start to goal.
        """
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _compute_leg_telemetry(
        self, start_name: str, end_name: str, path: List[Tuple[int, int]]
    ) -> LegTelemetry:
        """Calculates distance, elevation profile, steep-step counts, energy, and time.

        Args:
            start_name: Name of origin waypoint.
            end_name: Name of destination waypoint.
            path: Ordered list of (row, col) grid coordinates.

        Returns:
            Populated LegTelemetry instance.
        """
        total_dist = 0.0
        elev_gain = 0.0
        elev_loss = 0.0
        max_slope = 0.0
        weighted_slope_sum = 0.0
        steps_over_15 = 0
        steps_over_20 = 0
        steep_dist_over_20 = 0.0
        total_time_sec = 0.0
        roughness_samples = []

        for i in range(len(path) - 1):
            r1, c1 = path[i]
            r2, c2 = path[i + 1]

            planar_dist = np.hypot(r2 - r1, c2 - c1) * self.config.resolution_m
            dz = self.terrain.dem[r2, c2] - self.terrain.dem[r1, c1]
            step_dist = np.hypot(planar_dist, dz)

            signed_slope = np.degrees(np.arctan2(dz, planar_dist))
            abs_slope = abs(signed_slope)
            step_rough = float(self.terrain.roughness_m[r2, c2])

            if abs_slope > 15.0:
                steps_over_15 += 1
            if abs_slope > 20.0:
                steps_over_20 += 1
                steep_dist_over_20 += float(step_dist)

            step_speed = self._step_walk_speed_mps(signed_slope, step_rough)
            total_time_sec += step_dist / step_speed

            total_dist += step_dist
            weighted_slope_sum += abs_slope * step_dist
            if dz > 0:
                elev_gain += dz
            else:
                elev_loss += abs(dz)

            max_slope = max(max_slope, abs_slope)
            roughness_samples.append(step_rough)

        mean_slope = (weighted_slope_sum / total_dist) if total_dist > 0 else 0.0

        # First-order metabolic energy model: horizontal locomotion + uphill climb work.
        # Downhill segments (dz < 0) contribute zero positive climb work in this baseline.
        base_locomotion_j = 2.8 * self.config.astronaut_total_mass_kg * total_dist
        climb_work_j = (
            self.config.astronaut_total_mass_kg * MARS_GRAVITY_MPS2 * elev_gain
        ) / 0.25
        energy_kj = (base_locomotion_j + climb_work_j) / 1000.0

        return LegTelemetry(
            start_name=start_name,
            end_name=end_name,
            path_coords=path,
            distance_m=float(total_dist),
            elevation_gain_m=float(elev_gain),
            elevation_loss_m=float(elev_loss),
            max_slope_deg=float(max_slope),
            mean_slope_deg=float(mean_slope),
            steps_over_15deg=steps_over_15,
            steps_over_20deg=steps_over_20,
            steep_distance_over_20deg_m=float(steep_dist_over_20),
            mean_roughness_m=float(np.mean(roughness_samples)) if roughness_samples else 0.0,
            estimated_energy_kj=float(energy_kj),
            estimated_duration_min=float(total_time_sec / 60.0),
        )


# ==============================================================================
# EXPORT & DIAGNOSTIC VISUALIZATION
# ==============================================================================
def export_mission_geojson(
    legs: List[LegTelemetry], terrain: TerrainAnalyzer, output_path: str = "eva_route.geojson"
) -> None:
    """Exports the solved multi-leg EVA traverse to a geospatial GeoJSON FeatureCollection.

    Args:
        legs: List of solved LegTelemetry objects.
        terrain: TerrainAnalyzer containing elevation data and spatial metadata.
        output_path: Destination GeoJSON file path.
    """
    features = []
    for idx, leg in enumerate(legs, start=1):
        coords_3d = []
        pixel_coords = []
        for r, c in leg.path_coords:
            x_geo, y_geo = terrain.pixel_to_map_xy(r, c)
            z_elev = round(float(terrain.dem[r, c]), 3)
            coords_3d.append([round(x_geo, 3), round(y_geo, 3), z_elev])
            pixel_coords.append([int(c), int(r)])

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "leg_number": idx,
                    "from": leg.start_name,
                    "to": leg.end_name,
                    "is_synthetic_layout": terrain.is_synthetic,
                    "terrain_source": terrain.source_label,
                    "distance_m": round(leg.distance_m, 2),
                    "elevation_gain_m": round(leg.elevation_gain_m, 2),
                    "mean_slope_deg": round(leg.mean_slope_deg, 2),
                    "max_slope_deg": round(leg.max_slope_deg, 2),
                    "steps_over_15deg": leg.steps_over_15deg,
                    "steps_over_20deg": leg.steps_over_20deg,
                    "steep_distance_over_20deg_m": round(leg.steep_distance_over_20deg_m, 2),
                    "estimated_energy_kj": round(leg.estimated_energy_kj, 2),
                    "estimated_duration_min": round(leg.estimated_duration_min, 2),
                    "grid_pixel_coords": pixel_coords,
                },
                "geometry": {"type": "LineString", "coordinates": coords_3d},
            }
        )

    collection = {
        "type": "FeatureCollection",
        "metadata": {
            "is_synthetic_layout": terrain.is_synthetic,
            "scale_disclaimer": (
                "SYNTHETIC LAYOUT (~250 m synthetic separation; verify real feature "
                "spacing on a calibrated HiRISE DTEEC DTM before operational use)."
                if terrain.is_synthetic
                else "Derived from calibrated HiRISE DTM GeoTIFF."
            ),
            "source_label": terrain.source_label,
            "crs": terrain.crs_wkt,
        },
        "features": features,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)
    logger.info("Exported geospatial route GeoJSON to %s", output_path)


def save_route_preview(
    terrain: TerrainAnalyzer,
    waypoints: List[Waypoint],
    legs: List[LegTelemetry],
    output_png: str = "eva_route_preview.png",
) -> None:
    """Saves a 2-panel diagnostic visualization (Elevation + Hazard/Route overlay).

    Args:
        terrain: TerrainAnalyzer instance.
        waypoints: Mission Waypoint list.
        legs: Solved LegTelemetry list.
        output_png: Destination PNG image path.
    """
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed; skipping PNG preview generation.")
        return

    tag = "[SYNTHETIC LAYOUT] " if terrain.is_synthetic else ""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)

    im0 = axes[0].imshow(terrain.dem, cmap="terrain", origin="upper")
    axes[0].set_title(f"{tag}Mount Sharp Boxwork + Gediz Vallis DEM (m)")
    axes[0].set_xlabel("East-West Distance (m)")
    axes[0].set_ylabel("North-South Distance (m)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="Elevation (m)")

    im1 = axes[1].imshow(terrain.slope_deg, cmap="inferno", vmin=0, vmax=25, origin="upper")
    axes[1].contour(
        terrain.impassable_mask, levels=[0.5], colors="cyan", linewidths=0.7, alpha=0.85
    )
    axes[1].set_title(f"{tag}Slope Hazard Map (deg) & Optimal EVA Route")
    axes[1].set_xlabel("East-West Distance (m)")
    fig.colorbar(im1, ax=axes[1], shrink=0.8, label="Slope (deg)")

    colors = ["#00FF7F", "#FFD700", "#00BFFF"]
    for idx, leg in enumerate(legs):
        rows = [pt[0] for pt in leg.path_coords]
        cols = [pt[1] for pt in leg.path_coords]
        axes[1].plot(
            cols,
            rows,
            color=colors[idx % len(colors)],
            linewidth=2.2,
            label=f"Leg {idx+1}: {leg.distance_m:.0f} m ({leg.estimated_duration_min:.1f} min)",
        )

    for wp in waypoints:
        axes[1].scatter(wp.col, wp.row, c="white", edgecolors="black", s=65, zorder=5)
        axes[1].annotate(
            wp.name,
            (wp.col + 4, wp.row - 4),
            color="white",
            fontsize=8,
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.65),
        )

    axes[1].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close(fig)
    logger.info("Saved diagnostic route plot to %s", output_png)


# ==============================================================================
# ENTRY POINT (DEMONSTRATION RUN)
# ==============================================================================
if __name__ == "__main__":
    try:
        eva_config = EVACostConfig(
            resolution_m=1.0,
            max_passable_slope_deg=22.0,
            caution_slope_deg=15.0,
            max_roughness_m=0.85,
            max_walkback_distance_m=1500.0,
            despike_hirise_seams=False,
        )

        # Swap with TerrainAnalyzer.from_geotiff("<verified_DTEEC_product>.tif", eva_config)
        # once your HiRISE DTM cutout is verified and downloaded from UA HiRISE / PDS.
        analyzer = TerrainAnalyzer.from_synthetic_mount_sharp(
            rows=350, cols=450, config=eva_config
        )

        mission_waypoints = [
            Waypoint(name="HAB / Rover Base", row=45, col=60, science_priority="BASE"),
            Waypoint(name="Boxwork Vein Node", row=195, col=180, science_priority="PRIMARY"),
            Waypoint(name="Gediz Sulfur Rim", row=255, col=298, science_priority="ANOMALY"),
            Waypoint(name="HAB / Rover Base", row=45, col=60, science_priority="BASE"),
        ]

        router = MartianEVARouter(terrain=analyzer)
        solved_legs = router.plan_mission(mission_waypoints)

        export_mission_geojson(solved_legs, analyzer, output_path="eva_route.geojson")
        save_route_preview(analyzer, mission_waypoints, solved_legs, output_png="eva_route_preview.png")

    except Exception as exc:
        logger.exception("Fatal error during Martian EVA route execution: %s", exc)
        sys.exit(1)