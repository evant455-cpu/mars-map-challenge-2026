"""Martian EVA Route Planning Engine (Consumes Precomputed Hazard Layers).

Reads `hazard_layers.npz` produced by `terrain_hazard_mapper.py` as its single
source of truth for elevation (`dem`), hazard tiers (`hazard_tier`), traversal
penalties (`cost_multiplier`), and impassable obstacles (`no_go_mask`).

Pathfinding Rules:
    1. All cell-level hazard classifications, rock protrusion detections, and
       edge-cell policies are inherited directly from `hazard_layers.npz`.
    2. Retains a directional step-by-step slope check (`arctan2(|dz|, dist)`)
       between adjacent cells so no single step exceeds `max_passable_slope_deg`.
    3. Caches solved `(start, goal)` legs so walkback contingency checks reuse
       already-computed return legs without duplicate A* runs.

Example:
    First run `terrain_hazard_mapper.py` to generate `hazard_layers.npz`, then:
        $ python3 martian_eva_router.py
"""

from dataclasses import dataclass, field
import heapq
import json
import logging
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple
import numpy as np

try:
    from matplotlib.colors import BoundaryNorm, ListedColormap
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# ==============================================================================
# LOGGING CONFIGURATION (MOBILE-FRIENDLY TERMINAL & FILE TELEMETRY)
# ==============================================================================
def setup_logger(log_file: str = "eva_router.log", level: int = logging.INFO) -> logging.Logger:
    """Configures a dual-output logger for terminal and mobile log inspection.

    Args:
        log_file: Destination path for persistent execution logs.
        level: Logging threshold (e.g., logging.INFO, logging.DEBUG).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger("MartianEVARouter")
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
# DATA STRUCTURES & MISSION ASSUMPTIONS
# ==============================================================================
MARS_GRAVITY_MPS2: float = 3.721  # Mars surface gravitational acceleration (m/s^2)


@dataclass
class EVACostConfig:
    """Biomechanical and mission contingency parameters for the route solver.

    Note:
        Cell-level terrain hazards are governed by `hazard_layers.npz`. The
        parameters below control adjacent-step slope limits, metabolic energy,
        walking speed integration, and suit walkback limits.

    Attributes:
        astronaut_total_mass_kg: [ASSUMPTION] Suited astronaut mass (180 kg).
        max_passable_slope_deg: [ASSUMPTION] Maximum allowable step-to-step
            slope transition (deg) between adjacent cells. If None, inherits
            `no_go_slope_deg` from `hazard_layers.npz`.
        nominal_walk_speed_mps: [ASSUMPTION] Flat-ground suited walking speed (m/s).
        max_walkback_distance_m: [ASSUMPTION] Maximum safe return path distance (m)
            from any science stop back to the HAB/Lander under suit contingency.
        uphill_climb_weight: [ASSUMPTION] Extra routing cost weight per meter of
            positive vertical climb (`dz > 0`).
    """
    astronaut_total_mass_kg: float = 180.0
    max_passable_slope_deg: Optional[float] = None
    nominal_walk_speed_mps: float = 0.85
    max_walkback_distance_m: float = 1500.0
    uphill_climb_weight: float = 1.6


@dataclass
class PrecomputedHazardSurface:
    """In-memory representation of the precomputed layers from `hazard_layers.npz`.

    Attributes:
        dem: 2D float64 elevation array (meters).
        slope_deg: 2D float64 terrain slope array (degrees).
        roughness_tri_m: 2D float64 3x3 Terrain Ruggedness Index array (meters).
        rock_protrusion_index_m: 2D float64 Rock Protrusion Index array (meters).
        hazard_tier: 2D uint8 array {0: Nominal, 1: Caution, 2: High-Cost, 3: No-Go}.
        cost_multiplier: 2D float64 traversal cost multiplier (>= 1.0, `np.inf` = No-Go).
        no_go_mask: 2D boolean mask (True = impassable).
        resolution_m: Pixel spacing in meters.
        no_go_slope_deg: Hard slope cutoff (deg) recorded by the mapper.
        geo_origin_xy: Top-left (Easting, Northing) map coordinate in meters.
        is_synthetic_layout: True if derived from the synthetic test generator.
        source_label: Dataset or synthetic layout identifier.
        crs_wkt: Coordinate Reference System string.
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
    geo_origin_xy: Tuple[float, float]
    is_synthetic_layout: bool
    source_label: str
    crs_wkt: str

    @property
    def rows(self) -> int:
        """Grid row count (Y-axis)."""
        return int(self.dem.shape[0])

    @property
    def cols(self) -> int:
        """Grid column count (X-axis)."""
        return int(self.dem.shape[1])

    def pixel_to_map_xy(self, row: int, col: int) -> Tuple[float, float]:
        """Converts grid (row, col) indices into geospatial (Easting, Northing) meters."""
        x_m = self.geo_origin_xy[0] + col * self.resolution_m
        y_m = self.geo_origin_xy[1] - row * self.resolution_m
        return float(x_m), float(y_m)

    @classmethod
    def from_npz(cls, npz_path: str = "hazard_layers.npz") -> "PrecomputedHazardSurface":
        """Loads `hazard_layers.npz` generated by `terrain_hazard_mapper.py`.

        Args:
            npz_path: Path to the compressed NumPy archive.

        Returns:
            Initialized PrecomputedHazardSurface instance.

        Raises:
            FileNotFoundError: If `npz_path` does not exist yet.
        """
        path = Path(npz_path)
        if not path.exists():
            msg = (
                f"Precomputed hazard file '{npz_path}' not found. "
                "Run `python3 terrain_hazard_mapper.py` first to generate it."
            )
            logger.error(msg)
            raise FileNotFoundError(msg)

        logger.info("Loading single-source-of-truth hazard layers from %s", npz_path)
        with np.load(path, allow_pickle=False) as data:
            protrusion_key = (
                "rock_protrusion_index_m"
                if "rock_protrusion_index_m" in data
                else "rock_protrusion_m"
            )
            resolution_m = float(data["resolution_m"][0])
            no_go_slope = (
                float(data["no_go_slope_deg"][0]) if "no_go_slope_deg" in data else 22.0
            )
            origin_arr = data["geo_origin_xy"]
            geo_origin_xy = (float(origin_arr[0]), float(origin_arr[1]))
            is_synthetic = (
                bool(data["is_synthetic_layout"][0])
                if "is_synthetic_layout" in data
                else True
            )
            source_label = (
                str(data["source_label"][0])
                if "source_label" in data
                else "UNKNOWN_HAZARD_ARCHIVE"
            )
            crs_wkt = str(data["crs_wkt"][0]) if "crs_wkt" in data else "LOCAL_MARS_METRIC"

            surface = cls(
                dem=data["dem"].astype(np.float64),
                slope_deg=data["slope_deg"].astype(np.float64),
                roughness_tri_m=data["roughness_tri_m"].astype(np.float64),
                rock_protrusion_index_m=data[protrusion_key].astype(np.float64),
                hazard_tier=data["hazard_tier"].astype(np.uint8),
                cost_multiplier=data["cost_multiplier"].astype(np.float64),
                no_go_mask=data["no_go_mask"].astype(bool),
                resolution_m=resolution_m,
                no_go_slope_deg=no_go_slope,
                geo_origin_xy=geo_origin_xy,
                is_synthetic_layout=is_synthetic,
                source_label=source_label,
                crs_wkt=crs_wkt,
            )

        logger.info(
            "Hazard surface loaded [%s] | SyntheticLayout=%s | Grid=%dx%d px (%.2f m/px) | No-Go=%.1f%%",
            surface.source_label,
            surface.is_synthetic_layout,
            surface.rows,
            surface.cols,
            surface.resolution_m,
            100.0 * float(np.mean(surface.no_go_mask)),
        )
        return surface


@dataclass
class Waypoint:
    """Represents a named mission coordinate on the DEM grid."""
    name: str
    row: int
    col: int
    science_priority: str = "PRIMARY"


@dataclass
class LegTelemetry:
    """Computed engineering and safety metrics for a single traverse leg.

    Note on Metabolic Energy Model:
        `estimated_energy_kj` combines horizontal suited locomotion (~2.8 J/kg/m)
        and positive gravitational climb work (`m * g_mars * max(0, dz) / 0.25`).
        Downhill segments (`dz < 0`) contribute zero positive climb work in this
        baseline model, making return legs lower energy than outbound climbs.
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
# A* ROUTE PLANNING ENGINE
# ==============================================================================
class MartianEVARouter:
    """Terrain-aware 8-connected A* route planner backed by `hazard_layers.npz`."""

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

    def __init__(
        self, surface: PrecomputedHazardSurface, config: Optional[EVACostConfig] = None
    ) -> None:
        """Initializes the router with precomputed hazard rasters and mission rules.

        Args:
            surface: PrecomputedHazardSurface loaded from `hazard_layers.npz`.
            config: Optional EVACostConfig; inherits `no_go_slope_deg` if unset.
        """
        self.surface = surface
        self.config = config or EVACostConfig()
        if self.config.max_passable_slope_deg is None:
            self.config.max_passable_slope_deg = self.surface.no_go_slope_deg

    def snap_to_safe_cell(self, wp: Waypoint, search_radius_px: int = 8) -> Waypoint:
        """Returns the waypoint if passable, or snaps to the nearest safe cell within radius.

        Args:
            wp: Candidate Waypoint.
            search_radius_px: Maximum pixel radius to search for passable ground.

        Returns:
            Validated (and possibly snapped) Waypoint.

        Raises:
            ValueError: If out of bounds or no passable ground exists within radius.
        """
        if not (0 <= wp.row < self.surface.rows and 0 <= wp.col < self.surface.cols):
            msg = f"Waypoint '{wp.name}' ({wp.row}, {wp.col}) is outside DEM bounds."
            logger.error(msg)
            raise ValueError(msg)

        if not self.surface.no_go_mask[wp.row, wp.col]:
            return wp

        best_cell: Optional[Tuple[int, int]] = None
        best_score = np.inf

        for dr in range(-search_radius_px, search_radius_px + 1):
            for dc in range(-search_radius_px, search_radius_px + 1):
                nr, nc = wp.row + dr, wp.col + dc
                if not (0 <= nr < self.surface.rows and 0 <= nc < self.surface.cols):
                    continue
                if self.surface.no_go_mask[nr, nc]:
                    continue
                dist = np.hypot(dr, dc)
                score = dist + 0.25 * self.surface.cost_multiplier[nr, nc]
                if score < best_score:
                    best_score = score
                    best_cell = (nr, nc)

        if best_cell is None:
            msg = (
                f"Waypoint '{wp.name}' ({wp.row}, {wp.col}) sits on No-Go terrain and "
                f"no passable cell was found within {search_radius_px} px."
            )
            logger.error(msg)
            raise ValueError(msg)

        logger.warning(
            "Snapped waypoint '%s' from (%d,%d) to passable cell (%d,%d) [tier=%d, slope=%.1f deg]",
            wp.name,
            wp.row,
            wp.col,
            best_cell[0],
            best_cell[1],
            int(self.surface.hazard_tier[best_cell[0], best_cell[1]]),
            float(self.surface.slope_deg[best_cell[0], best_cell[1]]),
        )
        return Waypoint(
            name=wp.name,
            row=best_cell[0],
            col=best_cell[1],
            science_priority=wp.science_priority,
        )

    def _heuristic(self, r1: int, c1: int, r2: int, c2: int) -> float:
        """Admissible octile-distance heuristic scaled to minimum flat-ground cost (1.0x)."""
        dr = abs(r1 - r2)
        dc = abs(c1 - c2)
        octile_steps = (dr + dc) + (1.41421356237 - 2.0) * min(dr, dc)
        return octile_steps * self.surface.resolution_m

    def _step_walk_speed_mps(self, signed_slope_deg: float, roughness_m: float) -> float:
        """Computes local astronaut walking speed (m/s) for a single terrain step."""
        slope_penalty = np.exp(-2.8 * abs(np.radians(signed_slope_deg + 3.0)))
        rough_penalty = max(0.45, 1.0 - 0.5 * (roughness_m / 0.85))
        speed = self.config.nominal_walk_speed_mps * slope_penalty * rough_penalty
        return float(max(0.25, speed))

    def _step_cost(self, r1: int, c1: int, r2: int, c2: int, dist_mult: float) -> float:
        """Computes step cost using precomputed `no_go_mask` and `cost_multiplier`.

        Applies:
            1. Hard block from mapper's `no_go_mask` / infinite `cost_multiplier`.
            2. Directional step-by-step slope check between `(r1, c1)` and `(r2, c2)`.
            3. Mapper's precomputed `cost_multiplier[r2, c2]` plus uphill climb work.
        """
        if self.surface.no_go_mask[r2, c2]:
            return np.inf

        cell_multiplier = self.surface.cost_multiplier[r2, c2]
        if np.isinf(cell_multiplier):
            return np.inf

        planar_dist = dist_mult * self.surface.resolution_m
        z1 = self.surface.dem[r1, c1]
        z2 = self.surface.dem[r2, c2]
        if np.isnan(z1) or np.isnan(z2):
            return np.inf

        dz = z2 - z1

        # Directional step-by-step slope check between adjacent cells
        step_slope_deg = np.degrees(np.arctan2(abs(dz), planar_dist))
        if step_slope_deg > float(self.config.max_passable_slope_deg):
            return np.inf

        surface_dist = np.hypot(planar_dist, dz)
        uphill_penalty = self.config.uphill_climb_weight * max(0.0, dz)

        return float(surface_dist * cell_multiplier + uphill_penalty)

    def plan_leg(self, start: Waypoint, goal: Waypoint) -> LegTelemetry:
        """Finds the optimal safe EVA path between two waypoints using A*."""
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
                if not (0 <= nr < self.surface.rows and 0 <= nc < self.surface.cols):
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
        """Plans a sequential multi-waypoint EVA mission with leg caching."""
        if len(waypoints) < 2:
            raise ValueError("Mission requires at least a start and end waypoint.")

        safe_waypoints = [self.snap_to_safe_cell(wp) for wp in waypoints]
        base_wp = safe_waypoints[0]
        base_node = (base_wp.row, base_wp.col)

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

        # True walkback check: reuse cached return legs whenever available
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
            "SYNTHETIC_LAYOUT" if self.surface.is_synthetic_layout else "REAL_DTM",
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
        """Reconstructs the ordered coordinate path from the A* parent map."""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _compute_leg_telemetry(
        self, start_name: str, end_name: str, path: List[Tuple[int, int]]
    ) -> LegTelemetry:
        """Calculates distance, elevation profile, steep-step counts, energy, and time."""
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

            planar_dist = np.hypot(r2 - r1, c2 - c1) * self.surface.resolution_m
            dz = self.surface.dem[r2, c2] - self.surface.dem[r1, c1]
            step_dist = np.hypot(planar_dist, dz)

            signed_slope = np.degrees(np.arctan2(dz, planar_dist))
            abs_slope = abs(signed_slope)
            step_rough = float(self.surface.roughness_tri_m[r2, c2])

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
    legs: List[LegTelemetry],
    surface: PrecomputedHazardSurface,
    output_path: str = "eva_route.geojson",
) -> None:
    """Exports the solved multi-leg EVA traverse to a geospatial GeoJSON FeatureCollection."""
    features = []
    for idx, leg in enumerate(legs, start=1):
        coords_3d = []
        pixel_coords = []
        for r, c in leg.path_coords:
            x_geo, y_geo = surface.pixel_to_map_xy(r, c)
            z_elev = round(float(surface.dem[r, c]), 3)
            coords_3d.append([round(x_geo, 3), round(y_geo, 3), z_elev])
            pixel_coords.append([int(c), int(r)])

        features.append(
            {
                "type": "Feature",
                "properties": {
                    "leg_number": idx,
                    "from": leg.start_name,
                    "to": leg.end_name,
                    "is_synthetic_layout": surface.is_synthetic_layout,
                    "terrain_source": surface.source_label,
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
            "is_synthetic_layout": surface.is_synthetic_layout,
            "scale_disclaimer": (
                "SYNTHETIC LAYOUT (~250 m synthetic separation; verify real feature "
                "spacing on a calibrated HiRISE DTEEC DTM before operational use)."
                if surface.is_synthetic_layout
                else "Derived from calibrated HiRISE DTM GeoTIFF."
            ),
            "source_label": surface.source_label,
            "crs": surface.crs_wkt,
        },
        "features": features,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2)
    logger.info("Exported geospatial route GeoJSON to %s", output_path)


def save_route_preview(
    surface: PrecomputedHazardSurface,
    waypoints: List[Waypoint],
    legs: List[LegTelemetry],
    output_png: str = "eva_route_preview.png",
) -> None:
    """Overlays the solved route directly onto the mapper's 4-Tier Hazard & Obstacle Map."""
    if not HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed; skipping PNG preview generation.")
        return

    tag = "[SYNTHETIC LAYOUT] " if surface.is_synthetic_layout else ""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

    # Panel 1: Elevation Model (DEM) from hazard_layers.npz
    im0 = axes[0].imshow(surface.dem, cmap="terrain", origin="upper")
    axes[0].set_title(f"{tag}Mount Sharp Boxwork + Gediz Vallis DEM (m)")
    axes[0].set_xlabel("East-West Distance (m)")
    axes[0].set_ylabel("North-South Distance (m)")
    fig.colorbar(im0, ax=axes[0], shrink=0.8, label="Elevation (m)")

    # Panel 2: Mapper's 4-Tier Hazard & Obstacle Map + Solved EVA Traverse
    tier_cmap = ListedColormap(["#1b4332", "#e9c46a", "#f4a261", "#d62828"])
    tier_norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], tier_cmap.N)
    im1 = axes[1].imshow(surface.hazard_tier, cmap=tier_cmap, norm=tier_norm, origin="upper")
    axes[1].set_title(f"{tag}4-Tier Hazard Map & Obstacle-Avoiding Route")
    axes[1].set_xlabel("East-West Distance (m)")
    cbar = fig.colorbar(im1, ax=axes[1], ticks=[0, 1, 2, 3], shrink=0.8)
    cbar.ax.set_yticklabels(["Nominal", "Caution", "High-Cost", "No-Go (Rock/Cliff)"], fontsize=8)

    colors = ["#00FFFF", "#FFFFFF", "#FF66FF"]
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
            bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.75),
        )

    axes[1].legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_png)
    plt.close(fig)
    logger.info("Saved unified route-on-hazard plot to %s", output_png)


# ==============================================================================
# ENTRY POINT (REQUIRES HAZARD_LAYERS.NPZ FROM TERRAIN_HAZARD_MAPPER.PY)
# ==============================================================================
if __name__ == "__main__":
    try:
        # 1. Load single-source-of-truth terrain & hazard rasters
        hazard_surface = PrecomputedHazardSurface.from_npz("hazard_layers.npz")

        # 2. Configure EVA mission rules (inherits max_passable_slope_deg from .npz)
        eva_config = EVACostConfig(
            astronaut_total_mass_kg=180.0,
            nominal_walk_speed_mps=0.85,
            max_walkback_distance_m=1500.0,
        )

        # 3. Solve dual-target EVA loop
        mission_waypoints = [
            Waypoint(name="HAB / Rover Base", row=45, col=60, science_priority="BASE"),
            Waypoint(name="Boxwork Vein Node", row=195, col=180, science_priority="PRIMARY"),
            Waypoint(name="Gediz Sulfur Rim", row=255, col=298, science_priority="ANOMALY"),
            Waypoint(name="HAB / Rover Base", row=45, col=60, science_priority="BASE"),
        ]

        router = MartianEVARouter(surface=hazard_surface, config=eva_config)
        solved_legs = router.plan_mission(mission_waypoints)

        export_mission_geojson(solved_legs, hazard_surface, output_path="eva_route.geojson")
        save_route_preview(
            hazard_surface, mission_waypoints, solved_legs, output_png="eva_route_preview.png"
        )

    except Exception as exc:
        logger.exception("Fatal error during Martian EVA route execution: %s", exc)
        sys.exit(1)