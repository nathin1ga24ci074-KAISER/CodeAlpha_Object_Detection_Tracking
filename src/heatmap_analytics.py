"""
Heatmap Analytics Module for CodeAlpha Task 4: Object Detection and Tracking.
Provides 2D spatial occupancy and movement density accumulation across video frames.

Documented Definition:
- This module generates an object-position spatial frequency density map.
- It measures the cumulative presence and dwell of detected entity centroids in pixel space.
- It is NOT a physical temperature map, thermal infrared reading, or calibrated motion-speed map.
- All spatial coordinates and accumulation footprints are evaluated in image pixel units.
"""

from typing import List, Dict, Any, Optional, Tuple, Sequence, Union
import math
import cv2
import numpy as np


def extract_centroid(obj: Any) -> Optional[Tuple[float, float]]:
    """
    Safely extracts a finite 2D centroid (cx, cy) in pixel coordinates from a
    TrackedObject, tuple, list, or bounding-box-bearing entity.
    Returns None if malformed, non-finite (NaN, Inf), or missing.
    """
    if obj is None:
        return None

    # 1. Direct coordinate sequence: (x, y)
    if isinstance(obj, (tuple, list)):
        if len(obj) >= 2:
            try:
                x = float(obj[0])
                y = float(obj[1])
                if math.isfinite(x) and math.isfinite(y):
                    return (x, y)
            except (ValueError, TypeError):
                return None
        return None

    # 2. TrackedObject with trajectory (prefer most recent trajectory centroid)
    raw_traj = getattr(obj, "trajectory", None)
    if raw_traj:
        try:
            pt = raw_traj[-1]
            if isinstance(pt, (tuple, list)) and len(pt) >= 2:
                x = float(pt[0])
                y = float(pt[1])
                if math.isfinite(x) and math.isfinite(y):
                    return (x, y)
        except (IndexError, TypeError, ValueError):
            pass

    # 3. current_centroid attribute / property
    cc = getattr(obj, "current_centroid", None)
    if isinstance(cc, (tuple, list)) and len(cc) >= 2:
        try:
            x = float(cc[0])
            y = float(cc[1])
            if math.isfinite(x) and math.isfinite(y):
                return (x, y)
        except (ValueError, TypeError):
            pass

    # 4. center property: [cx, cy]
    c = getattr(obj, "center", None)
    if isinstance(c, (tuple, list)) and len(c) >= 2:
        try:
            x = float(c[0])
            y = float(c[1])
            if math.isfinite(x) and math.isfinite(y):
                return (x, y)
        except (ValueError, TypeError):
            pass

    # 5. Bounding box fallback: [x1, y1, x2, y2]
    box = getattr(obj, "box", None)
    if isinstance(box, (tuple, list)) and len(box) >= 4:
        try:
            x = float((box[0] + box[2]) / 2.0)
            y = float((box[1] + box[3]) / 2.0)
            if math.isfinite(x) and math.isfinite(y):
                return (x, y)
        except (ValueError, TypeError):
            pass

    return None


def generate_gaussian_kernel(radius: int) -> np.ndarray:
    """
    Generates a 2D radial Gaussian footprint kernel of size (2*radius + 1, 2*radius + 1).
    Normalized such that center peak equals 1.0, and circular cutoff is applied at radius.
    """
    r = max(1, int(radius))
    y, x = np.ogrid[-r:r + 1, -r:r + 1]
    dist_sq = x * x + y * y
    sigma = max(0.5, r / 2.0)
    kernel = np.exp(-dist_sq / (2.0 * sigma * sigma)).astype(np.float32)
    kernel[dist_sq > (r * r)] = 0.0
    max_val = float(np.max(kernel))
    if max_val > 0.0:
        kernel /= max_val
    return kernel


class HeatmapAnalytics:
    """
    Standalone 2D object spatial position density accumulation and visualization engine.

    Accumulates detected object centroids into a continuous float32 density map,
    supporting kernel splatting, decay, colormapped rendering, and non-destructive
    frame overlay blending.
    """

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        radius_px: int = 20,
        intensity_per_point: float = 1.0,
        decay_factor: float = 1.0,
        colormap: int = cv2.COLORMAP_JET,
        overlay_alpha: float = 0.5,
        min_density_threshold: float = 0.05
    ):
        """
        Initializes the heatmap accumulation engine.

        Args:
            frame_width: Horizontal frame resolution in pixels (must be > 0).
            frame_height: Vertical frame resolution in pixels (must be > 0).
            radius_px: Footprint radius in pixels around each centroid (must be > 0).
            intensity_per_point: Accumulation weight added per observation (must be > 0).
            decay_factor: Multiplier applied per frame update in (0.0, 1.0] (1.0 = no decay).
            colormap: OpenCV colormap constant (e.g. cv2.COLORMAP_JET).
            overlay_alpha: Blending weight for heatmap overlay in [0.0, 1.0].
            min_density_threshold: Minimum normalized density fraction (0.0..1.0) required
                                   for overlay coloring. Pixels below this threshold remain
                                   transparent/uncolored.
        """
        if int(frame_width) <= 0:
            raise ValueError(f"frame_width must be a positive integer, got {frame_width}")
        if int(frame_height) <= 0:
            raise ValueError(f"frame_height must be a positive integer, got {frame_height}")
        if int(radius_px) <= 0:
            raise ValueError(f"radius_px must be a positive integer, got {radius_px}")
        if float(intensity_per_point) <= 0.0:
            raise ValueError(f"intensity_per_point must be positive, got {intensity_per_point}")
        if not (0.0 < float(decay_factor) <= 1.0):
            raise ValueError(f"decay_factor must be in (0.0, 1.0], got {decay_factor}")
        if not (0.0 <= float(overlay_alpha) <= 1.0):
            raise ValueError(f"overlay_alpha must be in [0.0, 1.0], got {overlay_alpha}")
        if not (0.0 <= float(min_density_threshold) <= 1.0):
            raise ValueError(f"min_density_threshold must be in [0.0, 1.0], got {min_density_threshold}")

        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.radius_px = int(radius_px)
        self.intensity_per_point = float(intensity_per_point)
        self.decay_factor = float(decay_factor)
        self.colormap = int(colormap)
        self.overlay_alpha = float(overlay_alpha)
        self.min_density_threshold = float(min_density_threshold)

        # Internal 2D density array
        self._density_map = np.zeros((self.frame_height, self.frame_width), dtype=np.float32)
        self._kernel = generate_gaussian_kernel(self.radius_px)

        # Telemetry counters
        self._total_points_accumulated: int = 0
        self._frames_processed: int = 0

    @property
    def total_points_accumulated(self) -> int:
        """Returns the total number of points accumulated."""
        return self._total_points_accumulated

    @property
    def total_points(self) -> int:
        """Alias for total_points_accumulated."""
        return self._total_points_accumulated

    @property
    def is_empty(self) -> bool:
        """Returns True if no density has been accumulated on the map."""
        return bool(np.max(self._density_map) <= 0.0)

    @property
    def max_density(self) -> float:
        """Returns the current peak density value in the accumulator."""
        return float(np.max(self._density_map))

    def reset(self) -> None:
        """Clears all accumulated density and resets telemetry counters."""
        self._density_map.fill(0.0)
        self._total_points_accumulated = 0
        self._frames_processed = 0

    def get_density_map(self, copy: bool = True) -> np.ndarray:
        """
        Retrieves the 2D float32 density map.

        Args:
            copy: If True (default), returns a deep copy to prevent external mutation.
                  If False, returns direct reference for zero-copy read-only operations.

        Returns:
            np.ndarray: 2D float32 array of shape (frame_height, frame_width).
        """
        return self._density_map.copy() if copy else self._density_map

    def accumulate_point(self, x: float, y: float, intensity: Optional[float] = None) -> bool:
        """
        Accumulates a single centroid coordinate (x, y) into the density map.
        Safely slices boundary regions if the centroid is near frame borders.

        Args:
            x: Horizontal pixel coordinate.
            y: Vertical pixel coordinate.
            intensity: Optional override for contribution weight.

        Returns:
            bool: True if point was within accumulation range and recorded, False if skipped.
        """
        try:
            fx = float(x)
            fy = float(y)
        except (ValueError, TypeError):
            return False

        if not (math.isfinite(fx) and math.isfinite(fy)):
            return False

        cx = int(round(fx))
        cy = int(round(fy))
        r = self.radius_px
        weight = self.intensity_per_point if intensity is None else max(0.0, float(intensity))

        # Check bounds: bounding box of footprint
        x_min = cx - r
        x_max = cx + r + 1
        y_min = cy - r
        y_max = cy + r + 1

        # Clamped map coordinates
        map_x1 = max(0, x_min)
        map_x2 = min(self.frame_width, x_max)
        map_y1 = max(0, y_min)
        map_y2 = min(self.frame_height, y_max)

        if map_x1 >= map_x2 or map_y1 >= map_y2:
            # Completely outside frame boundaries
            return False

        # Corresponding kernel sub-slice
        k_x1 = map_x1 - x_min
        k_x2 = k_x1 + (map_x2 - map_x1)
        k_y1 = map_y1 - y_min
        k_y2 = k_y1 + (map_y2 - map_y1)

        # Splat kernel into density map in-place
        self._density_map[map_y1:map_y2, map_x1:map_x2] += self._kernel[k_y1:k_y2, k_x1:k_x2] * weight
        self._total_points_accumulated += 1
        return True

    def accumulate_points(
        self,
        points: Sequence[Any],
        intensity: Optional[float] = None
    ) -> int:
        """
        Accumulates multiple centroid observations.

        Args:
            points: Sequence of (x, y) tuples or coordinate pairs.
            intensity: Optional contribution weight per point.

        Returns:
            int: Number of points successfully accumulated.
        """
        if not points:
            return 0

        count = 0
        for pt in points:
            c = extract_centroid(pt)
            if c is not None and self.accumulate_point(c[0], c[1], intensity=intensity):
                count += 1
        return count

    def update(
        self,
        tracks: Sequence[Any],
        frame_idx: Optional[int] = None,
        timestamp_sec: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Updates the heatmap with the active tracks observed in a video frame.

        Args:
            tracks: Sequence of TrackedObject instances or centroid pairs.
            frame_idx: Optional frame sequence counter (1-indexed).
            timestamp_sec: Optional stream timestamp in seconds.

        Returns:
            Dict[str, Any]: Frame-level accumulation telemetry snapshot.
        """
        self._frames_processed += 1

        # Optional decay factor
        if self.decay_factor < 1.0:
            self._density_map *= self.decay_factor

        accumulated_count = 0
        for t in tracks or []:
            c = extract_centroid(t)
            if c is not None and self.accumulate_point(c[0], c[1]):
                accumulated_count += 1

        return {
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "points_accumulated": accumulated_count,
            "total_points_accumulated": self._total_points_accumulated,
            "max_density": round(self.max_density, 4),
            "is_empty": self.is_empty
        }

    def generate_heatmap_image(
        self,
        colormap: Optional[int] = None,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Renders a 3-channel BGR heatmap visualization image (np.uint8).

        Args:
            colormap: Optional OpenCV colormap constant override.
            normalize: If True, stretches peak density to 255.
                       If False, clamps raw values directly into 0..255.

        Returns:
            np.ndarray: 3-channel BGR image of shape (frame_height, frame_width, 3).
                        Returns a solid black image when density is zero (division by zero safe).
        """
        cmap = self.colormap if colormap is None else int(colormap)
        max_val = float(np.max(self._density_map))

        if max_val <= 0.0 or not math.isfinite(max_val):
            return np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

        if normalize:
            norm_map = np.clip((self._density_map / max_val) * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            norm_map = np.clip(self._density_map, 0.0, 255.0).astype(np.uint8)

        return cv2.applyColorMap(norm_map, cmap)

    def apply_overlay(
        self,
        frame: np.ndarray,
        alpha: Optional[float] = None,
        colormap: Optional[int] = None,
        min_density_threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Blends the spatial density heatmap onto an input video frame non-destructively.
        The input frame array is NEVER modified in-place.

        Args:
            frame: 3-channel BGR numpy array matching (frame_height, frame_width, 3).
            alpha: Optional blend weight override in [0.0, 1.0].
            colormap: Optional OpenCV colormap override.
            min_density_threshold: Optional threshold override in [0.0, 1.0].
                                   Pixels with relative density below this threshold
                                   remain pristine without color tinting.

        Returns:
            np.ndarray: New blended frame of identical dimensions and dtype (np.uint8).
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise ValueError("Input frame must be a valid numpy ndarray")

        if frame.shape[:2] != (self.frame_height, self.frame_width):
            raise ValueError(
                f"Frame dimensions {frame.shape[:2]} do not match heatmap configuration "
                f"{(self.frame_height, self.frame_width)}"
            )

        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(f"Input frame must be a 3-channel BGR image, got shape {frame.shape}")

        # Non-destructive copy
        output = frame.copy()

        max_val = float(np.max(self._density_map))
        if max_val <= 0.0 or not math.isfinite(max_val):
            return output

        a = self.overlay_alpha if alpha is None else max(0.0, min(1.0, float(alpha)))
        if a <= 0.0:
            return output

        thresh = (
            self.min_density_threshold
            if min_density_threshold is None
            else max(0.0, min(1.0, float(min_density_threshold)))
        )

        norm_density = np.clip(self._density_map / max_val, 0.0, 1.0)
        norm_uint8 = (norm_density * 255.0).astype(np.uint8)

        cmap = self.colormap if colormap is None else int(colormap)
        colored_heatmap = cv2.applyColorMap(norm_uint8, cmap)

        if thresh > 0.0:
            mask = norm_density >= thresh
            if np.any(mask):
                blended = cv2.addWeighted(output, 1.0 - a, colored_heatmap, a, 0)
                output[mask] = blended[mask]
        else:
            output = cv2.addWeighted(output, 1.0 - a, colored_heatmap, a, 0)

        return output

    def get_summary(self) -> Dict[str, Any]:
        """
        Returns structured summary metadata and density telemetry.
        """
        max_val = float(np.max(self._density_map))
        mean_val = float(np.mean(self._density_map))
        return {
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "radius_px": self.radius_px,
            "intensity_per_point": self.intensity_per_point,
            "decay_factor": self.decay_factor,
            "colormap": self.colormap,
            "overlay_alpha": self.overlay_alpha,
            "min_density_threshold": self.min_density_threshold,
            "total_points_accumulated": self._total_points_accumulated,
            "frames_processed": self._frames_processed,
            "max_density": round(max_val, 4),
            "mean_density": round(mean_val, 4),
            "is_empty": self.is_empty
        }
