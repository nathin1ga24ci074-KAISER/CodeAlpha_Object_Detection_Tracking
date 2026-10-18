"""
Behavior Analytics Module for CodeAlpha Task 4: Object Detection and Tracking.
Provides explainable, structured trajectory metrics:
- Path length in pixels
- Net displacement in pixels
- Average path speed in pixels per second (when reliable timing is available)
- Direction changes with jitter suppression
- Tortuosity with numerical stability guards
- Multi-track session aggregation and bounded memory management
- Observation-aligned timing windows and multi-lifecycle track ID reuse

Measurements are reported in pixel units only (no uncalibrated real-world speed claims).
Curved or meandering paths are analyzed purely geometrically without subjective labeling.
"""

from typing import List, Dict, Any, Optional, Tuple, Sequence, Union
from collections import Counter, defaultdict, deque
import math


class TrajectoryMetrics:
    """Structured per-track trajectory analysis metrics."""

    def __init__(
        self,
        track_id: int,
        class_name: str,
        total_points: int,
        valid_points: int,
        is_sufficient: bool,
        path_length_px: float,
        net_displacement_px: float,
        tortuosity: float,
        direction_changes_count: int,
        direction_changes: List[Dict[str, Any]],
        avg_speed_px_per_sec: Optional[float],
        speed_unavailable_reason: Optional[str] = None,
        start_centroid: Optional[Tuple[int, int]] = None,
        end_centroid: Optional[Tuple[int, int]] = None,
        first_seen_frame: Optional[int] = None,
        last_seen_frame: Optional[int] = None,
        elapsed_time_sec: Optional[float] = None,
        lifecycle_id: Optional[int] = None
    ):
        self.track_id = int(track_id)
        self.lifecycle_id = int(lifecycle_id) if lifecycle_id is not None else None
        self.class_name = str(class_name)
        self.total_points = int(total_points)
        self.valid_points = int(valid_points)
        self.is_sufficient = bool(is_sufficient)
        self.path_length_px = float(path_length_px)
        self.net_displacement_px = float(net_displacement_px)
        self.tortuosity = float(tortuosity)
        self.direction_changes_count = int(direction_changes_count)
        self.direction_changes = direction_changes or []
        self.avg_speed_px_per_sec = float(avg_speed_px_per_sec) if avg_speed_px_per_sec is not None else None
        self.speed_unavailable_reason = str(speed_unavailable_reason) if speed_unavailable_reason else None
        self.start_centroid = start_centroid
        self.end_centroid = end_centroid
        self.first_seen_frame = first_seen_frame
        self.last_seen_frame = last_seen_frame
        self.elapsed_time_sec = float(elapsed_time_sec) if elapsed_time_sec is not None else None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "lifecycle_id": self.lifecycle_id,
            "class_name": self.class_name,
            "total_points": self.total_points,
            "valid_points": self.valid_points,
            "is_sufficient": self.is_sufficient,
            "path_length_px": self.path_length_px,
            "net_displacement_px": self.net_displacement_px,
            "tortuosity": self.tortuosity,
            "direction_changes_count": self.direction_changes_count,
            "direction_changes": self.direction_changes,
            "avg_speed_px_per_sec": self.avg_speed_px_per_sec,
            "speed_unavailable_reason": self.speed_unavailable_reason,
            "start_centroid": self.start_centroid,
            "end_centroid": self.end_centroid,
            "first_seen_frame": self.first_seen_frame,
            "last_seen_frame": self.last_seen_frame,
            "elapsed_time_sec": self.elapsed_time_sec
        }

    def __repr__(self) -> str:
        speed_str = f"{self.avg_speed_px_per_sec:.2f} px/s" if self.avg_speed_px_per_sec is not None else "N/A"
        lid_str = f"#{self.track_id} (L{self.lifecycle_id})" if self.lifecycle_id is not None else f"#{self.track_id}"
        return (
            f"TrajectoryMetrics({lid_str} {self.class_name}: "
            f"pts={self.valid_points}, len={self.path_length_px:.1f}px, disp={self.net_displacement_px:.1f}px, "
            f"tort={self.tortuosity:.2f}, turns={self.direction_changes_count}, speed={speed_str})"
        )


def validate_trajectory_points(raw_points: Any) -> Tuple[List[Tuple[float, float]], int]:
    """
    Validates and extracts finite 2D coordinate pairs from any sequence.
    Rejects or skips non-finite (NaN, Inf), non-numeric, or malformed points.
    Returns: (valid_points, total_points_received)
    """
    if raw_points is None:
        return [], 0

    try:
        points_list = list(raw_points)
    except TypeError:
        return [], 0

    total_count = len(points_list)
    valid_points: List[Tuple[float, float]] = []

    for pt in points_list:
        if isinstance(pt, (list, tuple)) and len(pt) >= 2:
            try:
                x = float(pt[0])
                y = float(pt[1])
                if math.isfinite(x) and math.isfinite(y):
                    valid_points.append((x, y))
            except (ValueError, TypeError):
                continue

    return valid_points, total_count


def compute_path_length_and_displacement(
    points: List[Tuple[float, float]]
) -> Tuple[float, float]:
    """
    Calculates cumulative path length and net displacement in pixels.
    - Path length: sum of Euclidean distances between consecutive points.
    - Net displacement: straight-line distance from first to last point.
    """
    if len(points) < 2:
        return 0.0, 0.0

    path_length = 0.0
    for i in range(1, len(points)):
        p_prev = points[i - 1]
        p_curr = points[i]
        step = math.hypot(p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        if math.isfinite(step):
            path_length += step

    p_first = points[0]
    p_last = points[-1]
    net_disp = math.hypot(p_last[0] - p_first[0], p_last[1] - p_first[1])
    if not math.isfinite(net_disp):
        net_disp = 0.0

    return max(0.0, round(path_length, 2)), max(0.0, round(net_disp, 2))


def compute_tortuosity(
    path_length_px: float,
    net_displacement_px: float,
    min_displacement_epsilon_px: float = 3.0,
    max_tortuosity_cap: float = 100.0
) -> float:
    """
    Computes trajectory tortuosity as path_length / net_displacement.

    Documented Definition:
    - Tortuosity measures the curvature/winding nature of a path.
    - Straight-line trajectories yield tortuosity == 1.0.
    - Curved, wandering, or meandering paths yield tortuosity > 1.0.
    - Stationary objects (path length < epsilon) yield a baseline tortuosity of 1.0.
    - Closed loops or return paths (displacement < epsilon while path length >= epsilon)
      use epsilon as the minimum denominator to prevent division by zero or infinite values,
      and are capped at max_tortuosity_cap (default: 100.0).
    - Output is strictly bounded >= 1.0 and finite (never NaN or Infinity).
    """
    eps = max(0.1, float(min_displacement_epsilon_px))
    if net_displacement_px < eps:
        if path_length_px < eps:
            return 1.0
        raw_tort = path_length_px / eps
        return max(1.0, round(min(raw_tort, max_tortuosity_cap), 2))

    raw_tort = path_length_px / net_displacement_px
    return max(1.0, round(min(raw_tort, max_tortuosity_cap), 2))


def compute_direction_changes(
    points: List[Tuple[float, float]],
    min_segment_length_px: float = 6.0,
    min_turn_angle_deg: float = 45.0
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Detects meaningful direction changes along a trajectory while suppressing pixel jitter.

    Algorithm:
    1. Spatially filters points by accumulating distance until reaching min_segment_length_px.
       Sub-pixel detector jitter and micro-oscillations below this threshold are discarded.
    2. Constructs movement vectors between consecutive filtered vertices.
    3. Calculates the angle between consecutive vectors via vector dot product.
    4. Registers a direction change if the angle equals or exceeds min_turn_angle_deg.
    """
    if len(points) < 3:
        return 0, []

    min_seg = max(1.0, float(min_segment_length_px))
    min_angle = max(1.0, min(180.0, float(min_turn_angle_deg)))

    # Step 1: Spatial filter to accumulate significant movement nodes
    simplified = [points[0]]
    for pt in points[1:]:
        last = simplified[-1]
        dist = math.hypot(pt[0] - last[0], pt[1] - last[1])
        if dist >= min_seg:
            simplified.append(pt)

    if len(simplified) < 3:
        return 0, []

    # Step 2: Form directional movement vectors
    vectors = []
    for i in range(len(simplified) - 1):
        p1 = simplified[i]
        p2 = simplified[i + 1]
        vx = p2[0] - p1[0]
        vy = p2[1] - p1[1]
        vectors.append((vx, vy))

    # Step 3: Compute angular deviations between consecutive vectors
    turn_count = 0
    turn_details: List[Dict[str, Any]] = []

    for i in range(len(vectors) - 1):
        v1 = vectors[i]
        v2 = vectors[i + 1]
        mag1 = math.hypot(v1[0], v1[1])
        mag2 = math.hypot(v2[0], v2[1])
        if mag1 == 0.0 or mag2 == 0.0:
            continue

        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_theta = dot / (mag1 * mag2)
        # Numerical clamp for floating-point safety
        cos_theta = max(-1.0, min(1.0, cos_theta))
        angle_deg = math.degrees(math.acos(cos_theta))

        if angle_deg >= min_angle:
            turn_count += 1
            turn_details.append({
                "vertex_index": i + 1,
                "vertex_point": (int(round(simplified[i + 1][0])), int(round(simplified[i + 1][1]))),
                "angle_deg": round(angle_deg, 1),
                "from_vector": (round(v1[0], 1), round(v1[1], 1)),
                "to_vector": (round(v2[0], 1), round(v2[1], 1))
            })

    return turn_count, turn_details


def compute_speed(
    path_length_px: float,
    elapsed_time_sec: Optional[float] = None
) -> Tuple[Optional[float], Optional[str]]:
    """
    Calculates average path speed in pixels per second.
    Returns: (speed_px_per_sec, reason_if_unavailable)
    """
    if elapsed_time_sec is None:
        return None, "Elapsed time was not provided"

    try:
        dt = float(elapsed_time_sec)
    except (ValueError, TypeError):
        return None, "Elapsed time is not a valid numeric value"

    if not math.isfinite(dt):
        return None, "Elapsed time is non-finite (NaN or Inf)"

    if dt <= 0.0:
        return None, "Elapsed time is non-positive (<= 0.0s)"

    speed = path_length_px / dt
    if not math.isfinite(speed):
        return None, "Calculated speed is non-finite"

    return max(0.0, round(speed, 2)), None


def compute_trajectory_metrics(
    trajectory: Any,
    track_id: int = 0,
    class_name: str = "object",
    elapsed_time_sec: Optional[float] = None,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
    first_seen_frame: Optional[int] = None,
    last_seen_frame: Optional[int] = None,
    fps: Optional[float] = None,
    lifecycle_id: Optional[int] = None,
    min_points_for_analysis: int = 3,
    min_segment_length_px: float = 6.0,
    min_turn_angle_deg: float = 45.0,
    min_displacement_epsilon_px: float = 3.0
) -> TrajectoryMetrics:
    """
    Computes complete, numerically safe trajectory metrics from a point sequence.

    Args:
        trajectory: Iterable of 2D coordinates (tuples, lists, or TrackedObject trajectory).
        track_id: Persistent identifier of the tracked entity.
        class_name: Semantic category designation.
        elapsed_time_sec: Explicit elapsed duration in seconds for this trajectory window.
        start_time: Starting timestamp of trajectory.
        end_time: Ending timestamp of trajectory.
        first_seen_frame: Earliest frame index of this trajectory window.
        last_seen_frame: Latest frame index of this trajectory window.
        fps: Video framerate for timing fallback.
        lifecycle_id: Unique lifecycle occurrence sequence for this track.
        min_points_for_analysis: Minimum points needed for curvature/direction analysis.
        min_segment_length_px: Pixel threshold to suppress detector jitter.
        min_turn_angle_deg: Angular threshold for meaningful turns.
        min_displacement_epsilon_px: Epsilon to guard against division by zero in tortuosity.

    Returns:
        TrajectoryMetrics: Fully populated metric record.
    """
    valid_points, total_points = validate_trajectory_points(trajectory)
    min_points = max(2, int(min_points_for_analysis))

    # Resolve timing basis
    resolved_dt: Optional[float] = None
    timing_unusable_reason: Optional[str] = None

    if elapsed_time_sec is not None:
        try:
            val = float(elapsed_time_sec)
            if math.isfinite(val) and val > 0.0:
                resolved_dt = val
            elif val <= 0.0:
                timing_unusable_reason = "Elapsed time is non-positive (<= 0.0s)"
            else:
                timing_unusable_reason = "Elapsed time is non-finite (NaN or Inf)"
        except (ValueError, TypeError):
            timing_unusable_reason = "Elapsed time is not a valid numeric value"
    elif start_time is not None and end_time is not None:
        try:
            st = float(start_time)
            et = float(end_time)
            if math.isfinite(st) and math.isfinite(et):
                if et > st:
                    resolved_dt = et - st
                else:
                    timing_unusable_reason = f"Ending time ({et}s) <= starting time ({st}s)"
            else:
                timing_unusable_reason = "Supplied timestamps are non-finite (NaN or Inf)"
        except (ValueError, TypeError):
            timing_unusable_reason = "Supplied timestamps are not valid numbers"
    elif first_seen_frame is not None and last_seen_frame is not None and fps is not None:
        try:
            f1 = int(first_seen_frame)
            f2 = int(last_seen_frame)
            fps_val = float(fps)
            if fps_val > 0:
                if f2 > f1:
                    resolved_dt = (f2 - f1) / fps_val
                else:
                    timing_unusable_reason = f"Last frame ({f2}) <= first frame ({f1})"
            else:
                timing_unusable_reason = f"FPS must be positive, got {fps_val}"
        except (ValueError, TypeError):
            timing_unusable_reason = "Frame indices or FPS are not valid numbers"
    elif fps is not None and len(valid_points) >= 2:
        try:
            fps_val = float(fps)
            if fps_val > 0:
                # Lowest-tier fallback: assumes contiguous consecutive frames
                resolved_dt = (len(valid_points) - 1) / fps_val
            else:
                timing_unusable_reason = f"FPS must be positive, got {fps_val}"
        except (ValueError, TypeError):
            timing_unusable_reason = "FPS is not a valid number"

    # Degenerate: 0 valid points
    if len(valid_points) == 0:
        return TrajectoryMetrics(
            track_id=track_id,
            lifecycle_id=lifecycle_id,
            class_name=class_name,
            total_points=total_points,
            valid_points=0,
            is_sufficient=False,
            path_length_px=0.0,
            net_displacement_px=0.0,
            tortuosity=1.0,
            direction_changes_count=0,
            direction_changes=[],
            avg_speed_px_per_sec=None,
            speed_unavailable_reason="No valid trajectory points",
            start_centroid=None,
            end_centroid=None,
            first_seen_frame=first_seen_frame,
            last_seen_frame=last_seen_frame,
            elapsed_time_sec=None
        )

    start_c = (int(round(valid_points[0][0])), int(round(valid_points[0][1])))
    end_c = (int(round(valid_points[-1][0])), int(round(valid_points[-1][1])))

    # Degenerate: 1 point
    if len(valid_points) == 1:
        return TrajectoryMetrics(
            track_id=track_id,
            lifecycle_id=lifecycle_id,
            class_name=class_name,
            total_points=total_points,
            valid_points=1,
            is_sufficient=False,
            path_length_px=0.0,
            net_displacement_px=0.0,
            tortuosity=1.0,
            direction_changes_count=0,
            direction_changes=[],
            avg_speed_px_per_sec=None,
            speed_unavailable_reason="Insufficient points for speed (minimum 2 points required)",
            start_centroid=start_c,
            end_centroid=end_c,
            first_seen_frame=first_seen_frame,
            last_seen_frame=last_seen_frame,
            elapsed_time_sec=None
        )

    # 2 or more valid points
    path_len, net_disp = compute_path_length_and_displacement(valid_points)
    tort = compute_tortuosity(path_len, net_disp, min_displacement_epsilon_px)
    speed, speed_reason = compute_speed(path_len, resolved_dt)
    if speed is None and timing_unusable_reason is not None:
        speed_reason = timing_unusable_reason

    if len(valid_points) < min_points:
        is_suff = False
        turns_cnt = 0
        turns_det: List[Dict[str, Any]] = []
    else:
        is_suff = True
        turns_cnt, turns_det = compute_direction_changes(valid_points, min_segment_length_px, min_turn_angle_deg)

    return TrajectoryMetrics(
        track_id=track_id,
        lifecycle_id=lifecycle_id,
        class_name=class_name,
        total_points=total_points,
        valid_points=len(valid_points),
        is_sufficient=is_suff,
        path_length_px=path_len,
        net_displacement_px=net_disp,
        tortuosity=tort,
        direction_changes_count=turns_cnt,
        direction_changes=turns_det,
        avg_speed_px_per_sec=speed,
        speed_unavailable_reason=speed_reason,
        start_centroid=start_c,
        end_centroid=end_c,
        first_seen_frame=first_seen_frame,
        last_seen_frame=last_seen_frame,
        elapsed_time_sec=resolved_dt
    )


class _TrackBehaviorState:
    """Internal session tracking state for an individual active track ID."""

    def __init__(
        self,
        track_id: int,
        class_name: str,
        first_frame: int,
        first_ts: Optional[float],
        lifecycle_id: int,
        max_trajectory_len: int = 45
    ):
        self.track_id = int(track_id)
        self.lifecycle_id = int(lifecycle_id)
        self.class_name = str(class_name)
        self.first_seen_frame = int(first_frame)
        self.last_seen_frame = int(first_frame)
        self.first_seen_timestamp: Optional[float] = float(first_ts) if first_ts is not None else None
        self.last_seen_timestamp: Optional[float] = float(first_ts) if first_ts is not None else None
        self.frames_unseen = 0
        self.is_active = True

        # Aligned observation buffer strictly matching trajectory length:
        # Each entry is: (centroid_xy, frame_idx, timestamp_sec_or_None)
        self.observations: deque[Tuple[Tuple[int, int], Optional[int], Optional[float]]] = deque(maxlen=max_trajectory_len)
        self.latest_metrics: Optional[TrajectoryMetrics] = None


class BehaviorAnalytics:
    """
    Session-level behavior analytics engine for multi-object tracking.
    Evaluates trajectory curvature, path length, net displacement, speed,
    and direction changes across multiple persistent object tracks.
    Maintains cumulative session aggregates alongside bounded retained history.
    """

    def __init__(
        self,
        min_points_for_analysis: int = 3,
        min_segment_length_px: float = 6.0,
        min_turn_angle_deg: float = 45.0,
        min_displacement_epsilon_px: float = 3.0,
        max_tracks_history: int = 500,
        max_trajectory_len: int = 45,
        disappearance_grace_frames: int = 15
    ):
        self.min_points_for_analysis = max(2, int(min_points_for_analysis))
        self.min_segment_length_px = max(1.0, float(min_segment_length_px))
        self.min_turn_angle_deg = max(1.0, float(min_turn_angle_deg))
        self.min_displacement_epsilon_px = max(0.1, float(min_displacement_epsilon_px))
        self.max_tracks_history = max(1, int(max_tracks_history))
        self.max_trajectory_len = max(10, int(max_trajectory_len))
        self.disappearance_grace_frames = max(1, int(disappearance_grace_frames))

        # Lifecycle identity generator
        self._lifecycle_seq: int = 0

        # Active tracks keyed by track_id
        self._active_tracks: Dict[int, _TrackBehaviorState] = {}

        # Retained completed tracks history (bounded to max_tracks_history)
        self._completed_tracks_history: List[TrajectoryMetrics] = []

        # Cumulative session aggregates (preserved against history pruning)
        self._cumulative_completed_tracks_count: int = 0
        self._cumulative_path_length_px: float = 0.0
        self._cumulative_net_displacement_px: float = 0.0
        self._cumulative_tortuosity_sum: float = 0.0
        self._cumulative_direction_changes: int = 0
        self._cumulative_speed_sum: float = 0.0
        self._cumulative_speed_count: int = 0
        self._cumulative_by_class: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
            "count": 0,
            "path_length_px": 0.0,
            "net_displacement_px": 0.0,
            "tortuosity_sum": 0.0,
            "direction_changes": 0,
            "speed_sum": 0.0,
            "speed_count": 0
        })

        self._latest_frame_idx: int = 0
        self._latest_timestamp: float = 0.0

    @property
    def _completed_tracks(self) -> Dict[int, TrajectoryMetrics]:
        """Backward-compatible mapping of latest completed metrics by track_id."""
        return {m.track_id: m for m in self._completed_tracks_history}

    def reset(self) -> None:
        """Resets all session tracking, cumulative aggregates, and history."""
        self._active_tracks.clear()
        self._completed_tracks_history.clear()
        self._lifecycle_seq = 0
        self._cumulative_completed_tracks_count = 0
        self._cumulative_path_length_px = 0.0
        self._cumulative_net_displacement_px = 0.0
        self._cumulative_tortuosity_sum = 0.0
        self._cumulative_direction_changes = 0
        self._cumulative_speed_sum = 0.0
        self._cumulative_speed_count = 0
        self._cumulative_by_class.clear()
        self._latest_frame_idx = 0
        self._latest_timestamp = 0.0

    def _archive_completed_track(self, metrics: TrajectoryMetrics) -> None:
        """Archives a completed track lifecycle into history and updates cumulative totals."""
        self._completed_tracks_history.append(metrics)
        if len(self._completed_tracks_history) > self.max_tracks_history:
            self._completed_tracks_history.pop(0)

        self._cumulative_completed_tracks_count += 1
        self._cumulative_path_length_px += metrics.path_length_px
        self._cumulative_net_displacement_px += metrics.net_displacement_px
        self._cumulative_tortuosity_sum += metrics.tortuosity
        self._cumulative_direction_changes += metrics.direction_changes_count
        if metrics.avg_speed_px_per_sec is not None:
            self._cumulative_speed_sum += metrics.avg_speed_px_per_sec
            self._cumulative_speed_count += 1

        c_data = self._cumulative_by_class[metrics.class_name]
        c_data["count"] += 1
        c_data["path_length_px"] += metrics.path_length_px
        c_data["net_displacement_px"] += metrics.net_displacement_px
        c_data["tortuosity_sum"] += metrics.tortuosity
        c_data["direction_changes"] += metrics.direction_changes_count
        if metrics.avg_speed_px_per_sec is not None:
            c_data["speed_sum"] += metrics.avg_speed_px_per_sec
            c_data["speed_count"] += 1

    def update(
        self,
        frame_idx: int,
        tracks: List[Any],
        timestamp_sec: Optional[float] = None,
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """
        Updates trajectory behavior analytics for all active tracks in the frame.
        Aligns observation window with timestamps and frame numbers.

        Args:
            frame_idx: Current frame counter (1-indexed).
            tracks: List of active TrackedObject instances.
            timestamp_sec: Optional stream time in seconds.
            fps: Video framerate for timestamp fallback.

        Returns:
            Dict[str, Any]: Frame behavior analytics snapshot.
        """
        self._latest_frame_idx = frame_idx
        ts: Optional[float] = None
        if timestamp_sec is not None:
            try:
                val = float(timestamp_sec)
                if math.isfinite(val) and val >= 0.0:
                    ts = val
            except (ValueError, TypeError):
                pass
        if ts is None:
            ts = (frame_idx / fps) if fps > 0 else float(frame_idx)
        self._latest_timestamp = ts

        active_track_ids = set()
        frame_metrics: Dict[int, Dict[str, Any]] = {}

        # 1. Update active tracks
        for t in tracks or []:
            tid = getattr(t, "track_id", None)
            cname = getattr(t, "class_name", "object")
            if tid is None:
                continue

            tid = int(tid)
            active_track_ids.add(tid)

            # Extract centroid
            centroid = None
            raw_traj = getattr(t, "trajectory", None)
            if raw_traj:
                try:
                    centroid = raw_traj[-1]
                except (IndexError, TypeError):
                    centroid = None

            if centroid is None or not (isinstance(centroid, (list, tuple)) and len(centroid) >= 2):
                centroid = getattr(t, "current_centroid", None)

            if centroid is None or not (isinstance(centroid, (list, tuple)) and len(centroid) >= 2):
                box = getattr(t, "box", None)
                if box and len(box) == 4:
                    centroid = (int((box[0] + box[2]) / 2.0), int((box[1] + box[3]) / 2.0))

            if tid not in self._active_tracks:
                self._lifecycle_seq += 1
                state = _TrackBehaviorState(
                    track_id=tid,
                    class_name=cname,
                    first_frame=frame_idx,
                    first_ts=ts,
                    lifecycle_id=self._lifecycle_seq,
                    max_trajectory_len=self.max_trajectory_len
                )
                self._active_tracks[tid] = state

                # If track already had prior trajectory points (e.g. from TrackedObject),
                # backfill earlier points with None timing so spatial calculations have full points
                raw_traj = getattr(t, "trajectory", None)
                if raw_traj:
                    traj_list = list(raw_traj)
                    if len(traj_list) > 1:
                        # Add earlier points without timing
                        for pt in traj_list[:-1]:
                            if isinstance(pt, (list, tuple)) and len(pt) >= 2:
                                pt_tup = (int(round(pt[0])), int(round(pt[1])))
                                state.observations.append((pt_tup, None, None))
            else:
                state = self._active_tracks[tid]
                state.last_seen_frame = frame_idx
                state.last_seen_timestamp = ts
                state.frames_unseen = 0

            if centroid is not None:
                c_tuple = (int(round(centroid[0])), int(round(centroid[1])))
                state.observations.append((c_tuple, frame_idx, ts))

            # Derive observation-aligned points and duration
            pts = [obs[0] for obs in state.observations]
            start_obs = state.observations[0]
            end_obs = state.observations[-1]

            # Timing is reliable only when both earliest and latest points in this window have timestamps or frames
            dt: Optional[float] = None
            if len(state.observations) >= 2:
                first_obs_ts = start_obs[2]
                last_obs_ts = end_obs[2]
                first_obs_frame = start_obs[1]
                last_obs_frame = end_obs[1]

                if first_obs_ts is not None and last_obs_ts is not None:
                    if last_obs_ts > first_obs_ts:
                        dt = last_obs_ts - first_obs_ts
                elif first_obs_frame is not None and last_obs_frame is not None and fps > 0:
                    if last_obs_frame > first_obs_frame:
                        dt = (last_obs_frame - first_obs_frame) / fps

            metrics = compute_trajectory_metrics(
                trajectory=pts,
                track_id=tid,
                class_name=cname,
                elapsed_time_sec=dt,
                first_seen_frame=start_obs[1],
                last_seen_frame=end_obs[1],
                fps=fps,
                lifecycle_id=state.lifecycle_id,
                min_points_for_analysis=self.min_points_for_analysis,
                min_segment_length_px=self.min_segment_length_px,
                min_turn_angle_deg=self.min_turn_angle_deg,
                min_displacement_epsilon_px=self.min_displacement_epsilon_px
            )
            state.latest_metrics = metrics
            frame_metrics[tid] = metrics.to_dict()

        # 2. Handle disappeared tracks
        for tid, state in list(self._active_tracks.items()):
            if tid not in active_track_ids:
                state.frames_unseen += 1
                if state.frames_unseen >= self.disappearance_grace_frames:
                    state.is_active = False
                    if state.latest_metrics is not None:
                        self._archive_completed_track(state.latest_metrics)
                    del self._active_tracks[tid]

        return {
            "frame_idx": frame_idx,
            "timestamp_sec": round(ts, 2),
            "active_tracks_count": len(self._active_tracks),
            "frame_track_metrics": frame_metrics,
            "total_tracks_logged": self._cumulative_completed_tracks_count + len(self._active_tracks)
        }

    def finalize_stream(
        self,
        last_frame_idx: Optional[int] = None,
        timestamp_sec: Optional[float] = None
    ) -> None:
        """
        Cleans up stream session, archiving remaining active tracks to completed history.
        Idempotent across repeated calls.
        """
        for tid, state in list(self._active_tracks.items()):
            state.is_active = False
            if state.latest_metrics is not None:
                self._archive_completed_track(state.latest_metrics)
        self._active_tracks.clear()

    def get_track_metrics(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Returns trajectory metrics for a specific track ID (active or latest completed)."""
        tid = int(track_id)
        if tid in self._active_tracks and self._active_tracks[tid].latest_metrics is not None:
            return self._active_tracks[tid].latest_metrics.to_dict()
        for m in reversed(self._completed_tracks_history):
            if m.track_id == tid:
                return m.to_dict()
        return None

    def get_completed_tracks(self, track_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Returns retained completed track records, optionally filtered by track_id."""
        if track_id is not None:
            tid = int(track_id)
            return [m.to_dict() for m in self._completed_tracks_history if m.track_id == tid]
        return [m.to_dict() for m in self._completed_tracks_history]

    def get_all_metrics(self) -> List[Dict[str, Any]]:
        """Returns metrics for all active tracks and all retained completed lifecycles."""
        results = []
        for state in self._active_tracks.values():
            if state.latest_metrics is not None:
                results.append(state.latest_metrics.to_dict())
        for m in self._completed_tracks_history:
            results.append(m.to_dict())
        return results

    def get_summary(self) -> Dict[str, Any]:
        """
        Computes structured session behavior summary.
        Combines cumulative session aggregates with retained history diagnostics.
        """
        # Active tracks
        active_list = [s.latest_metrics for s in self._active_tracks.values() if s.latest_metrics is not None]
        active_count = len(active_list)
        total_analyzed = self._cumulative_completed_tracks_count + active_count

        if total_analyzed == 0:
            return {
                "total_tracks_analyzed": 0,
                "active_tracks_count": 0,
                "completed_tracks_count": 0,
                "retained_completed_tracks_count": 0,
                "retained_history_limit": self.max_tracks_history,
                "avg_path_length_px": 0.0,
                "avg_net_displacement_px": 0.0,
                "avg_tortuosity": 1.0,
                "avg_speed_px_per_sec": None,
                "total_direction_changes": 0,
                "class_breakdown": {}
            }

        # Combine cumulative completed sums + current active tracks
        total_path_len = self._cumulative_path_length_px + sum(m.path_length_px for m in active_list)
        total_disp = self._cumulative_net_displacement_px + sum(m.net_displacement_px for m in active_list)
        total_tort = self._cumulative_tortuosity_sum + sum(m.tortuosity for m in active_list)
        total_turns = self._cumulative_direction_changes + sum(m.direction_changes_count for m in active_list)

        active_speeds = [m.avg_speed_px_per_sec for m in active_list if m.avg_speed_px_per_sec is not None]
        total_speed_sum = self._cumulative_speed_sum + sum(active_speeds)
        total_speed_count = self._cumulative_speed_count + len(active_speeds)
        overall_avg_speed = round(total_speed_sum / total_speed_count, 2) if total_speed_count > 0 else None

        # Per-class summary
        class_breakdown: Dict[str, Dict[str, Any]] = {}
        all_classes = set(self._cumulative_by_class.keys()) | {m.class_name for m in active_list}

        for cname in all_classes:
            c_cum = self._cumulative_by_class[cname]
            c_active = [m for m in active_list if m.class_name == cname]

            c_count = c_cum["count"] + len(c_active)
            if c_count == 0:
                continue

            c_path = c_cum["path_length_px"] + sum(m.path_length_px for m in c_active)
            c_disp = c_cum["net_displacement_px"] + sum(m.net_displacement_px for m in c_active)
            c_tort = c_cum["tortuosity_sum"] + sum(m.tortuosity for m in c_active)
            c_turns = c_cum["direction_changes"] + sum(m.direction_changes_count for m in c_active)

            c_act_speeds = [m.avg_speed_px_per_sec for m in c_active if m.avg_speed_px_per_sec is not None]
            c_speed_sum = c_cum["speed_sum"] + sum(c_act_speeds)
            c_speed_count = c_cum["speed_count"] + len(c_act_speeds)

            class_breakdown[cname] = {
                "track_count": c_count,
                "avg_path_length_px": round(c_path / c_count, 2),
                "avg_net_displacement_px": round(c_disp / c_count, 2),
                "avg_tortuosity": round(c_tort / c_count, 2),
                "total_direction_changes": c_turns,
                "avg_speed_px_per_sec": round(c_speed_sum / c_speed_count, 2) if c_speed_count > 0 else None
            }

        return {
            "total_tracks_analyzed": total_analyzed,
            "active_tracks_count": active_count,
            "completed_tracks_count": self._cumulative_completed_tracks_count,
            "retained_completed_tracks_count": len(self._completed_tracks_history),
            "retained_history_limit": self.max_tracks_history,
            "avg_path_length_px": round(total_path_len / total_analyzed, 2),
            "avg_net_displacement_px": round(total_disp / total_analyzed, 2),
            "avg_tortuosity": round(total_tort / total_analyzed, 2),
            "avg_speed_px_per_sec": overall_avg_speed,
            "total_direction_changes": total_turns,
            "class_breakdown": class_breakdown
        }
