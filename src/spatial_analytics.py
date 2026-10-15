"""
Spatial Analytics Module for CodeAlpha Task 4: Object Detection and Tracking.
Provides explainable virtual line-crossing detection, rectangular region of interest (ROI)
occupancy analytics, directional classification, boundary hysteresis, and structured spatial events.
"""

from typing import List, Dict, Any, Optional, Tuple, Set, Union
from collections import Counter, deque
import math
import cv2
import numpy as np


class CrossingDirection:
    """Standardized directional designations for line-crossing occurrences."""
    A_TO_B = "A_TO_B"
    B_TO_A = "B_TO_A"
    TOP_TO_BOTTOM = "TOP_TO_BOTTOM"
    BOTTOM_TO_TOP = "BOTTOM_TO_TOP"
    LEFT_TO_RIGHT = "LEFT_TO_RIGHT"
    RIGHT_TO_LEFT = "RIGHT_TO_LEFT"


class RegionState:
    """State designations for object positions relative to a rectangular region."""
    OUTSIDE = "OUTSIDE"
    INSIDE = "INSIDE"
    TERMINATED_INSIDE = "TERMINATED_INSIDE"


class SpatialEvent:
    """Represents a structured geometric occurrence (line crossing, ROI entry/exit)."""

    def __init__(
        self,
        event_type: str,
        track_id: int,
        class_name: str,
        frame_idx: int,
        timestamp_sec: float,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.event_type = str(event_type)  # "LINE_CROSS", "ROI_ENTRY", "ROI_EXIT"
        self.track_id = int(track_id)
        self.class_name = str(class_name)
        self.frame_idx = int(frame_idx)
        self.timestamp_sec = round(float(timestamp_sec), 2)
        self.description = str(description)
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_sec": self.timestamp_sec,
            "frame_idx": self.frame_idx,
            "event_type": self.event_type,
            "track_id": self.track_id,
            "class_name": self.class_name,
            "description": self.description,
            "details": self.details
        }

    def __repr__(self) -> str:
        return (
            f"SpatialEvent({self.event_type}: #{self.track_id} {self.class_name} "
            f"at frame {self.frame_idx} [{self.timestamp_sec}s] - {self.description})"
        )


def validate_line_coords(
    coords: Union[List[int], Tuple[int, int, int, int]],
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None
) -> Tuple[int, int, int, int]:
    """
    Validates virtual line endpoints (x1, y1, x2, y2).
    Ensures non-zero length and boundary compliance.
    """
    if len(coords) != 4:
        raise ValueError(f"Line coordinates must contain exactly 4 integers (x1, y1, x2, y2), got {len(coords)}")

    x1, y1, x2, y2 = [int(v) for v in coords]
    if x1 == x2 and y1 == y2:
        raise ValueError(f"Virtual line endpoints cannot be identical: ({x1}, {y1}) to ({x2}, {y2}) has zero length.")

    if frame_width is not None and frame_height is not None:
        if not (0 <= x1 <= frame_width and 0 <= x2 <= frame_width and 0 <= y1 <= frame_height and 0 <= y2 <= frame_height):
            raise ValueError(
                f"Line coordinates ({x1}, {y1})->({x2}, {y2}) extend outside frame bounds ({frame_width}x{frame_height})."
            )

    return (x1, y1, x2, y2)


def validate_roi_coords(
    coords: Union[List[int], Tuple[int, int, int, int]],
    frame_width: Optional[int] = None,
    frame_height: Optional[int] = None
) -> Tuple[int, int, int, int]:
    """
    Validates rectangular region of interest (x1, y1, x2, y2).
    Ensures positive area and boundary compliance.
    """
    if len(coords) != 4:
        raise ValueError(f"ROI coordinates must contain exactly 4 integers (x1, y1, x2, y2), got {len(coords)}")

    x1, y1, x2, y2 = [int(v) for v in coords]
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"ROI coordinates must satisfy x1 < x2 and y1 < y2, got ({x1}, {y1}, {x2}, {y2}).")

    if x1 < 0 or y1 < 0:
        raise ValueError(f"ROI coordinates cannot be negative, got ({x1}, {y1}, {x2}, {y2}).")

    if frame_width is not None and frame_height is not None:
        if x2 > frame_width or y2 > frame_height:
            raise ValueError(
                f"ROI ({x1}, {y1}, {x2}, {y2}) exceeds frame dimensions ({frame_width}x{frame_height})."
            )

    return (x1, y1, x2, y2)


def parse_spatial_coords_string(coord_str: str) -> Tuple[int, int, int, int]:
    """
    Parses comma- or space-separated coordinates string into a 4-integer tuple.
    Accepts integers and whole-valued floats (e.g. '100,200,500,200' or '100.0, 200.0, 500.0, 200.0').
    Rejects malformed, non-finite (NaN, Inf), or fractional numeric values.
    """
    clean_str = coord_str.replace(",", " ").strip()
    parts = clean_str.split()
    if len(parts) != 4:
        raise ValueError(f"Expected 4 coordinate values, received '{coord_str}' ({len(parts)} items).")

    parsed_coords = []
    for p in parts:
        try:
            val_f = float(p)
        except ValueError as err:
            raise ValueError(f"Coordinates must be integers or whole-valued floats, received invalid '{p}' in '{coord_str}': {err}") from err

        if not math.isfinite(val_f):
            raise ValueError(f"Coordinates must be finite integers, received non-finite '{p}' in '{coord_str}'.")

        if not val_f.is_integer():
            raise ValueError(
                f"Coordinates must be integers or whole-valued floats, received fractional value '{p}' in '{coord_str}'."
            )

        parsed_coords.append(int(round(val_f)))

    return (parsed_coords[0], parsed_coords[1], parsed_coords[2], parsed_coords[3])


class TrackSpatialState:
    """Internal spatial tracking state maintained for each persistent object ID."""

    def __init__(self, track_id: int, class_name: str, initial_centroid: Tuple[int, int], frame_idx: int):
        self.track_id = int(track_id)
        self.class_name = str(class_name)
        self.last_seen_frame = int(frame_idx)
        self.frames_unseen = 0
        self.centroid_history: deque = deque(maxlen=20)
        self.centroid_history.append(initial_centroid)

        # Line crossing states
        self.last_solid_side: Optional[int] = None  # +1 (Side A), -1 (Side B), None (indeterminate)
        self.last_line_cross_frame: int = -999

        # ROI states
        self.region_state: str = RegionState.OUTSIDE
        self.last_roi_event_frame: int = -999
        self.terminated_inside: bool = False


class SpatialAnalytics:
    """
    Modular, explainable spatial analytics engine for object detection and tracking.
    Computes:
    - Virtual line crossings with directional classification and hysteresis
    - Rectangular Region of Interest (ROI) occupancy, entry, and exit
    - Class-specific cumulative and instantaneous telemetry
    - Chronological, bounded event logging
    """

    def __init__(
        self,
        virtual_line: Optional[Union[List[int], Tuple[int, int, int, int]]] = None,
        roi_region: Optional[Union[List[int], Tuple[int, int, int, int]]] = None,
        line_hysteresis_px: float = 8.0,
        roi_hysteresis_px: float = 8.0,
        line_cooldown_frames: int = 5,
        roi_cooldown_frames: int = 5,
        disappearance_grace_frames: int = 10,
        max_events_history: int = 250,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None
    ):
        self.line_hysteresis_px = max(1.0, float(line_hysteresis_px))
        self.roi_hysteresis_px = max(1.0, float(roi_hysteresis_px))
        self.line_cooldown_frames = max(1, int(line_cooldown_frames))
        self.roi_cooldown_frames = max(1, int(roi_cooldown_frames))
        self.disappearance_grace_frames = max(1, int(disappearance_grace_frames))
        self.max_events_history = max(1, int(max_events_history))
        self.frame_width = frame_width
        self.frame_height = frame_height

        self.virtual_line: Optional[Tuple[int, int, int, int]] = None
        if virtual_line is not None:
            self.set_virtual_line(virtual_line)

        self.roi_region: Optional[Tuple[int, int, int, int]] = None
        if roi_region is not None:
            self.set_roi_region(roi_region)

        # Track history mapping
        self._tracks: Dict[int, TrackSpatialState] = {}
        self._events: List[SpatialEvent] = []

        # Line crossing statistics
        self._line_crossings_total: int = 0
        self._line_crossings_by_direction: Counter = Counter()
        self._line_crossings_by_class: Counter = Counter()
        self._line_crossings_detailed: Counter = Counter()  # (class_name, direction)

        # ROI statistics
        self._roi_current_inside: Set[int] = set()
        self._roi_entries_total: int = 0
        self._roi_entries_by_class: Counter = Counter()
        self._roi_exits_total: int = 0
        self._roi_exits_by_class: Counter = Counter()
        self._roi_terminated_inside_total: int = 0

        self._latest_frame_idx: int = 0
        self._latest_timestamp: float = 0.0

    def set_virtual_line(self, coords: Union[List[int], Tuple[int, int, int, int]]) -> None:
        """Sets and validates virtual line endpoints (x1, y1, x2, y2)."""
        self.virtual_line = validate_line_coords(coords, self.frame_width, self.frame_height)

    def set_roi_region(self, coords: Union[List[int], Tuple[int, int, int, int]]) -> None:
        """Sets and validates rectangular ROI coordinates (x1, y1, x2, y2)."""
        self.roi_region = validate_roi_coords(coords, self.frame_width, self.frame_height)

    def reset(self) -> None:
        """Resets all session tracking, cumulative counts, and event logs."""
        self._tracks.clear()
        self._events.clear()
        self._line_crossings_total = 0
        self._line_crossings_by_direction.clear()
        self._line_crossings_by_class.clear()
        self._line_crossings_detailed.clear()
        self._roi_current_inside.clear()
        self._roi_entries_total = 0
        self._roi_entries_by_class.clear()
        self._roi_exits_total = 0
        self._roi_exits_by_class.clear()
        self._roi_terminated_inside_total = 0
        self._latest_frame_idx = 0
        self._latest_timestamp = 0.0

    def _log_event(
        self,
        event_type: str,
        track_id: int,
        class_name: str,
        frame_idx: int,
        timestamp_sec: float,
        description: str,
        details: Optional[Dict[str, Any]] = None
    ) -> SpatialEvent:
        event = SpatialEvent(
            event_type=event_type,
            track_id=track_id,
            class_name=class_name,
            frame_idx=frame_idx,
            timestamp_sec=timestamp_sec,
            description=description,
            details=details
        )
        self._events.append(event)
        if len(self._events) > self.max_events_history:
            self._events.pop(0)
        return event

    def _compute_line_side(self, pt: Tuple[float, float]) -> Tuple[float, float]:
        """
        Computes signed perpendicular pixel distance and projection parameter t
        from point pt=(x, y) to the directed virtual line (x1, y1)->(x2, y2).
        Returns: (signed_distance_px, projection_t)
        """
        if self.virtual_line is None:
            return 0.0, 0.0

        x1, y1, x2, y2 = self.virtual_line
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = math.hypot(dx, dy)
        if length == 0.0:
            return 0.0, 0.0

        # Signed distance via 2D cross product
        # Positive: Side A (left/top relative to directed vector)
        # Negative: Side B (right/bottom relative to directed vector)
        cross = dx * (pt[1] - y1) - dy * (pt[0] - x1)
        signed_dist = cross / length

        # Projection along line segment
        dot = (pt[0] - x1) * dx + (pt[1] - y1) * dy
        t = dot / (length * length)

        return signed_dist, t

    def _segments_intersect(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        p3: Tuple[float, float],
        p4: Tuple[float, float],
        endpoint_tolerance: float = 0.15
    ) -> bool:
        """
        Determines if line segment p1-p2 intersects segment p3-p4.
        p3-p4 is the virtual line, with a small endpoint tolerance allowance.
        """
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        # Strict straddle check for trajectory p1-p2 across infinite line p3-p4
        d34 = math.hypot(p4[0] - p3[0], p4[1] - p3[1])
        if d34 == 0.0:
            return False

        # Projection parameter of trajectory intersection on line segment
        rx = p4[0] - p3[0]
        ry = p4[1] - p3[1]
        qx = p2[0] - p1[0]
        qy = p2[1] - p1[1]
        denom = rx * qy - ry * qx
        if abs(denom) < 1e-6:
            return False

        t_line = ((p1[0] - p3[0]) * qy - (p1[1] - p3[1]) * qx) / denom
        u_traj = ((p1[0] - p3[0]) * ry - (p1[1] - p3[1]) * rx) / denom

        # Check trajectory crosses (0 <= u <= 1) and crosses within segment length (+ tolerance)
        return (0.0 <= u_traj <= 1.0) and (-endpoint_tolerance <= t_line <= 1.0 + endpoint_tolerance)

    def _check_line_crossing(
        self,
        state: TrackSpatialState,
        curr_centroid: Tuple[int, int],
        frame_idx: int,
        timestamp_sec: float
    ) -> Optional[SpatialEvent]:
        """
        Evaluates centroid movement across the virtual line using hysteresis and segment intersection.
        """
        if self.virtual_line is None:
            return None

        signed_dist, t_proj = self._compute_line_side(curr_centroid)

        # Determine current solid side based on hysteresis threshold
        current_solid_side = None
        if signed_dist >= self.line_hysteresis_px:
            current_solid_side = 1  # Side A
        elif signed_dist <= -self.line_hysteresis_px:
            current_solid_side = -1  # Side B

        # Initial side registration
        if state.last_solid_side is None:
            if current_solid_side is not None:
                state.last_solid_side = current_solid_side
            return None

        # Cooldown guard: prevent jitter re-triggers
        if (frame_idx - state.last_line_cross_frame) < self.line_cooldown_frames:
            if current_solid_side is not None:
                state.last_solid_side = current_solid_side
            return None

        # Check for confirmed transition between opposite solid sides
        if current_solid_side is not None and current_solid_side != state.last_solid_side:
            prev_centroid = state.centroid_history[-2] if len(state.centroid_history) >= 2 else curr_centroid
            p3 = (float(self.virtual_line[0]), float(self.virtual_line[1]))
            p4 = (float(self.virtual_line[2]), float(self.virtual_line[3]))

            # Verify that the trajectory intersected the line segment
            crossed_segment = self._segments_intersect(prev_centroid, curr_centroid, p3, p4)

            # Fallback check: if projection t is within segment bounds [0, 1]
            if not crossed_segment:
                crossed_segment = -0.15 <= t_proj <= 1.15

            if crossed_segment:
                direction_id = CrossingDirection.A_TO_B if state.last_solid_side == 1 else CrossingDirection.B_TO_A

                # Screen coordinate descriptive label
                x1, y1, x2, y2 = self.virtual_line
                is_horizontal = abs(x2 - x1) >= abs(y2 - y1)
                if is_horizontal:
                    human_dir = CrossingDirection.TOP_TO_BOTTOM if prev_centroid[1] < curr_centroid[1] else CrossingDirection.BOTTOM_TO_TOP
                else:
                    human_dir = CrossingDirection.LEFT_TO_RIGHT if prev_centroid[0] < curr_centroid[0] else CrossingDirection.RIGHT_TO_LEFT

                self._line_crossings_total += 1
                self._line_crossings_by_direction[direction_id] += 1
                self._line_crossings_by_class[state.class_name] += 1
                self._line_crossings_detailed[(state.class_name, direction_id)] += 1

                state.last_line_cross_frame = frame_idx
                state.last_solid_side = current_solid_side

                desc = (
                    f"{state.class_name.capitalize()} #{state.track_id} crossed virtual line "
                    f"({direction_id} / {human_dir}) at frame {frame_idx}."
                )
                ev = self._log_event(
                    event_type="LINE_CROSS",
                    track_id=state.track_id,
                    class_name=state.class_name,
                    frame_idx=frame_idx,
                    timestamp_sec=timestamp_sec,
                    description=desc,
                    details={
                        "direction": direction_id,
                        "screen_direction": human_dir,
                        "centroid": curr_centroid,
                        "line": self.virtual_line
                    }
                )
                return ev

        if current_solid_side is not None:
            state.last_solid_side = current_solid_side

        return None

    def _check_roi_occupancy(
        self,
        state: TrackSpatialState,
        curr_centroid: Tuple[int, int],
        frame_idx: int,
        timestamp_sec: float
    ) -> Optional[SpatialEvent]:
        """
        Evaluates rectangular ROI occupancy using boundary hysteresis.
        Prevents jitter toggling along the perimeter.
        """
        if self.roi_region is None:
            return None

        rx1, ry1, rx2, ry2 = self.roi_region
        cx, cy = curr_centroid
        h = self.roi_hysteresis_px

        # Inner boundary: solid entry
        is_solidly_inside = (rx1 + h <= cx <= rx2 - h) and (ry1 + h <= cy <= ry2 - h)
        # Outer boundary: solid exit
        is_solidly_outside = (cx < rx1 - h) or (cx > rx2 + h) or (cy < ry1 - h) or (cy > ry2 + h)

        # Recovery Transition: TERMINATED_INSIDE -> OUTSIDE
        # If a track was marked TERMINATED_INSIDE and is later observed reliably outside the ROI
        # beyond the exit hysteresis boundary, safely recover its region state to OUTSIDE.
        # This allows subsequent genuine re-entries without emitting a false exit.
        if state.region_state == RegionState.TERMINATED_INSIDE and is_solidly_outside:
            state.region_state = RegionState.OUTSIDE
            state.terminated_inside = False

        # Transition 1: OUTSIDE -> INSIDE
        if state.region_state == RegionState.OUTSIDE and is_solidly_inside:
            state.region_state = RegionState.INSIDE
            state.last_roi_event_frame = frame_idx
            self._roi_current_inside.add(state.track_id)
            self._roi_entries_total += 1
            self._roi_entries_by_class[state.class_name] += 1

            desc = (
                f"{state.class_name.capitalize()} #{state.track_id} entered ROI "
                f"at ({cx}, {cy})."
            )
            return self._log_event(
                event_type="ROI_ENTRY",
                track_id=state.track_id,
                class_name=state.class_name,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                description=desc,
                details={"centroid": curr_centroid, "roi": self.roi_region}
            )

        # Transition 2: INSIDE -> OUTSIDE (Geometric boundary crossing only)
        elif state.region_state == RegionState.INSIDE and is_solidly_outside:
            state.region_state = RegionState.OUTSIDE
            state.last_roi_event_frame = frame_idx
            self._roi_current_inside.discard(state.track_id)
            self._roi_exits_total += 1
            self._roi_exits_by_class[state.class_name] += 1

            desc = (
                f"{state.class_name.capitalize()} #{state.track_id} exited ROI "
                f"at ({cx}, {cy})."
            )
            return self._log_event(
                event_type="ROI_EXIT",
                track_id=state.track_id,
                class_name=state.class_name,
                frame_idx=frame_idx,
                timestamp_sec=timestamp_sec,
                description=desc,
                details={"centroid": curr_centroid, "roi": self.roi_region}
            )

        return None

    def update(
        self,
        frame_idx: int,
        tracks: List[Any],
        timestamp_sec: Optional[float] = None,
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """
        Processes detections/tracks for the current frame and updates line/ROI spatial analytics.

        Args:
            frame_idx: Current frame counter (1-indexed).
            tracks: List of active TrackedObject instances.
            timestamp_sec: Optional stream time in seconds.
            fps: Video framerate for timestamp fallback.

        Returns:
            Dict[str, Any]: Frame spatial analytics snapshot.
        """
        self._latest_frame_idx = frame_idx
        if timestamp_sec is not None:
            ts = float(timestamp_sec)
        else:
            ts = (frame_idx / fps) if fps > 0 else float(frame_idx)
        self._latest_timestamp = ts

        active_track_ids = set()
        new_events = []

        # 1. Process active tracks in current frame
        for t in tracks or []:
            tid = getattr(t, "track_id", None)
            cname = getattr(t, "class_name", "object")
            centroid = getattr(t, "current_centroid", None)
            box = getattr(t, "box", None)

            if tid is None:
                continue

            tid = int(tid)
            active_track_ids.add(tid)

            if centroid is None or not (isinstance(centroid, (list, tuple)) and len(centroid) == 2):
                if box and len(box) == 4:
                    cx = int((box[0] + box[2]) / 2.0)
                    cy = int((box[1] + box[3]) / 2.0)
                    centroid = (cx, cy)
                else:
                    continue

            centroid = (int(centroid[0]), int(centroid[1]))

            if tid not in self._tracks:
                state = TrackSpatialState(tid, cname, centroid, frame_idx)
                self._tracks[tid] = state
            else:
                state = self._tracks[tid]
                state.last_seen_frame = frame_idx
                state.frames_unseen = 0
                state.centroid_history.append(centroid)

            # Evaluate virtual line crossing
            line_ev = self._check_line_crossing(state, centroid, frame_idx, ts)
            if line_ev:
                new_events.append(line_ev)

            # Evaluate ROI entry/exit
            roi_ev = self._check_roi_occupancy(state, centroid, frame_idx, ts)
            if roi_ev:
                new_events.append(roi_ev)

        # 2. Handle missing tracks (Temporary detection loss and track termination)
        for tid, state in self._tracks.items():
            if tid not in active_track_ids:
                state.frames_unseen += 1
                if state.frames_unseen >= self.disappearance_grace_frames:
                    # If track terminates while inside ROI, do NOT report an exit
                    if state.region_state == RegionState.INSIDE:
                        state.region_state = RegionState.TERMINATED_INSIDE
                        state.terminated_inside = True
                        self._roi_current_inside.discard(tid)
                        self._roi_terminated_inside_total += 1

        # Current counts inside ROI by class
        current_roi_classes = Counter()
        for tid in self._roi_current_inside:
            if tid in self._tracks:
                current_roi_classes[self._tracks[tid].class_name] += 1

        return {
            "frame_idx": frame_idx,
            "timestamp_sec": round(ts, 2),
            "virtual_line_configured": self.virtual_line is not None,
            "roi_configured": self.roi_region is not None,
            "line_crossings_total": self._line_crossings_total,
            "line_crossings_by_direction": dict(self._line_crossings_by_direction),
            "line_crossings_by_class": dict(self._line_crossings_by_class),
            "roi_current_occupancy": len(self._roi_current_inside),
            "roi_current_classes": dict(current_roi_classes),
            "roi_entries_total": self._roi_entries_total,
            "roi_entries_by_class": dict(self._roi_entries_by_class),
            "roi_exits_total": self._roi_exits_total,
            "roi_exits_by_class": dict(self._roi_exits_by_class),
            "roi_terminated_inside": self._roi_terminated_inside_total,
            "new_events": [e.to_dict() for e in new_events],
            "total_events_logged": len(self._events)
        }

    def finalize_stream(
        self,
        last_frame_idx: Optional[int] = None,
        timestamp_sec: Optional[float] = None
    ) -> List[SpatialEvent]:
        """
        Cleans up stream session. Crucially, objects remaining inside ROI at end of stream
        are marked as TERMINATED_INSIDE without false ROI_EXIT emissions.
        """
        for tid, state in self._tracks.items():
            if state.region_state == RegionState.INSIDE:
                state.region_state = RegionState.TERMINATED_INSIDE
                state.terminated_inside = True
                self._roi_current_inside.discard(tid)
                self._roi_terminated_inside_total += 1
        return []

    def get_events(
        self,
        limit: Optional[int] = None,
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Returns structured spatial event records, optionally filtered."""
        evs = self._events
        if event_type:
            evs = [e for e in evs if e.event_type == event_type.upper()]
        if limit:
            evs = evs[-limit:]
        return [e.to_dict() for e in evs]

    def get_summary(self) -> Dict[str, Any]:
        """Returns full structured session report for spatial analytics."""
        current_roi_classes = Counter()
        for tid in self._roi_current_inside:
            if tid in self._tracks:
                current_roi_classes[self._tracks[tid].class_name] += 1

        return {
            "total_frames": self._latest_frame_idx,
            "duration_sec": round(self._latest_timestamp, 2),
            "virtual_line": self.virtual_line,
            "roi_region": self.roi_region,
            "line_crossings": {
                "total": self._line_crossings_total,
                "by_direction": dict(self._line_crossings_by_direction),
                "by_class": dict(self._line_crossings_by_class)
            },
            "roi_occupancy": {
                "current_inside": len(self._roi_current_inside),
                "current_classes": dict(current_roi_classes),
                "entries_total": self._roi_entries_total,
                "entries_by_class": dict(self._roi_entries_by_class),
                "exits_total": self._roi_exits_total,
                "exits_by_class": dict(self._roi_exits_by_class),
                "terminated_inside": self._roi_terminated_inside_total
            },
            "total_events_logged": len(self._events),
            "recent_events": self.get_events(limit=10)
        }

    def draw_overlays(self, frame: np.ndarray) -> np.ndarray:
        """
        Renders visual indicators for virtual line and ROI directly onto the frame.
        """
        if frame is None or frame.size == 0:
            return frame

        # 1. Render Virtual Line
        if self.virtual_line is not None:
            x1, y1, x2, y2 = self.virtual_line
            line_color = (0, 215, 255)  # Gold / Amber in BGR
            cv2.line(frame, (x1, y1), (x2, y2), line_color, 2, cv2.LINE_AA)
            cv2.circle(frame, (x1, y1), 5, (0, 180, 255), -1)
            cv2.circle(frame, (x2, y2), 5, (0, 180, 255), -1)

            # Label banner
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            label = f"LINE: {self._line_crossings_total} crosses"
            cv2.putText(
                frame,
                label,
                (mid_x + 8, mid_y - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                line_color,
                1,
                cv2.LINE_AA
            )

        # 2. Render ROI Region
        if self.roi_region is not None:
            rx1, ry1, rx2, ry2 = self.roi_region
            roi_color = (255, 180, 50)  # Cyan/Sky blue in BGR

            # Translucent fill
            overlay = frame.copy()
            cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), roi_color, -1)
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

            # Solid boundary
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), roi_color, 2)

            # ROI Header Badge
            current_inside = len(self._roi_current_inside)
            roi_badge = f"ROI: {current_inside} inside | In: {self._roi_entries_total} Out: {self._roi_exits_total}"
            cv2.putText(
                frame,
                roi_badge,
                (rx1 + 6, max(18, ry1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                roi_color,
                1,
                cv2.LINE_AA
            )

        return frame
