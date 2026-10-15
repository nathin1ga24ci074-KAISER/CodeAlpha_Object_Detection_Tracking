"""
Scene Intelligence Module for CodeAlpha Task 4: Object Detection and Tracking.
Provides explainable, rule-based scene analytics, motion direction estimation,
event timeline logging, and deterministic natural-language summaries.
"""

from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, deque
import math
import time


class MotionState:
    """Coarse motion categories based on measured pixel displacements."""
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
    STATIONARY = "STATIONARY"
    MOVING_LEFT = "MOVING_LEFT"
    MOVING_RIGHT = "MOVING_RIGHT"
    MOVING_UP = "MOVING_UP"
    MOVING_DOWN = "MOVING_DOWN"


class SceneEvent:
    """Represents a structured, timestamped occurrence in the video stream."""

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
        self.event_type = event_type  # "ENTERED", "MOVED", "EXITED"
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
            f"SceneEvent({self.event_type}: #{self.track_id} {self.class_name} "
            f"at frame {self.frame_idx} [{self.timestamp_sec}s] - {self.description})"
        )


class TrackMotionRecord:
    """Maintains motion and observation history for a single persistent track ID."""

    def __init__(
        self,
        track_id: int,
        class_name: str,
        first_frame: int,
        first_timestamp: float,
        initial_centroid: Tuple[int, int],
        window_size: int = 15
    ):
        self.track_id = track_id
        self.class_name = class_name
        self.first_frame = first_frame
        self.last_seen_frame = first_frame
        self.first_timestamp = first_timestamp
        self.last_timestamp = first_timestamp
        self.centroid_history: deque = deque(maxlen=window_size)
        self.centroid_history.append(initial_centroid)
        self.current_motion_state = MotionState.INSUFFICIENT_DATA
        self.total_frames_seen = 1
        self.frames_unseen = 0
        self.has_logged_exit = False
        self.last_logged_move_frame = first_frame
        self.last_logged_direction: Optional[str] = None
        self.cumulative_displacement_px: float = 0.0

    def add_observation(
        self,
        frame_idx: int,
        timestamp: float,
        centroid: Tuple[int, int]
    ) -> None:
        """Records a new frame detection and updates centroid trail."""
        if len(self.centroid_history) > 0:
            prev = self.centroid_history[-1]
            step_dist = math.hypot(centroid[0] - prev[0], centroid[1] - prev[1])
            self.cumulative_displacement_px += step_dist

        self.centroid_history.append(centroid)
        self.last_seen_frame = frame_idx
        self.last_timestamp = timestamp
        self.total_frames_seen += 1
        self.frames_unseen = 0


class SceneIntelligence:
    """
    Explainable scene-intelligence engine.
    Analyzes active tracks across consecutive frames to produce:
    - Current vs cumulative object statistics
    - Coarse motion vectors (Stationary, Left, Right, Up, Down)
    - Structured timeline events (ENTERED, MOVED, EXITED)
    - Verifiable, deterministic natural-language scene summaries
    """

    def __init__(
        self,
        min_movement_threshold_px: float = 12.0,
        min_frames_for_motion: int = 3,
        disappearance_grace_frames: int = 15,
        movement_throttle_frames: int = 20,
        max_events_history: int = 250
    ):
        self.min_movement_threshold_px = float(min_movement_threshold_px)
        self.min_frames_for_motion = int(min_frames_for_motion)
        self.disappearance_grace_frames = int(disappearance_grace_frames)
        self.movement_throttle_frames = int(movement_throttle_frames)
        self.max_events_history = int(max_events_history)

        self._track_records: Dict[int, TrackMotionRecord] = {}
        self._events: List[SceneEvent] = []
        self._cumulative_classes: Counter = Counter()
        self._latest_counts: Counter = Counter()
        self._latest_frame_idx: int = 0
        self._latest_timestamp: float = 0.0

    def reset(self) -> None:
        """Resets all session tracking, events, and temporal state."""
        self._track_records.clear()
        self._events.clear()
        self._cumulative_classes.clear()
        self._latest_counts.clear()
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
    ) -> SceneEvent:
        event = SceneEvent(
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

    def _estimate_motion(
        self,
        record: TrackMotionRecord
    ) -> Tuple[str, float, float, float]:
        """
        Estimates coarse motion direction and displacement across the recent centroid window.
        Returns: (motion_state, net_displacement_px, delta_x, delta_y)
        """
        history = list(record.centroid_history)
        if len(history) < self.min_frames_for_motion:
            return MotionState.INSUFFICIENT_DATA, 0.0, 0.0, 0.0

        first_cx, first_cy = history[0]
        curr_cx, curr_cy = history[-1]
        dx = float(curr_cx - first_cx)
        dy = float(curr_cy - first_cy)
        net_disp = math.hypot(dx, dy)

        if net_disp < self.min_movement_threshold_px:
            return MotionState.STATIONARY, net_disp, dx, dy

        # Determine dominant direction axis
        if abs(dx) >= abs(dy):
            direction = MotionState.MOVING_LEFT if dx < 0 else MotionState.MOVING_RIGHT
        else:
            direction = MotionState.MOVING_UP if dy < 0 else MotionState.MOVING_DOWN

        return direction, net_disp, dx, dy

    def update(
        self,
        frame_idx: int,
        tracks: List[Any],
        timestamp_sec: Optional[float] = None,
        fps: float = 30.0
    ) -> Dict[str, Any]:
        """
        Processes detections/tracks for the current frame and updates scene analytics.

        Args:
            frame_idx: Current 1-indexed video frame counter.
            tracks: List of TrackedObject instances active in this frame.
            timestamp_sec: Optional stream time in seconds (falls back to frame_idx / fps).
            fps: Video framerate for timestamp estimation.

        Returns:
            Dict[str, Any]: Structured frame scene intelligence snapshot.
        """
        self._latest_frame_idx = frame_idx
        if timestamp_sec is not None:
            ts = float(timestamp_sec)
        else:
            ts = (frame_idx / fps) if fps > 0 else float(frame_idx)
        self._latest_timestamp = ts

        current_counts = Counter()
        active_track_ids = set()
        new_events = []

        # 1. Process active tracks in current frame
        for t in tracks or []:
            # Safe attribute extraction handling both TrackedObject and duck-typed dicts
            tid = getattr(t, "track_id", None)
            cname = getattr(t, "class_name", "object")
            centroid = getattr(t, "current_centroid", None)
            box = getattr(t, "box", [0, 0, 0, 0])

            if tid is None:
                continue

            tid = int(tid)
            active_track_ids.add(tid)
            current_counts[cname] += 1

            if centroid is None or not (isinstance(centroid, (list, tuple)) and len(centroid) == 2):
                # Calculate centroid fallback from box
                cx = int((box[0] + box[2]) / 2.0)
                cy = int((box[1] + box[3]) / 2.0)
                centroid = (cx, cy)

            if tid not in self._track_records:
                # Newly observed object
                self._cumulative_classes[cname] += 1
                rec = TrackMotionRecord(
                    track_id=tid,
                    class_name=cname,
                    first_frame=frame_idx,
                    first_timestamp=ts,
                    initial_centroid=centroid
                )
                self._track_records[tid] = rec
                ev = self._log_event(
                    event_type="ENTERED",
                    track_id=tid,
                    class_name=cname,
                    frame_idx=frame_idx,
                    timestamp_sec=ts,
                    description=f"{cname.capitalize()} #{tid} entered the scene at ({centroid[0]}, {centroid[1]}).",
                    details={"centroid": centroid, "box": box}
                )
                new_events.append(ev)
            else:
                # Existing track: add observation and evaluate motion
                rec = self._track_records[tid]
                rec.add_observation(frame_idx, ts, centroid)

                motion_state, net_disp, dx, dy = self._estimate_motion(rec)
                rec.current_motion_state = motion_state

                # Check if we should log a MOVED event (avoid flood)
                is_directional = motion_state not in (MotionState.INSUFFICIENT_DATA, MotionState.STATIONARY)
                frames_since_move_log = frame_idx - rec.last_logged_move_frame

                if is_directional:
                    direction_changed = motion_state != rec.last_logged_direction
                    exceeded_throttle = frames_since_move_log >= self.movement_throttle_frames

                    if direction_changed or exceeded_throttle:
                        dir_label = motion_state.replace("MOVING_", "").lower()
                        ev = self._log_event(
                            event_type="MOVED",
                            track_id=tid,
                            class_name=cname,
                            frame_idx=frame_idx,
                            timestamp_sec=ts,
                            description=f"{cname.capitalize()} #{tid} moving {dir_label} (displacement: {net_disp:.1f}px).",
                            details={
                                "direction": dir_label,
                                "displacement_px": round(net_disp, 1),
                                "delta_x": round(dx, 1),
                                "delta_y": round(dy, 1),
                                "centroid": centroid
                            }
                        )
                        new_events.append(ev)
                        rec.last_logged_move_frame = frame_idx
                        rec.last_logged_direction = motion_state

        # 2. Check for disappeared objects
        for tid, rec in self._track_records.items():
            if tid not in active_track_ids and not rec.has_logged_exit:
                rec.frames_unseen += 1
                if rec.frames_unseen >= self.disappearance_grace_frames:
                    rec.has_logged_exit = True
                    rec.current_motion_state = MotionState.STATIONARY
                    ev = self._log_event(
                        event_type="EXITED",
                        track_id=tid,
                        class_name=rec.class_name,
                        frame_idx=frame_idx,
                        timestamp_sec=ts,
                        description=(
                            f"{rec.class_name.capitalize()} #{tid} exited the scene "
                            f"(visible for {rec.total_frames_seen} frames)."
                        ),
                        details={
                            "total_frames_seen": rec.total_frames_seen,
                            "lifetime_sec": round(ts - rec.first_timestamp, 2)
                        }
                    )
                    new_events.append(ev)

        self._latest_counts = current_counts

        # Prepare per-track motion summary
        active_motion_summary = {}
        for tid in active_track_ids:
            r = self._track_records[tid]
            active_motion_summary[tid] = {
                "track_id": tid,
                "class_name": r.class_name,
                "motion_state": r.current_motion_state,
                "frames_observed": r.total_frames_seen,
                "cumulative_dist_px": round(r.cumulative_displacement_px, 1)
            }

        return {
            "frame_idx": frame_idx,
            "timestamp_sec": round(ts, 2),
            "visible_count": len(active_track_ids),
            "visible_classes": dict(current_counts),
            "cumulative_unique_ids": len(self._track_records),
            "cumulative_classes": dict(self._cumulative_classes),
            "active_motion": active_motion_summary,
            "new_events": [e.to_dict() for e in new_events],
            "total_events_logged": len(self._events)
        }

    def finalize_stream(self, last_frame_idx: Optional[int] = None, timestamp_sec: Optional[float] = None) -> List[SceneEvent]:
        """Emits EXITED events for any remaining un-exited tracks at stream completion."""
        frame = last_frame_idx or self._latest_frame_idx
        ts = timestamp_sec or self._latest_timestamp
        closing_events = []

        for tid, rec in self._track_records.items():
            if not rec.has_logged_exit:
                rec.has_logged_exit = True
                ev = self._log_event(
                    event_type="EXITED",
                    track_id=tid,
                    class_name=rec.class_name,
                    frame_idx=frame,
                    timestamp_sec=ts,
                    description=f"{rec.class_name.capitalize()} #{tid} exited at end of stream (observed for {rec.total_frames_seen} frames).",
                    details={"total_frames_seen": rec.total_frames_seen}
                )
                closing_events.append(ev)
        return closing_events

    def get_events(self, limit: Optional[int] = None, event_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Returns structured event records, optionally filtered by type or count."""
        evs = self._events
        if event_type:
            evs = [e for e in evs if e.event_type == event_type.upper()]
        if limit:
            evs = evs[-limit:]
        return [e.to_dict() for e in evs]

    def generate_natural_language_summary(self) -> str:
        """
        Constructs a concise, deterministic, and verifiable natural-language summary
        based strictly on measured observations.
        """
        if not self._track_records:
            return "No objects were detected during the analysis session."

        total_unique = len(self._track_records)
        cls_summary = ", ".join(f"{cnt} {cls}{'s' if cnt > 1 and not cls.endswith('s') else ''}" for cls, cnt in sorted(self._cumulative_classes.items()))

        lines = [
            f"Observed {total_unique} distinct object{'s' if total_unique > 1 else ''} ({cls_summary}) across {self._latest_frame_idx} frames."
        ]

        # Currently visible status
        currently_visible = [r for r in self._track_records.values() if not r.has_logged_exit and r.frames_unseen == 0]
        if currently_visible:
            vis_details = []
            for r in currently_visible[:4]:
                state_desc = r.current_motion_state.replace("MOVING_", "moving ").replace("_", " ").lower()
                vis_details.append(f"#{r.track_id} {r.class_name} ({state_desc})")
            lines.append(f"Currently visible ({len(currently_visible)}): {', '.join(vis_details)}.")
        else:
            lines.append("All tracked objects have departed or the stream has concluded.")

        # Movement highlights
        moved_records = [
            r for r in self._track_records.values()
            if r.cumulative_displacement_px >= self.min_movement_threshold_px
        ]
        if moved_records:
            move_notes = []
            for r in moved_records[:3]:
                move_notes.append(
                    f"#{r.track_id} {r.class_name} covered ~{int(r.cumulative_displacement_px)}px"
                )
            lines.append(f"Motion highlights: {'; '.join(move_notes)}.")
        else:
            lines.append("Detected entities exhibited minimal or stationary displacement.")

        return " ".join(lines)

    def get_session_report(self) -> Dict[str, Any]:
        """Returns complete structured session report suitable for JSON export or GUI inspection."""
        return {
            "total_frames": self._latest_frame_idx,
            "duration_sec": round(self._latest_timestamp, 2),
            "cumulative_unique_objects": len(self._track_records),
            "cumulative_class_counts": dict(self._cumulative_classes),
            "currently_visible_count": len(self._latest_counts),
            "currently_visible_classes": dict(self._latest_counts),
            "total_events_logged": len(self._events),
            "recent_events": self.get_events(limit=10),
            "natural_language_summary": self.generate_natural_language_summary()
        }
