"""
Thread-safe background streaming worker for the web dashboard.
Integrates VideoSource, YOLODetector, ObjectTracker, SceneIntelligence,
SpatialAnalytics, BehaviorAnalytics, and HeatmapAnalytics with the Flask web server.
"""

import threading
import time
import math
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from collections import Counter

import cv2
import numpy as np

from src.detector import YOLODetector
from src.tracker import ObjectTracker, TrackedObject
from src.video_source import VideoSource
from src.scene_intelligence import SceneIntelligence
from src.spatial_analytics import SpatialAnalytics
from src.behavior_analytics import BehaviorAnalytics
from src.heatmap_analytics import HeatmapAnalytics
from src.utils import (
    draw_bounding_box,
    draw_trajectory,
    draw_hud,
    ColorPalette,
    FPSMeter
)


def create_standby_frame(
    width: int = 640,
    height: int = 480,
    message: str = "Stream Idle"
) -> np.ndarray:
    """Creates a dark-themed placeholder frame with a centered status banner."""
    frame = np.full((height, width, 3), 20, dtype=np.uint8)
    # Subtle grid lines
    for y in range(0, height, 40):
        cv2.line(frame, (0, y), (width, y), (30, 30, 30), 1)
    for x in range(0, width, 40):
        cv2.line(frame, (x, 0), (x, height), (30, 30, 30), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    text = f"CodeAlpha Vision Engine - {message}"
    scale = 0.6
    thick = 1
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    tx = max(10, (width - tw) // 2)
    ty = (height + th) // 2
    cv2.putText(frame, text, (tx, ty), font, scale, (140, 140, 140), thick, cv2.LINE_AA)
    return frame


def draw_dashed_segment(
    img: np.ndarray,
    pt1: Tuple[int, int],
    pt2: Tuple[int, int],
    color: Tuple[int, int, int],
    thickness: int = 2,
    dash_len: int = 12,
    gap_len: int = 6
) -> None:
    """Draws an anti-aliased dashed line between two points."""
    dist = math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])
    if dist <= 0:
        return
    dx = (pt2[0] - pt1[0]) / dist
    dy = (pt2[1] - pt1[1]) / dist
    curr = 0.0
    while curr < dist:
        sx = int(round(pt1[0] + dx * curr))
        sy = int(round(pt1[1] + dy * curr))
        ec = min(curr + dash_len, dist)
        ex = int(round(pt1[0] + dx * ec))
        ey = int(round(pt1[1] + dy * ec))
        cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
        curr += dash_len + gap_len


def draw_web_spatial_overlays(frame: np.ndarray, spatial_analytics: Optional[SpatialAnalytics]) -> np.ndarray:
    """
    Renders clean, web-tailored overlays for virtual line and ROI.
    Renders ROI with a dashed border, corner accents, and a distinct header badge,
    ensuring it is visually unmistakable from object detection bounding boxes.
    """
    if frame is None or frame.size == 0 or spatial_analytics is None:
        return frame

    # 1. Virtual Line
    if spatial_analytics.virtual_line is not None:
        x1, y1, x2, y2 = spatial_analytics.virtual_line
        line_color = (0, 215, 255)  # Gold/Amber in BGR
        cv2.line(frame, (x1, y1), (x2, y2), line_color, 2, cv2.LINE_AA)
        cv2.circle(frame, (x1, y1), 5, (0, 180, 255), -1)
        cv2.circle(frame, (x2, y2), 5, (0, 180, 255), -1)
        mid_x = (x1 + x2) // 2
        mid_y = (y1 + y2) // 2
        label = f"LINE: {spatial_analytics._line_crossings_total} crosses"
        cv2.putText(frame, label, (mid_x + 8, mid_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, line_color, 1, cv2.LINE_AA)

    # 2. Distinct Dashed ROI Boundary
    if spatial_analytics.roi_region is not None:
        rx1, ry1, rx2, ry2 = spatial_analytics.roi_region
        roi_color = (255, 210, 0)  # Bright Cyan/Teal in BGR

        # Very subtle translucent fill (10%)
        overlay = frame.copy()
        cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), roi_color, -1)
        cv2.addWeighted(overlay, 0.10, frame, 0.90, 0, frame)

        # Dashed boundary on all four sides
        draw_dashed_segment(frame, (rx1, ry1), (rx2, ry1), roi_color, thickness=2, dash_len=12, gap_len=6)
        draw_dashed_segment(frame, (rx2, ry1), (rx2, ry2), roi_color, thickness=2, dash_len=12, gap_len=6)
        draw_dashed_segment(frame, (rx2, ry2), (rx1, ry2), roi_color, thickness=2, dash_len=12, gap_len=6)
        draw_dashed_segment(frame, (rx1, ry2), (rx1, ry1), roi_color, thickness=2, dash_len=12, gap_len=6)

        # Solid corner emphasis markers (length up to 16px)
        corner_len = min(16, (rx2 - rx1) // 4, (ry2 - ry1) // 4)
        if corner_len > 0:
            for (cx, cy), (dx, dy) in [
                ((rx1, ry1), (corner_len, corner_len)),
                ((rx2, ry1), (-corner_len, corner_len)),
                ((rx1, ry2), (corner_len, -corner_len)),
                ((rx2, ry2), (-corner_len, -corner_len))
            ]:
                cv2.line(frame, (cx, cy), (cx + dx, cy), roi_color, 3, cv2.LINE_AA)
                cv2.line(frame, (cx, cy), (cx, cy + dy), roi_color, 3, cv2.LINE_AA)

        # Prominent high-contrast ROI Header Badge
        current_inside = len(spatial_analytics._roi_current_inside)
        roi_badge = f"[ REGION OF INTEREST (ROI) ] Inside: {current_inside} | Total In: {spatial_analytics._roi_entries_total}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.45
        (text_w, text_h), baseline = cv2.getTextSize(roi_badge, font, font_scale, 1)
        badge_y2 = max(text_h + 8, ry1 - 4)
        badge_y1 = badge_y2 - text_h - 6
        cv2.rectangle(frame, (rx1, badge_y1), (rx1 + text_w + 8, badge_y2), (20, 20, 20), -1)
        cv2.rectangle(frame, (rx1, badge_y1), (rx1 + text_w + 8, badge_y2), roi_color, 1)
        cv2.putText(frame, roi_badge, (rx1 + 4, badge_y2 - 4), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

    return frame


class WebStreamWorker(threading.Thread):
    """
    Background worker thread running YOLO detection, ByteTrack tracking,
    and multi-engine computer vision analytics for web browser streaming.
    """

    def __init__(
        self,
        source: str = "0",
        model_name: str = "yolo11n.pt",
        tracker_type: str = "bytetrack.yaml",
        conf_threshold: float = 0.40,
        device: str = "cpu",
        virtual_line: Optional[Tuple[int, int, int, int]] = None,
        roi_region: Optional[Tuple[int, int, int, int]] = None,
        enable_heatmap: bool = False,
        heatmap_radius: int = 20,
        heatmap_alpha: float = 0.5,
        save_output: bool = False,
        output_dir: Optional[str] = None
    ):
        super().__init__(daemon=True)
        self.source = str(source).strip()
        self.model_name = str(model_name).strip()
        self.tracker_type = str(tracker_type).strip()
        self.conf_threshold = float(conf_threshold)
        self.device = str(device).strip()
        self.virtual_line = virtual_line
        self.roi_region = roi_region
        self.enable_heatmap = bool(enable_heatmap)
        self.heatmap_radius = int(heatmap_radius)
        self.heatmap_alpha = float(heatmap_alpha)
        self.save_output = bool(save_output)
        self.output_dir = Path(output_dir) if output_dir else Path(__file__).resolve().parent.parent.parent / "outputs"

        self.is_webcam = self.source.isdigit()

        # Thread lifecycle synchronization
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()
        self._frame_condition = threading.Condition(self._state_lock)

        # Worker status: "initialized", "running", "stopping", "stopped", "completed", "error"
        self._status: str = "initialized"
        self._error_message: Optional[str] = None

        # Telemetry state cache (protected by _state_lock)
        self._frame_count: int = 0
        self._fps: float = 0.0
        self._active_tracks_count: int = 0
        self._total_tracks_count: int = 0
        self._visible_classes: Dict[str, int] = {}
        self._active_tracks_info: List[Dict[str, Any]] = []
        self._scene_summary: str = "Stream initialized."
        self._recent_events: List[Dict[str, Any]] = []
        self._spatial_summary: Optional[Dict[str, Any]] = None
        self._behavior_summary: Optional[Dict[str, Any]] = None
        self._heatmap_summary: Optional[Dict[str, Any]] = None

        # Encoded frame caches
        self._latest_jpeg: Optional[bytes] = None
        self._last_raw_frame: Optional[np.ndarray] = None
        self._saved_video_path: Optional[str] = None
        self._saved_heatmap_path: Optional[str] = None
        self._saved_overlay_path: Optional[str] = None

        # Consumer tracking & idle watchdog
        self._consumer_count: int = 0
        self._last_consumer_activity: float = time.time()
        self.idle_timeout_sec: float = 30.0

    @property
    def status(self) -> str:
        """Returns current lifecycle status."""
        with self._state_lock:
            return self._status

    @property
    def is_running(self) -> bool:
        """Returns True if the thread is alive and in running state."""
        with self._state_lock:
            return self.is_alive() and self._status == "running"

    def stop(self, timeout: float = 2.5) -> bool:
        """
        Signals the worker thread to stop and waits up to timeout seconds.
        Returns True if the thread has fully terminated, False if still alive.
        """
        self._stop_event.set()
        with self._state_lock:
            if self._status == "running":
                self._status = "stopping"
            self._frame_condition.notify_all()

        if self.is_alive():
            self.join(timeout=timeout)

        with self._state_lock:
            if self.is_alive():
                # Still running after timeout — report stopping truthfully
                self._status = "stopping"
                return False
            else:
                if self._status not in ("completed", "error"):
                    self._status = "stopped"
                self._frame_condition.notify_all()
                return True

    def register_consumer(self) -> None:
        """Registers an active stream consumer and refreshes activity timestamp."""
        with self._state_lock:
            self._consumer_count += 1
            self._last_consumer_activity = time.time()

    def unregister_consumer(self) -> None:
        """Unregisters a stream consumer."""
        with self._state_lock:
            self._consumer_count = max(0, self._consumer_count - 1)
            self._last_consumer_activity = time.time()

    def touch_activity(self) -> None:
        """Refreshes consumer activity timestamp (e.g. from telemetry polling)."""
        with self._state_lock:
            self._last_consumer_activity = time.time()

    def wait_for_frame(self, timeout: float = 0.5) -> Optional[bytes]:
        """
        Waits for a newly encoded JPEG frame via condition variable notification.
        Avoids CPU-intensive busy-waiting loops.
        """
        with self._state_lock:
            self._last_consumer_activity = time.time()
            if self._latest_jpeg is not None:
                return self._latest_jpeg
            # Wait for next frame update
            self._frame_condition.wait(timeout=timeout)
            return self._latest_jpeg

    def get_telemetry(self) -> Dict[str, Any]:
        """Returns a complete, thread-safe snapshot of analytics telemetry."""
        with self._state_lock:
            self._last_consumer_activity = time.time()
            return {
                "active": self.is_alive() and self._status == "running",
                "status": self._status,
                "error_message": self._error_message,
                "source": self.source,
                "is_webcam": self.is_webcam,
                "model_name": self.model_name,
                "confidence_threshold": self.conf_threshold,
                "frames_processed": self._frame_count,
                "fps": round(self._fps, 1),
                "active_tracks_count": self._active_tracks_count,
                "total_unique_tracks": self._total_tracks_count,
                "visible_classes": dict(self._visible_classes),
                "active_tracks": list(self._active_tracks_info),
                "scene_summary": self._scene_summary,
                "recent_events": list(self._recent_events),
                "spatial": self._spatial_summary,
                "behavior": self._behavior_summary,
                "heatmap": self._heatmap_summary,
                "saved_video_path": self._saved_video_path,
                "saved_heatmap_path": self._saved_heatmap_path,
                "saved_overlay_path": self._saved_overlay_path,
            }

    def run(self) -> None:
        """
        Main execution loop.
        Acquires video frames, runs YOLO inference and ByteTrack association,
        updates analytics engines, encodes annotated frames, and guarantees resource cleanup.
        """
        with self._state_lock:
            self._status = "running"
            self._frame_condition.notify_all()

        vs: Optional[VideoSource] = None
        writer: Optional[cv2.VideoWriter] = None
        tracker: Optional[ObjectTracker] = None
        scene_intel: Optional[SceneIntelligence] = None
        spatial_analytics: Optional[SpatialAnalytics] = None
        behavior_analytics: Optional[BehaviorAnalytics] = None
        heatmap_analytics: Optional[HeatmapAnalytics] = None
        fps_meter = FPSMeter()
        frame_idx = 0

        try:
            # 1. Initialize video source
            vs = VideoSource(self.source)
            vs.open()

            # 2. Initialize tracking engine
            tracker = ObjectTracker(
                tracker_type=self.tracker_type,
                model_name=self.model_name,
                confidence_threshold=self.conf_threshold,
                device=self.device
            )
            tracker.load_model()

            # 3. Initialize analytics engines
            scene_intel = SceneIntelligence()
            behavior_analytics = BehaviorAnalytics()

            v_line = self.virtual_line
            r_roi = self.roi_region

            # Coordinate safety: adapt coordinates if default or clamp if frame dimensions known
            if vs.width > 0 and vs.height > 0:
                if v_line is not None:
                    # If using the default 640x480 line on a different resolution, scale proportionately
                    if v_line == (0, 240, 640, 240) and (vs.width != 640 or vs.height != 480):
                        mid_y = int(round(vs.height * 0.5))
                        v_line = (0, mid_y, vs.width, mid_y)
                    else:
                        # Clamp safely to frame boundaries
                        lx1 = max(0, min(v_line[0], vs.width))
                        ly1 = max(0, min(v_line[1], vs.height))
                        lx2 = max(0, min(v_line[2], vs.width))
                        ly2 = max(0, min(v_line[3], vs.height))
                        v_line = (lx1, ly1, lx2, ly2) if not (lx1 == lx2 and ly1 == ly2) else None

                if r_roi is not None:
                    # If using the default 640x480 roi on a different resolution, scale proportionately
                    if r_roi == (100, 100, 540, 380) and (vs.width != 640 or vs.height != 480):
                        rx1 = int(round(vs.width * 0.15))
                        ry1 = int(round(vs.height * 0.20))
                        rx2 = int(round(vs.width * 0.85))
                        ry2 = int(round(vs.height * 0.80))
                        r_roi = (rx1, ry1, rx2, ry2) if (rx1 < rx2 and ry1 < ry2) else None
                    else:
                        # Clamp safely to frame boundaries
                        rx1 = max(0, min(r_roi[0], vs.width - 10))
                        ry1 = max(0, min(r_roi[1], vs.height - 10))
                        rx2 = max(rx1 + 10, min(r_roi[2], vs.width))
                        ry2 = max(ry1 + 10, min(r_roi[3], vs.height))
                        r_roi = (rx1, ry1, rx2, ry2) if (rx1 < rx2 and ry1 < ry2) else None

            if v_line is not None or r_roi is not None:
                spatial_analytics = SpatialAnalytics(
                    virtual_line=v_line,
                    roi_region=r_roi,
                    frame_width=vs.width if vs.width > 0 else None,
                    frame_height=vs.height if vs.height > 0 else None
                )

            if self.enable_heatmap and vs.width > 0 and vs.height > 0:
                heatmap_analytics = HeatmapAnalytics(
                    frame_width=vs.width,
                    frame_height=vs.height,
                    radius_px=self.heatmap_radius,
                    decay_factor=1.0,
                    overlay_alpha=self.heatmap_alpha,
                    min_density_threshold=0.05
                )

            # 4. Optional video output writer
            if self.save_output and vs.width > 0 and vs.height > 0:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                stem = f"webcam_{self.source}" if self.is_webcam else Path(self.source).stem
                out_path = self.output_dir / f"web_tracked_{stem}_{int(time.time())}.mp4"
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps_val = vs.fps if vs.fps > 0 else 30.0
                writer = cv2.VideoWriter(str(out_path.resolve()), fourcc, fps_val, (vs.width, vs.height))
                with self._state_lock:
                    self._saved_video_path = str(out_path.resolve())

            # 5. Continuous frame processing loop
            while not self._stop_event.is_set():
                # Check idle watchdog for webcam streams
                if self.is_webcam:
                    with self._state_lock:
                        idle_duration = time.time() - self._last_consumer_activity
                        if self._consumer_count == 0 and idle_duration > self.idle_timeout_sec:
                            # Auto-stop idle webcam
                            break

                ret, frame = vs.read()
                if not ret or frame is None:
                    # End of video reached
                    break

                frame_idx += 1
                self._last_raw_frame = frame

                # Lazy heatmap initialization if dimensions were unknown before first frame
                if self.enable_heatmap and heatmap_analytics is None:
                    heatmap_analytics = HeatmapAnalytics(
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                        radius_px=self.heatmap_radius,
                        decay_factor=1.0,
                        overlay_alpha=self.heatmap_alpha,
                        min_density_threshold=0.05
                    )

                # Detection & Tracking
                tracks = tracker.track(frame)
                current_fps = fps_meter.tick()
                stream_fps = vs.fps if vs.fps > 0 else 30.0
                timestamp_sec = frame_idx / stream_fps

                # Analytics updates
                intel_snap = scene_intel.update(
                    frame_idx=frame_idx,
                    tracks=tracks,
                    timestamp_sec=timestamp_sec,
                    fps=stream_fps
                )

                spatial_snap = None
                if spatial_analytics is not None:
                    spatial_snap = spatial_analytics.update(
                        frame_idx=frame_idx,
                        tracks=tracks,
                        timestamp_sec=timestamp_sec,
                        fps=stream_fps
                    )

                behavior_analytics.update(
                    frame_idx=frame_idx,
                    tracks=tracks,
                    timestamp_sec=timestamp_sec,
                    fps=stream_fps
                )

                if heatmap_analytics is not None:
                    heatmap_analytics.update(
                        tracks=tracks,
                        frame_idx=frame_idx,
                        timestamp_sec=timestamp_sec
                    )

                # Render overlays
                annotated = frame.copy()
                active_classes: Counter = Counter()
                tracks_info = []

                for t in tracks:
                    active_classes[t.class_name] += 1
                    color = ColorPalette.get_color(t.track_id)
                    label = f"#{t.track_id} {t.class_name} {t.confidence:.2f}"
                    if len(t.trajectory) >= 2:
                        draw_trajectory(annotated, list(t.trajectory), color, thickness=2)
                    draw_bounding_box(annotated, t.box, label, color=color, thickness=2)
                    tracks_info.append({
                        "track_id": t.track_id,
                        "class_name": t.class_name,
                        "confidence": round(t.confidence, 2),
                        "box": [round(v, 1) for v in t.box],
                        "centroid": t.current_centroid
                    })

                if spatial_analytics is not None:
                    draw_web_spatial_overlays(annotated, spatial_analytics)

                draw_hud(
                    annotated,
                    fps=current_fps,
                    active_count=tracker.get_active_count(),
                    total_count=tracker.get_total_count(),
                    class_counts=dict(active_classes)
                )

                if writer is not None:
                    writer.write(annotated)

                # Encode frame to JPEG
                encode_success, jpeg_buf = cv2.imencode(
                    ".jpg",
                    annotated,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                jpeg_bytes = jpeg_buf.tobytes() if encode_success else None

                # Update telemetry state and notify waiters
                with self._state_lock:
                    self._frame_count = frame_idx
                    self._fps = current_fps
                    self._active_tracks_count = tracker.get_active_count()
                    self._total_tracks_count = tracker.get_total_count()
                    self._visible_classes = dict(active_classes)
                    self._active_tracks_info = tracks_info
                    self._scene_summary = scene_intel.generate_natural_language_summary()
                    self._recent_events = scene_intel.get_events(limit=8)
                    if spatial_analytics is not None:
                        self._spatial_summary = spatial_analytics.get_summary()
                    if frame_idx % 10 == 0 or frame_idx == 1:
                        self._behavior_summary = behavior_analytics.get_summary()
                    if heatmap_analytics is not None:
                        self._heatmap_summary = heatmap_analytics.get_summary()

                    if jpeg_bytes is not None:
                        self._latest_jpeg = jpeg_bytes
                    self._frame_condition.notify_all()

        except Exception as err:
            with self._state_lock:
                self._status = "error"
                self._error_message = str(err)
        finally:
            # 6. Guaranteed resource cleanup
            if vs is not None:
                try:
                    vs.release()
                except Exception:
                    pass
            if writer is not None:
                try:
                    writer.release()
                except Exception:
                    pass

            # Finalize analytics
            if scene_intel is not None:
                try:
                    scene_intel.finalize_stream(last_frame_idx=frame_idx)
                except Exception:
                    pass
            if spatial_analytics is not None:
                try:
                    spatial_analytics.finalize_stream(last_frame_idx=frame_idx)
                except Exception:
                    pass
            if behavior_analytics is not None:
                try:
                    behavior_analytics.finalize_stream(last_frame_idx=frame_idx)
                except Exception:
                    pass

            # Export heatmap images if enabled and frames were processed
            if self.enable_heatmap and heatmap_analytics is not None and frame_idx > 0:
                try:
                    self.output_dir.mkdir(parents=True, exist_ok=True)
                    stem = f"webcam_{self.source}" if self.is_webcam else Path(self.source).stem
                    ts_now = int(time.time())
                    hm_path = self.output_dir / f"heatmap_web_{stem}_{ts_now}.png"
                    ov_path = self.output_dir / f"heatmap_overlay_web_{stem}_{ts_now}.png"

                    hm_img = heatmap_analytics.generate_heatmap_image(normalize=True)
                    base_bg = self._last_raw_frame if self._last_raw_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                    ov_img = heatmap_analytics.apply_overlay(base_bg, alpha=self.heatmap_alpha, min_density_threshold=0.05)

                    cv2.imwrite(str(hm_path.resolve()), hm_img)
                    cv2.imwrite(str(ov_path.resolve()), ov_img)

                    with self._state_lock:
                        self._saved_heatmap_path = str(hm_path.resolve())
                        self._saved_overlay_path = str(ov_path.resolve())
                except Exception:
                    pass

            # Final status transition
            with self._state_lock:
                if self._status != "error":
                    if self._stop_event.is_set():
                        self._status = "stopped"
                    else:
                        self._status = "completed"
                # Update final summaries
                if behavior_analytics is not None:
                    try:
                        self._behavior_summary = behavior_analytics.get_summary()
                    except Exception:
                        pass
                if spatial_analytics is not None:
                    try:
                        self._spatial_summary = spatial_analytics.get_summary()
                    except Exception:
                        pass
                if heatmap_analytics is not None:
                    try:
                        self._heatmap_summary = heatmap_analytics.get_summary()
                    except Exception:
                        pass

                self._frame_condition.notify_all()


class StreamSessionManager:
    """
    Thread-safe session controller managing the lifecycle of the single active WebStreamWorker.
    Guarantees no conflicting sessions, safe restartability, and atomic state inspection.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._worker: Optional[WebStreamWorker] = None
        self._last_completed_telemetry: Optional[Dict[str, Any]] = None

    @property
    def active_worker(self) -> Optional[WebStreamWorker]:
        """Returns the currently active worker if running, or None."""
        with self._lock:
            if self._worker is not None and self._worker.is_running:
                return self._worker
            return None

    @property
    def is_busy(self) -> bool:
        """Returns True if a worker thread is alive (running, starting, or shutting down)."""
        with self._lock:
            return self._worker is not None and self._worker.is_alive()

    def start_stream(
        self,
        source: str = "0",
        model_name: str = "yolo11n.pt",
        tracker_type: str = "bytetrack.yaml",
        conf_threshold: float = 0.40,
        device: str = "cpu",
        virtual_line: Optional[Tuple[int, int, int, int]] = None,
        roi_region: Optional[Tuple[int, int, int, int]] = None,
        enable_heatmap: bool = False,
        heatmap_radius: int = 20,
        heatmap_alpha: float = 0.5,
        save_output: bool = False,
        **kwargs
    ) -> Tuple[bool, str]:
        """
        Starts a new stream worker session.
        Guards against starting while a session is running or shutting down.
        """
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                if self._worker.status == "stopping":
                    return False, "Previous stream is still shutting down. Please wait a moment."
                return False, "Stream is already running"

            # Clear previous session telemetry when starting fresh
            self._last_completed_telemetry = None

            # Instantiate and start fresh worker
            self._worker = WebStreamWorker(
                source=source,
                model_name=model_name,
                tracker_type=tracker_type,
                conf_threshold=conf_threshold,
                device=device,
                virtual_line=virtual_line,
                roi_region=roi_region,
                enable_heatmap=enable_heatmap,
                heatmap_radius=heatmap_radius,
                heatmap_alpha=heatmap_alpha,
                save_output=save_output,
            )
            self._worker.start()

            # Bounded brief wait for thread initialization
            for _ in range(10):
                if self._worker.is_alive():
                    break
                time.sleep(0.01)

            return True, "Stream started successfully"

    def stop_stream(self, timeout: float = 2.5) -> Tuple[bool, str]:
        """
        Stops the active streaming worker.
        Returns (True, msg) on successful termination, or (False, msg) if not running / stopping.
        """
        with self._lock:
            if self._worker is None:
                return False, "No active stream to stop"

            worker = self._worker
            if worker.is_alive():
                stopped = worker.stop(timeout=timeout)
                if not stopped:
                    return False, "Stream stop requested but worker is still finalizing resources"

            self._last_completed_telemetry = worker.get_telemetry()
            self._worker = None
            return True, "Stream stopped successfully"

    def get_status(self) -> Dict[str, Any]:
        """Returns structured status of the active session or last recorded state."""
        with self._lock:
            if self._worker is not None:
                # Return worker telemetry whether running, completed via EOF, or failed with error
                return self._worker.get_telemetry()

            if self._last_completed_telemetry is not None:
                telemetry = dict(self._last_completed_telemetry)
                telemetry["active"] = False
                return telemetry

            return {
                "active": False,
                "status": "idle",
                "message": "No active streaming session",
                "source": None,
                "frames_processed": 0,
                "fps": 0.0,
                "active_tracks_count": 0,
                "total_unique_tracks": 0,
                "visible_classes": {},
                "active_tracks": [],
                "scene_summary": "Session idle. Start a webcam stream or upload a video.",
                "recent_events": [],
                "spatial": None,
                "behavior": None,
                "heatmap": None,
                "saved_video_path": None,
                "saved_heatmap_path": None,
                "saved_overlay_path": None,
            }

    def get_latest_frame(self) -> Optional[bytes]:
        """Retrieves the latest encoded JPEG frame from the active worker."""
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return self._worker.wait_for_frame(timeout=0.2)
            return None
