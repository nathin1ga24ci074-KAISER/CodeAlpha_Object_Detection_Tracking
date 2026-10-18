"""
Desktop GUI Interface for CodeAlpha Task 4: Object Detection and Tracking.
Built with Tkinter and Pillow, offering a responsive dark-themed dashboard.
Separates UI controls from the background inference and tracking worker thread.
"""

import os
import sys
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
from collections import Counter
import cv2
import numpy as np

if sys.platform.startswith("win"):
    if "TCL_LIBRARY" not in os.environ:
        _tcl_dir = os.path.join(sys.base_prefix, "tcl", "tcl8.6")
        if os.path.exists(_tcl_dir):
            os.environ["TCL_LIBRARY"] = _tcl_dir
    if "TK_LIBRARY" not in os.environ:
        _tk_dir = os.path.join(sys.base_prefix, "tcl", "tk8.6")
        if os.path.exists(_tk_dir):
            os.environ["TK_LIBRARY"] = _tk_dir

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

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
    FPSMeter,
    create_synthetic_frame
)


class TrackingWorker(threading.Thread):
    """
    Background worker thread running YOLO detection and ByteTrack association.
    Communicates with the Tkinter UI through a thread-safe Queue.
    """

    def __init__(
        self,
        source: str,
        model_name: str,
        tracker_type: str,
        conf_threshold: float,
        device: str,
        save_output: bool,
        output_queue: queue.Queue,
        status_queue: queue.Queue,
        virtual_line: Optional[Tuple[int, int, int, int]] = None,
        roi_region: Optional[Tuple[int, int, int, int]] = None,
        enable_heatmap: bool = False,
        heatmap_radius: int = 20,
        heatmap_alpha: float = 0.5,
        heatmap_output_dir: Optional[str] = None
    ):
        super().__init__(daemon=True)
        self.source = source
        self.model_name = model_name
        self.tracker_type = tracker_type
        self.conf_threshold = conf_threshold
        self.device = device
        self.save_output = save_output
        self.output_queue = output_queue
        self.status_queue = status_queue
        self.virtual_line = virtual_line
        self.roi_region = roi_region
        self.enable_heatmap = enable_heatmap
        self.heatmap_radius = heatmap_radius
        self.heatmap_alpha = heatmap_alpha
        self.heatmap_output_dir = heatmap_output_dir
        self.is_running = threading.Event()
        self.is_running.set()

    def stop(self) -> None:
        """Signals the worker thread to terminate."""
        self.is_running.clear()

    def run(self) -> None:
        vs = None
        writer = None
        out_file_path = None
        heatmap_analytics = None
        last_raw_frame = None
        spatial_analytics = None
        scene_intel = None
        behavior_analytics = None
        frame_idx = 0
        try:
            self.status_queue.put({"type": "status", "message": f"Opening source: {self.source}..."})
            vs = VideoSource(self.source)
            vs.open()

            self.status_queue.put({"type": "status", "message": f"Loading model '{self.model_name}'..."})
            tracker = ObjectTracker(
                tracker_type=self.tracker_type,
                model_name=self.model_name,
                confidence_threshold=self.conf_threshold,
                device=self.device
            )
            tracker.load_model()

            spatial_analytics = None
            if self.virtual_line is not None or self.roi_region is not None:
                spatial_analytics = SpatialAnalytics(
                    virtual_line=self.virtual_line,
                    roi_region=self.roi_region,
                    frame_width=vs.width if vs.width > 0 else None,
                    frame_height=vs.height if vs.height > 0 else None
                )

            # Spatial Heatmap initialization if stream dimensions are known at start
            if self.enable_heatmap and vs.width > 0 and vs.height > 0:
                heatmap_analytics = HeatmapAnalytics(
                    frame_width=vs.width,
                    frame_height=vs.height,
                    radius_px=self.heatmap_radius,
                    decay_factor=1.0,
                    overlay_alpha=self.heatmap_alpha,
                    min_density_threshold=0.05
                )

            if self.save_output:
                out_dir = Path(__file__).resolve().parent.parent / "outputs"
                out_dir.mkdir(parents=True, exist_ok=True)
                timestamp = int(time.time())
                stem = Path(self.source).stem if not str(self.source).isdigit() else f"webcam_{self.source}"
                out_file_path = str((out_dir / f"gui_tracked_{stem}_{timestamp}.mp4").resolve())
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                fps_val = vs.fps if vs.fps > 0 else 30.0
                writer = cv2.VideoWriter(out_file_path, fourcc, fps_val, (vs.width, vs.height))
                self.status_queue.put({"type": "status", "message": f"Saving to: {out_file_path}"})

            self.status_queue.put({"type": "status", "message": "Inference and tracking active."})
            fps_meter = FPSMeter()
            scene_intel = SceneIntelligence()
            behavior_analytics = BehaviorAnalytics()
            frame_idx = 0

            while self.is_running.is_set():
                ret, frame = vs.read()
                if not ret or frame is None:
                    self.status_queue.put({"type": "status", "message": "End of video stream reached."})
                    break

                frame_idx += 1
                last_raw_frame = frame

                # Lazy heatmap initialization if dimensions were unknown prior to first decoded frame
                if self.enable_heatmap and heatmap_analytics is None:
                    heatmap_analytics = HeatmapAnalytics(
                        frame_width=frame.shape[1],
                        frame_height=frame.shape[0],
                        radius_px=self.heatmap_radius,
                        decay_factor=1.0,
                        overlay_alpha=self.heatmap_alpha,
                        min_density_threshold=0.05
                    )

                tracks = tracker.track(frame)
                current_fps = fps_meter.tick()

                fps_for_time = vs.fps if vs.fps > 0 else 30.0
                intel_snap = scene_intel.update(
                    frame_idx=frame_idx,
                    tracks=tracks,
                    timestamp_sec=frame_idx / fps_for_time,
                    fps=fps_for_time
                )

                spatial_snap = None
                spatial_events = []
                if spatial_analytics is not None:
                    spatial_snap = spatial_analytics.update(
                        frame_idx=frame_idx,
                        tracks=tracks,
                        timestamp_sec=frame_idx / fps_for_time,
                        fps=fps_for_time
                    )
                    spatial_events = spatial_snap.get("new_events", [])

                behavior_analytics.update(
                    frame_idx=frame_idx,
                    tracks=tracks,
                    timestamp_sec=frame_idx / fps_for_time,
                    fps=fps_for_time
                )
                behavior_summary = behavior_analytics.get_summary() if (frame_idx == 1 or frame_idx % 10 == 0) else None

                # Heatmap update exactly once per processed frame
                heatmap_density_preview = None
                heatmap_overlay_preview = None
                if heatmap_analytics is not None:
                    heatmap_analytics.update(
                        tracks=tracks,
                        frame_idx=frame_idx,
                        timestamp_sec=frame_idx / fps_for_time
                    )
                    # Throttled preview rendering: generate preview images at frame 1 and every 15 frames
                    if frame_idx == 1 or frame_idx % 15 == 0:
                        heatmap_density_preview = heatmap_analytics.generate_heatmap_image(normalize=True)
                        heatmap_overlay_preview = heatmap_analytics.apply_overlay(
                            frame,
                            alpha=self.heatmap_alpha,
                            min_density_threshold=0.05
                        )

                annotated = frame.copy()
                active_classes = Counter()

                for t in tracks:
                    active_classes[t.class_name] += 1
                    color = ColorPalette.get_color(t.track_id)
                    label = f"#{t.track_id} {t.class_name} {t.confidence:.2f}"
                    if len(t.trajectory) >= 2:
                        draw_trajectory(annotated, list(t.trajectory), color, thickness=2)
                    draw_bounding_box(annotated, t.box, label, color=color, thickness=2)

                if spatial_analytics is not None:
                    spatial_analytics.draw_overlays(annotated)

                draw_hud(
                    annotated,
                    fps=current_fps,
                    active_count=tracker.get_active_count(),
                    total_count=tracker.get_total_count(),
                    class_counts=dict(active_classes)
                )

                if writer is not None:
                    writer.write(annotated)

                # Send frame to UI queue (drop old frames if UI is lagging)
                payload = {
                    "frame": annotated,
                    "fps": round(current_fps, 1),
                    "active_count": tracker.get_active_count(),
                    "total_count": tracker.get_total_count(),
                    "class_counts": dict(active_classes),
                    "frame_idx": frame_idx,
                    "new_events": intel_snap.get("new_events", []),
                    "scene_summary": scene_intel.generate_natural_language_summary(),
                    "spatial_snap": spatial_snap,
                    "spatial_events": spatial_events,
                    "behavior_summary": behavior_summary,
                    "heatmap_preview": heatmap_density_preview,
                    "heatmap_overlay": heatmap_overlay_preview,
                    "heatmap_points": heatmap_analytics.total_points_accumulated if heatmap_analytics is not None else 0,
                    "heatmap_max_density": round(heatmap_analytics.max_density, 2) if heatmap_analytics is not None else 0.0
                }

                try:
                    if self.output_queue.full():
                        try:
                            self.output_queue.get_nowait()
                        except queue.Empty:
                            pass
                    self.output_queue.put_nowait(payload)
                except Exception:
                    pass

        except Exception as err:
            self.status_queue.put({"type": "error", "message": f"Worker error: {err}"})
        finally:
            if vs is not None:
                vs.release()
            if writer is not None:
                writer.release()
            final_summary = "Tracking completed."
            if scene_intel is not None:
                try:
                    scene_intel.finalize_stream(frame_idx)
                    final_summary = scene_intel.generate_natural_language_summary()
                except Exception:
                    pass
            if spatial_analytics is not None:
                try:
                    spatial_analytics.finalize_stream(frame_idx)
                except Exception:
                    pass
            final_behavior = None
            if behavior_analytics is not None:
                try:
                    behavior_analytics.finalize_stream(frame_idx)
                    final_behavior = behavior_analytics.get_summary()
                except Exception:
                    pass

            heatmap_result = None
            if self.enable_heatmap:
                if heatmap_analytics is not None and frame_idx > 0 and last_raw_frame is not None:
                    try:
                        if self.heatmap_output_dir:
                            hm_out_dir = Path(self.heatmap_output_dir)
                        else:
                            hm_out_dir = Path(__file__).resolve().parent.parent / "outputs"

                        hm_out_dir.mkdir(parents=True, exist_ok=True)
                        timestamp = int(time.time())
                        is_cam = str(self.source).isdigit()
                        hm_stem = f"webcam_{self.source}" if is_cam else Path(self.source).stem

                        base_hm_name = f"heatmap_gui_{hm_stem}_{timestamp}.png"
                        base_ov_name = f"heatmap_overlay_gui_{hm_stem}_{timestamp}.png"
                        hm_img_path = hm_out_dir / base_hm_name
                        ov_img_path = hm_out_dir / base_ov_name

                        counter = 1
                        while hm_img_path.exists() or ov_img_path.exists():
                            hm_img_path = hm_out_dir / f"heatmap_gui_{hm_stem}_{timestamp}_{counter}.png"
                            ov_img_path = hm_out_dir / f"heatmap_overlay_gui_{hm_stem}_{timestamp}_{counter}.png"
                            counter += 1

                        if out_file_path is not None:
                            video_out_res = Path(out_file_path).resolve()
                            if hm_img_path.resolve() == video_out_res or ov_img_path.resolve() == video_out_res:
                                raise ValueError(f"Heatmap output path collides with output video: {out_file_path}")

                        if not str(self.source).isdigit():
                            src_res = None
                            try:
                                src_res = Path(self.source).resolve()
                            except (ValueError, OSError):
                                pass
                            if src_res is not None:
                                if hm_img_path.resolve() == src_res or ov_img_path.resolve() == src_res:
                                    raise ValueError(f"Heatmap output path collides with source video: {self.source}")

                        heatmap_img = heatmap_analytics.generate_heatmap_image(normalize=True)
                        fallback_shape = (vs.height, vs.width, 3) if vs is not None and vs.height > 0 and vs.width > 0 else (480, 640, 3)
                        base_frame = (
                            last_raw_frame
                            if last_raw_frame is not None
                            else np.zeros(fallback_shape, dtype=np.uint8)
                        )
                        overlay_img = heatmap_analytics.apply_overlay(
                            base_frame,
                            alpha=self.heatmap_alpha,
                            min_density_threshold=0.05
                        )

                        first_created = False
                        first_existed = hm_img_path.exists()
                        success_hm = cv2.imwrite(str(hm_img_path.resolve()), heatmap_img)
                        if not success_hm:
                            raise IOError(f"Failed to write GUI heatmap image to: {hm_img_path}")
                        first_created = not first_existed and hm_img_path.exists()

                        try:
                            success_ov = cv2.imwrite(str(ov_img_path.resolve()), overlay_img)
                            if not success_ov:
                                raise IOError(f"Failed to write GUI heatmap overlay image to: {ov_img_path}")
                        except Exception as err:
                            if first_created and hm_img_path.exists():
                                try:
                                    hm_img_path.unlink()
                                except Exception:
                                    pass
                            if isinstance(err, IOError):
                                raise
                            raise IOError(f"Failed to write GUI heatmap overlay image to: {ov_img_path}") from err

                        heatmap_result = {
                            "enabled": True,
                            "heatmap_image_path": str(hm_img_path.resolve()),
                            "overlay_image_path": str(ov_img_path.resolve()),
                            "output_dir": str(hm_out_dir.resolve()),
                            "summary": heatmap_analytics.get_summary(),
                            "final_heatmap_img": heatmap_img,
                            "final_overlay_img": overlay_img
                        }
                    except Exception as hm_err:
                        self.status_queue.put({"type": "error", "message": f"Heatmap export error: {hm_err}"})
                        heatmap_result = {
                            "enabled": True,
                            "error": str(hm_err),
                            "heatmap_image_path": None,
                            "overlay_image_path": None,
                            "summary": heatmap_analytics.get_summary() if heatmap_analytics is not None else None
                        }
                else:
                    heatmap_result = {
                        "enabled": True,
                        "heatmap_image_path": None,
                        "overlay_image_path": None,
                        "summary": {
                            "status": "No usable frames were captured",
                            "frames_processed": 0,
                            "total_points_accumulated": 0,
                            "is_empty": True
                        }
                    }

            self.status_queue.put({
                "type": "finished",
                "output_file": out_file_path,
                "summary": final_summary,
                "behavior_summary": final_behavior,
                "heatmap_result": heatmap_result,
                "message": "Tracking stopped. Resources released safely."
            })


class ObjectTrackerApp:
    """
    Main Tkinter application window providing a professional dashboard for
    Object Detection and Tracking.
    """

    DARK_BG = "#1e1e2e"
    PANEL_BG = "#252538"
    CARD_BG = "#2e2e44"
    ACCENT_COLOR = "#89b4fa"
    TEXT_COLOR = "#cdd6f4"
    SUCCESS_COLOR = "#a6e3a1"
    DANGER_COLOR = "#f38ba8"

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("CodeAlpha — Object Detection & Multi-Object Tracking")
        self.root.geometry("1180x760")
        self.root.minsize(980, 680)
        self.root.configure(bg=self.DARK_BG)

        self.worker: Optional[TrackingWorker] = None
        self.output_queue: queue.Queue = queue.Queue(maxsize=3)
        self.status_queue: queue.Queue = queue.Queue()
        self.current_photo_image: Optional[ImageTk.PhotoImage] = None

        self.latest_stream_frame: Optional[np.ndarray] = None
        self.latest_heatmap_preview: Optional[np.ndarray] = None
        self.latest_heatmap_overlay: Optional[np.ndarray] = None
        self.saved_heatmap_paths: Optional[Tuple[str, str, Optional[str]]] = None
        self._is_closing: bool = False
        self._poll_job: Optional[str] = None

        self._configure_styles()
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._poll_queues()

    def _configure_styles(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame", background=self.DARK_BG)
        style.configure("Card.TFrame", background=self.PANEL_BG)
        style.configure("TLabel", background=self.DARK_BG, foreground=self.TEXT_COLOR, font=("Segoe UI", 10))
        style.configure("Header.TLabel", font=("Segoe UI", 14, "bold"), foreground=self.ACCENT_COLOR)
        style.configure("SubHeader.TLabel", font=("Segoe UI", 10, "bold"), foreground=self.TEXT_COLOR)
        style.configure("StatNum.TLabel", font=("Segoe UI", 18, "bold"), foreground=self.ACCENT_COLOR)
        style.configure("StatLabel.TLabel", font=("Segoe UI", 9), foreground="#a6adc8")

    def _build_ui(self) -> None:
        # Top Header Banner
        header_frame = tk.Frame(self.root, bg=self.PANEL_BG, height=54)
        header_frame.pack(fill=tk.X, side=tk.TOP, padx=0, pady=0)

        title_lbl = tk.Label(
            header_frame,
            text="🎯 CodeAlpha AI Internship — Object Detection & Tracking",
            font=("Segoe UI", 13, "bold"),
            bg=self.PANEL_BG,
            fg=self.ACCENT_COLOR
        )
        title_lbl.pack(side=tk.LEFT, padx=16, pady=12)

        self.status_badge = tk.Label(
            header_frame,
            text="IDLE",
            font=("Segoe UI", 9, "bold"),
            bg="#45475a",
            fg="#ffffff",
            padx=10,
            pady=3
        )
        self.status_badge.pack(side=tk.RIGHT, padx=16, pady=12)

        # Main Layout (Sidebar Left, Center Video, Right Stats)
        main_content = tk.Frame(self.root, bg=self.DARK_BG)
        main_content.pack(fill=tk.BOTH, expand=True, padx=12, pady=10)

        # Left Sidebar (Controls)
        sidebar = tk.Frame(main_content, bg=self.PANEL_BG, width=300)
        sidebar.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        # Right Stats Panel
        stats_panel = tk.Frame(main_content, bg=self.PANEL_BG, width=240)
        stats_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        stats_panel.pack_propagate(False)
        self._build_stats_panel(stats_panel)

        # Center Video Canvas
        center_frame = tk.Frame(main_content, bg=self.CARD_BG)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # View Mode Toolbar
        view_bar = tk.Frame(center_frame, bg=self.CARD_BG)
        view_bar.pack(fill=tk.X, side=tk.TOP, padx=6, pady=(4, 2))

        tk.Label(
            view_bar, text="View:", font=("Segoe UI", 9, "bold"),
            bg=self.CARD_BG, fg=self.ACCENT_COLOR
        ).pack(side=tk.LEFT, padx=(4, 8))

        self.view_mode = tk.StringVar(value="stream")
        rb_stream = tk.Radiobutton(
            view_bar, text="Tracking Stream", variable=self.view_mode, value="stream",
            command=self._on_view_mode_changed, bg=self.CARD_BG, fg=self.TEXT_COLOR,
            selectcolor=self.PANEL_BG, activebackground=self.CARD_BG, font=("Segoe UI", 9)
        )
        rb_stream.pack(side=tk.LEFT, padx=4)

        rb_ov = tk.Radiobutton(
            view_bar, text="Heatmap Overlay", variable=self.view_mode, value="heatmap_overlay",
            command=self._on_view_mode_changed, bg=self.CARD_BG, fg=self.TEXT_COLOR,
            selectcolor=self.PANEL_BG, activebackground=self.CARD_BG, font=("Segoe UI", 9)
        )
        rb_ov.pack(side=tk.LEFT, padx=4)

        rb_den = tk.Radiobutton(
            view_bar, text="Heatmap Density", variable=self.view_mode, value="heatmap_density",
            command=self._on_view_mode_changed, bg=self.CARD_BG, fg=self.TEXT_COLOR,
            selectcolor=self.PANEL_BG, activebackground=self.CARD_BG, font=("Segoe UI", 9)
        )
        rb_den.pack(side=tk.LEFT, padx=4)

        self.video_canvas = tk.Canvas(center_frame, bg="#11111b", highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=(2, 6))

        # Initial canvas placeholder
        self._draw_placeholder("Click 'Start Tracking' to begin video processing")

    def _build_sidebar(self, parent: tk.Frame) -> None:
        pad_x = 12
        pad_y = 6

        lbl = tk.Label(parent, text="Stream Source", font=("Segoe UI", 11, "bold"), bg=self.PANEL_BG, fg=self.ACCENT_COLOR)
        lbl.pack(anchor=tk.W, padx=pad_x, pady=(12, 4))

        self.source_mode = tk.StringVar(value="webcam")
        rb1 = tk.Radiobutton(
            parent, text="Webcam", variable=self.source_mode, value="webcam",
            command=self._on_source_mode_changed, bg=self.PANEL_BG, fg=self.TEXT_COLOR,
            selectcolor=self.CARD_BG, activebackground=self.PANEL_BG
        )
        rb1.pack(anchor=tk.W, padx=pad_x)

        rb2 = tk.Radiobutton(
            parent, text="Video File", variable=self.source_mode, value="video",
            command=self._on_source_mode_changed, bg=self.PANEL_BG, fg=self.TEXT_COLOR,
            selectcolor=self.CARD_BG, activebackground=self.PANEL_BG
        )
        rb2.pack(anchor=tk.W, padx=pad_x)

        # Webcam input options
        self.webcam_frame = tk.Frame(parent, bg=self.PANEL_BG)
        self.webcam_frame.pack(fill=tk.X, padx=pad_x, pady=4)
        tk.Label(self.webcam_frame, text="Camera Index:", bg=self.PANEL_BG, fg=self.TEXT_COLOR).pack(side=tk.LEFT)
        self.webcam_idx_var = tk.StringVar(value="0")
        self.webcam_spin = tk.Spinbox(self.webcam_frame, from_=0, to=5, width=4, textvariable=self.webcam_idx_var)
        self.webcam_spin.pack(side=tk.LEFT, padx=8)

        # Video file input options
        self.file_frame = tk.Frame(parent, bg=self.PANEL_BG)
        self.file_path_var = tk.StringVar(value="assets/sample_clip.mp4")
        tk.Entry(self.file_frame, textvariable=self.file_path_var, width=20).pack(side=tk.LEFT, fill=tk.X, expand=True)
        btn_browse = tk.Button(self.file_frame, text="Browse", command=self._browse_file, bg=self.CARD_BG, fg=self.TEXT_COLOR)
        btn_browse.pack(side=tk.LEFT, padx=(4, 0))

        # Model & Tracker Settings
        tk.Label(parent, text="Model Configuration", font=("Segoe UI", 11, "bold"), bg=self.PANEL_BG, fg=self.ACCENT_COLOR).pack(anchor=tk.W, padx=pad_x, pady=(16, 4))

        tk.Label(parent, text="YOLO Model:", bg=self.PANEL_BG, fg=self.TEXT_COLOR).pack(anchor=tk.W, padx=pad_x)
        self.model_var = tk.StringVar(value="yolo11n.pt")
        model_combo = ttk.Combobox(parent, textvariable=self.model_var, values=["yolo11n.pt", "yolov8n.pt"], state="readonly")
        model_combo.pack(fill=tk.X, padx=pad_x, pady=(0, 6))

        tk.Label(parent, text="Tracker Type:", bg=self.PANEL_BG, fg=self.TEXT_COLOR).pack(anchor=tk.W, padx=pad_x)
        self.tracker_var = tk.StringVar(value="bytetrack.yaml")
        tracker_combo = ttk.Combobox(parent, textvariable=self.tracker_var, values=["bytetrack.yaml", "botsort.yaml"], state="readonly")
        tracker_combo.pack(fill=tk.X, padx=pad_x, pady=(0, 6))

        # Confidence Slider
        tk.Label(parent, text="Confidence Threshold:", bg=self.PANEL_BG, fg=self.TEXT_COLOR).pack(anchor=tk.W, padx=pad_x)
        self.conf_slider = tk.Scale(
            parent, from_=0.05, to=1.0, resolution=0.05, orient=tk.HORIZONTAL,
            bg=self.PANEL_BG, fg=self.TEXT_COLOR, highlightthickness=0
        )
        self.conf_slider.set(0.25)
        self.conf_slider.pack(fill=tk.X, padx=pad_x, pady=(0, 6))

        # Save Output Checkbox
        self.save_var = tk.BooleanVar(value=True)
        cb_save = tk.Checkbutton(
            parent, text="Save Annotated Output", variable=self.save_var,
            bg=self.PANEL_BG, fg=self.TEXT_COLOR, selectcolor=self.CARD_BG, activebackground=self.PANEL_BG
        )
        cb_save.pack(anchor=tk.W, padx=pad_x, pady=(4, 2))

        # Spatial Analytics Checkbox
        self.spatial_var = tk.BooleanVar(value=True)
        cb_spatial = tk.Checkbutton(
            parent, text="Virtual Line & ROI Analytics", variable=self.spatial_var,
            bg=self.PANEL_BG, fg=self.TEXT_COLOR, selectcolor=self.CARD_BG, activebackground=self.PANEL_BG
        )
        cb_spatial.pack(anchor=tk.W, padx=pad_x, pady=(0, 6))

        # Spatial Heatmap Section
        heatmap_frame = tk.LabelFrame(
            parent, text="Spatial Heatmap", font=("Segoe UI", 9, "bold"),
            bg=self.PANEL_BG, fg=self.ACCENT_COLOR, padx=6, pady=4
        )
        heatmap_frame.pack(fill=tk.X, padx=pad_x, pady=(0, 8))

        self.heatmap_var = tk.BooleanVar(value=False)
        cb_heatmap = tk.Checkbutton(
            heatmap_frame, text="Enable Heatmap", variable=self.heatmap_var,
            command=self._on_heatmap_toggle, bg=self.PANEL_BG, fg=self.TEXT_COLOR,
            selectcolor=self.CARD_BG, activebackground=self.PANEL_BG, font=("Segoe UI", 9)
        )
        cb_heatmap.pack(anchor=tk.W)

        # Config frame (radius and opacity)
        row_params = tk.Frame(heatmap_frame, bg=self.PANEL_BG)
        row_params.pack(fill=tk.X, pady=(2, 2))

        tk.Label(row_params, text="Radius:", font=("Segoe UI", 8), bg=self.PANEL_BG, fg=self.TEXT_COLOR).pack(side=tk.LEFT)
        self.heatmap_radius_var = tk.StringVar(value="20")
        self.heatmap_radius_spin = tk.Spinbox(
            row_params, from_=5, to=100, increment=5, width=4,
            textvariable=self.heatmap_radius_var, bg="#11111b", fg=self.TEXT_COLOR
        )
        self.heatmap_radius_spin.pack(side=tk.LEFT, padx=(2, 6))

        tk.Label(row_params, text="Opacity:", font=("Segoe UI", 8), bg=self.PANEL_BG, fg=self.TEXT_COLOR).pack(side=tk.LEFT)
        self.heatmap_alpha_var = tk.StringVar(value="0.5")
        self.heatmap_alpha_spin = tk.Spinbox(
            row_params, from_=0.1, to=1.0, increment=0.1, width=4,
            textvariable=self.heatmap_alpha_var, bg="#11111b", fg=self.TEXT_COLOR
        )
        self.heatmap_alpha_spin.pack(side=tk.LEFT, padx=(2, 0))

        # Output dir row
        row_dir = tk.Frame(heatmap_frame, bg=self.PANEL_BG)
        row_dir.pack(fill=tk.X, pady=(2, 2))
        self.heatmap_dir_var = tk.StringVar(value="")
        self.heatmap_dir_entry = tk.Entry(
            row_dir, textvariable=self.heatmap_dir_var, font=("Segoe UI", 8),
            bg="#11111b", fg=self.TEXT_COLOR, width=16
        )
        self.heatmap_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        btn_hm_browse = tk.Button(
            row_dir, text="Dir", font=("Segoe UI", 8),
            command=self._browse_heatmap_dir, bg=self.CARD_BG, fg=self.TEXT_COLOR, padx=3
        )
        btn_hm_browse.pack(side=tk.LEFT, padx=(2, 0))

        # Status and Open folder row
        row_status = tk.Frame(heatmap_frame, bg=self.PANEL_BG)
        row_status.pack(fill=tk.X, pady=(2, 2))
        self.lbl_heatmap_status = tk.Label(
            row_status, text="Status: Disabled", font=("Segoe UI", 8, "bold"),
            bg=self.PANEL_BG, fg="#a6adc8"
        )
        self.lbl_heatmap_status.pack(side=tk.LEFT)

        self.btn_open_heatmap_dir = tk.Button(
            row_status, text="📂 Open", font=("Segoe UI", 8),
            command=self.open_heatmap_folder, bg=self.CARD_BG, fg=self.TEXT_COLOR,
            state=tk.DISABLED, padx=4
        )
        self.btn_open_heatmap_dir.pack(side=tk.RIGHT)

        # Start / Stop Control Buttons
        self.btn_start = tk.Button(
            parent, text="▶  Start Tracking", font=("Segoe UI", 11, "bold"),
            bg="#2ecc71", fg="#ffffff", activebackground="#27ae60",
            relief=tk.FLAT, pady=6, command=self.start_tracking
        )
        self.btn_start.pack(fill=tk.X, padx=pad_x, pady=4)

        self.btn_stop = tk.Button(
            parent, text="⏹  Stop Tracking", font=("Segoe UI", 11, "bold"),
            bg="#e74c3c", fg="#ffffff", activebackground="#c0392b",
            relief=tk.FLAT, pady=6, state=tk.DISABLED, command=self.stop_tracking
        )
        self.btn_stop.pack(fill=tk.X, padx=pad_x, pady=4)

        # Log box at bottom of sidebar
        tk.Label(parent, text="Activity Log", font=("Segoe UI", 10, "bold"), bg=self.PANEL_BG, fg=self.ACCENT_COLOR).pack(anchor=tk.W, padx=pad_x, pady=(12, 2))
        self.log_text = tk.Text(parent, height=5, bg="#11111b", fg="#a6adc8", font=("Consolas", 8), relief=tk.FLAT)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=pad_x, pady=(0, 10))

    def _build_stats_panel(self, parent: tk.Frame) -> None:
        pad_x = 10
        tk.Label(parent, text="Live Analytics", font=("Segoe UI", 11, "bold"), bg=self.PANEL_BG, fg=self.ACCENT_COLOR).pack(anchor=tk.W, padx=pad_x, pady=(12, 8))

        # Card: FPS
        card_fps = tk.Frame(parent, bg=self.CARD_BG, pady=6, padx=8)
        card_fps.pack(fill=tk.X, padx=pad_x, pady=4)
        tk.Label(card_fps, text="Processing FPS", font=("Segoe UI", 9), bg=self.CARD_BG, fg="#a6adc8").pack(anchor=tk.W)
        self.lbl_fps = tk.Label(card_fps, text="0.0", font=("Segoe UI", 18, "bold"), bg=self.CARD_BG, fg="#2ecc71")
        self.lbl_fps.pack(anchor=tk.W)

        # Card: Active Tracks
        card_act = tk.Frame(parent, bg=self.CARD_BG, pady=6, padx=8)
        card_act.pack(fill=tk.X, padx=pad_x, pady=4)
        tk.Label(card_act, text="Active Objects", font=("Segoe UI", 9), bg=self.CARD_BG, fg="#a6adc8").pack(anchor=tk.W)
        self.lbl_active = tk.Label(card_act, text="0", font=("Segoe UI", 18, "bold"), bg=self.CARD_BG, fg=self.ACCENT_COLOR)
        self.lbl_active.pack(anchor=tk.W)

        # Card: Total Unique Tracks
        card_tot = tk.Frame(parent, bg=self.CARD_BG, pady=6, padx=8)
        card_tot.pack(fill=tk.X, padx=pad_x, pady=4)
        tk.Label(card_tot, text="Total Unique IDs", font=("Segoe UI", 9), bg=self.CARD_BG, fg="#a6adc8").pack(anchor=tk.W)
        self.lbl_total = tk.Label(card_tot, text="0", font=("Segoe UI", 18, "bold"), bg=self.CARD_BG, fg="#f1c40f")
        self.lbl_total.pack(anchor=tk.W)

        # Card: Scene Intelligence Summary
        card_scene = tk.Frame(parent, bg=self.CARD_BG, pady=6, padx=8)
        card_scene.pack(fill=tk.X, padx=pad_x, pady=4)
        tk.Label(card_scene, text="Scene Intelligence", font=("Segoe UI", 9, "bold"), bg=self.CARD_BG, fg=self.ACCENT_COLOR).pack(anchor=tk.W)
        self.lbl_scene_summary = tk.Label(
            card_scene,
            text="Awaiting stream...",
            font=("Segoe UI", 8),
            bg=self.CARD_BG,
            fg="#cdd6f4",
            wraplength=210,
            justify=tk.LEFT
        )
        self.lbl_scene_summary.pack(anchor=tk.W, pady=(2, 0))

        # Card: Spatial Analytics
        card_spatial = tk.Frame(parent, bg=self.CARD_BG, pady=6, padx=8)
        card_spatial.pack(fill=tk.X, padx=pad_x, pady=4)
        tk.Label(card_spatial, text="Spatial Analytics", font=("Segoe UI", 9, "bold"), bg=self.CARD_BG, fg=self.ACCENT_COLOR).pack(anchor=tk.W)
        self.lbl_spatial_line = tk.Label(
            card_spatial,
            text="Line: 0 crosses",
            font=("Segoe UI", 8),
            bg=self.CARD_BG,
            fg="#cdd6f4"
        )
        self.lbl_spatial_line.pack(anchor=tk.W, pady=(2, 0))
        self.lbl_spatial_roi = tk.Label(
            card_spatial,
            text="ROI: 0 inside (In: 0 | Out: 0)",
            font=("Segoe UI", 8),
            bg=self.CARD_BG,
            fg="#a6adc8"
        )
        self.lbl_spatial_roi.pack(anchor=tk.W, pady=(1, 0))

        # Card: Trajectory Behavior Analytics
        card_behavior = tk.Frame(parent, bg=self.CARD_BG, pady=6, padx=8)
        card_behavior.pack(fill=tk.X, padx=pad_x, pady=4)
        tk.Label(card_behavior, text="Trajectory Analytics", font=("Segoe UI", 9, "bold"), bg=self.CARD_BG, fg=self.ACCENT_COLOR).pack(anchor=tk.W)
        self.lbl_behavior_motion = tk.Label(
            card_behavior,
            text="Avg Len: 0.0px | Disp: 0.0px",
            font=("Segoe UI", 8),
            bg=self.CARD_BG,
            fg="#cdd6f4"
        )
        self.lbl_behavior_motion.pack(anchor=tk.W, pady=(2, 0))
        self.lbl_behavior_dynamics = tk.Label(
            card_behavior,
            text="Tort: 1.00 | Turns: 0 | Speed: N/A",
            font=("Segoe UI", 8),
            bg=self.CARD_BG,
            fg="#a6adc8"
        )
        self.lbl_behavior_dynamics.pack(anchor=tk.W, pady=(1, 0))

        # Card: Spatial Heatmap
        card_heatmap = tk.Frame(parent, bg=self.CARD_BG, pady=6, padx=8)
        card_heatmap.pack(fill=tk.X, padx=pad_x, pady=4)
        tk.Label(card_heatmap, text="Spatial Heatmap", font=("Segoe UI", 9, "bold"), bg=self.CARD_BG, fg=self.ACCENT_COLOR).pack(anchor=tk.W)
        self.lbl_heatmap_points = tk.Label(
            card_heatmap,
            text="Points: 0 | Max: 0.0",
            font=("Segoe UI", 8),
            bg=self.CARD_BG,
            fg="#cdd6f4"
        )
        self.lbl_heatmap_points.pack(anchor=tk.W, pady=(2, 0))
        self.lbl_heatmap_export_info = tk.Label(
            card_heatmap,
            text="Exports: Disabled",
            font=("Segoe UI", 8),
            bg=self.CARD_BG,
            fg="#a6adc8"
        )
        self.lbl_heatmap_export_info.pack(anchor=tk.W, pady=(1, 0))

        # Class Distribution list
        tk.Label(parent, text="Class Counts", font=("Segoe UI", 10, "bold"), bg=self.PANEL_BG, fg=self.ACCENT_COLOR).pack(anchor=tk.W, padx=pad_x, pady=(6, 4))
        self.class_listbox = tk.Listbox(parent, bg="#11111b", fg=self.TEXT_COLOR, font=("Segoe UI", 9), height=3, relief=tk.FLAT)
        self.class_listbox.pack(fill=tk.BOTH, expand=True, padx=pad_x, pady=(0, 10))

    def _on_source_mode_changed(self) -> None:
        if self.source_mode.get() == "webcam":
            self.file_frame.pack_forget()
            self.webcam_frame.pack(fill=tk.X, padx=12, pady=4)
        else:
            self.webcam_frame.pack_forget()
            self.file_frame.pack(fill=tk.X, padx=12, pady=4)

    def _browse_file(self) -> None:
        file_selected = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.webm"), ("All files", "*.*")]
        )
        if file_selected:
            self.file_path_var.set(file_selected)

    def _on_heatmap_toggle(self) -> None:
        if self.heatmap_var.get():
            self.lbl_heatmap_status.config(text="Status: Ready", fg=self.ACCENT_COLOR)
            self.lbl_heatmap_export_info.config(text="Exports: Ready")
        else:
            self.lbl_heatmap_status.config(text="Status: Disabled", fg="#a6adc8")
            self.lbl_heatmap_export_info.config(text="Exports: Disabled")

    def _browse_heatmap_dir(self) -> None:
        dir_selected = filedialog.askdirectory(title="Select Heatmap Output Directory")
        if dir_selected:
            self.heatmap_dir_var.set(dir_selected)

    def open_heatmap_folder(self) -> None:
        """Safely opens the heatmap output folder in the system file manager without shell strings."""
        if self.saved_heatmap_paths and self.saved_heatmap_paths[2]:
            target_dir = Path(self.saved_heatmap_paths[2])
        elif self.heatmap_dir_var.get().strip():
            target_dir = Path(self.heatmap_dir_var.get().strip())
        else:
            target_dir = Path(__file__).resolve().parent.parent / "outputs"

        if not target_dir.exists():
            messagebox.showwarning("Directory Not Found", f"Heatmap output directory does not exist:\n{target_dir}")
            return

        resolved_str = str(target_dir.resolve())
        try:
            if hasattr(os, "startfile"):
                os.startfile(resolved_str)
            elif sys.platform == "darwin":
                import subprocess
                subprocess.run(["open", resolved_str], check=False)
            else:
                import subprocess
                subprocess.run(["xdg-open", resolved_str], check=False)
        except Exception as err:
            messagebox.showerror("System Error", f"Failed to open directory:\n{err}")

    def _on_view_mode_changed(self) -> None:
        self._refresh_canvas_view()

    def _refresh_canvas_view(self) -> None:
        mode = self.view_mode.get()
        if mode == "stream":
            if self.latest_stream_frame is not None:
                self._render_frame(self.latest_stream_frame)
            else:
                self._draw_placeholder("Click 'Start Tracking' to begin video processing")
        elif mode == "heatmap_overlay":
            if self.latest_heatmap_overlay is not None:
                self._render_frame(self.latest_heatmap_overlay)
            elif self.heatmap_var.get():
                self._draw_placeholder("Heatmap overlay will appear as frames are processed")
            else:
                self._draw_placeholder("Enable 'Spatial Heatmap' in sidebar to preview overlay")
        elif mode == "heatmap_density":
            if self.latest_heatmap_preview is not None:
                self._render_frame(self.latest_heatmap_preview)
            elif self.heatmap_var.get():
                self._draw_placeholder("Heatmap density will appear as frames are processed")
            else:
                self._draw_placeholder("Enable 'Spatial Heatmap' in sidebar to preview density")

    def log(self, message: str) -> None:
        now_str = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{now_str}] {message}\n")
        self.log_text.see(tk.END)

    def _draw_placeholder(self, text: str) -> None:
        self.video_canvas.delete("all")
        w = self.video_canvas.winfo_width() or 640
        h = self.video_canvas.winfo_height() or 480
        self.video_canvas.create_text(
            w // 2, h // 2,
            text=text,
            fill="#6c7086",
            font=("Segoe UI", 12)
        )

    def start_tracking(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            return

        if self.source_mode.get() == "webcam":
            source = self.webcam_idx_var.get().strip()
        else:
            source = self.file_path_var.get().strip()
            if not Path(source).exists():
                messagebox.showerror("File Error", f"Video file not found:\n{source}")
                return

        # Validate Heatmap settings if enabled
        enable_hm = self.heatmap_var.get()
        hm_radius = 20
        hm_alpha = 0.5
        hm_dir = None
        if enable_hm:
            try:
                hm_radius = int(self.heatmap_radius_var.get().strip())
                if not (5 <= hm_radius <= 200):
                    raise ValueError("Radius must be between 5 and 200.")
            except Exception as e:
                messagebox.showerror("Heatmap Config Error", f"Invalid heatmap radius: {e}\nPlease enter an integer between 5 and 200.")
                return

            try:
                hm_alpha = float(self.heatmap_alpha_var.get().strip())
                if not (0.0 <= hm_alpha <= 1.0):
                    raise ValueError("Opacity must be between 0.0 and 1.0.")
            except Exception as e:
                messagebox.showerror("Heatmap Config Error", f"Invalid heatmap opacity: {e}\nPlease enter a number between 0.0 and 1.0.")
                return

            custom_dir = self.heatmap_dir_var.get().strip()
            if custom_dir:
                try:
                    p = Path(custom_dir)
                    p.mkdir(parents=True, exist_ok=True)
                    probe = p / ".probe_check"
                    probe.touch()
                    probe.unlink()
                    hm_dir = str(p.resolve())
                except Exception as e:
                    messagebox.showerror("Heatmap Config Error", f"Cannot write to heatmap output directory:\n{custom_dir}\nError: {e}")
                    return

        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_badge.config(text="RUNNING", bg="#2ecc71")
        self.lbl_scene_summary.config(text="Analyzing scene activity...")
        self.log(f"Starting tracking on {source}...")

        virt_line = None
        roi_reg = None
        if self.spatial_var.get():
            virt_line = (50, 240, 590, 240)
            roi_reg = (120, 100, 520, 380)

        self.lbl_spatial_line.config(text="Line: 0 crosses" if virt_line else "Line: inactive")
        self.lbl_spatial_roi.config(text="ROI: 0 inside (In: 0 | Out: 0)" if roi_reg else "ROI: inactive")
        self.lbl_behavior_motion.config(text="Avg Len: 0.0px | Disp: 0.0px")
        self.lbl_behavior_dynamics.config(text="Tort: 1.00 | Turns: 0 | Speed: N/A")

        # Session Reset for Heatmap state
        self.latest_stream_frame = None
        self.latest_heatmap_preview = None
        self.latest_heatmap_overlay = None
        self.saved_heatmap_paths = None
        self.btn_open_heatmap_dir.config(state=tk.DISABLED)
        if enable_hm:
            self.lbl_heatmap_status.config(text="Status: Active", fg="#2ecc71")
            self.lbl_heatmap_points.config(text="Points: 0 | Max: 0.0")
            self.lbl_heatmap_export_info.config(text="Exports: Active")
        else:
            self.lbl_heatmap_status.config(text="Status: Disabled", fg="#a6adc8")
            self.lbl_heatmap_points.config(text="Points: 0 | Max: 0.0")
            self.lbl_heatmap_export_info.config(text="Exports: Disabled")

        self.worker = TrackingWorker(
            source=source,
            model_name=self.model_var.get(),
            tracker_type=self.tracker_var.get(),
            conf_threshold=float(self.conf_slider.get()),
            device="cpu",
            save_output=self.save_var.get(),
            output_queue=self.output_queue,
            status_queue=self.status_queue,
            virtual_line=virt_line,
            roi_region=roi_reg,
            enable_heatmap=enable_hm,
            heatmap_radius=hm_radius,
            heatmap_alpha=hm_alpha,
            heatmap_output_dir=hm_dir
        )
        self.worker.start()

    def stop_tracking(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.log("Stopping tracking worker...")
            self.status_badge.config(text="STOPPING", bg="#e67e22")

    def _poll_queues(self) -> None:
        if self._is_closing:
            return

        # Check status updates
        try:
            while True:
                msg = self.status_queue.get_nowait()
                if not isinstance(msg, dict):
                    continue
                mtype = msg.get("type")
                if mtype == "status":
                    status_text = msg.get("message")
                    if status_text:
                        self.log(str(status_text))
                elif mtype == "error":
                    err_text = msg.get("message", "An unexpected tracking error occurred.")
                    self.log(f"ERROR: {err_text}")
                    try:
                        messagebox.showerror("Tracking Error", str(err_text))
                    except Exception:
                        pass
                elif mtype == "finished":
                    if msg.get("message"):
                        self.log(str(msg["message"]))
                    if msg.get("output_file"):
                        self.log(f"Saved: {msg['output_file']}")
                    if msg.get("summary"):
                        self.lbl_scene_summary.config(text=str(msg["summary"]))
                    if msg.get("behavior_summary") and isinstance(msg["behavior_summary"], dict):
                        b = msg["behavior_summary"]
                        sp_val = b.get("avg_speed_px_per_sec")
                        sp_str = f"{sp_val:.1f} px/s" if sp_val is not None else "N/A"
                        avg_len = b.get("avg_path_length_px", 0.0)
                        disp = b.get("avg_net_displacement_px", 0.0)
                        tort = b.get("avg_tortuosity", 1.0)
                        turns = b.get("total_direction_changes", 0)
                        tot_tracks = b.get("total_tracks_analyzed", 0)
                        self.lbl_behavior_motion.config(
                            text=f"Avg Len: {avg_len:.1f}px | Disp: {disp:.1f}px"
                        )
                        self.lbl_behavior_dynamics.config(
                            text=f"Tort: {tort:.2f} | Turns: {turns} | Speed: {sp_str}"
                        )
                        self.log(f"[BEHAVIOR] {tot_tracks} tracks | Avg Len: {avg_len}px | Disp: {disp}px | Speed: {sp_str}")

                    hm_res = msg.get("heatmap_result")
                    if hm_res and isinstance(hm_res, dict) and hm_res.get("enabled"):
                        if hm_res.get("error"):
                            self.lbl_heatmap_status.config(text="Status: Failed", fg="#e74c3c")
                            self.lbl_heatmap_export_info.config(text="Export Failed")
                        elif hm_res.get("heatmap_image_path") and hm_res.get("overlay_image_path"):
                            self.lbl_heatmap_status.config(text="Status: Completed", fg="#a6e3a1")
                            hm_p = str(hm_res["heatmap_image_path"])
                            ov_p = str(hm_res["overlay_image_path"])
                            self.saved_heatmap_paths = (hm_p, ov_p, hm_res.get("output_dir"))
                            self.btn_open_heatmap_dir.config(state=tk.NORMAL)
                            self.log(f"[HEATMAP] Saved colormap: {hm_p}")
                            self.log(f"[HEATMAP] Saved overlay:  {ov_p}")
                            self.lbl_heatmap_export_info.config(text="Saved to outputs")
                            if "final_heatmap_img" in hm_res and hm_res["final_heatmap_img"] is not None:
                                self.latest_heatmap_preview = hm_res["final_heatmap_img"]
                            if "final_overlay_img" in hm_res and hm_res["final_overlay_img"] is not None:
                                self.latest_heatmap_overlay = hm_res["final_overlay_img"]
                            self._refresh_canvas_view()
                        else:
                            self.lbl_heatmap_status.config(text="Status: No frames", fg="#f1c40f")
                            self.lbl_heatmap_export_info.config(text="No frames")

                    self.btn_start.config(state=tk.NORMAL)
                    self.btn_stop.config(state=tk.DISABLED)
                    self.status_badge.config(text="IDLE", bg="#45475a")
        except queue.Empty:
            pass

        # Check latest frame update and accumulate events
        new_events_batch = []
        spatial_events_batch = []
        try:
            latest = None
            while True:
                item = self.output_queue.get_nowait()
                if not isinstance(item, dict):
                    continue
                if "new_events" in item and item["new_events"]:
                    new_events_batch.extend(item["new_events"])
                if "spatial_events" in item and item["spatial_events"]:
                    spatial_events_batch.extend(item["spatial_events"])
                latest = item
        except queue.Empty:
            pass

        if latest is not None:
            self.latest_stream_frame = latest["frame"]
            if latest.get("heatmap_preview") is not None:
                self.latest_heatmap_preview = latest["heatmap_preview"]
            if latest.get("heatmap_overlay") is not None:
                self.latest_heatmap_overlay = latest["heatmap_overlay"]

            if latest.get("heatmap_points") is not None and self.heatmap_var.get():
                pts = latest["heatmap_points"]
                max_d = latest.get("heatmap_max_density", 0.0)
                self.lbl_heatmap_points.config(text=f"Points: {pts} | Max: {max_d}")

            self._refresh_canvas_view()
            self.lbl_fps.config(text=f"{latest['fps']}")
            self.lbl_active.config(text=f"{latest['active_count']}")
            self.lbl_total.config(text=f"{latest['total_count']}")

            # Update scene intelligence summary card
            if latest.get("scene_summary"):
                self.lbl_scene_summary.config(text=latest["scene_summary"])

            # Update trajectory behavior analytics card
            behavior_snap = latest.get("behavior_summary")
            if behavior_snap:
                sp_val = behavior_snap.get("avg_speed_px_per_sec")
                sp_str = f"{sp_val:.1f} px/s" if sp_val is not None else "N/A"
                self.lbl_behavior_motion.config(
                    text=f"Avg Len: {behavior_snap['avg_path_length_px']:.1f}px | Disp: {behavior_snap['avg_net_displacement_px']:.1f}px"
                )
                self.lbl_behavior_dynamics.config(
                    text=f"Tort: {behavior_snap['avg_tortuosity']:.2f} | Turns: {behavior_snap['total_direction_changes']} | Speed: {sp_str}"
                )

            # Update spatial analytics summary card
            spatial_snap = latest.get("spatial_snap")
            if spatial_snap:
                if spatial_snap.get("virtual_line_configured"):
                    tot = spatial_snap.get("line_crossings_total", 0)
                    self.lbl_spatial_line.config(text=f"Line: {tot} crossings")
                else:
                    self.lbl_spatial_line.config(text="Line: not configured")

                if spatial_snap.get("roi_configured"):
                    occ = spatial_snap.get("roi_current_occupancy", 0)
                    entries = spatial_snap.get("roi_entries_total", 0)
                    exits = spatial_snap.get("roi_exits_total", 0)
                    self.lbl_spatial_roi.config(text=f"ROI: {occ} inside (In: {entries} | Out: {exits})")
                else:
                    self.lbl_spatial_roi.config(text="ROI: not configured")

            # Log events
            for evt in new_events_batch:
                desc = evt.get("description") if isinstance(evt, dict) else getattr(evt, "description", str(evt))
                etype = evt.get("event_type", "EVENT") if isinstance(evt, dict) else getattr(evt, "event_type", "EVENT")
                self.log(f"[{etype.upper()}] {desc}")

            for sevt in spatial_events_batch:
                sdesc = sevt.get("description") if isinstance(sevt, dict) else getattr(sevt, "description", str(sevt))
                stype = sevt.get("event_type", "SPATIAL") if isinstance(sevt, dict) else getattr(sevt, "event_type", "SPATIAL")
                self.log(f"[{stype.upper()}] {sdesc}")

            self.class_listbox.delete(0, tk.END)
            for cls_name, count in sorted(latest["class_counts"].items()):
                self.class_listbox.insert(tk.END, f"{cls_name}: {count}")

        # Re-schedule next poll only if not closing
        if not self._is_closing:
            try:
                self._poll_job = self.root.after(30, self._poll_queues)
            except Exception:
                self._poll_job = None

    def _render_frame(self, cv_frame: np.ndarray) -> None:
        canvas_w = self.video_canvas.winfo_width()
        canvas_h = self.video_canvas.winfo_height()
        if canvas_w < 50 or canvas_h < 50:
            return

        h, w = cv_frame.shape[:2]
        scale = min(canvas_w / w, canvas_h / h)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))

        resized = cv2.resize(cv_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb_frame)
        self.current_photo_image = ImageTk.PhotoImage(pil_img)

        self.video_canvas.delete("all")
        self.video_canvas.create_image(
            canvas_w // 2, canvas_h // 2,
            image=self.current_photo_image,
            anchor=tk.CENTER
        )

    def on_close(self) -> None:
        if self._is_closing:
            return
        self._is_closing = True

        if self._poll_job is not None:
            try:
                self.root.after_cancel(self._poll_job)
            except Exception:
                pass
            self._poll_job = None

        if self.worker is not None and self.worker.is_alive():
            self.worker.stop()
            self.worker.join(timeout=1.0)
            if self.worker.is_alive():
                # Worker is still performing cleanup in background daemon thread.
                # Allow it to complete its finally blocks safely without destroying
                # resources it may still be writing to.
                pass

        try:
            self.root.destroy()
        except Exception:
            pass


def launch_gui() -> None:
    """Launches the Tkinter Object Detection and Tracking dashboard."""
    root = tk.Tk()
    app = ObjectTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
