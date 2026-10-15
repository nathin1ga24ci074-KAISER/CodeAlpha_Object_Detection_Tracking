"""
Desktop GUI Interface for CodeAlpha Task 4: Object Detection and Tracking.
Built with Tkinter and Pillow, offering a responsive dark-themed dashboard.
Separates UI controls from the background inference and tracking worker thread.
"""

import sys
import time
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Any, List
from collections import Counter
import cv2
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

from src.detector import YOLODetector
from src.tracker import ObjectTracker, TrackedObject
from src.video_source import VideoSource
from src.scene_intelligence import SceneIntelligence
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
        status_queue: queue.Queue
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
        self.is_running = threading.Event()
        self.is_running.set()

    def stop(self) -> None:
        """Signals the worker thread to terminate."""
        self.is_running.clear()

    def run(self) -> None:
        vs = None
        writer = None
        out_file_path = None
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
            frame_idx = 0

            while self.is_running.is_set():
                ret, frame = vs.read()
                if not ret or frame is None:
                    self.status_queue.put({"type": "status", "message": "End of video stream reached."})
                    break

                frame_idx += 1
                tracks = tracker.track(frame)
                current_fps = fps_meter.tick()

                fps_for_time = vs.fps if vs.fps > 0 else 30.0
                intel_snap = scene_intel.update(
                    frame_idx=frame_idx,
                    tracks=tracks,
                    timestamp_sec=frame_idx / fps_for_time,
                    fps=fps_for_time
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
                    "scene_summary": scene_intel.generate_natural_language_summary()
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
            try:
                scene_intel.finalize_stream(frame_idx)
                final_summary = scene_intel.generate_natural_language_summary()
            except Exception:
                pass
            self.status_queue.put({
                "type": "finished",
                "output_file": out_file_path,
                "summary": final_summary,
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

        self.video_canvas = tk.Canvas(center_frame, bg="#11111b", highlightthickness=0)
        self.video_canvas.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)

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
        cb_save.pack(anchor=tk.W, padx=pad_x, pady=(4, 12))

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
        tk.Label(parent, text="Activity Log", font=("Segoe UI", 10, "bold"), bg=self.PANEL_BG, fg=self.ACCENT_COLOR).pack(anchor=tk.W, padx=pad_x, pady=(16, 2))
        self.log_text = tk.Text(parent, height=8, bg="#11111b", fg="#a6adc8", font=("Consolas", 8), relief=tk.FLAT)
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

        # Class Distribution list
        tk.Label(parent, text="Class Counts", font=("Segoe UI", 10, "bold"), bg=self.PANEL_BG, fg=self.ACCENT_COLOR).pack(anchor=tk.W, padx=pad_x, pady=(12, 4))
        self.class_listbox = tk.Listbox(parent, bg="#11111b", fg=self.TEXT_COLOR, font=("Segoe UI", 9), height=5, relief=tk.FLAT)
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

        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.status_badge.config(text="RUNNING", bg="#2ecc71")
        self.lbl_scene_summary.config(text="Analyzing scene activity...")
        self.log(f"Starting tracking on {source}...")

        self.worker = TrackingWorker(
            source=source,
            model_name=self.model_var.get(),
            tracker_type=self.tracker_var.get(),
            conf_threshold=float(self.conf_slider.get()),
            device="cpu",
            save_output=self.save_var.get(),
            output_queue=self.output_queue,
            status_queue=self.status_queue
        )
        self.worker.start()

    def stop_tracking(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.log("Stopping tracking worker...")
            self.status_badge.config(text="STOPPING", bg="#e67e22")

    def _poll_queues(self) -> None:
        # Check status updates
        try:
            while True:
                msg = self.status_queue.get_nowait()
                mtype = msg.get("type")
                if mtype == "status":
                    self.log(msg["message"])
                elif mtype == "error":
                    self.log(f"ERROR: {msg['message']}")
                    messagebox.showerror("Tracking Error", msg["message"])
                elif mtype == "finished":
                    self.log(msg["message"])
                    if msg.get("output_file"):
                        self.log(f"Saved: {msg['output_file']}")
                    if msg.get("summary"):
                        self.lbl_scene_summary.config(text=msg["summary"])
                    self.btn_start.config(state=tk.NORMAL)
                    self.btn_stop.config(state=tk.DISABLED)
                    self.status_badge.config(text="IDLE", bg="#45475a")
        except queue.Empty:
            pass

        # Check latest frame update and accumulate events
        new_events_batch = []
        try:
            latest = None
            while True:
                item = self.output_queue.get_nowait()
                if "new_events" in item and item["new_events"]:
                    new_events_batch.extend(item["new_events"])
                latest = item
        except queue.Empty:
            pass

        if latest is not None:
            self._render_frame(latest["frame"])
            self.lbl_fps.config(text=f"{latest['fps']}")
            self.lbl_active.config(text=f"{latest['active_count']}")
            self.lbl_total.config(text=f"{latest['total_count']}")

            # Update scene intelligence summary card
            if latest.get("scene_summary"):
                self.lbl_scene_summary.config(text=latest["scene_summary"])

            # Log events
            for evt in new_events_batch:
                desc = evt.get("description") if isinstance(evt, dict) else getattr(evt, "description", str(evt))
                etype = evt.get("event_type", "EVENT") if isinstance(evt, dict) else getattr(evt, "event_type", "EVENT")
                self.log(f"[{etype.upper()}] {desc}")

            self.class_listbox.delete(0, tk.END)
            for cls_name, count in sorted(latest["class_counts"].items()):
                self.class_listbox.insert(tk.END, f"{cls_name}: {count}")

        # Re-schedule next poll
        self.root.after(30, self._poll_queues)

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
        if self.worker is not None and self.worker.is_alive():
            self.worker.stop()
            self.worker.join(timeout=1.0)
        self.root.destroy()


def launch_gui() -> None:
    """Launches the Tkinter Object Detection and Tracking dashboard."""
    root = tk.Tk()
    app = ObjectTrackerApp(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
