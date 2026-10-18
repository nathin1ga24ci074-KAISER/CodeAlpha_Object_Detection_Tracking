"""
GUI Heatmap Integration Test Suite for CodeAlpha Task 4: Object Detection & Tracking.
Verifies Phase C — Stage 3, Step 4:
- Heatmap GUI controls initial defaults and toggle behavior
- Validation of radius, opacity, and output directory inputs
- Worker thread lifecycle, start, stop, and clean teardown
- Throttled preview generation and thread-safe queue dispatching
- View mode switching between Live Stream, Heatmap Overlay, and Heatmap Density
- Safe file open callbacks without shell injection
- Export naming, collision disambiguation, and partial-write cleanup
- Session reset and stale state isolation
- Concurrent compatibility with Scene Intelligence, spatial line/ROI, and behavior metrics
"""

import sys
import os

if sys.platform.startswith("win"):
    if "TCL_LIBRARY" not in os.environ:
        _tcl_dir = os.path.join(sys.base_prefix, "tcl", "tcl8.6")
        if os.path.exists(_tcl_dir):
            os.environ["TCL_LIBRARY"] = _tcl_dir
    if "TK_LIBRARY" not in os.environ:
        _tk_dir = os.path.join(sys.base_prefix, "tcl", "tk8.6")
        if os.path.exists(_tk_dir):
            os.environ["TK_LIBRARY"] = _tk_dir

import time
import queue
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import numpy as np
import cv2
import tkinter as tk

from src.gui import TrackingWorker, ObjectTrackerApp
from src.tracker import TrackedObject
from src.heatmap_analytics import HeatmapAnalytics


class MockSyntheticVideoSource:
    """Provides a synthetic video stream for deterministic GUI worker testing."""

    def __init__(self, frames, width=100, height=80, fps=30.0):
        self._frames = list(frames)
        self.width = width
        self.height = height
        self.fps = fps
        self.is_opened = False
        self._idx = 0

    def open(self):
        self.is_opened = True
        return self

    def read(self):
        if not self.is_opened or self._idx >= len(self._frames):
            return False, None
        frame = self._frames[self._idx]
        self._idx += 1
        return True, frame

    def release(self):
        self.is_opened = False


class MockTrackerEngine:
    """Deterministic tracker stub yielding pre-defined tracks per frame."""

    def __init__(self, frame_tracks):
        self._frame_tracks = list(frame_tracks)
        self._idx = 0
        self._total_count = 0
        self._active_count = 0

    def load_model(self):
        pass

    def track(self, frame):
        if self._idx < len(self._frame_tracks):
            tracks = self._frame_tracks[self._idx]
            self._idx += 1
        else:
            tracks = []
        self._active_count = len(tracks)
        self._total_count = max(self._total_count, self._active_count)
        return tracks

    def get_total_count(self):
        return self._total_count

    def get_active_count(self):
        return self._active_count


@pytest.fixture(scope="module")
def shared_tk_root():
    root = tk.Tk()
    root.withdraw()
    yield root
    try:
        root.destroy()
    except Exception:
        pass


@pytest.fixture
def gui_app(shared_tk_root):
    """Initializes and yields ObjectTrackerApp with a hidden isolated window."""
    win = tk.Toplevel(shared_tk_root)
    win.withdraw()
    app = ObjectTrackerApp(win)
    yield app, win
    try:
        app.on_close()
    except Exception:
        try:
            win.destroy()
        except Exception:
            pass


class TestGUIHeatmapControlsAndValidation:
    """Verifies GUI controls layout, initial state, and input validation."""

    def test_heatmap_controls_default_state(self, gui_app):
        app, root = gui_app
        assert app.heatmap_var.get() is False
        assert app.heatmap_radius_var.get() == "20"
        assert app.heatmap_alpha_var.get() == "0.5"
        assert app.heatmap_dir_var.get() == ""
        assert app.lbl_heatmap_status.cget("text") == "Status: Disabled"
        assert app.lbl_heatmap_export_info.cget("text") == "Exports: Disabled"
        assert app.btn_open_heatmap_dir.cget("state") == "disabled"
        assert app.view_mode.get() == "stream"
        assert app.saved_heatmap_paths is None

    def test_heatmap_toggle_updates_status_labels(self, gui_app):
        app, root = gui_app
        app.heatmap_var.set(True)
        app._on_heatmap_toggle()
        assert app.lbl_heatmap_status.cget("text") == "Status: Ready"
        assert app.lbl_heatmap_export_info.cget("text") == "Exports: Ready"

        app.heatmap_var.set(False)
        app._on_heatmap_toggle()
        assert app.lbl_heatmap_status.cget("text") == "Status: Disabled"
        assert app.lbl_heatmap_export_info.cget("text") == "Exports: Disabled"

    def test_radius_validation_rejects_invalid_values(self, gui_app):
        app, root = gui_app
        app.heatmap_var.set(True)

        for invalid_val in ["abc", "-10", "3", "300"]:
            app.heatmap_radius_var.set(invalid_val)
            with patch("tkinter.messagebox.showerror") as mock_err, \
                 patch("src.gui.TrackingWorker") as mock_worker:
                app.start_tracking()
                mock_err.assert_called_once()
                mock_worker.assert_not_called()

    def test_opacity_validation_rejects_invalid_values(self, gui_app):
        app, root = gui_app
        app.heatmap_var.set(True)
        app.heatmap_radius_var.set("25")

        for invalid_val in ["xyz", "-0.1", "1.5", "nan"]:
            app.heatmap_alpha_var.set(invalid_val)
            with patch("tkinter.messagebox.showerror") as mock_err, \
                 patch("src.gui.TrackingWorker") as mock_worker:
                app.start_tracking()
                mock_err.assert_called_once()
                mock_worker.assert_not_called()

    def test_output_dir_validation_rejects_unwritable_path(self, gui_app):
        app, root = gui_app
        app.heatmap_var.set(True)
        app.heatmap_radius_var.set("20")
        app.heatmap_alpha_var.set("0.5")

        # Simulate non-writable directory
        with patch("pathlib.Path.mkdir", side_effect=PermissionError("Read-only volume")), \
             patch("tkinter.messagebox.showerror") as mock_err, \
             patch("src.gui.TrackingWorker") as mock_worker:
            app.heatmap_dir_var.set("Z:/read_only_dir")
            app.start_tracking()
            mock_err.assert_called_once()
            mock_worker.assert_not_called()

    def test_enabling_heatmap_passes_validated_config_to_worker(self, gui_app, tmp_path):
        app, root = gui_app
        app.heatmap_var.set(True)
        app.heatmap_radius_var.set("35")
        app.heatmap_alpha_var.set("0.7")
        app.heatmap_dir_var.set(str(tmp_path))

        with patch("src.gui.TrackingWorker") as mock_worker_cls:
            app.start_tracking()
            mock_worker_cls.assert_called_once()
            _, kwargs = mock_worker_cls.call_args
            assert kwargs["enable_heatmap"] is True
            assert kwargs["heatmap_radius"] == 35
            assert kwargs["heatmap_alpha"] == 0.7
            assert kwargs["heatmap_output_dir"] == str(tmp_path.resolve())


class TestGUIWorkerHeatmapLifecycleAndThrottling:
    """Verifies background worker heatmap accumulation, throttling, and exports."""

    def test_disabled_mode_does_not_instantiate_or_export_heatmaps(self, tmp_path):
        frames = [np.full((60, 80, 3), 100, dtype=np.uint8) for _ in range(3)]
        mock_vs = MockSyntheticVideoSource(frames, width=80, height=60)
        mock_tr = MockTrackerEngine([[] for _ in range(3)])

        out_q = queue.Queue()
        stat_q = queue.Queue()
        worker = TrackingWorker(
            source="0",
            model_name="yolo11n.pt",
            tracker_type="bytetrack.yaml",
            conf_threshold=0.25,
            device="cpu",
            save_output=False,
            output_queue=out_q,
            status_queue=stat_q,
            enable_heatmap=False,
            heatmap_output_dir=str(tmp_path)
        )

        with patch("src.gui.VideoSource", return_value=mock_vs), \
             patch("src.gui.ObjectTracker", return_value=mock_tr):
            worker.run()

        # Output payload has None previews
        assert not out_q.empty()
        item = out_q.get_nowait()
        assert item["heatmap_preview"] is None
        assert item["heatmap_overlay"] is None
        assert item["heatmap_points"] == 0

        # Status finished message has None heatmap_result
        fin_msg = None
        while not stat_q.empty():
            msg = stat_q.get_nowait()
            if msg.get("type") == "finished":
                fin_msg = msg
        assert fin_msg is not None
        assert fin_msg["heatmap_result"] is None
        assert len(list(tmp_path.glob("heatmap_gui_*"))) == 0

    def test_enabled_mode_throttled_preview_generation(self, tmp_path):
        # 20 frames: previews should only be generated at frame 1 and frame 15
        frames = [np.full((50, 50, 3), 80, dtype=np.uint8) for _ in range(20)]
        t = TrackedObject(1, [15, 15, 35, 35], 0.9, 0, "person")
        mock_vs = MockSyntheticVideoSource(frames, width=50, height=50)
        mock_tr = MockTrackerEngine([[t] for _ in range(20)])

        out_q = queue.Queue(maxsize=100)
        stat_q = queue.Queue()
        worker = TrackingWorker(
            source="0",
            model_name="yolo11n.pt",
            tracker_type="bytetrack.yaml",
            conf_threshold=0.25,
            device="cpu",
            save_output=False,
            output_queue=out_q,
            status_queue=stat_q,
            enable_heatmap=True,
            heatmap_radius=10,
            heatmap_alpha=0.5,
            heatmap_output_dir=str(tmp_path)
        )

        with patch("src.gui.VideoSource", return_value=mock_vs), \
             patch("src.gui.ObjectTracker", return_value=mock_tr):
            worker.run()

        # Collect items from output queue
        items = []
        while not out_q.empty():
            items.append(out_q.get_nowait())

        assert len(items) == 20
        # Frame 1 (first item) must have generated preview
        assert items[0]["frame_idx"] == 1
        assert items[0]["heatmap_preview"] is not None
        assert items[0]["heatmap_overlay"] is not None

        # Frame 2..14 should have None (throttled)
        for i in range(1, 14):
            assert items[i]["heatmap_preview"] is None
            assert items[i]["heatmap_overlay"] is None

        # Frame 15 (15th item, index 14) must have generated preview
        assert items[14]["frame_idx"] == 15
        assert items[14]["heatmap_preview"] is not None
        assert items[14]["heatmap_overlay"] is not None

        # Frame 16..20 should be None
        for i in range(15, 20):
            assert items[i]["heatmap_preview"] is None
            assert items[i]["heatmap_overlay"] is None

    def test_enabled_mode_full_export_and_file_generation(self, tmp_path):
        frames = [np.full((60, 60, 3), 80, dtype=np.uint8) for _ in range(3)]
        t = TrackedObject(1, [20, 20, 40, 40], 0.9, 0, "person")
        mock_vs = MockSyntheticVideoSource(frames, width=60, height=60)
        mock_tr = MockTrackerEngine([[t], [t], [t]])

        out_q = queue.Queue()
        stat_q = queue.Queue()
        worker = TrackingWorker(
            source="assets/sample_clip.mp4",
            model_name="yolo11n.pt",
            tracker_type="bytetrack.yaml",
            conf_threshold=0.25,
            device="cpu",
            save_output=False,
            output_queue=out_q,
            status_queue=stat_q,
            enable_heatmap=True,
            heatmap_radius=12,
            heatmap_alpha=0.6,
            heatmap_output_dir=str(tmp_path)
        )

        with patch("src.gui.VideoSource", return_value=mock_vs), \
             patch("src.gui.ObjectTracker", return_value=mock_tr):
            worker.run()

        fin_msg = None
        while not stat_q.empty():
            msg = stat_q.get_nowait()
            if msg.get("type") == "finished":
                fin_msg = msg

        assert fin_msg is not None
        hm_res = fin_msg["heatmap_result"]
        assert hm_res is not None
        assert hm_res["enabled"] is True

        hm_p = Path(hm_res["heatmap_image_path"])
        ov_p = Path(hm_res["overlay_image_path"])
        assert hm_p.exists()
        assert ov_p.exists()
        assert "heatmap_gui_sample_clip_" in hm_p.name
        assert "heatmap_overlay_gui_sample_clip_" in ov_p.name

        read_hm = cv2.imread(str(hm_p))
        read_ov = cv2.imread(str(ov_p))
        assert read_hm.shape == (60, 60, 3)
        assert read_ov.shape == (60, 60, 3)

    def test_worker_partial_write_failure_cleans_up_new_artifact(self, tmp_path):
        frames = [np.full((40, 40, 3), 100, dtype=np.uint8)]
        mock_vs = MockSyntheticVideoSource(frames, width=40, height=40)
        mock_tr = MockTrackerEngine([[]])

        real_imwrite = cv2.imwrite
        write_count = 0
        created_first_path = None

        def fake_imwrite(path_str, img):
            nonlocal write_count, created_first_path
            write_count += 1
            if write_count == 1:
                created_first_path = Path(path_str)
                return real_imwrite(path_str, img)
            return False

        out_q = queue.Queue()
        stat_q = queue.Queue()
        worker = TrackingWorker(
            source="0",
            model_name="yolo11n.pt",
            tracker_type="bytetrack.yaml",
            conf_threshold=0.25,
            device="cpu",
            save_output=False,
            output_queue=out_q,
            status_queue=stat_q,
            enable_heatmap=True,
            heatmap_output_dir=str(tmp_path)
        )

        with patch("src.gui.VideoSource", return_value=mock_vs), \
             patch("src.gui.ObjectTracker", return_value=mock_tr), \
             patch("cv2.imwrite", side_effect=fake_imwrite):
            worker.run()

        assert write_count == 2
        assert created_first_path is not None
        # Cleaned up on overlay failure
        assert not created_first_path.exists()

        # Check error status was delivered
        err_msg = None
        while not stat_q.empty():
            m = stat_q.get_nowait()
            if m.get("type") == "error":
                err_msg = m
        assert err_msg is not None
        assert "Heatmap export error" in err_msg["message"]


class TestGUIDispatchPreviewAndTeardown:
    """Verifies main-thread queue polling, preview switching, folder opening, and shutdown."""

    def test_stream_completion_updates_status_and_enables_folder_button(self, gui_app, tmp_path):
        app, root = gui_app
        app.heatmap_var.set(True)

        hm_p = str(tmp_path / "heatmap_gui_clip_100.png")
        ov_p = str(tmp_path / "heatmap_overlay_gui_clip_100.png")
        Path(hm_p).write_bytes(b"dummy_hm")
        Path(ov_p).write_bytes(b"dummy_ov")

        app.status_queue.put({
            "type": "finished",
            "output_file": None,
            "summary": "Tracking completed normally.",
            "behavior_summary": None,
            "heatmap_result": {
                "enabled": True,
                "heatmap_image_path": hm_p,
                "overlay_image_path": ov_p,
                "output_dir": str(tmp_path),
                "summary": {"total_points_accumulated": 42},
                "final_heatmap_img": np.zeros((40, 40, 3), dtype=np.uint8),
                "final_overlay_img": np.zeros((40, 40, 3), dtype=np.uint8)
            }
        })

        app._poll_queues()

        assert app.lbl_heatmap_status.cget("text") == "Status: Completed"
        assert app.btn_open_heatmap_dir.cget("state") == "normal"
        assert app.saved_heatmap_paths == (hm_p, ov_p, str(tmp_path))
        log_content = app.log_text.get("1.0", tk.END)
        assert "[HEATMAP] Saved colormap:" in log_content
        assert "[HEATMAP] Saved overlay:" in log_content

    def test_session_reset_clears_previous_heatmap_state(self, gui_app, tmp_path):
        app, root = gui_app
        # Setup pre-existing state from a prior run
        app.latest_stream_frame = np.zeros((30, 30, 3), dtype=np.uint8)
        app.latest_heatmap_preview = np.zeros((30, 30, 3), dtype=np.uint8)
        app.latest_heatmap_overlay = np.zeros((30, 30, 3), dtype=np.uint8)
        app.saved_heatmap_paths = ("map.png", "ov.png", str(tmp_path))
        app.btn_open_heatmap_dir.config(state=tk.NORMAL)
        app.lbl_heatmap_status.config(text="Status: Completed")

        # Disable heatmap and start new session
        app.heatmap_var.set(False)
        with patch("src.gui.TrackingWorker"):
            app.start_tracking()

        assert app.latest_stream_frame is None
        assert app.latest_heatmap_preview is None
        assert app.latest_heatmap_overlay is None
        assert app.saved_heatmap_paths is None
        assert app.btn_open_heatmap_dir.cget("state") == "disabled"
        assert app.lbl_heatmap_status.cget("text") == "Status: Disabled"
        assert app.lbl_heatmap_export_info.cget("text") == "Exports: Disabled"

    def test_view_mode_toggle_switches_canvas_display(self, gui_app):
        app, root = gui_app
        app.latest_stream_frame = np.full((40, 40, 3), 10, dtype=np.uint8)
        app.latest_heatmap_overlay = np.full((40, 40, 3), 100, dtype=np.uint8)
        app.latest_heatmap_preview = np.full((40, 40, 3), 200, dtype=np.uint8)

        rendered_frames = []

        def spy_render(frame):
            rendered_frames.append(int(frame[0, 0, 0]))

        app._render_frame = spy_render

        # 1. Stream mode
        app.view_mode.set("stream")
        app._on_view_mode_changed()
        assert rendered_frames[-1] == 10

        # 2. Overlay mode
        app.view_mode.set("heatmap_overlay")
        app._on_view_mode_changed()
        assert rendered_frames[-1] == 100

        # 3. Density mode
        app.view_mode.set("heatmap_density")
        app._on_view_mode_changed()
        assert rendered_frames[-1] == 200

    def test_open_heatmap_folder_safe_invocation(self, gui_app, tmp_path):
        app, root = gui_app
        app.saved_heatmap_paths = ("a.png", "b.png", str(tmp_path))

        with patch("os.startfile", create=True) as mock_startfile, \
             patch("subprocess.run") as mock_subproc:
            app.open_heatmap_folder()
            if hasattr(os, "startfile"):
                mock_startfile.assert_called_once_with(str(tmp_path.resolve()))
            else:
                mock_subproc.assert_called_once()
                args, kwargs = mock_subproc.call_args
                assert str(tmp_path.resolve()) in args[0]
                assert kwargs.get("check") is False

    def test_worker_stop_and_window_close_clean_shutdown(self, gui_app):
        app, root = gui_app
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        app.worker = mock_worker

        # Stop tracking triggers worker.stop()
        app.stop_tracking()
        mock_worker.stop.assert_called_once()

        # Window close joins worker and destroys window
        with patch.object(root, "destroy") as mock_destroy:
            app.on_close()
            mock_worker.join.assert_called_once_with(timeout=1.0)
            mock_destroy.assert_called_once()

    def test_compatibility_with_spatial_and_behavior_analytics(self, tmp_path):
        frames = [np.full((80, 80, 3), 100, dtype=np.uint8) for _ in range(3)]
        t = TrackedObject(1, [20, 20, 40, 40], 0.9, 0, "person")
        mock_vs = MockSyntheticVideoSource(frames, width=80, height=80)
        mock_tr = MockTrackerEngine([[t], [t], [t]])

        out_q = queue.Queue(maxsize=10)
        stat_q = queue.Queue()
        worker = TrackingWorker(
            source="0",
            model_name="yolo11n.pt",
            tracker_type="bytetrack.yaml",
            conf_threshold=0.25,
            device="cpu",
            save_output=False,
            output_queue=out_q,
            status_queue=stat_q,
            virtual_line=(10, 30, 70, 30),
            roi_region=(10, 10, 70, 70),
            enable_heatmap=True,
            heatmap_output_dir=str(tmp_path)
        )

        with patch("src.gui.VideoSource", return_value=mock_vs), \
             patch("src.gui.ObjectTracker", return_value=mock_tr):
            worker.run()

        assert not out_q.empty()
        item = out_q.get_nowait()
        # All analytics modules present in payload
        assert "spatial_snap" in item
        assert "behavior_summary" in item
        assert "heatmap_points" in item
        assert "scene_summary" in item


class TestGUIHardeningAndDefensiveMechanisms:
    """Verifies Phase C — Stage 3, Step 5 hardening: safe shutdown, defensive queues, collision guards, zero-frame handling."""

    def test_on_close_prevents_subsequent_polling_and_cancels_jobs(self, gui_app):
        app, root = gui_app
        assert app._is_closing is False

        # Close the app
        app.on_close()
        assert app._is_closing is True
        assert app._poll_job is None

        # Subsequent calls to _poll_queues return immediately without crashing or scheduling
        app.status_queue.put({"type": "status", "message": "Late message"})
        app._poll_queues()
        assert app._poll_job is None

    def test_on_close_while_worker_running_signals_stop_without_freezing(self, gui_app):
        app, root = gui_app
        mock_worker = MagicMock()
        mock_worker.is_alive.return_value = True
        app.worker = mock_worker

        with patch.object(root, "destroy") as mock_destroy:
            app.on_close()
            mock_worker.stop.assert_called_once()
            mock_worker.join.assert_called_once_with(timeout=1.0)
            mock_destroy.assert_called_once()

    def test_worker_startup_exception_releases_resources_and_reports_error(self):
        mock_vs = MagicMock()
        mock_vs.open.return_value = mock_vs
        mock_vs.release = MagicMock()

        out_q = queue.Queue()
        stat_q = queue.Queue()
        worker = TrackingWorker(
            source="0",
            model_name="nonexistent_model.pt",
            tracker_type="bytetrack.yaml",
            conf_threshold=0.25,
            device="cpu",
            save_output=False,
            output_queue=out_q,
            status_queue=stat_q
        )

        with patch("src.gui.VideoSource", return_value=mock_vs), \
             patch("src.gui.ObjectTracker.load_model", side_effect=RuntimeError("Model weights corrupted")):
            worker.run()

        # Capture device was released in finally
        mock_vs.release.assert_called_once()

        # Error status was queued
        err_msg = None
        while not stat_q.empty():
            m = stat_q.get_nowait()
            if m.get("type") == "error":
                err_msg = m
        assert err_msg is not None
        assert "Model weights corrupted" in err_msg["message"]

    def test_defensive_queue_polling_handles_malformed_and_missing_keys(self, gui_app):
        app, root = gui_app

        # 1. Non-dict message in status queue
        app.status_queue.put(None)
        app.status_queue.put("unexpected string")
        app.status_queue.put(12345)

        # 2. Status message missing "message" key
        app.status_queue.put({"type": "status"})

        # 3. Error message missing "message" key
        app.status_queue.put({"type": "error"})

        # 4. Finished message with non-dict behavior_summary and heatmap_result
        app.status_queue.put({
            "type": "finished",
            "behavior_summary": "malformed_string_summary",
            "heatmap_result": "malformed_heatmap_result"
        })

        # 5. Non-dict items in output_queue
        app.output_queue.put("not a dict frame")
        app.output_queue.put(None)

        # Should poll through all bad messages without raising KeyError or AttributeError
        with patch("tkinter.messagebox.showerror") as mock_err:
            app._poll_queues()
            # Error dialog handled default error text safely
            mock_err.assert_called_once_with("Tracking Error", "An unexpected tracking error occurred.")

    def test_export_path_collision_with_source_file_rejected(self, tmp_path):
        src_file = tmp_path / "my_source_video.mp4"
        src_file.write_bytes(b"dummy_video_data")

        frames = [np.full((40, 40, 3), 100, dtype=np.uint8)]
        mock_vs = MockSyntheticVideoSource(frames, width=40, height=40)
        mock_tr = MockTrackerEngine([[]])

        out_q = queue.Queue()
        stat_q = queue.Queue()
        worker = TrackingWorker(
            source=str(src_file),
            model_name="yolo11n.pt",
            tracker_type="bytetrack.yaml",
            conf_threshold=0.25,
            device="cpu",
            save_output=False,
            output_queue=out_q,
            status_queue=stat_q,
            enable_heatmap=True,
            heatmap_output_dir=str(tmp_path)
        )

        # Force path collision with source file
        with patch("src.gui.VideoSource", return_value=mock_vs), \
             patch("src.gui.ObjectTracker", return_value=mock_tr), \
             patch("pathlib.Path.resolve", return_value=src_file.resolve()):
            worker.run()

        # Check that error was caught and reported
        err_msg = None
        while not stat_q.empty():
            m = stat_q.get_nowait()
            if m.get("type") == "error" and "collides with source video" in m.get("message", ""):
                err_msg = m
        assert err_msg is not None

        # Source file remained intact and was not overwritten
        assert src_file.read_bytes() == b"dummy_video_data"

    def test_export_path_collision_with_output_video_rejected(self, tmp_path):
        out_video = tmp_path / "gui_tracked_sample.mp4"
        out_video.write_bytes(b"dummy_output_data")

        frames = [np.full((40, 40, 3), 100, dtype=np.uint8)]
        mock_vs = MockSyntheticVideoSource(frames, width=40, height=40)
        mock_tr = MockTrackerEngine([[]])

        out_q = queue.Queue()
        stat_q = queue.Queue()
        worker = TrackingWorker(
            source="assets/sample_clip.mp4",
            model_name="yolo11n.pt",
            tracker_type="bytetrack.yaml",
            conf_threshold=0.25,
            device="cpu",
            save_output=True,
            output_queue=out_q,
            status_queue=stat_q,
            enable_heatmap=True,
            heatmap_output_dir=str(tmp_path)
        )

        with patch("src.gui.VideoSource", return_value=mock_vs), \
             patch("src.gui.ObjectTracker", return_value=mock_tr), \
             patch("cv2.VideoWriter"), \
             patch("pathlib.Path.resolve", return_value=out_video.resolve()):
            worker.run()

        # Check that error was caught and reported
        err_msg = None
        while not stat_q.empty():
            m = stat_q.get_nowait()
            if m.get("type") == "error" and "collides with output video" in m.get("message", ""):
                err_msg = m
        assert err_msg is not None

    def test_zero_frame_source_releases_resources_and_avoids_misleading_export(self, gui_app, tmp_path):
        app, root = gui_app

        # Video source returns False, None on the very first read
        mock_vs = MockSyntheticVideoSource([], width=60, height=60)
        mock_tr = MockTrackerEngine([])

        out_q = queue.Queue()
        stat_q = queue.Queue()
        worker = TrackingWorker(
            source="assets/empty_clip.mp4",
            model_name="yolo11n.pt",
            tracker_type="bytetrack.yaml",
            conf_threshold=0.25,
            device="cpu",
            save_output=True,
            output_queue=out_q,
            status_queue=stat_q,
            enable_heatmap=True,
            heatmap_output_dir=str(tmp_path)
        )

        with patch("src.gui.VideoSource", return_value=mock_vs), \
             patch("src.gui.ObjectTracker", return_value=mock_tr), \
             patch("cv2.VideoWriter") as mock_writer_cls:
            mock_writer = MagicMock()
            mock_writer_cls.return_value = mock_writer
            worker.run()

        # 1. Verify capture and writer resources were released
        assert mock_vs.is_opened is False
        mock_writer.release.assert_called_once()

        # 2. Verify no misleading heatmap files were created in tmp_path
        exported_pngs = list(tmp_path.glob("*.png"))
        assert len(exported_pngs) == 0

        # 3. Verify finished message was produced with no usable frames summary
        fin_msg = None
        while not stat_q.empty():
            m = stat_q.get_nowait()
            if m.get("type") == "finished":
                fin_msg = m

        assert fin_msg is not None
        hm_res = fin_msg["heatmap_result"]
        assert hm_res is not None
        assert hm_res["enabled"] is True
        assert hm_res["heatmap_image_path"] is None
        assert hm_res["overlay_image_path"] is None
        assert hm_res["summary"]["status"] == "No usable frames were captured"
        assert hm_res["summary"]["frames_processed"] == 0

        # 4. Verify GUI displays "No frames" status
        app.heatmap_var.set(True)
        app.status_queue.put(fin_msg)
        app._poll_queues()

        assert app.lbl_heatmap_status.cget("text") == "Status: No frames"
        assert app.lbl_heatmap_export_info.cget("text") == "No frames"
        assert app.btn_open_heatmap_dir.cget("state") == "disabled"

