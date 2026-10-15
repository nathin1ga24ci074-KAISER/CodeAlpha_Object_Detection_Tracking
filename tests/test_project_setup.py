"""
Environment and initial architecture setup verification tests.
Ensures modules are correctly structured and reusable components function as expected.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.detector import DetectionResult, YOLODetector
from src.tracker import TrackedObject, ObjectTracker
from src.video_source import VideoSource
from src.utils import (
    ColorPalette,
    FPSMeter,
    draw_bounding_box,
    draw_trajectory,
    draw_hud,
    create_synthetic_frame
)
import app


class TestEnvironmentAndPackages:
    """Verifies Python runtime environment and core dependency versions."""

    def test_python_version_is_compatible(self):
        assert sys.version_info >= (3, 10), "Requires Python 3.10+"

    def test_core_dependencies_importable(self):
        import cv2
        import ultralytics
        import torch
        import torchvision
        import PIL
        import matplotlib
        import yaml
        import psutil
        import tqdm

        assert hasattr(cv2, "__version__")
        assert hasattr(ultralytics, "__version__")
        assert hasattr(torch, "__version__")
        assert hasattr(torchvision, "__version__")


class TestArchitectureModules:
    """Tests the skeleton architecture modules and their data models."""

    def test_detection_result_model(self):
        det = DetectionResult(
            box=[10.0, 20.0, 100.0, 150.0],
            confidence=0.89512,
            class_id=0,
            class_name="person"
        )
        assert det.class_name == "person"
        assert det.class_id == 0
        assert det.confidence == pytest.approx(0.8951, rel=1e-3)
        assert det.width == 90.0
        assert det.height == 130.0
        assert det.area == 90.0 * 130.0
        assert det.center == [55.0, 85.0]
        dict_data = det.to_dict()
        assert dict_data["class_name"] == "person"
        assert dict_data["confidence"] == 0.8951
        assert dict_data["area"] == 11700.0

    def test_yolo_detector_lazy_initialization(self):
        detector = YOLODetector(model_name="yolo11n.pt", confidence_threshold=0.4)
        assert detector.model_name == "yolo11n.pt"
        assert detector.confidence_threshold == 0.4
        assert not detector.is_loaded

        # Should raise RuntimeError if detect() is called before load_model()
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with pytest.raises(RuntimeError, match="Model is not loaded"):
            detector.detect(dummy_frame)

    def test_yolo_detector_invalid_frame_validation(self):
        detector = YOLODetector()
        detector._model = "mock_model"  # mock loaded
        with pytest.raises(ValueError, match="Invalid frame"):
            detector.detect(None)
        with pytest.raises(ValueError, match="Invalid frame"):
            detector.detect(np.array([]))

    def test_tracked_object_centroid_and_trajectory(self):
        tracked = TrackedObject(
            track_id=1,
            box=[10.0, 20.0, 50.0, 60.0],
            confidence=0.92,
            class_id=2,
            class_name="car"
        )
        # Centroid: ((10+50)/2, (20+60)/2) = (30, 40)
        assert tracked.current_centroid == (30, 40)
        assert len(tracked.trajectory) == 1

        # Move box
        tracked.box = [20.0, 30.0, 60.0, 70.0]
        tracked.update_centroid()
        assert tracked.current_centroid == (40, 50)
        assert len(tracked.trajectory) == 2

    def test_object_tracker_configuration(self):
        tracker = ObjectTracker(tracker_type="bytetrack.yaml")
        assert tracker.tracker_type == "bytetrack.yaml"
        assert not tracker.is_loaded
        assert tracker.get_total_count() == 0
        assert tracker.get_active_count() == 0

        # Unsupported tracker should raise ValueError
        with pytest.raises(ValueError, match="Unsupported tracker"):
            ObjectTracker(tracker_type="invalid_tracker.yaml")

    def test_video_source_input_resolution(self):
        # Webcam index as int
        vs_cam = VideoSource(source=0)
        assert vs_cam.is_webcam is True
        assert vs_cam.source_id == 0

        # Webcam index as string "1"
        vs_cam_str = VideoSource(source="1")
        assert vs_cam_str.is_webcam is True
        assert vs_cam_str.source_id == 1

        # Non-existent file should raise FileNotFoundError
        with pytest.raises(FileNotFoundError):
            VideoSource(source="non_existent_video_file_xyz.mp4")

    def test_color_palette_consistency(self):
        c1 = ColorPalette.get_color(1)
        c2 = ColorPalette.get_color(1)
        assert c1 == c2
        assert len(c1) == 3

    def test_fps_meter(self):
        fps_meter = FPSMeter(moving_avg_window=5)
        val = fps_meter.tick()
        assert isinstance(val, float)

    def test_drawing_and_hud_pipeline_on_synthetic_frame(self):
        frame = create_synthetic_frame(width=320, height=240)
        assert frame.shape == (240, 320, 3)

        # Draw box
        frame = draw_bounding_box(frame, [20, 20, 80, 80], "person 0.95")
        # Draw trajectory
        frame = draw_trajectory(frame, [(30, 30), (40, 40), (50, 50)], (0, 255, 0))
        # Draw HUD
        frame = draw_hud(frame, fps=30.0, active_count=1, total_count=1, class_counts={"person": 1})
        assert frame is not None
        assert frame.shape == (240, 320, 3)

    def test_app_entrypoint_exists(self):
        assert hasattr(app, "main")
        assert hasattr(app, "parse_args")
        assert hasattr(app, "run_image_detection")

    def test_app_file_type_helpers(self):
        assert app.is_image_file("frame.jpg") is True
        assert app.is_image_file("frame.PNG") is True
        assert app.is_image_file("video.mp4") is False
        assert app.is_video_file("clip.mp4") is True
        assert app.is_video_file("clip.AVI") is True
        assert app.is_video_file("photo.jpg") is False

    def test_run_image_detection_missing_file(self):
        with pytest.raises(FileNotFoundError, match="Input image not found"):
            app.run_image_detection("non_existent_image_12345.jpg")


class TestTrackingArchitecture:
    """Verifies ByteTrack tracking configuration, mocked association, empty frames, and stream isolation."""

    def test_tracker_configuration_and_init(self):
        tracker = ObjectTracker(
            tracker_type="bytetrack.yaml",
            model_name="yolo11n.pt",
            confidence_threshold=0.35,
            device="cpu"
        )
        assert tracker.tracker_type == "bytetrack.yaml"
        assert tracker.model_name == "yolo11n.pt"
        assert tracker.confidence_threshold == 0.35
        assert tracker.device == "cpu"
        assert not tracker.is_loaded
        assert tracker.get_total_count() == 0
        assert tracker.get_active_count() == 0

    def test_tracked_object_full_lifecycle(self):
        obj = TrackedObject(
            track_id=10,
            box=[100.0, 150.0, 300.0, 450.0],
            confidence=0.88,
            class_id=2,
            class_name="car"
        )
        assert obj.track_id == 10
        assert obj.class_name == "car"
        assert obj.width == 200.0
        assert obj.height == 300.0
        assert obj.area == 60000.0
        assert obj.center == [200.0, 300.0]
        assert obj.current_centroid == (200, 300)
        assert len(obj.trajectory) == 1

        # Frame 2: object moves
        obj.update([110.0, 160.0, 310.0, 460.0], 0.92, 2, "car")
        assert obj.current_centroid == (210, 310)
        assert len(obj.trajectory) == 2
        data = obj.to_dict()
        assert data["track_id"] == 10
        assert data["trajectory_points"] == 2

    def test_tracker_empty_detection_frame_handling(self):
        tracker = ObjectTracker(tracker_type="bytetrack.yaml")

        class MockEmptyResults:
            boxes = None

        class MockModel:
            names = {0: "person"}

            def track(self, **kwargs):
                return [MockEmptyResults()]

        tracker._model = MockModel()
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        tracks = tracker.track(frame)
        assert tracks == []
        assert tracker.get_active_count() == 0

    def test_tracker_stream_isolation_and_reset(self):
        tracker_a = ObjectTracker(tracker_type="bytetrack.yaml")
        tracker_b = ObjectTracker(tracker_type="bytetrack.yaml")

        obj_a = TrackedObject(1, [10, 10, 50, 50], 0.9, 0, "person")
        tracker_a._active_tracks[1] = obj_a
        tracker_a._total_unique_ids.add(1)
        tracker_a._current_frame_tracks = [obj_a]

        assert tracker_a.get_active_count() == 1
        assert tracker_a.get_total_count() == 1
        # Tracker B must remain completely isolated
        assert tracker_b.get_active_count() == 0
        assert tracker_b.get_total_count() == 0

        # Reset tracker A
        tracker_a.reset()
        assert tracker_a.get_active_count() == 0
        assert tracker_a.get_total_count() == 0
        assert len(tracker_a.get_all_tracked_objects()) == 0

    def test_run_video_tracking_missing_file_error(self):
        with pytest.raises(FileNotFoundError, match="Video file not found"):
            app.run_video_tracking("non_existent_clip_xyz.mp4")

    def test_real_tracker_on_sample_clip(self):
        sample_p = Path(__file__).resolve().parent.parent / "assets" / "sample_clip.mp4"
        if not sample_p.exists():
            pytest.skip("sample_clip.mp4 not present in assets")

        res = app.run_video_tracking(
            video_path=str(sample_p),
            model_name="yolo11n.pt",
            tracker_type="bytetrack.yaml",
            conf_threshold=0.25,
            device="cpu",
            save_path=None,
            max_frames=5
        )
        assert res["success"] is True
        assert res["frames_processed"] == 5
        assert res["total_unique_tracks"] >= 1


class TestStage5WebcamAndGUI:
    """Verifies Stage 5 webcam CLI error handling and GUI dashboard architecture."""

    def test_run_webcam_tracking_invalid_device(self):
        with pytest.raises(IOError, match="Could not open webcam"):
            app.run_webcam_tracking(webcam_id=9999, max_frames=1, headless=True)

    def test_gui_worker_lifecycle(self):
        import queue
        from src.gui import TrackingWorker

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
            status_queue=stat_q
        )
        assert worker.is_running.is_set()
        worker.stop()
        assert not worker.is_running.is_set()

    def test_gui_application_init_and_teardown(self):
        import tkinter as tk
        from src.gui import ObjectTrackerApp

        root = tk.Tk()
        root.withdraw()
        gui = ObjectTrackerApp(root)
        assert gui.status_badge.cget("text") == "IDLE"
        assert gui.model_var.get() == "yolo11n.pt"
        assert gui.tracker_var.get() == "bytetrack.yaml"
        root.update()
        gui.on_close()
