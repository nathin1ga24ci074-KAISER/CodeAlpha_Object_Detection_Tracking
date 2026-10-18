"""
Unit and Integration Tests for Standalone Heatmap Engine (Phase C, Stage 3).
Validates 2D spatial density accumulation, kernel splatting, parameter validation,
numerical safety (NaN/Inf guards), normalization, non-destructive overlay blending,
and session reset behavior.
"""

import sys
from pathlib import Path
from typing import Sequence, List, Dict, Any, Optional, Union, Tuple
import math
from unittest.mock import patch, MagicMock
import cv2
import numpy as np
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.heatmap_analytics import (
    HeatmapAnalytics,
    extract_centroid,
    generate_gaussian_kernel
)
from src.tracker import TrackedObject
import app


class MockTrack:
    """Mock object compatible with TrackedObject centroid interface."""
    def __init__(self, track_id: int, trajectory: list):
        self.track_id = int(track_id)
        self.trajectory = list(trajectory)
        self.class_name = "person"

    @property
    def current_centroid(self):
        return self.trajectory[-1] if self.trajectory else (0, 0)


class TestHeatmapParameterValidation:
    """Validates constructor dimension bounds and parameter guards."""

    def test_valid_initialization(self):
        engine = HeatmapAnalytics(frame_width=640, frame_height=480, radius_px=15)
        assert engine.frame_width == 640
        assert engine.frame_height == 480
        assert engine.radius_px == 15
        assert engine.is_empty is True
        assert engine.max_density == 0.0

    def test_invalid_dimensions(self):
        # Non-positive width
        with pytest.raises(ValueError, match="frame_width must be a positive integer"):
            HeatmapAnalytics(frame_width=0, frame_height=480)
        with pytest.raises(ValueError, match="frame_width must be a positive integer"):
            HeatmapAnalytics(frame_width=-100, frame_height=480)

        # Non-positive height
        with pytest.raises(ValueError, match="frame_height must be a positive integer"):
            HeatmapAnalytics(frame_width=640, frame_height=0)
        with pytest.raises(ValueError, match="frame_height must be a positive integer"):
            HeatmapAnalytics(frame_width=640, frame_height=-50)

    def test_invalid_radius(self):
        with pytest.raises(ValueError, match="radius_px must be a positive integer"):
            HeatmapAnalytics(frame_width=100, frame_height=100, radius_px=0)
        with pytest.raises(ValueError, match="radius_px must be a positive integer"):
            HeatmapAnalytics(frame_width=100, frame_height=100, radius_px=-10)

    def test_invalid_parameters(self):
        # Non-positive intensity
        with pytest.raises(ValueError, match="intensity_per_point must be positive"):
            HeatmapAnalytics(100, 100, intensity_per_point=0.0)
        with pytest.raises(ValueError, match="intensity_per_point must be positive"):
            HeatmapAnalytics(100, 100, intensity_per_point=-1.0)

        # Decay factor out of (0.0, 1.0]
        with pytest.raises(ValueError, match="decay_factor must be in"):
            HeatmapAnalytics(100, 100, decay_factor=0.0)
        with pytest.raises(ValueError, match="decay_factor must be in"):
            HeatmapAnalytics(100, 100, decay_factor=1.2)

        # Overlay alpha out of [0.0, 1.0]
        with pytest.raises(ValueError, match="overlay_alpha must be in"):
            HeatmapAnalytics(100, 100, overlay_alpha=-0.1)
        with pytest.raises(ValueError, match="overlay_alpha must be in"):
            HeatmapAnalytics(100, 100, overlay_alpha=1.5)

        # Threshold out of [0.0, 1.0]
        with pytest.raises(ValueError, match="min_density_threshold must be in"):
            HeatmapAnalytics(100, 100, min_density_threshold=-0.5)
        with pytest.raises(ValueError, match="min_density_threshold must be in"):
            HeatmapAnalytics(100, 100, min_density_threshold=2.0)


class TestHeatmapCentroidExtraction:
    """Validates coordinate extraction and malformed input handling."""

    def test_extract_from_tuple_and_list(self):
        assert extract_centroid((100, 200)) == (100.0, 200.0)
        assert extract_centroid([50.5, 75.2]) == (50.5, 75.2)

    def test_extract_from_tracked_object(self):
        t = TrackedObject(track_id=1, box=[10, 20, 50, 60], confidence=0.9, class_id=0, class_name="person")
        assert extract_centroid(t) == (30.0, 40.0)

    def test_extract_from_mock_track(self):
        m = MockTrack(2, [(10, 10), (50, 70)])
        assert extract_centroid(m) == (50.0, 70.0)

    def test_malformed_and_non_finite_inputs(self):
        assert extract_centroid(None) is None
        assert extract_centroid("invalid_string") is None
        assert extract_centroid([]) is None
        assert extract_centroid([10]) is None
        assert extract_centroid([float("nan"), 100.0]) is None
        assert extract_centroid([100.0, float("inf")]) is None
        assert extract_centroid([-float("inf"), 50.0]) is None
        assert extract_centroid(("not_a_number", 50)) is None


class TestHeatmapDensityAccumulation:
    """Validates density map accumulation, repeated observations, and boundaries."""

    def test_initial_empty_state(self):
        engine = HeatmapAnalytics(frame_width=200, frame_height=150)
        dmap = engine.get_density_map()
        assert dmap.shape == (150, 200)
        assert dmap.dtype == np.float32
        assert np.all(dmap == 0.0)
        assert engine.is_empty is True
        assert engine.max_density == 0.0

        summary = engine.get_summary()
        assert summary["total_points_accumulated"] == 0
        assert summary["frames_processed"] == 0
        assert summary["is_empty"] is True

    def test_single_centroid_accumulation(self):
        engine = HeatmapAnalytics(frame_width=100, frame_height=100, radius_px=10, intensity_per_point=1.0)
        success = engine.accumulate_point(50, 50)
        assert success is True
        assert engine.is_empty is False
        assert engine.max_density == pytest.approx(1.0, rel=1e-3)

        dmap = engine.get_density_map()
        # Peak at center (50, 50)
        assert dmap[50, 50] == pytest.approx(1.0, rel=1e-3)
        # Nearby point has positive Gaussian falloff
        assert dmap[52, 50] > 0.0
        assert dmap[52, 50] < 1.0
        # Point outside radius is zero
        assert dmap[70, 70] == 0.0

    def test_repeated_observations_increase_density(self):
        engine = HeatmapAnalytics(frame_width=100, frame_height=100, radius_px=10, intensity_per_point=2.0)
        for _ in range(3):
            engine.accumulate_point(40, 40)

        assert engine.max_density == pytest.approx(6.0, rel=1e-3)
        dmap = engine.get_density_map()
        assert dmap[40, 40] == pytest.approx(6.0, rel=1e-3)

    def test_multiple_centroids_and_accumulate_points(self):
        engine = HeatmapAnalytics(frame_width=200, frame_height=200, radius_px=10)
        pts = [(30, 30), (100, 100), (170, 170)]
        count = engine.accumulate_points(pts)
        assert count == 3
        assert engine.get_summary()["total_points_accumulated"] == 3

        dmap = engine.get_density_map()
        assert dmap[30, 30] > 0.9
        assert dmap[100, 100] > 0.9
        assert dmap[170, 170] > 0.9

    def test_partially_out_of_bounds_centroid_splatting(self):
        engine = HeatmapAnalytics(frame_width=100, frame_height=100, radius_px=15)
        # Centroid at (5, 5): footprint extends into negative coordinates [-10..20]
        # Must not crash and should accumulate the visible slice
        success = engine.accumulate_point(5, 5)
        assert success is True
        dmap = engine.get_density_map()
        assert dmap[5, 5] > 0.9
        assert dmap[0, 0] > 0.0

    def test_completely_out_of_bounds_points_ignored(self):
        engine = HeatmapAnalytics(frame_width=100, frame_height=100, radius_px=10)
        # Far out of bounds
        assert engine.accumulate_point(-100, -100) is False
        assert engine.accumulate_point(500, 500) is False
        assert engine.accumulate_point(50, 1000) is False
        assert engine.is_empty is True
        assert engine.get_summary()["total_points_accumulated"] == 0

    def test_safe_copy_preserves_internal_density(self):
        engine = HeatmapAnalytics(frame_width=50, frame_height=50)
        engine.accumulate_point(25, 25)

        copy_map = engine.get_density_map(copy=True)
        # Mutate the copy
        copy_map[25, 25] = 9999.0
        copy_map[0, 0] = 8888.0

        # Internal map must remain pristine
        internal_map = engine.get_density_map()
        assert internal_map[25, 25] != 9999.0
        assert internal_map[0, 0] == 0.0

    def test_reset_behavior(self):
        engine = HeatmapAnalytics(frame_width=100, frame_height=100)
        engine.accumulate_point(50, 50)
        engine.update([MockTrack(1, [(30, 30)])])
        assert engine.is_empty is False
        assert engine.get_summary()["total_points_accumulated"] >= 2

        engine.reset()
        assert engine.is_empty is True
        assert engine.max_density == 0.0
        assert engine.get_summary()["total_points_accumulated"] == 0
        assert engine.get_summary()["frames_processed"] == 0
        assert np.all(engine.get_density_map() == 0.0)

    def test_decay_factor_behavior(self):
        engine = HeatmapAnalytics(frame_width=100, frame_height=100, radius_px=10, decay_factor=0.5)
        engine.accumulate_point(50, 50)
        initial_peak = engine.max_density
        assert initial_peak == pytest.approx(1.0, rel=1e-3)

        # Update frame with no tracks -> density decays by 0.5
        res = engine.update([])
        assert res["points_accumulated"] == 0
        assert engine.max_density == pytest.approx(0.5, rel=1e-3)


class TestHeatmapVisualizationAndOverlay:
    """Validates colormapping, normalization, and non-destructive frame blending."""

    def test_generate_heatmap_image_empty_map(self):
        engine = HeatmapAnalytics(frame_width=120, frame_height=90)
        # Empty map should return all black without division by zero
        img = engine.generate_heatmap_image()
        assert isinstance(img, np.ndarray)
        assert img.shape == (90, 120, 3)
        assert img.dtype == np.uint8
        assert np.all(img == 0)

    def test_generate_heatmap_image_populated_map(self):
        engine = HeatmapAnalytics(frame_width=100, frame_height=100, radius_px=15)
        engine.accumulate_point(50, 50)

        img = engine.generate_heatmap_image()
        assert img.shape == (100, 100, 3)
        assert img.dtype == np.uint8
        # Should not be all black
        assert np.any(img > 0)

    def test_apply_overlay_preserves_original_frame(self):
        engine = HeatmapAnalytics(frame_width=100, frame_height=100, radius_px=15)
        engine.accumulate_point(50, 50)

        # Create clean synthetic test frame (gray: 128)
        frame = np.full((100, 100, 3), 128, dtype=np.uint8)
        frame_backup = frame.copy()

        overlaid = engine.apply_overlay(frame, alpha=0.5)

        # 1. Returned array is a new object
        assert overlaid is not frame

        # 2. Original input frame was NOT modified in-place
        assert np.array_equal(frame, frame_backup)

        # 3. Output has correct shape and dtype
        assert overlaid.shape == (100, 100, 3)
        assert overlaid.dtype == np.uint8

        # 4. Center region was blended
        assert not np.array_equal(overlaid[50, 50], frame[50, 50])

    def test_apply_overlay_on_empty_heatmap(self):
        engine = HeatmapAnalytics(frame_width=80, frame_height=60)
        frame = np.full((60, 80, 3), 200, dtype=np.uint8)

        # On empty map, overlay returns an exact copy of the frame
        out = engine.apply_overlay(frame)
        assert np.array_equal(out, frame)
        assert out is not frame

    def test_apply_overlay_thresholding_leaves_unvisited_pixels_clean(self):
        engine = HeatmapAnalytics(
            frame_width=100, frame_height=100,
            radius_px=10, min_density_threshold=0.10
        )
        engine.accumulate_point(20, 20)

        frame = np.full((100, 100, 3), 50, dtype=np.uint8)
        overlaid = engine.apply_overlay(frame, alpha=0.6)

        # Center of accumulation is blended
        assert not np.array_equal(overlaid[20, 20], [50, 50, 50])

        # Far unvisited pixel (80, 80) is untouched because density == 0.0 < threshold
        assert np.array_equal(overlaid[80, 80], [50, 50, 50])

    def test_apply_overlay_dimension_validation(self):
        engine = HeatmapAnalytics(frame_width=100, frame_height=80)

        # Wrong dimensions (80, 100) instead of (100, 80)
        wrong_size = np.zeros((100, 80, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="Frame dimensions"):
            engine.apply_overlay(wrong_size)

        # Non-3-channel frame
        single_channel = np.zeros((80, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="3-channel BGR image"):
            engine.apply_overlay(single_channel)

        # Non-ndarray
        with pytest.raises(ValueError, match="valid numpy ndarray"):
            engine.apply_overlay("not_an_array")  # type: ignore


class TestHeatmapMultiFrameStreamIntegration:
    """Validates multi-frame update cycles with mock and real TrackedObjects."""

    def test_sequential_frame_updates(self):
        engine = HeatmapAnalytics(frame_width=150, frame_height=150, radius_px=12)

        # Frame 1: Person at (30, 30)
        t1 = MockTrack(1, [(30, 30)])
        res1 = engine.update([t1], frame_idx=1, timestamp_sec=0.1)
        assert res1["points_accumulated"] == 1
        assert res1["total_points_accumulated"] == 1
        assert res1["frame_idx"] == 1

        # Frame 2: Person moves to (45, 30), Car at (100, 100)
        t1.trajectory.append((45, 30))
        t2 = MockTrack(2, [(100, 100)])
        res2 = engine.update([t1, t2], frame_idx=2, timestamp_sec=0.2)
        assert res2["points_accumulated"] == 2
        assert res2["total_points_accumulated"] == 3
        assert res2["frame_idx"] == 2

        summary = engine.get_summary()
        assert summary["frames_processed"] == 2
        assert summary["total_points_accumulated"] == 3
        assert summary["is_empty"] is False


class MockSyntheticVideoSource:
    """Synthetic VideoSource mock for fast, deterministic video pipeline tests."""

    def __init__(
        self,
        frames: Sequence[np.ndarray],
        width: int = 120,
        height: int = 100,
        fps: float = 30.0
    ):
        self.frames = list(frames)
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self._idx = 0
        self.is_opened = False

    def open(self):
        self.is_opened = True
        return self

    def read(self):
        if self._idx < len(self.frames):
            f = self.frames[self._idx]
            self._idx += 1
            return True, f
        return False, None

    def release(self):
        self.is_opened = False


class MockTrackerEngine:
    """Mock ObjectTracker providing controlled TrackedObject outputs per frame."""

    def __init__(self, detections_per_frame: Sequence[Sequence[TrackedObject]]):
        self.detections_per_frame = list(detections_per_frame)
        self._frame_idx = 0

    def load_model(self):
        pass

    def track(self, frame):
        if self._frame_idx < len(self.detections_per_frame):
            res = list(self.detections_per_frame[self._frame_idx])
            self._frame_idx += 1
            return res
        return []

    def get_active_count(self):
        return 0

    def get_total_count(self):
        return 1


class TestHeatmapVideoIntegration:
    """
    Validates Phase C, Stage 3 Step 2: Prerecorded video pipeline integration,
    CLI argument propagation, output image creation, error handling, session isolation,
    and concurrent analytics compatibility using synthetic frames.
    """

    def test_video_tracking_heatmap_disabled_backward_compatibility(self, tmp_path):
        dummy_video = tmp_path / "dummy_clip.mp4"
        dummy_video.write_bytes(b"dummy_content")

        frames = [np.full((100, 120, 3), 50, dtype=np.uint8) for _ in range(3)]
        t1 = TrackedObject(track_id=1, box=[30, 30, 50, 50], confidence=0.9, class_id=0, class_name="person")
        mock_vs = MockSyntheticVideoSource(frames, width=120, height=100)
        mock_tr = MockTrackerEngine([[t1], [t1], [t1]])

        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr):
            res = app.run_video_tracking(str(dummy_video), enable_heatmap=False)

        assert res["success"] is True
        assert res["frames_processed"] == 3
        assert res["heatmap"] is None
        # Verify no heatmap image files were written
        assert len(list(tmp_path.glob("heatmap_*.png"))) == 0

    def test_video_tracking_heatmap_enabled_generates_images_and_density(self, tmp_path):
        dummy_video = tmp_path / "sample_test.mp4"
        dummy_video.write_bytes(b"dummy_content")
        out_dir = tmp_path / "custom_heatmaps"

        frames = [np.full((100, 120, 3), 100, dtype=np.uint8) for _ in range(3)]
        # Track centroid at (40, 50)
        t = TrackedObject(track_id=1, box=[30, 40, 50, 60], confidence=0.95, class_id=0, class_name="person")
        mock_vs = MockSyntheticVideoSource(frames, width=120, height=100)
        mock_tr = MockTrackerEngine([[t], [t], [t]])

        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr):
            res = app.run_video_tracking(
                str(dummy_video),
                enable_heatmap=True,
                heatmap_output_dir=out_dir,
                heatmap_radius=10,
                heatmap_alpha=0.6
            )

        assert res["success"] is True
        assert res["frames_processed"] == 3
        assert res["heatmap"] is not None
        hm = res["heatmap"]
        assert hm["enabled"] is True
        assert hm["summary"]["frame_width"] == 120
        assert hm["summary"]["frame_height"] == 100
        assert hm["summary"]["radius_px"] == 10
        assert hm["summary"]["total_points_accumulated"] == 3
        assert hm["summary"]["is_empty"] is False

        # Verify density map peak location (cy=50, cx=40)
        dmap = hm["density_map"]
        assert dmap.shape == (100, 120)
        assert dmap.dtype == np.float32
        assert dmap[50, 40] == pytest.approx(3.0, rel=1e-3)

        # Verify written files
        hm_img_path = Path(hm["heatmap_image_path"])
        ov_img_path = Path(hm["overlay_image_path"])
        assert hm_img_path.exists()
        assert ov_img_path.exists()
        assert hm_img_path.name == "heatmap_sample_test.png"
        assert ov_img_path.name == "heatmap_overlay_sample_test.png"

        # Verify image dimensions and channels
        read_hm = cv2.imread(str(hm_img_path))
        read_ov = cv2.imread(str(ov_img_path))
        assert read_hm.shape == (100, 120, 3)
        assert read_ov.shape == (100, 120, 3)

    def test_one_update_per_processed_frame(self, tmp_path):
        dummy_video = tmp_path / "frame_count_test.mp4"
        dummy_video.write_bytes(b"dummy_content")

        frames = [np.full((80, 80, 3), 120, dtype=np.uint8) for _ in range(5)]
        t = TrackedObject(track_id=1, box=[20, 20, 40, 40], confidence=0.9, class_id=0, class_name="person")
        mock_vs = MockSyntheticVideoSource(frames, width=80, height=80)
        mock_tr = MockTrackerEngine([[t] for _ in range(5)])

        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr):
            res = app.run_video_tracking(
                str(dummy_video),
                enable_heatmap=True,
                heatmap_output_dir=tmp_path
            )

        assert res["frames_processed"] == 5
        summary = res["heatmap"]["summary"]
        assert summary["frames_processed"] == 5
        assert summary["total_points_accumulated"] == 5

    def test_empty_stream_handling(self, tmp_path):
        dummy_video = tmp_path / "empty_stream.mp4"
        dummy_video.write_bytes(b"dummy_content")

        mock_vs = MockSyntheticVideoSource([], width=100, height=80)
        mock_tr = MockTrackerEngine([])

        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr):
            res = app.run_video_tracking(
                str(dummy_video),
                enable_heatmap=True,
                heatmap_output_dir=tmp_path
            )

        assert res["success"] is True
        assert res["frames_processed"] == 0
        summary = res["heatmap"]["summary"]
        assert summary["frames_processed"] == 0
        assert summary["total_points_accumulated"] == 0
        assert summary["is_empty"] is True

        # Verify solid black heatmap image was generated without error
        read_hm = cv2.imread(res["heatmap"]["heatmap_image_path"])
        assert read_hm.shape == (80, 100, 3)
        assert np.all(read_hm == 0)

    def test_stream_with_no_detections(self, tmp_path):
        dummy_video = tmp_path / "no_detections.mp4"
        dummy_video.write_bytes(b"dummy_content")

        frames = [np.full((60, 60, 3), 80, dtype=np.uint8) for _ in range(4)]
        mock_vs = MockSyntheticVideoSource(frames, width=60, height=60)
        mock_tr = MockTrackerEngine([[], [], [], []])

        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr):
            res = app.run_video_tracking(
                str(dummy_video),
                enable_heatmap=True,
                heatmap_output_dir=tmp_path
            )

        assert res["frames_processed"] == 4
        summary = res["heatmap"]["summary"]
        assert summary["frames_processed"] == 4
        assert summary["total_points_accumulated"] == 0
        assert summary["is_empty"] is True

    def test_image_write_failure_raises_io_error(self, tmp_path):
        dummy_video = tmp_path / "write_fail.mp4"
        dummy_video.write_bytes(b"dummy_content")

        frames = [np.full((50, 50, 3), 100, dtype=np.uint8)]
        mock_vs = MockSyntheticVideoSource(frames, width=50, height=50)
        mock_tr = MockTrackerEngine([[]])

        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr), \
             patch("cv2.imwrite", return_value=False):
            with pytest.raises(IOError, match="Failed to write heatmap image"):
                app.run_video_tracking(
                    str(dummy_video),
                    enable_heatmap=True,
                    heatmap_output_dir=tmp_path
                )

    def test_output_collision_with_source_video_rejected(self, tmp_path):
        dummy_video = tmp_path / "clip.mp4"
        dummy_video.write_bytes(b"dummy")

        mock_vs = MockSyntheticVideoSource([], width=50, height=50)
        mock_tr = MockTrackerEngine([])

        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr):
            with pytest.raises(ValueError, match="Output video path collides with source video"):
                app.run_video_tracking(
                    str(dummy_video),
                    save_path=str(dummy_video),
                    enable_heatmap=True,
                    heatmap_output_dir=tmp_path
                )

    def test_session_isolation_between_consecutive_runs(self, tmp_path):
        dummy_video1 = tmp_path / "video1.mp4"
        dummy_video1.write_bytes(b"dummy1")
        dummy_video2 = tmp_path / "video2.mp4"
        dummy_video2.write_bytes(b"dummy2")

        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(2)]
        # Track 1 at (25, 25)
        t1 = TrackedObject(1, [15, 15, 35, 35], 0.9, 0, "person")
        # Track 2 at (75, 75)
        t2 = TrackedObject(2, [65, 65, 85, 85], 0.9, 0, "car")

        vs1 = MockSyntheticVideoSource(frames, width=100, height=100)
        tr1 = MockTrackerEngine([[t1], [t1]])
        with patch("app.VideoSource", return_value=vs1), \
             patch("app.ObjectTracker", return_value=tr1):
            res1 = app.run_video_tracking(str(dummy_video1), enable_heatmap=True, heatmap_output_dir=tmp_path)

        vs2 = MockSyntheticVideoSource(frames, width=100, height=100)
        tr2 = MockTrackerEngine([[t2], [t2]])
        with patch("app.VideoSource", return_value=vs2), \
             patch("app.ObjectTracker", return_value=tr2):
            res2 = app.run_video_tracking(str(dummy_video2), enable_heatmap=True, heatmap_output_dir=tmp_path)

        dmap1 = res1["heatmap"]["density_map"]
        dmap2 = res2["heatmap"]["density_map"]

        # In run 1, peak is at (25, 25), (75, 75) is 0
        assert dmap1[25, 25] > 1.0
        assert dmap1[75, 75] == 0.0

        # In run 2, peak is at (75, 75), (25, 25) is 0 (no state leaked from run 1)
        assert dmap2[75, 75] > 1.0
        assert dmap2[25, 25] == 0.0

    def test_compatibility_with_spatial_and_behavior_analytics(self, tmp_path):
        dummy_video = tmp_path / "all_analytics.mp4"
        dummy_video.write_bytes(b"dummy")

        frames = [np.full((120, 120, 3), 40, dtype=np.uint8) for _ in range(4)]
        # Moving entity crossing from (60, 20) to (60, 80) across virtual line at y=50
        tracks = [
            [TrackedObject(1, [50, 10, 70, 30], 0.9, 0, "person")],
            [TrackedObject(1, [50, 35, 70, 55], 0.9, 0, "person")],
            [TrackedObject(1, [50, 60, 70, 80], 0.9, 0, "person")],
            [TrackedObject(1, [50, 75, 70, 95], 0.9, 0, "person")]
        ]
        mock_vs = MockSyntheticVideoSource(frames, width=120, height=120)
        mock_tr = MockTrackerEngine(tracks)

        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr):
            res = app.run_video_tracking(
                str(dummy_video),
                virtual_line=(10, 50, 110, 50),
                roi_region=(20, 20, 100, 100),
                enable_heatmap=True,
                heatmap_output_dir=tmp_path
            )

        assert res["success"] is True
        assert res["scene_intelligence"] is not None
        assert res["spatial_analytics"] is not None
        assert res["behavior_analytics"] is not None
        assert res["heatmap"] is not None

        # Verify all four modules have active results
        assert res["spatial_analytics"]["line_crossings"]["total"] >= 1
        assert res["behavior_analytics"]["total_tracks_analyzed"] >= 1
        assert res["heatmap"]["summary"]["total_points_accumulated"] == 4

    def test_cli_argument_parsing(self):
        # Case 1: Heatmap flag enabled with custom directory
        with patch("sys.argv", ["app.py", "--source", "sample.mp4", "--heatmap", "--heatmap-dir", "custom_dir"]):
            args = app.parse_args()
            assert args.heatmap is True
            assert args.heatmap_dir == "custom_dir"

        # Case 2: Heatmap flag omitted
        with patch("sys.argv", ["app.py", "--source", "sample.mp4"]):
            args = app.parse_args()
            assert args.heatmap is False
            assert args.heatmap_dir is None


class TestHeatmapWebcamIntegration:
    """
    Validates Phase C, Stage 3 Step 3: Webcam tracking heatmap integration,
    CLI routing, fresh session isolation, image output generation, collision safety,
    partial-write transactional cleanup, empty stream handling, and resource teardown.
    """

    def test_webcam_tracking_heatmap_disabled_backward_compatibility(self, tmp_path):
        frames = [np.full((100, 120, 3), 60, dtype=np.uint8) for _ in range(3)]
        t = TrackedObject(1, [20, 20, 40, 40], 0.9, 0, "person")
        mock_vs = MockSyntheticVideoSource(frames, width=120, height=100)
        mock_tr = MockTrackerEngine([[t], [t], [t]])

        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr):
            res = app.run_webcam_tracking(
                webcam_id=0,
                enable_heatmap=False,
                heatmap_output_dir=tmp_path,
                headless=True,
                max_frames=3
            )

        assert res["success"] is True
        assert res["frames_processed"] == 3
        assert res["heatmap"] is None
        assert res["scene_intelligence"] is not None
        assert res["behavior_analytics"] is not None
        # Assert no heatmap image files created
        assert len(list(tmp_path.glob("heatmap_webcam_*.png"))) == 0

    def test_webcam_tracking_heatmap_enabled_generates_images_and_density(self, tmp_path):
        frames = [np.full((100, 120, 3), 100, dtype=np.uint8) for _ in range(3)]
        t = TrackedObject(1, [30, 40, 50, 60], 0.95, 0, "person")  # Centroid at (40, 50)
        mock_vs = MockSyntheticVideoSource(frames, width=120, height=100)
        mock_tr = MockTrackerEngine([[t], [t], [t]])

        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr):
            res = app.run_webcam_tracking(
                webcam_id=0,
                enable_heatmap=True,
                heatmap_output_dir=tmp_path,
                heatmap_radius=10,
                heatmap_alpha=0.6,
                headless=True,
                max_frames=3
            )

        assert res["success"] is True
        assert res["frames_processed"] == 3
        assert res["heatmap"] is not None
        hm = res["heatmap"]
        assert hm["enabled"] is True
        assert hm["summary"]["frame_width"] == 120
        assert hm["summary"]["frame_height"] == 100
        assert hm["summary"]["total_points_accumulated"] == 3
        assert hm["summary"]["is_empty"] is False

        # Peak location at (cy=50, cx=40)
        dmap = hm["density_map"]
        assert dmap.shape == (100, 120)
        assert dmap.dtype == np.float32
        assert dmap[50, 40] == pytest.approx(3.0, rel=1e-3)

        # Output image files exist and are readable
        hm_path = Path(hm["heatmap_image_path"])
        ov_path = Path(hm["overlay_image_path"])
        assert hm_path.exists()
        assert ov_path.exists()
        assert "heatmap_webcam_0_" in hm_path.name
        assert "heatmap_overlay_webcam_0_" in ov_path.name

        read_hm = cv2.imread(str(hm_path))
        read_ov = cv2.imread(str(ov_path))
        assert read_hm.shape == (100, 120, 3)
        assert read_ov.shape == (100, 120, 3)

    def test_webcam_exactly_one_update_per_processed_frame(self, tmp_path):
        frames = [np.full((80, 80, 3), 100, dtype=np.uint8) for _ in range(4)]
        t = TrackedObject(1, [20, 20, 40, 40], 0.9, 0, "person")
        mock_vs = MockSyntheticVideoSource(frames, width=80, height=80)
        mock_tr = MockTrackerEngine([[t] for _ in range(4)])

        orig_update = HeatmapAnalytics.update
        update_calls = []

        def spy_update(self_obj, *args, **kwargs):
            update_calls.append({"args": args, "kwargs": kwargs})
            return orig_update(self_obj, *args, **kwargs)

        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr), \
             patch.object(HeatmapAnalytics, "update", spy_update):
            res = app.run_webcam_tracking(
                webcam_id=0,
                enable_heatmap=True,
                heatmap_output_dir=tmp_path,
                headless=True,
                max_frames=4
            )

        assert res["frames_processed"] == 4
        # Assert update was called exactly 4 times (once per frame)
        assert len(update_calls) == 4
        # Verify frame indices passed to update() were 1, 2, 3, 4
        passed_indices = [call["kwargs"].get("frame_idx") for call in update_calls]
        assert passed_indices == [1, 2, 3, 4]

    def test_webcam_camera_startup_failure_clean_exit(self):
        mock_vs = MagicMock()
        mock_vs.open.side_effect = IOError("Could not open webcam index 99")

        with patch("app.VideoSource", return_value=mock_vs):
            with pytest.raises(IOError, match="Could not open webcam index 99"):
                app.run_webcam_tracking(webcam_id=99, enable_heatmap=True)

    def test_webcam_empty_reads_and_zero_usable_frames(self, tmp_path):
        mock_vs = MockSyntheticVideoSource([], width=80, height=60)
        mock_tr = MockTrackerEngine([])

        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr):
            res = app.run_webcam_tracking(
                webcam_id=0,
                enable_heatmap=True,
                heatmap_output_dir=tmp_path,
                headless=True
            )

        assert res["success"] is True
        assert res["frames_processed"] == 0
        assert res["heatmap"] is not None
        summary = res["heatmap"]["summary"]
        assert summary["frames_processed"] == 0
        assert summary["total_points_accumulated"] == 0
        assert summary["is_empty"] is True

        # When dimensions were known, black images were written without crash
        read_hm = cv2.imread(res["heatmap"]["heatmap_image_path"])
        assert read_hm.shape == (60, 80, 3)
        assert np.all(read_hm == 0)

    def test_webcam_session_isolation_across_consecutive_runs(self, tmp_path):
        frames = [np.zeros((100, 100, 3), dtype=np.uint8) for _ in range(2)]
        t1 = TrackedObject(1, [10, 10, 30, 30], 0.9, 0, "person")  # (20, 20)
        t2 = TrackedObject(2, [70, 70, 90, 90], 0.9, 0, "car")     # (80, 80)

        vs1 = MockSyntheticVideoSource(frames, width=100, height=100)
        tr1 = MockTrackerEngine([[t1], [t1]])
        with patch("app.VideoSource", return_value=vs1), \
             patch("app.ObjectTracker", return_value=tr1):
            res1 = app.run_webcam_tracking(webcam_id=0, enable_heatmap=True, heatmap_output_dir=tmp_path, headless=True)

        vs2 = MockSyntheticVideoSource(frames, width=100, height=100)
        tr2 = MockTrackerEngine([[t2], [t2]])
        with patch("app.VideoSource", return_value=vs2), \
             patch("app.ObjectTracker", return_value=tr2):
            res2 = app.run_webcam_tracking(webcam_id=0, enable_heatmap=True, heatmap_output_dir=tmp_path, headless=True)

        dmap1 = res1["heatmap"]["density_map"]
        dmap2 = res2["heatmap"]["density_map"]

        assert dmap1[20, 20] > 1.0
        assert dmap1[80, 80] == 0.0

        assert dmap2[80, 80] > 1.0
        assert dmap2[20, 20] == 0.0

    def test_webcam_output_filename_collision_safety(self, tmp_path):
        frames = [np.full((50, 50, 3), 100, dtype=np.uint8)]
        mock_vs1 = MockSyntheticVideoSource(frames, width=50, height=50)
        mock_vs2 = MockSyntheticVideoSource(frames, width=50, height=50)
        mock_tr = MockTrackerEngine([[]])

        # Run 1 at t=1000
        with patch("app.VideoSource", return_value=mock_vs1), \
             patch("app.ObjectTracker", return_value=mock_tr), \
             patch("time.time", return_value=1000.0):
            res1 = app.run_webcam_tracking(webcam_id=0, enable_heatmap=True, heatmap_output_dir=tmp_path, headless=True)

        # Run 2 at t=2000
        with patch("app.VideoSource", return_value=mock_vs2), \
             patch("app.ObjectTracker", return_value=mock_tr), \
             patch("time.time", return_value=2000.0):
            res2 = app.run_webcam_tracking(webcam_id=0, enable_heatmap=True, heatmap_output_dir=tmp_path, headless=True)

        path1 = Path(res1["heatmap"]["heatmap_image_path"])
        path2 = Path(res2["heatmap"]["heatmap_image_path"])
        assert path1 != path2
        assert path1.exists()
        assert path2.exists()
        assert "1000" in path1.name
        assert "2000" in path2.name

    def test_webcam_first_image_write_failure_raises_io_error(self, tmp_path):
        frames = [np.full((50, 50, 3), 100, dtype=np.uint8)]
        mock_vs = MockSyntheticVideoSource(frames, width=50, height=50)
        mock_tr = MockTrackerEngine([[]])

        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr), \
             patch("cv2.imwrite", return_value=False):
            with pytest.raises(IOError, match="Failed to write webcam heatmap image"):
                app.run_webcam_tracking(
                    webcam_id=0,
                    enable_heatmap=True,
                    heatmap_output_dir=tmp_path,
                    headless=True
                )

    def test_webcam_second_image_write_failure_cleans_up_partial_artifact(self, tmp_path):
        frames = [np.full((50, 50, 3), 100, dtype=np.uint8)]
        mock_vs = MockSyntheticVideoSource(frames, width=50, height=50)
        mock_tr = MockTrackerEngine([[]])

        real_imwrite = cv2.imwrite
        write_call_count = 0
        created_first_path = None

        def fake_imwrite(path_str, img):
            nonlocal write_call_count, created_first_path
            write_call_count += 1
            if write_call_count == 1:
                created_first_path = Path(path_str)
                return real_imwrite(path_str, img)
            # Second call fails
            return False

        # Sub-case A: First image was newly created by this run -> cleaned up on second failure
        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr), \
             patch("cv2.imwrite", side_effect=fake_imwrite):
            with pytest.raises(IOError, match="Failed to write webcam heatmap overlay image"):
                app.run_webcam_tracking(
                    webcam_id=0,
                    enable_heatmap=True,
                    heatmap_output_dir=tmp_path,
                    headless=True,
                    max_frames=1
                )

        assert write_call_count == 2
        assert created_first_path is not None
        # File was deleted to avoid leaving orphaned partial artifact
        assert not created_first_path.exists()

        # Sub-case B: Pre-existing unrelated file must NOT be deleted
        pre_existing = tmp_path / "heatmap_webcam_0_55555.png"
        pre_existing.write_bytes(b"preserved_content")
        write_call_count = 0

        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr), \
             patch("time.time", return_value=55555.0), \
             patch("cv2.imwrite", side_effect=fake_imwrite):
            with pytest.raises(IOError, match="Failed to write webcam heatmap overlay image"):
                app.run_webcam_tracking(
                    webcam_id=0,
                    enable_heatmap=True,
                    heatmap_output_dir=tmp_path,
                    headless=True,
                    max_frames=1
                )

        # Pre-existing file is safe and intact
        assert pre_existing.exists()
        assert pre_existing.read_bytes() == b"preserved_content"

    def test_webcam_resource_release_on_completion_interruption_and_exceptions(self, tmp_path):
        frames = [np.zeros((50, 50, 3), dtype=np.uint8) for _ in range(3)]
        mock_tr = MockTrackerEngine([[] for _ in range(3)])

        # 1. Normal completion
        vs1 = MockSyntheticVideoSource(frames, width=50, height=50)
        with patch("app.VideoSource", return_value=vs1), \
             patch("app.ObjectTracker", return_value=mock_tr):
            app.run_webcam_tracking(webcam_id=0, enable_heatmap=True, heatmap_output_dir=tmp_path, headless=True, max_frames=2)
        assert vs1.is_opened is False

        # 2. Interruption simulation (quit key 'q' pressed in cv2 live window)
        vs2 = MockSyntheticVideoSource(frames, width=50, height=50)
        with patch("app.VideoSource", return_value=vs2), \
             patch("app.ObjectTracker", return_value=mock_tr), \
             patch("cv2.imshow"), \
             patch("cv2.waitKey", return_value=ord("q")):
            app.run_webcam_tracking(webcam_id=0, enable_heatmap=True, heatmap_output_dir=tmp_path, headless=False)
        assert vs2.is_opened is False

        # 3. Exception in tracking loop
        vs3 = MockSyntheticVideoSource(frames, width=50, height=50)
        mock_tr_err = MagicMock()
        mock_tr_err.load_model.return_value = None
        mock_tr_err.track.side_effect = RuntimeError("Fatal GPU/model error")
        with patch("app.VideoSource", return_value=vs3), \
             patch("app.ObjectTracker", return_value=mock_tr_err):
            with pytest.raises(RuntimeError, match="Fatal GPU/model error"):
                app.run_webcam_tracking(webcam_id=0, enable_heatmap=True, heatmap_output_dir=tmp_path, headless=True)
        assert vs3.is_opened is False

    def test_webcam_compatibility_with_scene_spatial_and_behavior_analytics(self, tmp_path):
        frames = [np.full((100, 100, 3), 40, dtype=np.uint8) for _ in range(4)]
        tracks = [
            [TrackedObject(1, [40, 10, 60, 30], 0.9, 0, "person")],
            [TrackedObject(1, [40, 35, 60, 55], 0.9, 0, "person")],
            [TrackedObject(1, [40, 60, 60, 80], 0.9, 0, "person")],
            [TrackedObject(1, [40, 75, 60, 95], 0.9, 0, "person")]
        ]
        mock_vs = MockSyntheticVideoSource(frames, width=100, height=100)
        mock_tr = MockTrackerEngine(tracks)

        with patch("app.VideoSource", return_value=mock_vs), \
             patch("app.ObjectTracker", return_value=mock_tr):
            res = app.run_webcam_tracking(
                webcam_id=0,
                virtual_line=(10, 45, 90, 45),
                roi_region=(20, 20, 80, 80),
                enable_heatmap=True,
                heatmap_output_dir=tmp_path,
                headless=True,
                max_frames=4
            )

        assert res["success"] is True
        assert res["scene_intelligence"] is not None
        assert res["spatial_analytics"] is not None
        assert res["behavior_analytics"] is not None
        assert res["heatmap"] is not None
        assert res["heatmap"]["summary"]["total_points_accumulated"] == 4

    def test_webcam_cli_argument_parsing_and_routing(self):
        # Case 1: Webcam source with heatmap enabled
        with patch("sys.argv", ["app.py", "--source", "0", "--heatmap", "--heatmap-dir", "webcam_reports"]):
            args = app.parse_args()
            assert args.source == "0"
            assert args.heatmap is True
            assert args.heatmap_dir == "webcam_reports"

        # Case 2: Webcam source with default options
        with patch("sys.argv", ["app.py", "--source", "1"]):
            args = app.parse_args()
            assert args.source == "1"
            assert args.heatmap is False
            assert args.heatmap_dir is None


