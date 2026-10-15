"""
Comprehensive Unit Tests for Scene Intelligence Module.
Validates scene snapshots, motion direction analysis, event timelines,
duplicate suppression, natural-language summaries, and session isolation.
"""

import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.scene_intelligence import SceneIntelligence, MotionState, SceneEvent, TrackMotionRecord
from src.tracker import TrackedObject


class MockTrack:
    """Lightweight duck-typed object representing a track."""
    def __init__(self, track_id, class_name, centroid, box=None):
        self.track_id = track_id
        self.class_name = class_name
        self.current_centroid = centroid
        self.box = box or [centroid[0] - 10, centroid[1] - 10, centroid[0] + 10, centroid[1] + 10]
        self.confidence = 0.90


class TestSceneIntelligence:
    """Test suite for explainable scene analytics and event generation."""

    def test_empty_detections_handling(self):
        intel = SceneIntelligence()
        res = intel.update(frame_idx=1, tracks=[], timestamp_sec=0.033)
        assert res["visible_count"] == 0
        assert res["visible_classes"] == {}
        assert res["cumulative_unique_ids"] == 0
        assert res["new_events"] == []
        summary = intel.generate_natural_language_summary()
        assert "No objects were detected" in summary

    def test_single_and_multiple_classes_snapshot(self):
        intel = SceneIntelligence()
        t1 = MockTrack(track_id=1, class_name="person", centroid=(100, 100))
        t2 = MockTrack(track_id=2, class_name="car", centroid=(200, 200))
        t3 = MockTrack(track_id=3, class_name="person", centroid=(300, 300))

        res = intel.update(frame_idx=1, tracks=[t1, t2, t3])
        assert res["visible_count"] == 3
        assert res["visible_classes"] == {"person": 2, "car": 1}
        assert res["cumulative_unique_ids"] == 3
        assert res["cumulative_classes"] == {"person": 2, "car": 1}
        assert len(res["new_events"]) == 3
        assert all(e["event_type"] == "ENTERED" for e in res["new_events"])

    def test_repeated_tracking_ids_no_duplicate_entered_events(self):
        intel = SceneIntelligence()
        t1 = MockTrack(track_id=1, class_name="person", centroid=(100, 100))

        # Frame 1: Object 1 enters
        res1 = intel.update(frame_idx=1, tracks=[t1])
        assert len(res1["new_events"]) == 1
        assert res1["new_events"][0]["event_type"] == "ENTERED"

        # Frame 2: Same Object 1 still present
        res2 = intel.update(frame_idx=2, tracks=[t1])
        # Should NOT log another ENTERED event
        entered_events = [e for e in res2["new_events"] if e["event_type"] == "ENTERED"]
        assert len(entered_events) == 0
        assert res2["cumulative_unique_ids"] == 1

    def test_small_centroid_jitter_classified_as_stationary(self):
        intel = SceneIntelligence(min_movement_threshold_px=15.0, min_frames_for_motion=3)
        # 4 frames with tiny jitter < 15px
        centroids = [(100, 100), (102, 101), (101, 103), (102, 102)]
        for idx, pt in enumerate(centroids, start=1):
            t = MockTrack(track_id=1, class_name="cup", centroid=pt)
            res = intel.update(frame_idx=idx, tracks=[t])

        # State should be STATIONARY, not moving
        motion_info = res["active_motion"][1]
        assert motion_info["motion_state"] == MotionState.STATIONARY
        # No MOVED events should have fired
        moved_events = intel.get_events(event_type="MOVED")
        assert len(moved_events) == 0

    def test_meaningful_movement_and_direction_estimation(self):
        intel = SceneIntelligence(min_movement_threshold_px=15.0, min_frames_for_motion=3)
        # Move distinctly right: x increases by 50px
        centroids_right = [(100, 100), (120, 100), (140, 100), (160, 100)]
        for idx, pt in enumerate(centroids_right, start=1):
            t = MockTrack(track_id=1, class_name="person", centroid=pt)
            res = intel.update(frame_idx=idx, tracks=[t])

        motion_info = res["active_motion"][1]
        assert motion_info["motion_state"] == MotionState.MOVING_RIGHT
        moved_events = intel.get_events(event_type="MOVED")
        assert len(moved_events) >= 1
        assert moved_events[0]["details"]["direction"] == "right"

        # Test Moving Down (y increases) on track 2
        intel.reset()
        centroids_down = [(100, 100), (100, 120), (100, 150), (100, 175)]
        for idx, pt in enumerate(centroids_down, start=1):
            t = MockTrack(track_id=2, class_name="dog", centroid=pt)
            res = intel.update(frame_idx=idx, tracks=[t])

        assert res["active_motion"][2]["motion_state"] == MotionState.MOVING_DOWN

    def test_insufficient_history_motion_state(self):
        intel = SceneIntelligence(min_frames_for_motion=5)
        # Only 2 observations provided
        t1 = MockTrack(track_id=1, class_name="person", centroid=(100, 100))
        intel.update(frame_idx=1, tracks=[t1])
        t1_moved = MockTrack(track_id=1, class_name="person", centroid=(150, 100))
        res = intel.update(frame_idx=2, tracks=[t1_moved])

        assert res["active_motion"][1]["motion_state"] == MotionState.INSUFFICIENT_DATA

    def test_disappearing_objects_and_exited_events(self):
        intel = SceneIntelligence(disappearance_grace_frames=3)
        t = MockTrack(track_id=5, class_name="bicycle", centroid=(50, 50))

        # Frame 1: present
        intel.update(frame_idx=1, tracks=[t])
        # Frames 2, 3: missing
        intel.update(frame_idx=2, tracks=[])
        intel.update(frame_idx=3, tracks=[])
        # Frame 4: hits grace threshold (3 missing frames) -> emits EXITED
        res4 = intel.update(frame_idx=4, tracks=[])

        exited = [e for e in res4["new_events"] if e["event_type"] == "EXITED"]
        assert len(exited) == 1
        assert exited[0]["track_id"] == 5
        assert exited[0]["class_name"] == "bicycle"

    def test_stream_finalization_exits_active_tracks(self):
        intel = SceneIntelligence(disappearance_grace_frames=20)
        t = MockTrack(track_id=7, class_name="car", centroid=(200, 200))
        intel.update(frame_idx=1, tracks=[t])

        # Still visible at stream end; finalize_stream should cleanly emit EXITED
        closing = intel.finalize_stream(last_frame_idx=10, timestamp_sec=0.33)
        assert len(closing) == 1
        assert closing[0].track_id == 7
        assert closing[0].event_type == "EXITED"

    def test_duplicate_movement_event_suppression(self):
        intel = SceneIntelligence(
            min_movement_threshold_px=10.0,
            min_frames_for_motion=2,
            movement_throttle_frames=10
        )
        # Move right continuously across 8 frames in same direction
        for i in range(1, 9):
            t = MockTrack(track_id=1, class_name="car", centroid=(100 + i * 20, 100))
            intel.update(frame_idx=i, tracks=[t])

        # Throttle is 10 frames, so only 1 MOVED event should be emitted despite continuous motion
        moved_events = intel.get_events(event_type="MOVED")
        assert len(moved_events) == 1

    def test_session_reset_and_isolation(self):
        intel = SceneIntelligence()
        t = MockTrack(track_id=1, class_name="person", centroid=(50, 50))
        intel.update(frame_idx=1, tracks=[t])
        assert intel.get_session_report()["cumulative_unique_objects"] == 1
        assert len(intel.get_events()) == 1

        # Reset session
        intel.reset()
        rep = intel.get_session_report()
        assert rep["total_frames"] == 0
        assert rep["cumulative_unique_objects"] == 0
        assert rep["total_events_logged"] == 0
        assert "No objects were detected" in rep["natural_language_summary"]

    def test_natural_language_summary_traceability(self):
        intel = SceneIntelligence(min_movement_threshold_px=10.0, min_frames_for_motion=2)
        # Track 1 (person, stationary), Track 2 (car, moves right 40px)
        t1 = MockTrack(track_id=1, class_name="person", centroid=(100, 100))
        t2 = MockTrack(track_id=2, class_name="car", centroid=(200, 200))
        intel.update(frame_idx=1, tracks=[t1, t2])

        t1_b = MockTrack(track_id=1, class_name="person", centroid=(101, 101))
        t2_b = MockTrack(track_id=2, class_name="car", centroid=(240, 200))
        intel.update(frame_idx=2, tracks=[t1_b, t2_b])

        summary = intel.generate_natural_language_summary()
        # Traceability assertions
        assert "2 distinct objects" in summary
        assert "1 car" in summary
        assert "1 person" in summary
        assert "#2 car covered" in summary

    def test_malformed_or_missing_tracking_fields(self):
        intel = SceneIntelligence()
        # Object without centroid attribute (only box)
        class PartialTrack:
            track_id = 9
            class_name = "bottle"
            box = [10, 10, 30, 50]  # centroid will be (20, 30)

        res = intel.update(frame_idx=1, tracks=[PartialTrack()])
        assert res["visible_count"] == 1
        assert res["visible_classes"] == {"bottle": 1}
        assert res["new_events"][0]["track_id"] == 9

    def test_run_video_tracking_includes_scene_intelligence(self):
        sample_p = Path(__file__).resolve().parent.parent / "assets" / "sample_clip.mp4"
        if not sample_p.exists():
            pytest.skip("sample_clip.mp4 not present in assets")

        import app
        res = app.run_video_tracking(
            video_path=str(sample_p),
            model_name="yolo11n.pt",
            tracker_type="bytetrack.yaml",
            conf_threshold=0.25,
            device="cpu",
            save_path=None,
            max_frames=3
        )
        assert res["success"] is True
        assert "scene_intelligence" in res
        intel = res["scene_intelligence"]
        assert "natural_language_summary" in intel
        assert "cumulative_unique_objects" in intel
        assert intel["total_frames"] == 3
