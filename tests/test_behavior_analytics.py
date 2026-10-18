"""
Unit and Integration Tests for Trajectory Behavior Analytics Module.
Validates path length, net displacement, average speed, direction changes,
tortuosity, numerical safety (NaN/Inf guards), timing fallbacks,
session aggregation, and memory bounds.
"""

import sys
from pathlib import Path
import math
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.behavior_analytics import (
    TrajectoryMetrics,
    BehaviorAnalytics,
    compute_trajectory_metrics,
    compute_path_length_and_displacement,
    compute_tortuosity,
    compute_direction_changes,
    compute_speed,
    validate_trajectory_points
)


class MockTrack:
    """Lightweight mock object compatible with TrackedObject interface."""
    def __init__(self, track_id: int, class_name: str, trajectory: list):
        self.track_id = int(track_id)
        self.class_name = str(class_name)
        self.trajectory = list(trajectory)
        self.confidence = 0.95

    @property
    def current_centroid(self):
        return self.trajectory[-1] if self.trajectory else (0, 0)


class TestTrajectoryGeometryAndCalculations:
    """Validates low-level geometric, angular, and numerical calculations."""

    def test_straight_line_movement(self):
        # Straight line along x-axis from x=0 to x=100
        points = [(0, 0), (25, 0), (50, 0), (75, 0), (100, 0)]
        path_len, net_disp = compute_path_length_and_displacement(points)
        assert path_len == 100.0
        assert net_disp == 100.0

        tort = compute_tortuosity(path_len, net_disp)
        assert tort == 1.0

        turns, details = compute_direction_changes(points)
        assert turns == 0
        assert details == []

        metrics = compute_trajectory_metrics(points, track_id=1, elapsed_time_sec=2.0)
        assert metrics.is_sufficient is True
        assert metrics.path_length_px == 100.0
        assert metrics.net_displacement_px == 100.0
        assert metrics.tortuosity == 1.0
        assert metrics.direction_changes_count == 0
        assert metrics.avg_speed_px_per_sec == 50.0

    def test_stationary_object(self):
        # Object stationary at (50, 50)
        points = [(50, 50), (50, 50), (50, 50), (50, 50)]
        path_len, net_disp = compute_path_length_and_displacement(points)
        assert path_len == 0.0
        assert net_disp == 0.0

        tort = compute_tortuosity(path_len, net_disp)
        assert tort == 1.0  # Safe stationary default

        turns, details = compute_direction_changes(points)
        assert turns == 0
        assert details == []

        metrics = compute_trajectory_metrics(points, track_id=2, elapsed_time_sec=1.0)
        assert metrics.path_length_px == 0.0
        assert metrics.net_displacement_px == 0.0
        assert metrics.tortuosity == 1.0
        assert metrics.direction_changes_count == 0
        assert metrics.avg_speed_px_per_sec == 0.0

    def test_curved_path_and_tortuosity(self):
        # S-like or detour path: goes right 50, down 50, right 50
        # Start: (0, 0), End: (100, 50)
        # Path length = 50 + 50 + 50 = 150
        # Net displacement = sqrt(100^2 + 50^2) = sqrt(12500) ≈ 111.80
        points = [(0, 0), (50, 0), (50, 50), (100, 50)]
        path_len, net_disp = compute_path_length_and_displacement(points)
        assert path_len == 150.0
        assert round(net_disp, 2) == 111.80

        tort = compute_tortuosity(path_len, net_disp)
        # 150 / 111.80 ≈ 1.34
        assert tort > 1.0
        assert round(tort, 2) == 1.34

    def test_meaningful_direction_changes(self):
        # Path makes two 90-degree turns (right, then down, then right)
        points = [
            (0, 0), (20, 0), (40, 0),       # moving East
            (40, 20), (40, 40),             # turn 1: 90 deg South
            (60, 40), (80, 40)              # turn 2: 90 deg East
        ]
        turns, details = compute_direction_changes(points, min_segment_length_px=10.0, min_turn_angle_deg=45.0)
        assert turns == 2
        assert len(details) == 2
        assert details[0]["angle_deg"] == 90.0
        assert details[1]["angle_deg"] == 90.0

    def test_small_jitter_does_not_count_as_turn(self):
        # Mostly stationary object with micro-jitter (1-2 px fluctuations)
        points = [
            (100, 100), (101, 100), (100, 101), (101, 101),
            (100, 100), (102, 101), (101, 100), (100, 101)
        ]
        # With default min_segment_length_px=6.0, micro-jitter must produce 0 turns
        turns, details = compute_direction_changes(points, min_segment_length_px=6.0)
        assert turns == 0
        assert details == []

    def test_too_few_trajectory_points(self):
        # 0 points
        m0 = compute_trajectory_metrics([])
        assert m0.valid_points == 0
        assert m0.is_sufficient is False
        assert m0.path_length_px == 0.0
        assert m0.tortuosity == 1.0
        assert m0.avg_speed_px_per_sec is None
        assert "No valid trajectory points" in m0.speed_unavailable_reason

        # 1 point
        m1 = compute_trajectory_metrics([(10, 20)])
        assert m1.valid_points == 1
        assert m1.is_sufficient is False
        assert m1.path_length_px == 0.0
        assert m1.direction_changes_count == 0
        assert m1.avg_speed_px_per_sec is None

        # 2 points (straight segment, but insufficient for curvature/turn analysis)
        m2 = compute_trajectory_metrics([(0, 0), (30, 40)], min_points_for_analysis=3)
        assert m2.valid_points == 2
        assert m2.is_sufficient is False
        assert m2.path_length_px == 50.0
        assert m2.net_displacement_px == 50.0
        assert m2.direction_changes_count == 0

    def test_invalid_missing_nan_and_infinite_coordinates(self):
        raw_pts = [
            (0, 0),
            (float("nan"), 10),
            (20, float("inf")),
            (-float("inf"), 20),
            "invalid_point",
            None,
            [40, 0],
            (40, 30)
        ]
        valid_pts, total_rec = validate_trajectory_points(raw_pts)
        assert total_rec == 8
        assert len(valid_pts) == 3
        assert valid_pts == [(0.0, 0.0), (40.0, 0.0), (40.0, 30.0)]

        metrics = compute_trajectory_metrics(raw_pts, track_id=5)
        assert metrics.valid_points == 3
        assert metrics.total_points == 8
        assert metrics.path_length_px == 70.0
        assert metrics.net_displacement_px == 50.0
        assert not math.isnan(metrics.tortuosity)
        assert not math.isinf(metrics.tortuosity)

    def test_zero_or_near_zero_displacement_closed_loop(self):
        # Closed loop: 40px right, 40px down, 40px left, 40px up (returns to start)
        points = [(0, 0), (40, 0), (40, 40), (0, 40), (0, 0)]
        path_len, net_disp = compute_path_length_and_displacement(points)
        assert path_len == 160.0
        assert net_disp == 0.0

        # Must avoid division by zero and cap gracefully
        tort = compute_tortuosity(path_len, net_disp, min_displacement_epsilon_px=4.0)
        assert not math.isinf(tort)
        assert not math.isnan(tort)
        assert tort > 1.0
        # 160 / 4.0 = 40.0
        assert tort == 40.0

    def test_repeated_or_duplicate_consecutive_points(self):
        points = [(10, 20), (10, 20), (30, 20), (30, 20), (30, 20)]
        path_len, net_disp = compute_path_length_and_displacement(points)
        assert path_len == 20.0
        assert net_disp == 20.0
        assert compute_tortuosity(path_len, net_disp) == 1.0


class TestSpeedAndTimingSafety:
    """Validates speed calculation, timing fallbacks, and error reasons."""

    def test_speed_calculation_with_valid_timestamps(self):
        # 60 pixels over 1.5 seconds -> 40.0 px/s
        speed, reason = compute_speed(path_length_px=60.0, elapsed_time_sec=1.5)
        assert speed == 40.0
        assert reason is None

    def test_missing_invalid_or_inconsistent_timing(self):
        # None
        s1, r1 = compute_speed(50.0, elapsed_time_sec=None)
        assert s1 is None
        assert "not provided" in r1

        # 0.0 seconds (divide by zero guard)
        s2, r2 = compute_speed(50.0, elapsed_time_sec=0.0)
        assert s2 is None
        assert "non-positive" in r2

        # Negative elapsed time
        s3, r3 = compute_speed(50.0, elapsed_time_sec=-2.0)
        assert s3 is None
        assert "non-positive" in r3

        # NaN / Inf
        s4, r4 = compute_speed(50.0, elapsed_time_sec=float("nan"))
        assert s4 is None
        assert "non-finite" in r4

    def test_compute_trajectory_metrics_timing_options(self):
        pts = [(0, 0), (30, 0), (60, 0)]
        # Via start_time and end_time
        m1 = compute_trajectory_metrics(pts, start_time=1.0, end_time=3.0)
        assert m1.avg_speed_px_per_sec == 30.0  # 60px / 2s

        # Via fps and point count
        m2 = compute_trajectory_metrics(pts, fps=10.0)
        # 3 points -> 2 intervals -> 2 / 10 = 0.2s -> 60 / 0.2 = 300.0 px/s
        assert m2.avg_speed_px_per_sec == 300.0

        # Inconsistent start/end (end <= start)
        m3 = compute_trajectory_metrics(pts, start_time=5.0, end_time=2.0)
        assert m3.avg_speed_px_per_sec is None
        assert m3.speed_unavailable_reason is not None


class TestBehaviorAnalyticsSessionAndLifecycle:
    """Validates multi-object session management, memory bounds, and isolation."""

    def test_multiple_track_ids_and_classes(self):
        engine = BehaviorAnalytics()

        # Frame 1: Person 1 at (0, 0), Car 2 at (100, 100)
        t1 = MockTrack(1, "person", [(0, 0)])
        t2 = MockTrack(2, "car", [(100, 100)])
        engine.update(frame_idx=1, tracks=[t1, t2], timestamp_sec=0.1)

        # Frame 2: Person 1 moves to (20, 0), Car 2 moves to (100, 150)
        t1.trajectory.append((20, 0))
        t2.trajectory.append((100, 150))
        engine.update(frame_idx=2, tracks=[t1, t2], timestamp_sec=0.2)

        # Frame 3: Person 1 moves to (40, 0), Car 2 moves to (100, 200)
        t1.trajectory.append((40, 0))
        t2.trajectory.append((100, 200))
        res3 = engine.update(frame_idx=3, tracks=[t1, t2], timestamp_sec=0.3)

        assert res3["active_tracks_count"] == 2
        assert 1 in res3["frame_track_metrics"]
        assert 2 in res3["frame_track_metrics"]

        m1 = engine.get_track_metrics(1)
        assert m1["class_name"] == "person"
        assert m1["path_length_px"] == 40.0
        assert m1["tortuosity"] == 1.0

        m2 = engine.get_track_metrics(2)
        assert m2["class_name"] == "car"
        assert m2["path_length_px"] == 100.0

        summary = engine.get_summary()
        assert summary["total_tracks_analyzed"] == 2
        assert "person" in summary["class_breakdown"]
        assert "car" in summary["class_breakdown"]
        assert summary["class_breakdown"]["person"]["track_count"] == 1
        assert summary["class_breakdown"]["car"]["track_count"] == 1

    def test_session_reset_and_isolation(self):
        engine_a = BehaviorAnalytics()
        engine_b = BehaviorAnalytics()

        t = MockTrack(1, "person", [(0, 0), (20, 0), (40, 0)])
        engine_a.update(frame_idx=1, tracks=[t], timestamp_sec=0.1)

        assert engine_a.get_summary()["total_tracks_analyzed"] == 1
        assert engine_b.get_summary()["total_tracks_analyzed"] == 0

        engine_a.reset()
        assert engine_a.get_summary()["total_tracks_analyzed"] == 0
        assert engine_a.get_all_metrics() == []
        assert engine_a.get_track_metrics(1) is None

    def test_bounded_history_pruning(self):
        engine = BehaviorAnalytics(max_tracks_history=2, disappearance_grace_frames=2)

        # Create and terminate tracks 1, 2, 3
        for tid in range(1, 4):
            t = MockTrack(tid, "person", [(0, 0), (20, 0), (40, 0)])
            engine.update(frame_idx=tid * 10, tracks=[t], timestamp_sec=float(tid))
            # Disappear
            engine.update(frame_idx=tid * 10 + 1, tracks=[])
            engine.update(frame_idx=tid * 10 + 2, tracks=[])

        # Track 1 should be pruned, tracks 2 and 3 remain in completed history
        assert len(engine._completed_tracks) <= 2
        assert engine.get_track_metrics(3) is not None

    def test_stream_finalization_archives_active_tracks(self):
        engine = BehaviorAnalytics()
        t = MockTrack(10, "bicycle", [(0, 0), (10, 10), (20, 20)])
        engine.update(frame_idx=1, tracks=[t], timestamp_sec=0.1)
        assert len(engine._active_tracks) == 1

        engine.finalize_stream(last_frame_idx=5, timestamp_sec=0.5)
        assert len(engine._active_tracks) == 0
        assert len(engine._completed_tracks) == 1

        m = engine.get_track_metrics(10)
        assert m is not None
        assert m["track_id"] == 10

        # Repeated finalize call is idempotent
        engine.finalize_stream()
        assert len(engine._completed_tracks) == 1

    def test_to_dict_has_no_nan_or_inf(self):
        pts = [(0, 0), (10, 0), (10, 20)]
        m = compute_trajectory_metrics(pts, track_id=99, elapsed_time_sec=1.5)
        d = m.to_dict()

        for k, v in d.items():
            if isinstance(v, (int, float)):
                assert not math.isnan(v), f"Key {k} is NaN"
                assert not math.isinf(v), f"Key {k} is Inf"

    def test_real_tracked_object_and_spatial_non_interference(self):
        from src.tracker import TrackedObject
        from src.spatial_analytics import SpatialAnalytics

        # 1. Real TrackedObject feeding BehaviorAnalytics
        t = TrackedObject(track_id=1, box=[0, 40, 20, 60], confidence=0.90, class_id=0, class_name="person")
        t.update([20, 40, 40, 60], 0.92, 0, "person")
        t.update([40, 40, 60, 60], 0.93, 0, "person")
        assert len(t.trajectory) == 3

        behavior = BehaviorAnalytics()
        res = behavior.update(frame_idx=3, tracks=[t], timestamp_sec=0.1, fps=30.0)
        assert res["active_tracks_count"] == 1
        m = behavior.get_track_metrics(1)
        assert m is not None
        assert m["class_name"] == "person"
        assert m["valid_points"] == 3
        assert m["path_length_px"] == 40.0
        assert m["tortuosity"] == 1.0

        # 2. Stage 1 SpatialAnalytics operates completely independently without interference
        spatial = SpatialAnalytics(roi_region=(10, 10, 100, 100))
        spatial_res = spatial.update(frame_idx=1, tracks=[t], timestamp_sec=0.1)
        assert spatial_res["roi_current_occupancy"] == 1
        dwell_summary = spatial.get_dwell_summary()
        assert dwell_summary["current_inside_count"] == 1
        assert dwell_summary["peak_occupancy"] == 1


class TestBehaviorAnalyticsTargetedCorrections:
    """Regression tests for targeted Phase C Stage 2 audit fixes."""

    def test_long_running_track_window_alignment(self):
        """Fix 1: Observation window duration aligns with retained trajectory points (not entire track lifespan)."""
        engine = BehaviorAnalytics(max_trajectory_len=45)

        # 60 frames of steady movement: 10 px/frame, 0.1s/frame -> 100.0 px/s
        traj = []
        for f in range(1, 61):
            centroid = (f * 10, 0)
            traj.append(centroid)
            t = MockTrack(1, "car", list(traj))
            res = engine.update(frame_idx=f, tracks=[t], timestamp_sec=f * 0.1)

        m = engine.get_track_metrics(1)
        assert m is not None
        assert m["valid_points"] == 45
        # The 45 points span from frame 16 (160, 0) to frame 60 (600, 0)
        # Distance = 440 px, dt = 6.0 - 1.6 = 4.4s -> 100.0 px/s
        assert m["path_length_px"] == 440.0
        assert m["net_displacement_px"] == 440.0
        assert m["elapsed_time_sec"] == pytest.approx(4.4, rel=1e-3)
        assert m["avg_speed_px_per_sec"] == pytest.approx(100.0, rel=1e-3)

    def test_dropped_and_irregular_frames_speed_accuracy(self):
        """Fix 1: Correct elapsed time used across dropped/irregular frames without assuming consecutive 1/fps steps."""
        engine = BehaviorAnalytics()

        # Frame 10 at ts=1.0s, centroid=(100, 100)
        t1 = MockTrack(1, "person", [(100, 100)])
        engine.update(frame_idx=10, tracks=[t1], timestamp_sec=1.0)

        # Frame 20 (frames 11-19 dropped) at ts=2.0s, centroid=(160, 100) -> 60 px in 1.0s = 60.0 px/s
        t2 = MockTrack(1, "person", [(100, 100), (160, 100)])
        engine.update(frame_idx=20, tracks=[t2], timestamp_sec=2.0)

        m = engine.get_track_metrics(1)
        assert m is not None
        assert m["path_length_px"] == 60.0
        assert m["elapsed_time_sec"] == 1.0
        assert m["avg_speed_px_per_sec"] == 60.0

        # Also verify frame fallback when timestamp_sec is None
        engine2 = BehaviorAnalytics()
        t_a = MockTrack(2, "person", [(100, 100)])
        engine2.update(frame_idx=10, tracks=[t_a], timestamp_sec=None, fps=10.0)

        t_b = MockTrack(2, "person", [(100, 100), (160, 100)])
        engine2.update(frame_idx=20, tracks=[t_b], timestamp_sec=None, fps=10.0)

        m2 = engine2.get_track_metrics(2)
        assert m2 is not None
        assert m2["path_length_px"] == 60.0
        # (20 - 10) / 10.0 = 1.0s
        assert m2["elapsed_time_sec"] == 1.0
        assert m2["avg_speed_px_per_sec"] == 60.0

    def test_missing_timestamps_speed_fallback_and_unavailable_reason(self):
        """Fix 1: Explicit unavailable reason when timing is incomplete or invalid."""
        pts = [(0, 0), (50, 0)]
        m = compute_trajectory_metrics(pts, elapsed_time_sec=None, fps=None)
        assert m.avg_speed_px_per_sec is None
        assert m.speed_unavailable_reason is not None
        assert "not provided" in m.speed_unavailable_reason

        # Negative time guard
        s_neg, r_neg = compute_speed(50.0, elapsed_time_sec=-5.0)
        assert s_neg is None
        assert "non-positive" in r_neg

    def test_reused_track_id_preserves_both_lifecycles(self):
        """Fix 2: Distinct lifecycle IDs preserve both lifecycles when ByteTrack reuses a track ID."""
        engine = BehaviorAnalytics(disappearance_grace_frames=2)

        # Lifecycle 1 of Track 7
        t1 = MockTrack(7, "person", [(0, 0), (20, 0), (40, 0)])
        engine.update(frame_idx=1, tracks=[t1], timestamp_sec=0.1)
        engine.update(frame_idx=2, tracks=[t1], timestamp_sec=0.2)
        # Disappears
        engine.update(frame_idx=3, tracks=[])
        engine.update(frame_idx=4, tracks=[])

        assert len(engine.get_completed_tracks(7)) == 1
        l1 = engine.get_completed_tracks(7)[0]
        assert l1["track_id"] == 7
        assert l1["lifecycle_id"] is not None

        # Lifecycle 2 of Track 7 (reused ID)
        t2 = MockTrack(7, "person", [(200, 200), (200, 250), (200, 300)])
        engine.update(frame_idx=10, tracks=[t2], timestamp_sec=1.0)
        engine.update(frame_idx=11, tracks=[t2], timestamp_sec=1.1)
        # Disappears
        engine.update(frame_idx=12, tracks=[])
        engine.update(frame_idx=13, tracks=[])

        all_completed = engine.get_completed_tracks(7)
        assert len(all_completed) == 2
        l1_rec = all_completed[0]
        l2_rec = all_completed[1]

        assert l1_rec["track_id"] == 7
        assert l2_rec["track_id"] == 7
        assert l1_rec["lifecycle_id"] != l2_rec["lifecycle_id"]
        assert l1_rec["path_length_px"] == 40.0
        assert l2_rec["path_length_px"] == 100.0

        # Verify get_all_metrics includes both
        metrics_all = engine.get_all_metrics()
        lids = [m["lifecycle_id"] for m in metrics_all if m["track_id"] == 7]
        assert len(lids) == 2
        assert len(set(lids)) == 2

    def test_summary_cumulative_aggregates_preserved_after_pruning(self):
        """Fix 3: Cumulative session metrics are preserved even after history buffer pruning."""
        engine = BehaviorAnalytics(max_tracks_history=2, disappearance_grace_frames=1)

        # Create 5 distinct completed tracks with known lengths: 10, 20, 30, 40, 50 px
        lengths = [10.0, 20.0, 30.0, 40.0, 50.0]
        for tid, l_val in enumerate(lengths, start=1):
            t = MockTrack(tid, "car", [(0, 0), (l_val, 0)])
            engine.update(frame_idx=tid * 10, tracks=[t], timestamp_sec=float(tid))
            engine.update(frame_idx=tid * 10 + 1, tracks=[])

        summary = engine.get_summary()
        # All 5 completed
        assert summary["total_tracks_analyzed"] == 5
        assert summary["completed_tracks_count"] == 5
        assert summary["retained_completed_tracks_count"] == 2
        assert summary["retained_history_limit"] == 2

        # Average path length should be (10+20+30+40+50) / 5 = 30.0 px
        # (NOT (40+50)/2 = 45.0 px that would occur if computed from retained history)
        assert summary["avg_path_length_px"] == 30.0
        assert summary["class_breakdown"]["car"]["track_count"] == 5
        assert summary["class_breakdown"]["car"]["avg_path_length_px"] == 30.0

        # Retained history list contains exactly the last 2
        retained = engine.get_completed_tracks()
        assert len(retained) == 2
        assert [r["track_id"] for r in retained] == [4, 5]

    def test_empty_and_reset_session_safety(self):
        """Fix 3: Fresh and reset sessions return zero-division-safe defaults."""
        engine = BehaviorAnalytics()

        # Fresh summary
        s1 = engine.get_summary()
        assert s1["total_tracks_analyzed"] == 0
        assert s1["active_tracks_count"] == 0
        assert s1["completed_tracks_count"] == 0
        assert s1["retained_completed_tracks_count"] == 0
        assert s1["avg_path_length_px"] == 0.0
        assert s1["avg_net_displacement_px"] == 0.0
        assert s1["avg_tortuosity"] == 1.0
        assert s1["avg_speed_px_per_sec"] is None
        assert s1["total_direction_changes"] == 0
        assert s1["class_breakdown"] == {}
        assert engine.get_all_metrics() == []
        assert engine.get_completed_tracks() == []
        assert engine.get_track_metrics(99) is None

        # Add track, then reset
        t = MockTrack(1, "person", [(0, 0), (50, 0)])
        engine.update(frame_idx=1, tracks=[t], timestamp_sec=0.1)
        assert engine.get_summary()["total_tracks_analyzed"] == 1

        engine.reset()
        s2 = engine.get_summary()
        assert s2["total_tracks_analyzed"] == 0
        assert s2["avg_path_length_px"] == 0.0
        assert engine.get_all_metrics() == []


class TestTrajectoryBehaviorIntegration:
    """Validates video pipeline, webcam pipeline, CLI, and GUI integration of BehaviorAnalytics."""

    def test_run_video_tracking_populates_behavior_analytics(self):
        import app
        sample_video = str(PROJECT_ROOT / "assets" / "sample_clip.mp4")
        if not Path(sample_video).exists():
            pytest.skip("sample_clip.mp4 not present in assets")

        res = app.run_video_tracking(
            video_path=sample_video,
            max_frames=5
        )
        assert res["success"] is True
        assert res["frames_processed"] == 5
        assert "behavior_analytics" in res
        ba = res["behavior_analytics"]
        assert ba is not None
        assert ba["total_tracks_analyzed"] >= 1
        assert ba["active_tracks_count"] == 0  # Finalized!
        assert ba["completed_tracks_count"] >= 1
        assert ba["avg_path_length_px"] >= 0.0
        assert ba["avg_tortuosity"] >= 1.0

    def test_run_video_tracking_session_isolation(self):
        import app
        sample_video = str(PROJECT_ROOT / "assets" / "sample_clip.mp4")
        if not Path(sample_video).exists():
            pytest.skip("sample_clip.mp4 not present in assets")

        res1 = app.run_video_tracking(video_path=sample_video, max_frames=3)
        res2 = app.run_video_tracking(video_path=sample_video, max_frames=3)

        assert res1["behavior_analytics"]["total_tracks_analyzed"] == res2["behavior_analytics"]["total_tracks_analyzed"]
        assert res2["behavior_analytics"]["completed_tracks_count"] == res1["behavior_analytics"]["completed_tracks_count"]

    def test_concurrent_intelligence_spatial_and_behavior_analytics(self):
        import app
        sample_video = str(PROJECT_ROOT / "assets" / "sample_clip.mp4")
        if not Path(sample_video).exists():
            pytest.skip("sample_clip.mp4 not present in assets")

        res = app.run_video_tracking(
            video_path=sample_video,
            max_frames=5,
            virtual_line=(50, 240, 590, 240),
            roi_region=(120, 100, 520, 380)
        )
        assert res["success"] is True
        assert res["scene_intelligence"] is not None
        assert res["spatial_analytics"] is not None
        assert res["behavior_analytics"] is not None
        assert "natural_language_summary" in res["scene_intelligence"]
        assert "line_crossings" in res["spatial_analytics"]
        assert "roi_occupancy" in res["spatial_analytics"]
        assert "total_tracks_analyzed" in res["behavior_analytics"]

    def test_cli_print_behavior_summary_formatting(self, capsys):
        import app
        # 1. Populated with speed and class breakdown
        populated = {
            "total_tracks_analyzed": 3,
            "active_tracks_count": 0,
            "completed_tracks_count": 3,
            "avg_path_length_px": 150.5,
            "avg_net_displacement_px": 120.0,
            "avg_tortuosity": 1.25,
            "total_direction_changes": 2,
            "avg_speed_px_per_sec": 45.2,
            "class_breakdown": {
                "car": {
                    "track_count": 2,
                    "avg_path_length_px": 160.0,
                    "avg_net_displacement_px": 130.0,
                    "avg_tortuosity": 1.23,
                    "total_direction_changes": 1,
                    "avg_speed_px_per_sec": 50.0
                },
                "person": {
                    "track_count": 1,
                    "avg_path_length_px": 131.5,
                    "avg_net_displacement_px": 100.0,
                    "avg_tortuosity": 1.31,
                    "total_direction_changes": 1,
                    "avg_speed_px_per_sec": 35.6
                }
            }
        }
        app.print_behavior_summary(populated)
        captured = capsys.readouterr().out
        assert "Trajectory Behavior Analytics:" in captured
        assert "Tracks Analyzed: 3 total (3 completed, 0 active at termination)" in captured
        assert "Average Path Length:     150.5 px" in captured
        assert "Average Speed:           45.2 px/s" in captured
        assert "car" in captured
        assert "person" in captured

        # 2. Unavailable speed formatting
        no_speed = {
            "total_tracks_analyzed": 1,
            "active_tracks_count": 0,
            "completed_tracks_count": 1,
            "avg_path_length_px": 10.0,
            "avg_net_displacement_px": 10.0,
            "avg_tortuosity": 1.0,
            "total_direction_changes": 0,
            "avg_speed_px_per_sec": None,
            "class_breakdown": {}
        }
        app.print_behavior_summary(no_speed)
        captured2 = capsys.readouterr().out
        assert "Average Speed:           N/A (reliable timing unavailable)" in captured2

        # 3. Empty summary safety
        app.print_behavior_summary({})
        captured3 = capsys.readouterr().out
        assert captured3 == ""

    def test_gui_tracking_worker_behavior_analytics_lifecycle(self):
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

    def test_gui_app_has_behavior_cards(self):
        import tkinter as tk
        from src.gui import ObjectTrackerApp

        root = tk.Tk()
        root.withdraw()
        gui = ObjectTrackerApp(root)
        assert hasattr(gui, "lbl_behavior_motion")
        assert hasattr(gui, "lbl_behavior_dynamics")
        assert "Avg Len:" in gui.lbl_behavior_motion.cget("text")
        assert "Tort:" in gui.lbl_behavior_dynamics.cget("text")
        root.update()
        gui.on_close()



