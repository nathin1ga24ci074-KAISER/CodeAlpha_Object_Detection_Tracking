"""
Unit and Integration Tests for Spatial Analytics Module.
Validates virtual line-crossing detection, rectangular ROI occupancy,
directional classifications, boundary hysteresis, duplicate event suppression,
track termination without false exits, coordinate validation, and session isolation.
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.spatial_analytics import (
    SpatialAnalytics,
    CrossingDirection,
    RegionState,
    ROIVisitStatus,
    ROIVisitRecord,
    SpatialEvent,
    validate_line_coords,
    validate_roi_coords,
    parse_spatial_coords_string
)
import app


class MockTrack:
    """Lightweight duck-typed object representing a tracked detection."""
    def __init__(self, track_id: int, class_name: str, centroid: tuple, box: list = None):
        self.track_id = int(track_id)
        self.class_name = str(class_name)
        self.current_centroid = (int(centroid[0]), int(centroid[1]))
        self.box = box or [centroid[0] - 15, centroid[1] - 15, centroid[0] + 15, centroid[1] + 15]
        self.confidence = 0.90


class TestSpatialAnalyticsGeometry:
    """Validates coordinate checking, string parsing, and boundary validation."""

    def test_valid_line_coordinates(self):
        coords = validate_line_coords((10, 20, 100, 200), frame_width=640, frame_height=480)
        assert coords == (10, 20, 100, 200)

    def test_zero_length_line_rejected(self):
        with pytest.raises(ValueError, match="zero length"):
            validate_line_coords((50, 50, 50, 50))

    def test_line_coordinates_out_of_bounds(self):
        with pytest.raises(ValueError, match="extend outside frame bounds"):
            validate_line_coords((10, 20, 700, 200), frame_width=640, frame_height=480)

    def test_invalid_line_length(self):
        with pytest.raises(ValueError, match="exactly 4 integers"):
            validate_line_coords((10, 20, 30))  # type: ignore

    def test_valid_roi_coordinates(self):
        coords = validate_roi_coords((50, 50, 300, 250), frame_width=640, frame_height=480)
        assert coords == (50, 50, 300, 250)

    def test_inverted_or_zero_roi_rejected(self):
        with pytest.raises(ValueError, match="x1 < x2 and y1 < y2"):
            validate_roi_coords((300, 50, 100, 250))
        with pytest.raises(ValueError, match="x1 < x2 and y1 < y2"):
            validate_roi_coords((100, 250, 300, 50))

    def test_negative_roi_rejected(self):
        with pytest.raises(ValueError, match="cannot be negative"):
            validate_roi_coords((-10, 50, 200, 250))

    def test_roi_out_of_bounds(self):
        with pytest.raises(ValueError, match="exceeds frame dimensions"):
            validate_roi_coords((50, 50, 700, 500), frame_width=640, frame_height=480)

    def test_parse_spatial_coords_string(self):
        res1 = parse_spatial_coords_string("100,200,500,200")
        assert res1 == (100, 200, 500, 200)
        res2 = parse_spatial_coords_string(" 100  200  500  200 ")
        assert res2 == (100, 200, 500, 200)

        with pytest.raises(ValueError, match="Expected 4 coordinate values"):
            parse_spatial_coords_string("100,200")
        with pytest.raises(ValueError, match="must be integers"):
            parse_spatial_coords_string("100,abc,300,400")

    def test_parse_spatial_coords_whole_valued_floats(self):
        res = parse_spatial_coords_string("100.0, 200.0, 500.0, 200.0")
        assert res == (100, 200, 500, 200)
        res_mixed = parse_spatial_coords_string("100  200.0  500  200.0")
        assert res_mixed == (100, 200, 500, 200)

    def test_parse_spatial_coords_fractional_rejected(self):
        with pytest.raises(ValueError, match="fractional"):
            parse_spatial_coords_string("100.5, 200.0, 500.0, 200.0")

    def test_parse_spatial_coords_non_finite_rejected(self):
        with pytest.raises(ValueError, match="non-finite"):
            parse_spatial_coords_string("nan, 200, 500, 200")
        with pytest.raises(ValueError, match="non-finite"):
            parse_spatial_coords_string("100, inf, 500, 200")
        with pytest.raises(ValueError, match="non-finite"):
            parse_spatial_coords_string("100, -inf, 500, 200")


class TestLineCrossingAnalytics:
    """Validates virtual line-crossing detection, directional classification, and hysteresis."""

    def test_no_tracked_objects_handling(self):
        analytics = SpatialAnalytics(virtual_line=(100, 200, 500, 200))
        res = analytics.update(frame_idx=1, tracks=[], timestamp_sec=0.033)
        assert res["line_crossings_total"] == 0
        assert res["new_events"] == []
        assert res["total_events_logged"] == 0

    def test_valid_crossing_top_to_bottom(self):
        # Horizontal line at y=200, from x=100 to 500
        analytics = SpatialAnalytics(
            virtual_line=(100, 200, 500, 200),
            line_hysteresis_px=8.0,
            line_cooldown_frames=2
        )
        # Frame 1: Object solidly above line (y=150) -> Side B
        t1 = MockTrack(track_id=1, class_name="person", centroid=(300, 150))
        res1 = analytics.update(frame_idx=1, tracks=[t1], timestamp_sec=0.033)
        assert res1["line_crossings_total"] == 0

        # Frame 2: Object moves solidly below line (y=250) -> Side A
        t2 = MockTrack(track_id=1, class_name="person", centroid=(300, 250))
        res2 = analytics.update(frame_idx=2, tracks=[t2], timestamp_sec=0.066)

        assert res2["line_crossings_total"] == 1
        assert len(res2["new_events"]) == 1
        ev = res2["new_events"][0]
        assert ev["event_type"] == "LINE_CROSS"
        assert ev["track_id"] == 1
        assert ev["class_name"] == "person"
        assert ev["details"]["direction"] == CrossingDirection.B_TO_A
        assert ev["details"]["screen_direction"] == CrossingDirection.TOP_TO_BOTTOM

    def test_valid_crossing_bottom_to_top(self):
        analytics = SpatialAnalytics(
            virtual_line=(100, 200, 500, 200),
            line_hysteresis_px=8.0,
            line_cooldown_frames=2
        )
        # Frame 1: Below line (y=260) -> Side A
        t1 = MockTrack(track_id=2, class_name="car", centroid=(300, 260))
        analytics.update(frame_idx=1, tracks=[t1])

        # Frame 2: Moves above line (y=140) -> Side B
        t2 = MockTrack(track_id=2, class_name="car", centroid=(300, 140))
        res2 = analytics.update(frame_idx=2, tracks=[t2])

        assert res2["line_crossings_total"] == 1
        ev = res2["new_events"][0]
        assert ev["details"]["direction"] == CrossingDirection.A_TO_B
        assert ev["details"]["screen_direction"] == CrossingDirection.BOTTOM_TO_TOP
        assert res2["line_crossings_by_class"]["car"] == 1

    def test_approach_and_reverse_without_crossing_not_counted(self):
        analytics = SpatialAnalytics(
            virtual_line=(100, 200, 500, 200),
            line_hysteresis_px=10.0
        )
        # Frame 1: At y=150 (above line)
        analytics.update(frame_idx=1, tracks=[MockTrack(1, "person", (300, 150))])
        # Frame 2: Approaches line at y=195 (within 10px hysteresis zone)
        analytics.update(frame_idx=2, tracks=[MockTrack(1, "person", (300, 195))])
        # Frame 3: Reverses direction back to y=160 (without ever reaching opposite solid side)
        res3 = analytics.update(frame_idx=3, tracks=[MockTrack(1, "person", (300, 160))])

        # Must NOT count as crossing
        assert res3["line_crossings_total"] == 0
        assert len(analytics.get_events(event_type="LINE_CROSS")) == 0

    def test_boundary_jitter_does_not_trigger_repeated_crossings(self):
        analytics = SpatialAnalytics(
            virtual_line=(100, 200, 500, 200),
            line_hysteresis_px=10.0,
            line_cooldown_frames=5
        )
        # Initial point on side B
        analytics.update(frame_idx=1, tracks=[MockTrack(1, "person", (300, 150))])

        # Hover right around the line within hysteresis strip (195 to 205)
        jitter_points = [(300, 202), (300, 198), (300, 204), (300, 199)]
        for idx, pt in enumerate(jitter_points, start=2):
            res = analytics.update(frame_idx=idx, tracks=[MockTrack(1, "person", pt)])
            assert res["line_crossings_total"] == 0


class TestRegionAnalytics:
    """Validates rectangular ROI entry, exit, boundary hysteresis, and disappearance handling."""

    def test_roi_entry_and_exit_cycle(self):
        # ROI: x from 100 to 400, y from 100 to 400
        analytics = SpatialAnalytics(
            roi_region=(100, 100, 400, 400),
            roi_hysteresis_px=8.0,
            roi_cooldown_frames=2
        )

        # Frame 1: Object is solidly outside (x=50, y=200)
        t1 = MockTrack(1, "person", (50, 200))
        res1 = analytics.update(frame_idx=1, tracks=[t1])
        assert res1["roi_current_occupancy"] == 0
        assert res1["roi_entries_total"] == 0

        # Frame 2: Object moves solidly inside (x=200, y=200)
        t2 = MockTrack(1, "person", (200, 200))
        res2 = analytics.update(frame_idx=2, tracks=[t2])
        assert res2["roi_current_occupancy"] == 1
        assert res2["roi_entries_total"] == 1
        assert len(res2["new_events"]) == 1
        assert res2["new_events"][0]["event_type"] == "ROI_ENTRY"

        # Frame 3: Object moves solidly outside (x=450, y=200)
        t3 = MockTrack(1, "person", (450, 200))
        res3 = analytics.update(frame_idx=3, tracks=[t3])
        assert res3["roi_current_occupancy"] == 0
        assert res3["roi_exits_total"] == 1
        assert res3["new_events"][0]["event_type"] == "ROI_EXIT"

    def test_objects_remaining_inside_no_duplicate_entries(self):
        analytics = SpatialAnalytics(roi_region=(100, 100, 400, 400))
        # Enters on frame 1
        analytics.update(frame_idx=1, tracks=[MockTrack(1, "dog", (250, 250))])

        # Stays inside across frames 2, 3, 4
        for idx in range(2, 5):
            res = analytics.update(frame_idx=idx, tracks=[MockTrack(1, "dog", (260 + idx, 250))])
            assert res["roi_current_occupancy"] == 1
            assert res["roi_entries_total"] == 1
            entry_events = [e for e in res["new_events"] if e["event_type"] == "ROI_ENTRY"]
            assert len(entry_events) == 0

    def test_temporary_detection_loss_grace_period(self):
        analytics = SpatialAnalytics(
            roi_region=(100, 100, 400, 400),
            disappearance_grace_frames=4
        )
        t = MockTrack(1, "person", (200, 200))
        analytics.update(frame_idx=1, tracks=[t])
        assert analytics.get_summary()["roi_occupancy"]["current_inside"] == 1

        # Missed for 2 frames (within grace period)
        analytics.update(frame_idx=2, tracks=[])
        analytics.update(frame_idx=3, tracks=[])

        # Reappears inside
        res4 = analytics.update(frame_idx=4, tracks=[t])
        assert res4["roi_current_occupancy"] == 1
        assert res4["roi_entries_total"] == 1  # No duplicate entry
        assert res4["roi_exits_total"] == 0    # No false exit

    def test_track_termination_inside_does_not_emit_false_exit(self):
        """CRITICAL: Object terminated inside ROI must NOT trigger an ROI_EXIT event."""
        analytics = SpatialAnalytics(
            roi_region=(100, 100, 400, 400),
            disappearance_grace_frames=3
        )
        analytics.update(frame_idx=1, tracks=[MockTrack(1, "bicycle", (250, 250))])
        assert analytics.get_summary()["roi_occupancy"]["entries_total"] == 1

        # Object stops being tracked (exceeds grace period of 3 frames)
        analytics.update(frame_idx=2, tracks=[])
        analytics.update(frame_idx=3, tracks=[])
        analytics.update(frame_idx=4, tracks=[])
        res5 = analytics.update(frame_idx=5, tracks=[])

        # Must NOT count as geometric exit
        assert res5["roi_exits_total"] == 0
        assert res5["roi_current_occupancy"] == 0
        assert res5["roi_terminated_inside"] == 1
        exit_events = analytics.get_events(event_type="ROI_EXIT")
        assert len(exit_events) == 0

    def test_stream_finalization_marks_terminated_without_false_exit(self):
        analytics = SpatialAnalytics(roi_region=(100, 100, 400, 400))
        analytics.update(frame_idx=1, tracks=[MockTrack(5, "person", (200, 200))])

        analytics.finalize_stream(last_frame_idx=10, timestamp_sec=0.33)
        summary = analytics.get_summary()["roi_occupancy"]
        assert summary["exits_total"] == 0
        assert summary["terminated_inside"] == 1
        assert summary["current_inside"] == 0

    def test_terminated_inside_track_state_recovery_and_reentry(self):
        """
        Validates that a track terminated inside ROI can recover to OUTSIDE when observed
        solidly outside without emitting a false ROI_EXIT, and subsequent re-entry counts cleanly.
        """
        analytics = SpatialAnalytics(
            roi_region=(100, 100, 400, 400),
            disappearance_grace_frames=3
        )
        # 1. Track enters ROI at frame 1
        analytics.update(frame_idx=1, tracks=[MockTrack(10, "person", (250, 250))])
        assert analytics.get_summary()["roi_occupancy"]["entries_total"] == 1
        assert analytics.get_summary()["roi_occupancy"]["current_inside"] == 1

        # 2. Track terminates while inside (unseen for 3 frames >= grace period)
        analytics.update(frame_idx=2, tracks=[])
        analytics.update(frame_idx=3, tracks=[])
        analytics.update(frame_idx=4, tracks=[])
        res5 = analytics.update(frame_idx=5, tracks=[])
        assert res5["roi_current_occupancy"] == 0
        assert res5["roi_terminated_inside"] == 1
        assert res5["roi_exits_total"] == 0

        # 3. Same track ID 10 is later observed solidly outside (x=50, y=250)
        res6 = analytics.update(frame_idx=15, tracks=[MockTrack(10, "person", (50, 250))])
        # State must recover to OUTSIDE without generating false ROI_EXIT
        assert res6["roi_current_occupancy"] == 0
        assert res6["roi_exits_total"] == 0
        assert len(analytics.get_events(event_type="ROI_EXIT")) == 0

        # 4. Same track ID 10 subsequently re-enters the ROI (x=250, y=250)
        res7 = analytics.update(frame_idx=20, tracks=[MockTrack(10, "person", (250, 250))])
        assert res7["roi_current_occupancy"] == 1
        assert res7["roi_entries_total"] == 2
        assert len(res7["new_events"]) == 1
        assert res7["new_events"][0]["event_type"] == "ROI_ENTRY"
        assert res7["new_events"][0]["track_id"] == 10

        # Invariant check: entries (2) == current (1) + exits (0) + terminated (1)
        summary = analytics.get_summary()["roi_occupancy"]
        assert summary["entries_total"] == summary["current_inside"] + summary["exits_total"] + summary["terminated_inside"]



class TestSessionIsolationAndTelemetry:
    """Validates session resets, multi-class counters, bounded history, and overlay drawing."""

    def test_multi_class_and_multi_id_counters(self):
        analytics = SpatialAnalytics(
            virtual_line=(100, 200, 500, 200),
            roi_region=(100, 100, 400, 400)
        )
        # Track 1 (person), Track 2 (car), Track 3 (person)
        t1 = MockTrack(1, "person", (200, 150))
        t2 = MockTrack(2, "car", (300, 150))
        t3 = MockTrack(3, "person", (50, 250))
        analytics.update(frame_idx=1, tracks=[t1, t2, t3])

        # Track 1 & 2 cross line and enter ROI
        t1_b = MockTrack(1, "person", (200, 250))
        t2_b = MockTrack(2, "car", (300, 250))
        res2 = analytics.update(frame_idx=2, tracks=[t1_b, t2_b, t3])

        assert res2["line_crossings_total"] == 2
        assert res2["line_crossings_by_class"]["person"] == 1
        assert res2["line_crossings_by_class"]["car"] == 1
        assert res2["roi_entries_total"] == 2
        assert res2["roi_current_occupancy"] == 2
        assert res2["roi_current_classes"] == {"person": 1, "car": 1}

    def test_bounded_event_history(self):
        analytics = SpatialAnalytics(
            virtual_line=(100, 200, 500, 200),
            max_events_history=5
        )
        # Generate 8 events by crossing back and forth
        for i in range(1, 9):
            y_pos = 250 if i % 2 == 0 else 150
            t = MockTrack(1, "person", (300, y_pos))
            analytics.update(frame_idx=i * 10, tracks=[t])

        # Events list must not exceed max_events_history
        assert len(analytics.get_events()) <= 5

    def test_session_reset_and_isolation(self):
        inst_a = SpatialAnalytics(virtual_line=(100, 200, 500, 200))
        inst_b = SpatialAnalytics(virtual_line=(100, 200, 500, 200))

        inst_a.update(frame_idx=1, tracks=[MockTrack(1, "person", (300, 150))])
        inst_a.update(frame_idx=2, tracks=[MockTrack(1, "person", (300, 250))])
        assert inst_a.get_summary()["line_crossings"]["total"] == 1

        # Instance B must be isolated
        assert inst_b.get_summary()["line_crossings"]["total"] == 0

        # Reset Instance A
        inst_a.reset()
        assert inst_a.get_summary()["line_crossings"]["total"] == 0
        assert inst_a.get_summary()["total_events_logged"] == 0

    def test_draw_overlays_on_synthetic_frame(self):
        analytics = SpatialAnalytics(
            virtual_line=(100, 200, 500, 200),
            roi_region=(150, 100, 450, 350)
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = analytics.draw_overlays(frame)
        assert annotated is not None
        assert annotated.shape == (480, 640, 3)
        # Should not be entirely black anymore due to drawn line and ROI
        assert np.any(annotated > 0)

    def test_run_video_tracking_with_spatial_analytics(self):
        sample_video = str(PROJECT_ROOT / "assets" / "sample_clip.mp4")
        if not Path(sample_video).exists():
            pytest.skip("sample_clip.mp4 not present")

        res = app.run_video_tracking(
            video_path=sample_video,
            max_frames=5,
            virtual_line=(50, 240, 590, 240),
            roi_region=(120, 100, 520, 380)
        )
        assert res["success"] is True
        assert res["frames_processed"] == 5
        assert "spatial_analytics" in res
        assert res["spatial_analytics"] is not None
        assert "line_crossings" in res["spatial_analytics"]
        assert "roi_occupancy" in res["spatial_analytics"]


class TestROIDwellAndPeakOccupancy:
    """
    Validates Stage 1: ROI dwell time analytics, peak occupancy tracking,
    bounded visit history, multi-class averages, and graceful stream finalization.
    """

    def test_roi_entry_creates_in_progress_visit(self):
        analytics = SpatialAnalytics(roi_region=(100, 100, 400, 400))
        t = MockTrack(1, "person", (200, 200))
        res = analytics.update(frame_idx=10, tracks=[t], timestamp_sec=1.5)

        assert res["roi_current_occupancy"] == 1
        active = analytics.get_active_roi_visits()
        assert len(active) == 1
        v = active[0]
        assert v["track_id"] == 1
        assert v["class_name"] == "person"
        assert v["status"] == ROIVisitStatus.IN_PROGRESS
        assert v["entry_frame"] == 10
        assert v["entry_timestamp"] == 1.5
        assert v["exit_frame"] is None
        assert v["exit_timestamp"] is None
        assert v["dwell_duration_sec"] == 0.0

        # At subsequent frame 20 (ts=2.5), elapsed duration is 1.0s
        analytics.update(frame_idx=20, tracks=[t], timestamp_sec=2.5)
        dwell_sum = analytics.get_dwell_summary()
        assert dwell_sum["current_inside_count"] == 1
        assert dwell_sum["active_visits"][0]["dwell_duration_sec"] == 1.0

    def test_geometric_exit_completes_visit_with_accurate_dwell(self):
        analytics = SpatialAnalytics(roi_region=(100, 100, 400, 400))
        # Entry at frame 10 (1.0s)
        analytics.update(frame_idx=10, tracks=[MockTrack(2, "car", (250, 250))], timestamp_sec=1.0)
        # Sustained inside at frame 20 (2.5s)
        analytics.update(frame_idx=20, tracks=[MockTrack(2, "car", (260, 250))], timestamp_sec=2.5)
        # Geometric exit at frame 30 (4.25s)
        res3 = analytics.update(frame_idx=30, tracks=[MockTrack(2, "car", (500, 250))], timestamp_sec=4.25)

        assert res3["roi_current_occupancy"] == 0
        assert res3["roi_exits_total"] == 1
        assert res3["roi_completed_visits"] == 1
        assert res3["roi_avg_dwell_sec"] == 3.25

        active = analytics.get_active_roi_visits()
        assert len(active) == 0

        completed = analytics.get_completed_roi_visits()
        assert len(completed) == 1
        c = completed[0]
        assert c["track_id"] == 2
        assert c["class_name"] == "car"
        assert c["status"] == ROIVisitStatus.COMPLETED
        assert c["entry_frame"] == 10
        assert c["entry_timestamp"] == 1.0
        assert c["exit_frame"] == 30
        assert c["exit_timestamp"] == 4.25
        assert c["dwell_duration_sec"] == 3.25

    def test_peak_occupancy_tracks_concurrent_objects_and_preserves_peak(self):
        analytics = SpatialAnalytics(roi_region=(100, 100, 400, 400))
        assert analytics.get_dwell_summary()["peak_occupancy"] == 0
        assert analytics.get_dwell_summary()["peak_frame"] is None

        # Frame 1: 1 object inside
        res1 = analytics.update(frame_idx=1, tracks=[MockTrack(1, "person", (200, 200))], timestamp_sec=0.1)
        assert res1["roi_peak_occupancy"] == 1
        assert res1["roi_peak_frame"] == 1

        # Frame 2: 3 objects inside concurrently
        res2 = analytics.update(
            frame_idx=2,
            tracks=[
                MockTrack(1, "person", (200, 200)),
                MockTrack(2, "person", (220, 220)),
                MockTrack(3, "person", (240, 240))
            ],
            timestamp_sec=0.2
        )
        assert res2["roi_peak_occupancy"] == 3
        assert res2["roi_peak_frame"] == 2
        assert res2["roi_peak_timestamp"] == 0.2

        # Frame 3: 2 objects exit (only 1 remains)
        res3 = analytics.update(
            frame_idx=3,
            tracks=[
                MockTrack(1, "person", (200, 200)),
                MockTrack(2, "person", (50, 220)),
                MockTrack(3, "person", (50, 240))
            ],
            timestamp_sec=0.3
        )
        assert res3["roi_current_occupancy"] == 1
        # Peak remains 3 at frame 2
        assert res3["roi_peak_occupancy"] == 3
        assert res3["roi_peak_frame"] == 2
        assert res3["roi_peak_timestamp"] == 0.2

        summary = analytics.get_summary()["roi_occupancy"]
        assert summary["peak_occupancy"] == 3
        assert summary["peak_frame"] == 2
        assert summary["peak_timestamp"] == 0.2

    def test_completed_average_excludes_in_progress_and_terminated_visits(self):
        analytics = SpatialAnalytics(roi_region=(100, 100, 400, 400), disappearance_grace_frames=2)
        # Track 1 enters at 1.0s, exits at 5.0s -> completed dwell 4.0s
        analytics.update(frame_idx=1, tracks=[MockTrack(1, "car", (200, 200))], timestamp_sec=1.0)
        analytics.update(frame_idx=5, tracks=[MockTrack(1, "car", (50, 200))], timestamp_sec=5.0)

        # Track 2 enters at 6.0s, remains inside (in-progress)
        analytics.update(frame_idx=6, tracks=[MockTrack(2, "car", (220, 220))], timestamp_sec=6.0)

        # Track 3 enters at 7.0s, disappears inside (terminated)
        analytics.update(
            frame_idx=7,
            tracks=[MockTrack(2, "car", (220, 220)), MockTrack(3, "car", (240, 240))],
            timestamp_sec=7.0
        )
        analytics.update(frame_idx=8, tracks=[MockTrack(2, "car", (220, 220))], timestamp_sec=8.0)
        analytics.update(frame_idx=9, tracks=[MockTrack(2, "car", (220, 220))], timestamp_sec=9.0)
        analytics.update(frame_idx=10, tracks=[MockTrack(2, "car", (220, 220))], timestamp_sec=10.0)

        dwell_sum = analytics.get_dwell_summary()
        assert dwell_sum["completed_visits_count"] == 1
        assert dwell_sum["avg_completed_dwell_sec"] == 4.0
        assert dwell_sum["total_completed_dwell_sec"] == 4.0
        assert dwell_sum["current_inside_count"] == 1  # Track 2
        assert dwell_sum["terminated_inside_count"] == 1  # Track 3
        assert len(analytics.get_active_roi_visits()) == 1
        assert len(analytics.get_terminated_roi_visits()) == 1

    def test_track_disappearance_creates_terminated_inside_visit(self):
        analytics = SpatialAnalytics(roi_region=(100, 100, 400, 400), disappearance_grace_frames=3)
        # Enters at frame 10 (1.0s), last seen at frame 12 (1.2s)
        analytics.update(frame_idx=10, tracks=[MockTrack(5, "truck", (200, 200))], timestamp_sec=1.0)
        analytics.update(frame_idx=11, tracks=[MockTrack(5, "truck", (205, 205))], timestamp_sec=1.1)
        analytics.update(frame_idx=12, tracks=[MockTrack(5, "truck", (210, 210))], timestamp_sec=1.2)

        # Disappears for 3 frames >= grace period
        analytics.update(frame_idx=13, tracks=[], timestamp_sec=1.3)
        analytics.update(frame_idx=14, tracks=[], timestamp_sec=1.4)
        res15 = analytics.update(frame_idx=15, tracks=[], timestamp_sec=1.5)

        assert res15["roi_exits_total"] == 0
        assert res15["roi_terminated_inside"] == 1
        assert res15["roi_completed_visits"] == 0

        terminated = analytics.get_terminated_roi_visits()
        assert len(terminated) == 1
        t = terminated[0]
        assert t["track_id"] == 5
        assert t["class_name"] == "truck"
        assert t["status"] == ROIVisitStatus.TERMINATED_INSIDE
        assert t["entry_frame"] == 10
        assert t["entry_timestamp"] == 1.0
        assert t["exit_frame"] == 12
        assert t["exit_timestamp"] == 1.2
        assert t["dwell_duration_sec"] == 0.2
        assert len(analytics.get_completed_roi_visits()) == 0

    def test_stream_finalization_finalizes_in_progress_visits(self):
        analytics = SpatialAnalytics(roi_region=(100, 100, 400, 400))
        analytics.update(frame_idx=1, tracks=[MockTrack(1, "person", (200, 200))], timestamp_sec=0.1)
        analytics.update(
            frame_idx=2,
            tracks=[MockTrack(1, "person", (200, 200)), MockTrack(2, "dog", (250, 250))],
            timestamp_sec=0.2
        )

        assert len(analytics.get_active_roi_visits()) == 2
        analytics.finalize_stream(last_frame_idx=20, timestamp_sec=2.0)

        assert len(analytics.get_active_roi_visits()) == 0
        assert analytics.get_summary()["roi_occupancy"]["current_inside"] == 0
        assert analytics.get_summary()["roi_occupancy"]["terminated_inside"] == 2
        assert analytics.get_summary()["roi_occupancy"]["exits_total"] == 0

        term_visits = analytics.get_terminated_roi_visits()
        assert len(term_visits) == 2
        for v in term_visits:
            assert v["status"] == ROIVisitStatus.TERMINATED_INSIDE
            assert v["exit_frame"] == 20
            assert v["exit_timestamp"] == 2.0

    def test_repeated_stream_finalization_is_idempotent(self):
        analytics = SpatialAnalytics(roi_region=(100, 100, 400, 400))
        analytics.update(frame_idx=1, tracks=[MockTrack(1, "person", (200, 200))], timestamp_sec=0.1)

        analytics.finalize_stream(last_frame_idx=10, timestamp_sec=1.0)
        assert analytics.get_summary()["roi_occupancy"]["terminated_inside"] == 1
        assert len(analytics.get_terminated_roi_visits()) == 1

        # Second finalize call must do nothing
        analytics.finalize_stream(last_frame_idx=10, timestamp_sec=1.0)
        assert analytics.get_summary()["roi_occupancy"]["terminated_inside"] == 1
        assert len(analytics.get_terminated_roi_visits()) == 1

    def test_terminated_inside_recovery_and_reentry_creates_new_visit(self):
        analytics = SpatialAnalytics(roi_region=(100, 100, 400, 400), disappearance_grace_frames=2)
        # 1. Track 10 enters at 1.0s, last seen at 2.0s
        analytics.update(frame_idx=1, tracks=[MockTrack(10, "person", (200, 200))], timestamp_sec=1.0)
        analytics.update(frame_idx=2, tracks=[MockTrack(10, "person", (200, 200))], timestamp_sec=2.0)

        # Disappears and terminates
        analytics.update(frame_idx=3, tracks=[], timestamp_sec=3.0)
        analytics.update(frame_idx=4, tracks=[], timestamp_sec=4.0)
        analytics.update(frame_idx=5, tracks=[], timestamp_sec=5.0)

        assert analytics.get_summary()["roi_occupancy"]["terminated_inside"] == 1
        term_visits = analytics.get_terminated_roi_visits()
        assert len(term_visits) == 1
        assert term_visits[0]["status"] == ROIVisitStatus.TERMINATED_INSIDE

        # 2. Track 10 recovered solidly outside at 10.0s
        analytics.update(frame_idx=10, tracks=[MockTrack(10, "person", (50, 200))], timestamp_sec=10.0)

        # 3. Track 10 re-enters at 12.0s
        analytics.update(frame_idx=12, tracks=[MockTrack(10, "person", (200, 200))], timestamp_sec=12.0)
        assert len(analytics.get_active_roi_visits()) == 1
        v2 = analytics.get_active_roi_visits()[0]
        assert v2["track_id"] == 10
        assert v2["entry_frame"] == 12
        assert v2["entry_timestamp"] == 12.0

        # 4. Track 10 exits at 15.0s
        analytics.update(frame_idx=15, tracks=[MockTrack(10, "person", (50, 200))], timestamp_sec=15.0)
        assert len(analytics.get_active_roi_visits()) == 0
        comp_visits = analytics.get_completed_roi_visits()
        assert len(comp_visits) == 1
        assert comp_visits[0]["dwell_duration_sec"] == 3.0

        # History integrity: 1 terminated visit, 1 completed visit, 0 active
        summary = analytics.get_summary()["roi_occupancy"]
        assert summary["entries_total"] == 2
        assert summary["exits_total"] == 1
        assert summary["terminated_inside"] == 1
        assert summary["current_inside"] == 0

    def test_multi_class_separate_dwell_averages(self):
        analytics = SpatialAnalytics(roi_region=(100, 100, 400, 400))
        # Person 1 dwells 2.0s
        analytics.update(frame_idx=1, tracks=[MockTrack(1, "person", (200, 200))], timestamp_sec=1.0)
        analytics.update(frame_idx=2, tracks=[MockTrack(1, "person", (50, 200))], timestamp_sec=3.0)

        # Person 2 dwells 4.0s
        analytics.update(frame_idx=3, tracks=[MockTrack(2, "person", (200, 200))], timestamp_sec=4.0)
        analytics.update(frame_idx=4, tracks=[MockTrack(2, "person", (50, 200))], timestamp_sec=8.0)

        # Car 3 dwells 10.0s
        analytics.update(frame_idx=5, tracks=[MockTrack(3, "car", (200, 200))], timestamp_sec=10.0)
        analytics.update(frame_idx=6, tracks=[MockTrack(3, "car", (50, 200))], timestamp_sec=20.0)

        dwell_sum = analytics.get_dwell_summary()
        assert dwell_sum["completed_visits_count"] == 3
        # Person avg: (2.0 + 4.0) / 2 = 3.0s
        assert dwell_sum["avg_completed_dwell_by_class"]["person"] == 3.0
        # Car avg: 10.0s
        assert dwell_sum["avg_completed_dwell_by_class"]["car"] == 10.0
        # Overall avg: (2.0 + 4.0 + 10.0) / 3 = 16.0 / 3 = 5.33s
        assert dwell_sum["avg_completed_dwell_sec"] == 5.33
        assert dwell_sum["min_completed_dwell_sec"] == 2.0
        assert dwell_sum["max_completed_dwell_sec"] == 10.0

    def test_timestamp_fallback_and_negative_duration_guard(self):
        analytics = SpatialAnalytics(roi_region=(100, 100, 400, 400))
        # Frame 30, fps=30.0 -> ts=1.0
        res1 = analytics.update(
            frame_idx=30,
            tracks=[MockTrack(1, "person", (200, 200))],
            timestamp_sec=None,
            fps=30.0
        )
        assert res1["timestamp_sec"] == 1.0

        # Frame 60, non-finite NaN timestamp falls back to 60/30 = 2.0
        res2 = analytics.update(
            frame_idx=60,
            tracks=[MockTrack(1, "person", (200, 200))],
            timestamp_sec=float("nan"),
            fps=30.0
        )
        assert res2["timestamp_sec"] == 2.0

        # Negative timestamp clamped / falls back to frame_idx / fps
        res3 = analytics.update(
            frame_idx=90,
            tracks=[MockTrack(1, "person", (50, 200))],
            timestamp_sec=-5.0,
            fps=30.0
        )
        assert res3["timestamp_sec"] == 3.0
        assert res3["roi_completed_visits"] == 1
        assert res3["roi_avg_dwell_sec"] == 2.0  # 3.0s - 1.0s = 2.0s

    def test_session_reset_clears_dwell_and_peak_state(self):
        analytics = SpatialAnalytics(roi_region=(100, 100, 400, 400))
        analytics.update(frame_idx=1, tracks=[MockTrack(1, "person", (200, 200))], timestamp_sec=1.0)
        analytics.update(frame_idx=2, tracks=[MockTrack(1, "person", (50, 200))], timestamp_sec=3.0)
        analytics.update(frame_idx=3, tracks=[MockTrack(2, "car", (200, 200))], timestamp_sec=4.0)

        dwell_before = analytics.get_dwell_summary()
        assert dwell_before["peak_occupancy"] == 1
        assert dwell_before["completed_visits_count"] == 1
        assert dwell_before["current_inside_count"] == 1

        analytics.reset()
        dwell_after = analytics.get_dwell_summary()
        assert dwell_after["peak_occupancy"] == 0
        assert dwell_after["peak_frame"] is None
        assert dwell_after["peak_timestamp"] is None
        assert dwell_after["completed_visits_count"] == 0
        assert dwell_after["total_completed_dwell_sec"] == 0.0
        assert dwell_after["avg_completed_dwell_sec"] == 0.0
        assert dwell_after["current_inside_count"] == 0
        assert dwell_after["active_visits"] == []
        assert dwell_after["recent_completed_visits"] == []
        assert dwell_after["recent_terminated_visits"] == []

    def test_bounded_history_pruning_preserves_cumulative_metrics(self):
        # Configure max_visits_history=2
        analytics = SpatialAnalytics(roi_region=(100, 100, 400, 400), max_visits_history=2)

        # Complete 5 visits of 3.0s each
        for i in range(1, 6):
            analytics.update(
                frame_idx=i * 10,
                tracks=[MockTrack(i, "person", (200, 200))],
                timestamp_sec=float(i * 10)
            )
            analytics.update(
                frame_idx=i * 10 + 3,
                tracks=[MockTrack(i, "person", (50, 200))],
                timestamp_sec=float(i * 10 + 3)
            )

        # Recent history list is bounded to 2
        assert len(analytics.get_completed_roi_visits()) == 2
        assert len(analytics._completed_roi_visits) == 2

        # Cumulative running metrics are NOT truncated or corrupted!
        dwell_sum = analytics.get_dwell_summary()
        assert dwell_sum["completed_visits_count"] == 5
        assert dwell_sum["total_completed_dwell_sec"] == 15.0
        assert dwell_sum["avg_completed_dwell_sec"] == 3.0
        assert dwell_sum["avg_completed_dwell_by_class"]["person"] == 3.0
