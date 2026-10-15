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
