"""
Unit and integration tests for the Web Backend & Dashboard (Phase 3).
Verifies Flask application factory, REST endpoints, input validation,
MJPEG streaming, telemetry, upload security, export security,
worker lifecycle, idle watchdog, and error handling without requiring real
camera hardware or downloading neural network models.
"""

import sys
import io
import time
import math
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.web.stream_worker import WebStreamWorker, StreamSessionManager, create_standby_frame
from src.web.app_server import create_app, validate_start_payload
import web_app


@pytest.fixture
def app(tmp_path):
    """Creates an isolated Flask test application instance with dedicated session manager and temp folders."""
    uploads_dir = tmp_path / "uploads"
    outputs_dir = tmp_path / "outputs"
    uploads_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    session_mgr = StreamSessionManager()
    test_app = create_app({
        "TESTING": True,
        "STREAM_SESSION_MANAGER": session_mgr,
        "UPLOAD_FOLDER": str(uploads_dir.resolve()),
        "OUTPUT_FOLDER": str(outputs_dir.resolve()),
    })
    yield test_app
    # Teardown: ensure any active worker is cleanly stopped
    session_mgr.stop_stream()


@pytest.fixture
def client(app):
    """Provides a Flask test client for HTTP requests."""
    return app.test_client()


class TestWebPackageIntegrity:
    """Tests that importing web components does not trigger hardware or heavy ML loading."""

    def test_import_does_not_initialize_hardware_or_yolo(self):
        """Verifies package import does not initialize camera or YOLO weights."""
        with patch("cv2.VideoCapture") as mock_cap, patch("ultralytics.YOLO") as mock_yolo:
            import src.web
            from src.web import create_app, WebStreamWorker, StreamSessionManager
            mock_cap.assert_not_called()
            mock_yolo.assert_not_called()

    def test_package_exports(self):
        """Verifies that __all__ exports expected symbols."""
        import src.web
        assert hasattr(src.web, "create_app")
        assert hasattr(src.web, "WebStreamWorker")
        assert hasattr(src.web, "StreamSessionManager")


class TestPayloadValidation:
    """Verifies strict input sanitization and rejection of invalid JSON payloads."""

    def test_validate_none_payload(self):
        valid, msg, _ = validate_start_payload(None)
        assert valid is False
        assert "valid JSON object" in msg

    def test_validate_non_dict_payload(self):
        valid, msg, _ = validate_start_payload([1, 2, 3])
        assert valid is False
        assert "valid JSON object" in msg

        valid_str, msg_str, _ = validate_start_payload("invalid_string")
        assert valid_str is False

    def test_validate_boolean_conf_threshold(self):
        # In Python isinstance(True, int) is True, so boolean rejection must be explicit
        valid, msg, _ = validate_start_payload({"conf_threshold": True})
        assert valid is False
        assert "conf_threshold" in msg

        valid_f, msg_f, _ = validate_start_payload({"conf_threshold": False})
        assert valid_f is False

    def test_validate_string_conf_threshold(self):
        valid, msg, _ = validate_start_payload({"conf_threshold": "0.35"})
        assert valid is False
        assert "conf_threshold" in msg

    def test_validate_out_of_bounds_conf_threshold(self):
        valid_neg, _, _ = validate_start_payload({"conf_threshold": -0.1})
        assert valid_neg is False

        valid_hi, _, _ = validate_start_payload({"conf_threshold": 1.1})
        assert valid_hi is False

    def test_validate_nan_and_inf_conf_threshold(self):
        valid_nan, _, _ = validate_start_payload({"conf_threshold": float("nan")})
        assert valid_nan is False

        valid_inf, _, _ = validate_start_payload({"conf_threshold": float("inf")})
        assert valid_inf is False

    def test_validate_path_traversal_source(self):
        valid, msg, _ = validate_start_payload({"source": "../../secret.txt"})
        assert valid is False
        assert "traversal" in msg.lower()

    def test_validate_valid_payload(self):
        valid, msg, cleaned = validate_start_payload({
            "source": "0",
            "conf_threshold": 0.45,
            "enable_heatmap": True,
            "virtual_line": [10, 20, 30, 40],
            "roi_region": [50, 60, 70, 80]
        })
        assert valid is True
        assert msg is None
        assert cleaned["conf_threshold"] == 0.45
        assert cleaned["source"] == "0"
        assert cleaned["enable_heatmap"] is True
        assert cleaned["virtual_line"] == (10, 20, 30, 40)
        assert cleaned["roi_region"] == (50, 60, 70, 80)

    def test_validate_payload_defaults_to_conf_040_and_no_overlays(self):
        """Verifies that omitting parameters cleanly defaults to 0.40 confidence and no overlays."""
        valid, msg, cleaned = validate_start_payload({"source": "0"})
        assert valid is True
        assert cleaned["conf_threshold"] == 0.40
        assert "virtual_line" not in cleaned
        assert "roi_region" not in cleaned
        assert cleaned["enable_heatmap"] is False

    def test_validate_payload_configurable_conf_threshold(self):
        """Verifies that custom confidence thresholds are preserved and configurable."""
        valid, msg, cleaned = validate_start_payload({"source": "0", "conf_threshold": 0.65})
        assert valid is True
        assert cleaned["conf_threshold"] == 0.65

    def test_validate_payload_explicitly_enabled_virtual_line_and_roi(self):
        """Verifies that explicitly enabled virtual line and ROI are properly parsed."""
        valid, msg, cleaned = validate_start_payload({
            "source": "0",
            "virtual_line": [0, 240, 640, 240],
            "roi_region": [100, 100, 540, 380]
        })
        assert valid is True
        assert cleaned["virtual_line"] == (0, 240, 640, 240)
        assert cleaned["roi_region"] == (100, 100, 540, 380)

    def test_validate_image_source_rejected(self):
        valid, msg, _ = validate_start_payload({"source": "photo.jpg"})
        assert valid is False
        assert "Image files" in msg

        valid_png, msg_png, _ = validate_start_payload({"source": "screenshot.png"})
        assert valid_png is False
        assert "Image files" in msg_png

    def test_validate_supported_video_sources(self):
        for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            valid, msg, cleaned = validate_start_payload({"source": f"sample{ext}"})
            assert valid is True, f"Expected {ext} to be valid, got error: {msg}"
            assert msg is None
            assert cleaned["source"].endswith(ext)


class TestWebEndpoints:
    """Tests the REST API and template endpoints using Flask test client."""

    def test_root_index_renders_html_dashboard(self, client):
        """Tests that GET / returns the full HTML5 dark-themed dashboard."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.content_type
        assert b"CodeAlpha Object Detection &amp; Tracking" in response.data
        assert b"Ultralytics YOLO11" in response.data
        assert b"Live Vision Stream" in response.data

    def test_api_health(self, client):
        """Tests GET /api/health response status and schema."""
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.get_json()
        assert data["status"] == "ok"
        assert data["service"] == "codealpha-object-detection-web"
        assert "active" in data

    def test_api_status_idle(self, client):
        """Tests GET /api/status when no stream session is active."""
        response = client.get("/api/status")
        assert response.status_code == 200
        data = response.get_json()
        assert data["active"] is False
        assert data["status"] == "idle"
        assert data["frames_processed"] == 0
        assert "scene_summary" in data

    def test_api_telemetry_schema(self, client):
        """Tests GET /api/telemetry returns complete telemetry schema."""
        response = client.get("/api/telemetry")
        assert response.status_code == 200
        data = response.get_json()
        assert "fps" in data
        assert "active_tracks_count" in data
        assert "total_unique_tracks" in data
        assert "visible_classes" in data
        assert "recent_events" in data

    def test_api_start_validation_errors(self, client):
        """Tests starting stream with invalid payloads returns HTTP 400 JSON."""
        # Non-dict payload
        res_non_dict = client.post("/api/start", json=[1, 2, 3])
        assert res_non_dict.status_code == 400
        assert res_non_dict.get_json()["success"] is False

        # Invalid conf_threshold string
        res_str_conf = client.post("/api/start", json={"conf_threshold": "invalid"})
        assert res_str_conf.status_code == 400

        # Invalid boolean conf_threshold
        res_bool_conf = client.post("/api/start", json={"conf_threshold": True})
        assert res_bool_conf.status_code == 400

        # Out of bounds conf_threshold
        res_oob = client.post("/api/start", json={"conf_threshold": 1.5})
        assert res_oob.status_code == 400

    def test_api_start_and_stop_lifecycle(self, client):
        """Tests starting and stopping streaming session via API."""
        with patch.object(WebStreamWorker, "start"), patch.object(WebStreamWorker, "is_alive", return_value=True):
            # 1. Start stream
            res_start = client.post("/api/start", json={"source": "0", "conf_threshold": 0.3})
            assert res_start.status_code == 200
            data_start = res_start.get_json()
            assert data_start["success"] is True

            # 2. Duplicate start returns 400
            res_dup = client.post("/api/start", json={"source": "0"})
            assert res_dup.status_code == 400
            assert res_dup.get_json()["success"] is False

        # 3. Stop stream
        with patch.object(WebStreamWorker, "is_alive", return_value=True), patch.object(WebStreamWorker, "stop", return_value=True):
            res_stop = client.post("/api/stop")
            assert res_stop.status_code == 200
            assert res_stop.get_json()["success"] is True

            # 4. Duplicate stop when idle returns 400
            res_stop_dup = client.post("/api/stop")
            assert res_stop_dup.status_code == 400

    def test_api_404_json_error(self, client):
        """Tests that unknown /api/ endpoints return JSON error responses."""
        response = client.get("/api/nonexistent_route")
        assert response.status_code == 404
        assert "application/json" in response.content_type
        data = response.get_json()
        assert data["error"] == "Not Found"
        assert data["status_code"] == 404

    def test_api_405_method_not_allowed(self, client):
        """Tests that unsupported HTTP methods return JSON 405 error."""
        response = client.post("/api/health")
        assert response.status_code == 405
        assert "application/json" in response.content_type
        data = response.get_json()
        assert data["error"] == "Method Not Allowed"
        assert data["status_code"] == 405


class TestVideoStreamingAndMJPEG:
    """Tests browser MJPEG endpoint and frame formatting."""

    def test_create_standby_frame(self):
        """Tests generated standby placeholder frame dimensions and channels."""
        frame = create_standby_frame(640, 480, message="Testing")
        assert isinstance(frame, np.ndarray)
        assert frame.shape == (480, 640, 3)

    def test_video_feed_endpoint_mimetype(self, client):
        """Verifies /video_feed returns multipart/x-mixed-replace boundary."""
        response = client.get("/video_feed")
        assert response.status_code == 200
        assert "multipart/x-mixed-replace" in response.content_type
        assert "boundary=frame" in response.content_type


class TestUploadSecurity:
    """Verifies file upload security, validation, and path traversal prevention."""

    def test_upload_missing_file_field(self, client):
        """Tests upload request with missing file field returns 400."""
        response = client.post("/api/upload", data={})
        assert response.status_code == 400
        assert "No file field" in response.get_json()["message"]

    def test_upload_empty_filename(self, client):
        """Tests upload request with empty filename returns 400."""
        data = {"file": (io.BytesIO(b""), "")}
        response = client.post("/api/upload", data=data, content_type="multipart/form-data")
        assert response.status_code == 400

    def test_upload_disallowed_extension(self, client):
        """Tests upload request with disallowed extension (.exe) returns 400."""
        data = {"file": (io.BytesIO(b"malicious_executable_bytes"), "payload.exe")}
        response = client.post("/api/upload", data=data, content_type="multipart/form-data")
        assert response.status_code == 400
        assert "Unsupported file extension" in response.get_json()["message"]

    def test_upload_valid_video_file(self, client):
        """Tests valid video upload returns 200 and sanitizes filename."""
        dummy_content = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 100
        data = {"file": (io.BytesIO(dummy_content), "my_test_video.mp4")}
        response = client.post("/api/upload", data=data, content_type="multipart/form-data")
        assert response.status_code == 200
        res_json = response.get_json()
        assert res_json["success"] is True
        assert res_json["original_name"] == "my_test_video.mp4"
        assert "my_test_video_" in res_json["filename"]
        assert Path(res_json["filepath"]).exists()

    def test_upload_path_traversal_sanitization(self, client):
        """Tests that path traversal characters in upload filename are neutralized."""
        dummy_content = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 50
        data = {"file": (io.BytesIO(dummy_content), "../../../evil.mp4")}
        response = client.post("/api/upload", data=data, content_type="multipart/form-data")
        assert response.status_code == 200
        res_json = response.get_json()
        # secure_filename strips ../
        assert ".." not in res_json["filename"]
        assert "evil_" in res_json["filename"]

    def test_upload_image_rejected(self, client):
        """Tests that uploading image files (.jpg, .png) is rejected with HTTP 400."""
        data_jpg = {"file": (io.BytesIO(b"fake_jpeg_bytes"), "test.jpg")}
        res_jpg = client.post("/api/upload", data=data_jpg, content_type="multipart/form-data")
        assert res_jpg.status_code == 400
        assert "Image files" in res_jpg.get_json()["message"]

        data_png = {"file": (io.BytesIO(b"fake_png_bytes"), "test.png")}
        res_png = client.post("/api/upload", data=data_png, content_type="multipart/form-data")
        assert res_png.status_code == 400
        assert "Image files" in res_png.get_json()["message"]

    def test_upload_supported_video_formats(self, client):
        """Tests that all supported video formats are accepted by upload."""
        for ext in (".mp4", ".avi", ".mov", ".mkv", ".webm"):
            dummy = b"\x00\x00\x00\x20ftypisom" + b"\x00" * 30
            data = {"file": (io.BytesIO(dummy), f"clip{ext}")}
            res = client.post("/api/upload", data=data, content_type="multipart/form-data")
            assert res.status_code == 200, f"Expected {ext} to be accepted, got: {res.get_json()}"
            assert res.get_json()["success"] is True


class TestExportEndpoints:
    """Verifies analytics report and heatmap image downloads."""

    def test_export_json_endpoint(self, client):
        """Tests /api/export/json returns downloadable JSON report."""
        response = client.get("/api/export/json")
        assert response.status_code == 200
        assert "application/json" in response.content_type
        data = response.get_json()
        assert data["application"] == "CodeAlpha Object Detection & Tracking"
        assert "session_data" in data

    def test_export_heatmap_not_found(self, client):
        """Tests /api/export/heatmap returns 404 if no heatmap exists."""
        response = client.get("/api/export/heatmap")
        assert response.status_code == 404
        assert response.get_json()["error"] == "Not Found"

    def test_export_heatmap_overlay_not_found(self, client):
        """Tests /api/export/heatmap_overlay returns 404 if no overlay exists."""
        response = client.get("/api/export/heatmap_overlay")
        assert response.status_code == 404


class TestWorkerLifecycleAndSafety:
    """Verifies thread safety, worker stop timeout, and resource cleanup."""

    def test_worker_initialization(self):
        """Verifies worker initializes without starting thread."""
        worker = WebStreamWorker(source="test.mp4", conf_threshold=0.35)
        assert not worker.is_running
        assert worker.status == "initialized"
        telemetry = worker.get_telemetry()
        assert telemetry["active"] is False

    def test_worker_stop_when_already_stopped(self):
        """Tests stopping an unstarted worker returns True without errors."""
        worker = WebStreamWorker()
        stopped = worker.stop(timeout=0.5)
        assert stopped is True
        assert worker.status == "stopped"

    def test_consumer_registration_and_watchdog_touch(self):
        """Tests consumer registration and activity timestamp updating."""
        worker = WebStreamWorker(source="0")
        assert worker._consumer_count == 0
        worker.register_consumer()
        assert worker._consumer_count == 1
        worker.touch_activity()
        worker.unregister_consumer()
        assert worker._consumer_count == 0

    def test_worker_default_confidence_and_overlays_disabled(self):
        """Verifies worker defaults to 0.40 confidence and None for spatial overlays."""
        worker = WebStreamWorker(source="0")
        assert worker.conf_threshold == 0.40
        assert worker.virtual_line is None
        assert worker.roi_region is None
        telemetry = worker.get_telemetry()
        assert telemetry["confidence_threshold"] == 0.40
        assert telemetry["spatial"] is None

    def test_draw_web_spatial_overlays_distinct_roi(self):
        """Verifies web-tailored distinct spatial overlays render properly."""
        from src.web.stream_worker import draw_web_spatial_overlays
        from src.spatial_analytics import SpatialAnalytics
        import numpy as np

        spatial = SpatialAnalytics(
            virtual_line=(0, 240, 640, 240),
            roi_region=(100, 100, 540, 380),
            frame_width=640,
            frame_height=480
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = draw_web_spatial_overlays(frame, spatial)
        assert annotated is not None
        assert annotated.shape == (480, 640, 3)
        assert np.any(annotated > 0)


class TestStreamSessionManager:
    """Tests session controller concurrency and startup guards."""

    def test_manager_initial_state(self):
        mgr = StreamSessionManager()
        assert mgr.active_worker is None
        assert mgr.is_busy is False
        status = mgr.get_status()
        assert status["active"] is False
        assert status["status"] == "idle"

    def test_duplicate_session_prevention(self):
        mgr = StreamSessionManager()
        with patch.object(WebStreamWorker, "start"), patch.object(WebStreamWorker, "is_alive", return_value=True):
            started, msg = mgr.start_stream(source="0")
            assert started is True

            # Duplicate start while alive must be rejected
            dup_started, dup_msg = mgr.start_stream(source="0")
            assert dup_started is False
            assert "already running" in dup_msg

    def test_final_telemetry_preserved_after_video_eof(self):
        """Tests that when a worker completes via EOF, manager preserves final telemetry."""
        mgr = StreamSessionManager()
        worker = WebStreamWorker(source="dummy.mp4")
        with worker._state_lock:
            worker._status = "completed"
            worker._frame_count = 142
            worker._total_tracks_count = 7
            worker._scene_summary = "Observed 7 entities across 142 frames."
        mgr._worker = worker

        status = mgr.get_status()
        assert status["active"] is False
        assert status["status"] == "completed"
        assert status["frames_processed"] == 142
        assert status["total_unique_tracks"] == 7
        assert "Observed 7 entities" in status["scene_summary"]

    def test_final_telemetry_preserved_after_worker_error(self):
        """Tests that when a worker terminates with an error, manager preserves error telemetry."""
        mgr = StreamSessionManager()
        worker = WebStreamWorker(source="corrupt.mp4")
        with worker._state_lock:
            worker._status = "error"
            worker._error_message = "Codec decode failure"
            worker._frame_count = 25
        mgr._worker = worker

        status = mgr.get_status()
        assert status["active"] is False
        assert status["status"] == "error"
        assert status["error_message"] == "Codec decode failure"
        assert status["frames_processed"] == 25

    def test_new_session_does_not_receive_previous_session_telemetry(self):
        """Tests that starting a new session starts clean without previous session metrics."""
        mgr = StreamSessionManager()
        # Old finished session
        old_worker = WebStreamWorker(source="old.mp4")
        with old_worker._state_lock:
            old_worker._status = "completed"
            old_worker._frame_count = 999
            old_worker._total_tracks_count = 50
        mgr._worker = old_worker

        # Now start new stream
        with patch.object(WebStreamWorker, "start"):
            started, _ = mgr.start_stream(source="0")
            assert started is True
            new_status = mgr.get_status()
            assert new_status["frames_processed"] == 0
            assert new_status["total_unique_tracks"] == 0
            assert new_status["frames_processed"] != 999

    def test_stop_after_eof_retains_completed_telemetry_and_cleans_up(self):
        """Tests calling stop_stream after natural EOF archives telemetry and clears worker."""
        mgr = StreamSessionManager()
        worker = WebStreamWorker(source="video.mp4")
        with worker._state_lock:
            worker._status = "completed"
            worker._frame_count = 88
        mgr._worker = worker

        # Worker is already dead (EOF reached)
        assert not worker.is_alive()

        stopped, msg = mgr.stop_stream()
        assert stopped is True
        assert mgr.active_worker is None
        assert mgr._worker is None

        # Archived telemetry is returned
        status = mgr.get_status()
        assert status["status"] == "completed"
        assert status["frames_processed"] == 88


class TestWebAppCLI:
    """Tests web_app.py argument parser defaults and overrides."""

    def test_cli_default_args(self):
        args = web_app.parse_args([])
        assert args.host == "127.0.0.1"
        assert args.port == 5000
        assert args.debug is False

    def test_cli_custom_args(self):
        args = web_app.parse_args(["--host", "127.0.0.1", "--port", "5050", "--debug"])
        assert args.host == "127.0.0.1"
        assert args.port == 5050
        assert args.debug is True
