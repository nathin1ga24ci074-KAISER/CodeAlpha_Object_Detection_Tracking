"""
Flask Application Factory for CodeAlpha Object Detection & Tracking Web Dashboard.
Provides web UI routes, MJPEG streaming, real-time telemetry, upload security, and export endpoints.
"""

import os
import sys
import time
import math
import uuid
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, Generator

from flask import (
    Flask,
    jsonify,
    request,
    render_template,
    Response,
    send_file,
    abort
)
from werkzeug.utils import secure_filename

from src.web.stream_worker import StreamSessionManager, WebStreamWorker, create_standby_frame
import cv2

ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}


def validate_start_payload(data: Any) -> Tuple[bool, Optional[str], Dict[str, Any]]:
    """
    Validates and sanitizes start_stream client payload.
    Rejects malformed JSON, booleans as numbers, out-of-range thresholds, and invalid types.
    """
    if data is None or not isinstance(data, dict):
        return False, "Request payload must be a valid JSON object", {}

    cleaned: Dict[str, Any] = {}

    # 1. Validate confidence threshold
    if "conf_threshold" in data:
        conf_val = data["conf_threshold"]
        # Explicitly reject boolean True/False which are subclasses of int in Python
        if isinstance(conf_val, bool) or not isinstance(conf_val, (int, float)):
            return False, "conf_threshold must be a numeric float or int between 0.0 and 1.0", {}

        float_conf = float(conf_val)
        if not math.isfinite(float_conf) or float_conf < 0.0 or float_conf > 1.0:
            return False, "conf_threshold must be a finite number between 0.0 and 1.0", {}
        cleaned["conf_threshold"] = float_conf
    else:
        cleaned["conf_threshold"] = 0.40

    # 2. Validate source
    raw_source = data.get("source", "0")
    if isinstance(raw_source, int):
        cleaned["source"] = str(raw_source)
    elif isinstance(raw_source, str):
        src_str = raw_source.strip()
        if not src_str:
            return False, "source parameter cannot be empty", {}
        if src_str.isdigit():
            cleaned["source"] = src_str
        else:
            # File path: prevent traversal attacks and restrict to video formats
            p = Path(src_str)
            if ".." in p.parts:
                return False, "Invalid source path: directory traversal not permitted", {}
            ext = p.suffix.lower()
            if ext in IMAGE_EXTENSIONS:
                return False, f"Image files ({ext}) are not supported for video tracking. Please select a video file.", {}
            if ext not in ALLOWED_VIDEO_EXTENSIONS:
                return False, f"Unsupported video file format '{ext}'. Allowed video formats: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}", {}
            cleaned["source"] = str(p.resolve())
    else:
        return False, "source parameter must be an integer index or file path string", {}

    # 3. Validate feature toggles
    cleaned["enable_heatmap"] = bool(data.get("enable_heatmap", False))
    cleaned["save_output"] = bool(data.get("save_output", False))

    if "heatmap_radius" in data:
        try:
            r = int(data["heatmap_radius"])
            if r > 0:
                cleaned["heatmap_radius"] = r
        except (ValueError, TypeError):
            pass

    if "heatmap_alpha" in data:
        try:
            a = float(data["heatmap_alpha"])
            if 0.0 <= a <= 1.0:
                cleaned["heatmap_alpha"] = a
        except (ValueError, TypeError):
            pass

    # 4. Optional spatial boundaries
    for key in ("virtual_line", "roi_region"):
        if key in data and data[key] is not None:
            region_val = data[key]
            if isinstance(region_val, (list, tuple)) and len(region_val) == 4:
                try:
                    cleaned[key] = tuple(int(v) for v in region_val)
                except (ValueError, TypeError):
                    return False, f"{key} must be a 4-tuple of integer coordinates", {}

    return True, None, cleaned


def create_app(test_config: Optional[Dict[str, Any]] = None) -> Flask:
    """
    Application factory creating and configuring the Flask web application.
    Decoupled from camera hardware and ML model weights on initial creation.
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    templates_dir = project_root / "templates"
    static_dir = project_root / "static"
    uploads_dir = project_root / "uploads"
    outputs_dir = project_root / "outputs"

    uploads_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    app = Flask(
        __name__,
        template_folder=str(templates_dir.resolve()),
        static_folder=str(static_dir.resolve()),
        static_url_path="/static"
    )

    # Core configuration
    app.config.from_mapping(
        SECRET_KEY="codealpha-cv-tracking-secret-key-prod",
        MAX_CONTENT_LENGTH=100 * 1024 * 1024,  # 100 MB max upload
        UPLOAD_FOLDER=str(uploads_dir.resolve()),
        OUTPUT_FOLDER=str(outputs_dir.resolve()),
        STREAM_SESSION_MANAGER=StreamSessionManager(),
    )

    if test_config:
        app.config.update(test_config)

    session_manager: StreamSessionManager = app.config["STREAM_SESSION_MANAGER"]

    # -------------------------------------------------------------
    # WEB UI & TEMPLATES
    # -------------------------------------------------------------

    @app.route("/", methods=["GET"])
    def index():
        """Renders the comprehensive, responsive dark-themed dashboard."""
        return render_template("index.html")

    # -------------------------------------------------------------
    # REST API ENDPOINTS
    # -------------------------------------------------------------

    @app.route("/api/health", methods=["GET"])
    def api_health():
        """Service health probe."""
        return jsonify({
            "status": "ok",
            "service": "codealpha-object-detection-web",
            "version": "1.0.0",
            "active": session_manager.is_busy
        }), 200

    @app.route("/api/status", methods=["GET"])
    def api_status():
        """Returns structured session telemetry snapshot."""
        status = session_manager.get_status()
        return jsonify(status), 200

    @app.route("/api/telemetry", methods=["GET"])
    def api_telemetry():
        """Real-time JSON polling telemetry endpoint."""
        worker = session_manager.active_worker
        if worker is not None:
            worker.touch_activity()
            telemetry = worker.get_telemetry()
        else:
            telemetry = session_manager.get_status()
        return jsonify(telemetry), 200

    @app.route("/api/start", methods=["POST"])
    def api_start_stream():
        """Validates payload and starts the computer vision streaming worker."""
        raw_json = request.get_json(silent=True)
        is_valid, err_msg, cleaned = validate_start_payload(raw_json)
        if not is_valid:
            return jsonify({
                "success": False,
                "error": "Bad Request",
                "message": err_msg
            }), 400

        success, msg = session_manager.start_stream(**cleaned)
        status_code = 200 if success else 400
        return jsonify({
            "success": success,
            "message": msg,
            "session": session_manager.get_status()
        }), status_code

    @app.route("/api/stop", methods=["POST"])
    def api_stop_stream():
        """Gracefully terminates the active streaming worker and finalizes metrics."""
        success, msg = session_manager.stop_stream(timeout=2.5)
        status_code = 200 if success else 400
        return jsonify({
            "success": success,
            "message": msg,
            "session": session_manager.get_status()
        }), status_code

    # -------------------------------------------------------------
    # MJPEG VIDEO STREAMING
    # -------------------------------------------------------------

    @app.route("/video_feed", methods=["GET"])
    def video_feed():
        """
        Browser-native multipart MJPEG video stream.
        Decodes in standard <img> tags with zero client latency or external JS dependencies.
        """
        def generate_mjpeg() -> Generator[bytes, None, None]:
            worker = session_manager.active_worker
            if worker is not None:
                worker.register_consumer()

            try:
                while True:
                    current_worker = session_manager.active_worker
                    if current_worker is not None and current_worker.is_running:
                        frame_bytes = current_worker.wait_for_frame(timeout=0.4)
                        if frame_bytes:
                            yield (
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
                            )
                        else:
                            time.sleep(0.04)
                    else:
                        # Output standby placeholder frame
                        msg = "STREAM IDLE"
                        if current_worker is not None:
                            msg = current_worker.status.upper()
                        standby = create_standby_frame(640, 480, message=msg)
                        _, s_jpeg = cv2.imencode(".jpg", standby, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n\r\n" + s_jpeg.tobytes() + b"\r\n"
                        )
                        time.sleep(0.5)
            finally:
                if worker is not None:
                    worker.unregister_consumer()

        return Response(
            generate_mjpeg(),
            mimetype="multipart/x-mixed-replace; boundary=frame"
        )

    # -------------------------------------------------------------
    # FILE UPLOAD & SECURITY
    # -------------------------------------------------------------

    @app.route("/api/upload", methods=["POST"])
    def api_upload():
        """
        Handles video and image uploads securely.
        Enforces filename sanitization, extension allowlisting, and size boundaries.
        """
        if "file" not in request.files:
            return jsonify({
                "success": False,
                "error": "Bad Request",
                "message": "No file field found in upload request"
            }), 400

        file = request.files["file"]
        if not file or not file.filename:
            return jsonify({
                "success": False,
                "error": "Bad Request",
                "message": "No file selected for upload"
            }), 400

        original_name = file.filename
        ext = Path(original_name).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return jsonify({
                "success": False,
                "error": "Bad Request",
                "message": f"Image files ({ext}) are not supported. Please upload a video file (.mp4, .avi, .mov, .mkv, .webm)."
            }), 400

        if ext not in ALLOWED_VIDEO_EXTENSIONS:
            return jsonify({
                "success": False,
                "error": "Bad Request",
                "message": f"Unsupported file extension '{ext}'. Allowed video formats: {', '.join(sorted(ALLOWED_VIDEO_EXTENSIONS))}"
            }), 400

        # Secure filename with unique collision-free prefix
        safe_base = secure_filename(Path(original_name).stem) or "uploaded"
        unique_name = f"{safe_base}_{uuid.uuid4().hex[:8]}{ext}"
        current_uploads = Path(app.config.get("UPLOAD_FOLDER", uploads_dir))
        current_uploads.mkdir(parents=True, exist_ok=True)
        dest_path = current_uploads / unique_name

        try:
            file.save(str(dest_path.resolve()))
            if not dest_path.exists() or dest_path.stat().st_size == 0:
                if dest_path.exists():
                    dest_path.unlink()
                return jsonify({
                    "success": False,
                    "error": "Bad Request",
                    "message": "Uploaded file is empty"
                }), 400

            return jsonify({
                "success": True,
                "message": "File uploaded successfully",
                "filename": unique_name,
                "filepath": str(dest_path.resolve()),
                "original_name": original_name,
                "size_bytes": dest_path.stat().st_size
            }), 200
        except Exception as err:
            return jsonify({
                "success": False,
                "error": "Internal Server Error",
                "message": f"Failed to save upload: {err}"
            }), 500

    # -------------------------------------------------------------
    # EXPORT ENDPOINTS
    # -------------------------------------------------------------

    @app.route("/api/export/json", methods=["GET"])
    def export_json():
        """Downloads complete structured session telemetry report as JSON."""
        status = session_manager.get_status()
        report = {
            "application": "CodeAlpha Object Detection & Tracking",
            "exported_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "session_data": status
        }
        return jsonify(report), 200

    @app.route("/api/export/heatmap", methods=["GET"])
    def export_heatmap():
        """Downloads the standalone Gaussian density heatmap PNG."""
        status = session_manager.get_status()
        hm_path_str = status.get("saved_heatmap_path")
        if not hm_path_str or not Path(hm_path_str).exists():
            return jsonify({
                "error": "Not Found",
                "message": "No generated heatmap available for this session"
            }), 404

        path = Path(hm_path_str).resolve()
        current_outputs = Path(app.config.get("OUTPUT_FOLDER", outputs_dir)).resolve()
        # Security: verify path resides strictly within outputs/ directory
        if current_outputs not in path.parents:
            return jsonify({"error": "Forbidden", "message": "Access denied"}), 403

        return send_file(str(path), mimetype="image/png", as_attachment=True, download_name=path.name)

    @app.route("/api/export/heatmap_overlay", methods=["GET"])
    def export_heatmap_overlay():
        """Downloads the blended frame heatmap overlay PNG."""
        status = session_manager.get_status()
        ov_path_str = status.get("saved_overlay_path")
        if not ov_path_str or not Path(ov_path_str).exists():
            return jsonify({
                "error": "Not Found",
                "message": "No generated heatmap overlay available for this session"
            }), 404

        path = Path(ov_path_str).resolve()
        current_outputs = Path(app.config.get("OUTPUT_FOLDER", outputs_dir)).resolve()
        # Security: verify path resides strictly within outputs/ directory
        if current_outputs not in path.parents:
            return jsonify({"error": "Forbidden", "message": "Access denied"}), 403

        return send_file(str(path), mimetype="image/png", as_attachment=True, download_name=path.name)

    # -------------------------------------------------------------
    # ERROR HANDLERS
    # -------------------------------------------------------------

    @app.errorhandler(400)
    def handle_bad_request(e):
        if request.path.startswith("/api/"):
            return jsonify({
                "error": "Bad Request",
                "message": getattr(e, "description", "Malformed request"),
                "status_code": 400
            }), 400
        return "<h1>400 Bad Request</h1>", 400

    @app.errorhandler(404)
    def handle_not_found(e):
        if request.path.startswith("/api/"):
            return jsonify({
                "error": "Not Found",
                "message": f"Endpoint '{request.path}' does not exist",
                "status_code": 404
            }), 404
        return render_template("index.html"), 404

    @app.errorhandler(405)
    def handle_method_not_allowed(e):
        if request.path.startswith("/api/"):
            return jsonify({
                "error": "Method Not Allowed",
                "message": f"HTTP {request.method} is not supported on endpoint '{request.path}'",
                "status_code": 405
            }), 405
        return "<h1>405 Method Not Allowed</h1>", 405

    @app.errorhandler(413)
    def handle_oversized_file(e):
        if request.path.startswith("/api/"):
            return jsonify({
                "error": "Payload Too Large",
                "message": "Uploaded file exceeds maximum limit of 100 MB",
                "status_code": 413
            }), 413
        return "<h1>413 Payload Too Large</h1>", 413

    @app.errorhandler(500)
    def handle_server_error(e):
        if request.path.startswith("/api/"):
            return jsonify({
                "error": "Internal Server Error",
                "message": "An unexpected server error occurred",
                "status_code": 500
            }), 500
        return "<h1>500 Internal Server Error</h1>", 500

    return app
