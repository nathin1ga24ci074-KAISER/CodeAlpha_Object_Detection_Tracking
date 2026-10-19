"""
Web package for CodeAlpha Object Detection & Tracking.
Provides Flask application factory and streaming worker architecture.
"""

from src.web.app_server import create_app
from src.web.stream_worker import WebStreamWorker, StreamSessionManager

__all__ = ["create_app", "WebStreamWorker", "StreamSessionManager"]
