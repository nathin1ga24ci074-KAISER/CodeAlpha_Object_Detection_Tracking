"""
Video source module for CodeAlpha Object Detection and Tracking.
Handles robust video acquisition from webcams or video files with resource cleanup and context management.
"""

from typing import Union, Optional, Tuple, Iterator
from pathlib import Path
import cv2
import numpy as np


class VideoSource:
    """
    Manages OpenCV video capture streams (webcam or prerecorded files) safely.
    Implements Python's context manager protocol for guaranteed resource cleanup.
    """

    def __init__(self, source: Union[int, str, Path] = 0):
        self.raw_source = source
        self.is_webcam: bool = False
        self.source_id: Union[int, str] = self._resolve_source(source)
        self.cap: Optional[cv2.VideoCapture] = None

    def _resolve_source(self, source: Union[int, str, Path]) -> Union[int, str]:
        """Validates and normalizes the stream identifier."""
        if isinstance(source, int):
            self.is_webcam = True
            return source

        source_str = str(source).strip()
        if source_str.isdigit():
            self.is_webcam = True
            return int(source_str)

        path = Path(source_str)
        if not path.exists():
            raise FileNotFoundError(f"Video file not found: {source_str}")
        self.is_webcam = False
        return str(path.resolve())

    def open(self) -> "VideoSource":
        """Opens the video capture stream and verifies connectivity."""
        if self.cap is not None and self.cap.isOpened():
            return self

        # Use DirectShow backend on Windows for reliable webcam initialization
        if self.is_webcam and hasattr(cv2, "CAP_DSHOW"):
            self.cap = cv2.VideoCapture(self.source_id, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(self.source_id)

        if not self.cap.isOpened():
            target_desc = f"Webcam (index {self.source_id})" if self.is_webcam else f"File ({self.source_id})"
            raise IOError(f"Failed to open video source: {target_desc}")

        return self

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Reads a single frame from the video source."""
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Video source is not open. Call open() first.")
        ret, frame = self.cap.read()
        return ret, frame

    def frames(self) -> Iterator[np.ndarray]:
        """Generator yielding frames until stream finishes or read fails."""
        self.open()
        try:
            while True:
                ret, frame = self.read()
                if not ret or frame is None:
                    break
                yield frame
        finally:
            self.release()

    @property
    def width(self) -> int:
        if self.cap is None or not self.cap.isOpened():
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))

    @property
    def height(self) -> int:
        if self.cap is None or not self.cap.isOpened():
            return 0
        return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    @property
    def fps(self) -> float:
        if self.cap is None or not self.cap.isOpened():
            return 0.0
        val = float(self.cap.get(cv2.CAP_PROP_FPS))
        return val if val > 0 else 30.0

    @property
    def frame_count(self) -> int:
        if self.cap is None or not self.cap.isOpened() or self.is_webcam:
            return -1
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def release(self) -> None:
        """Releases the underlying OpenCV capture object."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
