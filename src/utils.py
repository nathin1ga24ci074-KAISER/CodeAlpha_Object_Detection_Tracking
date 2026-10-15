"""
Utility module for CodeAlpha Object Detection and Tracking.
Provides reusable drawing, overlay rendering, FPS estimation, and live metrics calculations.
"""

from typing import List, Tuple, Dict, Any, Optional
from collections import Counter
import time
import cv2
import numpy as np


class ColorPalette:
    """Generates consistent, visually distinct colors for track IDs and classes."""

    # 16 distinct RGB palette colors
    PALETTE: List[Tuple[int, int, int]] = [
        (46, 204, 113),   # Green
        (52, 152, 219),   # Blue
        (155, 89, 182),   # Purple
        (241, 196, 15),   # Yellow
        (230, 126, 34),   # Orange
        (231, 76, 60),    # Red
        (26, 188, 156),   # Turquoise
        (52, 73, 94),     # Dark Slate
        (255, 107, 129),  # Coral
        (10, 189, 227),   # Cyan
        (255, 159, 243),  # Pink
        (84, 160, 255),   # Light Blue
        (95, 39, 205),    # Violet
        (0, 210, 211),    # Teal
        (254, 202, 87),   # Gold
        (255, 107, 107),  # Light Red
    ]

    @classmethod
    def get_color(cls, identifier: int) -> Tuple[int, int, int]:
        """Returns a consistent BGR color tuple for a given integer ID."""
        r, g, b = cls.PALETTE[abs(identifier) % len(cls.PALETTE)]
        return (b, g, r)  # Return BGR for OpenCV


class FPSMeter:
    """Calculates smoothed frames-per-second (FPS)."""

    def __init__(self, moving_avg_window: int = 15):
        self.window = moving_avg_window
        self.timestamps: List[float] = []

    def tick(self) -> float:
        """Call on each processed frame; returns smoothed current FPS."""
        now = time.time()
        self.timestamps.append(now)
        if len(self.timestamps) > self.window:
            self.timestamps.pop(0)

        if len(self.timestamps) < 2:
            return 0.0

        elapsed = self.timestamps[-1] - self.timestamps[0]
        if elapsed <= 0:
            return 0.0
        return (len(self.timestamps) - 1) / elapsed


def draw_bounding_box(
    frame: np.ndarray,
    box: List[float],
    label: str,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2
) -> np.ndarray:
    """
    Draws an annotated bounding box with a filled header label banner on the frame.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w - 1, x2)
    y1, y2 = max(0, y1), min(h - 1, y2)

    # Bounding rectangle
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label background badge
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
    badge_y1 = max(0, y1 - text_h - 8)
    badge_y2 = y1

    cv2.rectangle(frame, (x1, badge_y1), (x1 + text_w + 6, badge_y2), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 3, badge_y2 - 4),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA
    )
    return frame


def draw_trajectory(
    frame: np.ndarray,
    trajectory: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    thickness: int = 2
) -> np.ndarray:
    """
    Draws a path trail tracing an object's centroid history over time.
    """
    if len(trajectory) < 2:
        return frame

    for i in range(1, len(trajectory)):
        pt1 = trajectory[i - 1]
        pt2 = trajectory[i]
        cv2.line(frame, pt1, pt2, color, thickness)
    return frame


def draw_hud(
    frame: np.ndarray,
    fps: float,
    active_count: int,
    total_count: int,
    class_counts: Optional[Dict[str, int]] = None
) -> np.ndarray:
    """
    Renders an informational heads-up display (HUD) banner with live system metrics.
    """
    overlay = frame.copy()
    hud_h = 75 if not class_counts else 75 + min(len(class_counts) * 20, 60)
    cv2.rectangle(overlay, (10, 10), (320, 10 + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, "CodeAlpha Object Tracker", (20, 32), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, 54), font, 0.45, (0, 255, 128), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Active: {active_count} | Total IDs: {total_count}", (20, 72), font, 0.45, (200, 200, 255), 1, cv2.LINE_AA)

    if class_counts:
        y_offset = 92
        items = list(class_counts.items())[:3]
        line = " | ".join(f"{k}: {v}" for k, v in items)
        cv2.putText(frame, line, (20, y_offset), font, 0.40, (180, 220, 255), 1, cv2.LINE_AA)

    return frame


def create_synthetic_frame(
    width: int = 640,
    height: int = 480,
    bg_color: Tuple[int, int, int] = (30, 30, 30)
) -> np.ndarray:
    """
    Creates a synthetic BGR frame for automated testing without requiring webcam hardware.
    """
    frame = np.full((height, width, 3), bg_color, dtype=np.uint8)
    # Add a mock object (white circle)
    cv2.circle(frame, (width // 2, height // 2), 40, (240, 240, 240), -1)
    return frame
