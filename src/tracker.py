"""
Tracker module for CodeAlpha Object Detection and Tracking.
Responsible for tracking detected objects across consecutive video frames and maintaining tracking IDs
using Ultralytics' supported ByteTrack and BoT-SORT integrations.
"""

from typing import List, Dict, Any, Tuple, Optional, Union
from collections import deque
from pathlib import Path
import numpy as np
import torch


class TrackedObject:
    """Represents an active tracked entity across frames."""

    def __init__(
        self,
        track_id: int,
        box: List[float],
        confidence: float,
        class_id: int,
        class_name: str,
        max_trajectory_len: int = 45
    ):
        self.track_id = int(track_id)
        self.box = [float(v) for v in box]  # [x1, y1, x2, y2]
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.trajectory: deque = deque(maxlen=max_trajectory_len)
        self.update_centroid()

    @property
    def width(self) -> float:
        """Returns bounding box width."""
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        """Returns bounding box height."""
        return max(0.0, self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        """Returns bounding box area in pixels."""
        return self.width * self.height

    @property
    def center(self) -> List[float]:
        """Returns center coordinates [cx, cy]."""
        return [
            (self.box[0] + self.box[2]) / 2.0,
            (self.box[1] + self.box[3]) / 2.0
        ]

    def update(
        self,
        box: List[float],
        confidence: float,
        class_id: int,
        class_name: str
    ) -> None:
        """Updates object position and classification with fresh frame observation."""
        self.box = [float(v) for v in box]
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.update_centroid()

    def update_centroid(self) -> Tuple[int, int]:
        """Calculates and appends integer pixel centroid to trajectory."""
        cx = int((self.box[0] + self.box[2]) / 2.0)
        cy = int((self.box[1] + self.box[3]) / 2.0)
        centroid = (cx, cy)
        self.trajectory.append(centroid)
        return centroid

    @property
    def current_centroid(self) -> Tuple[int, int]:
        """Returns the most recent centroid coordinate."""
        return self.trajectory[-1] if self.trajectory else (0, 0)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "track_id": self.track_id,
            "box": [round(v, 2) for v in self.box],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "centroid": self.current_centroid,
            "area": round(self.area, 2),
            "trajectory_points": len(self.trajectory)
        }

    def __repr__(self) -> str:
        return (
            f"TrackedObject(ID={self.track_id}, class='{self.class_name}', "
            f"conf={self.confidence:.2f}, box={self.to_dict()['box']})"
        )


class ObjectTracker:
    """
    Manages multi-object tracking across video frames.
    Wraps Ultralytics tracking engine with ByteTrack / BoT-SORT configuration.
    """

    SUPPORTED_TRACKERS = ("bytetrack.yaml", "botsort.yaml")
    DEFAULT_MODEL = "yolo11n.pt"

    def __init__(
        self,
        tracker_type: str = "bytetrack.yaml",
        model_name: str = DEFAULT_MODEL,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        weights_dir: Optional[Union[str, Path]] = None
    ):
        if tracker_type not in self.SUPPORTED_TRACKERS:
            raise ValueError(
                f"Unsupported tracker '{tracker_type}'. Must be one of {self.SUPPORTED_TRACKERS}"
            )
        self.tracker_type = tracker_type
        self.model_name = model_name
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        self.device = self._resolve_device(device)
        self.weights_dir = Path(weights_dir) if weights_dir else Path(__file__).resolve().parent.parent / "models"
        self._model = None
        self._class_names: Dict[int, str] = {}
        self._active_tracks: Dict[int, TrackedObject] = {}
        self._total_unique_ids: set = set()
        self._current_frame_tracks: List[TrackedObject] = []
        self._actual_weights_path: Optional[str] = None

    def _resolve_device(self, requested_device: Optional[str]) -> str:
        """Selects computation device safely based on hardware availability."""
        if requested_device is None or requested_device.lower() in ("auto", ""):
            return "cuda" if torch.cuda.is_available() else "cpu"

        clean_device = requested_device.lower().strip()
        if clean_device.startswith("cuda"):
            if not torch.cuda.is_available():
                return "cpu"
            return clean_device
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        """Returns True if the underlying tracking model has been loaded."""
        return self._model is not None

    def _resolve_weights_path(self) -> str:
        """Determines where weights should be loaded from or saved to."""
        model_p = Path(self.model_name)
        if model_p.exists():
            return str(model_p.resolve())

        in_weights_dir = self.weights_dir / self.model_name
        if in_weights_dir.exists():
            return str(in_weights_dir.resolve())

        self.weights_dir.mkdir(parents=True, exist_ok=True)
        return str(in_weights_dir.resolve())

    def load_model(self) -> None:
        """Loads the tracking model on demand."""
        if self._model is not None:
            return

        target_path = self._resolve_weights_path()
        try:
            from ultralytics import YOLO
            self._model = YOLO(target_path)
            self._actual_weights_path = target_path
            self._class_names = getattr(self._model, "names", {}) or {}
        except Exception as err:
            self._model = None
            raise RuntimeError(
                f"Failed to load tracking model '{self.model_name}' (target path: {target_path}): {err}"
            ) from err

    def track(self, frame: np.ndarray) -> List[TrackedObject]:
        """
        Processes a video frame, updating tracking state and returning active tracked objects.

        Args:
            frame: np.ndarray BGR image.

        Returns:
            List[TrackedObject]: active tracks with assigned persistent track IDs.
        """
        if not self.is_loaded:
            raise RuntimeError("Tracking model is not loaded. Call load_model() first.")

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame: input must be a non-empty numpy.ndarray")

        try:
            results = self._model.track(
                source=frame,
                persist=True,
                tracker=self.tracker_type,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
        except Exception as err:
            raise RuntimeError(f"Tracker execution failed: {err}") from err

        current_frame_tracks: List[TrackedObject] = []
        if not results:
            self._current_frame_tracks = []
            return current_frame_tracks

        first_res = results[0]
        boxes = getattr(first_res, "boxes", None)
        if boxes is None or getattr(boxes, "id", None) is None:
            self._current_frame_tracks = []
            return current_frame_tracks

        names = self._class_names or getattr(self._model, "names", {}) or {}
        ids = boxes.id.int().cpu().tolist()
        xyxys = boxes.xyxy.cpu().numpy().tolist()
        confs = boxes.conf.cpu().tolist()
        clss = boxes.cls.int().cpu().tolist()

        for track_id, xyxy, conf, cls_id in zip(ids, xyxys, confs, clss):
            self._total_unique_ids.add(track_id)
            cls_name = names.get(cls_id, f"class_{cls_id}")

            if track_id in self._active_tracks:
                tracked_obj = self._active_tracks[track_id]
                tracked_obj.update(
                    box=xyxy,
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=cls_name
                )
            else:
                tracked_obj = TrackedObject(
                    track_id=track_id,
                    box=xyxy,
                    confidence=float(conf),
                    class_id=cls_id,
                    class_name=cls_name
                )
                self._active_tracks[track_id] = tracked_obj

            current_frame_tracks.append(tracked_obj)

        self._current_frame_tracks = current_frame_tracks
        return current_frame_tracks

    def get_total_count(self) -> int:
        """Returns the total number of unique object IDs observed so far."""
        return len(self._total_unique_ids)

    def get_active_count(self) -> int:
        """Returns the number of objects currently tracked in the latest frame."""
        return len(self._current_frame_tracks)

    def get_current_tracks(self) -> List[TrackedObject]:
        """Returns the list of objects tracked in the latest frame."""
        return list(self._current_frame_tracks)

    def get_all_tracked_objects(self) -> Dict[int, TrackedObject]:
        """Returns mapping of all tracked objects seen across the stream."""
        return dict(self._active_tracks)

    def get_class_names(self) -> Dict[int, str]:
        """Returns mapping of class IDs to class names."""
        return self._class_names

    def reset(self) -> None:
        """Resets tracking history, counters, and predictor state."""
        self._active_tracks.clear()
        self._total_unique_ids.clear()
        self._current_frame_tracks.clear()
        if self._model is not None:
            predictor = getattr(self._model, "predictor", None)
            if predictor is not None and hasattr(predictor, "trackers"):
                predictor.trackers = None
