"""
Detector module for CodeAlpha Object Detection and Tracking.
Responsible for loading pretrained YOLO models and running inference on images and video frames.
"""

from typing import List, Dict, Any, Optional, Union
from pathlib import Path
import numpy as np
import torch


class DetectionResult:
    """Represents a single detected object in a frame."""

    def __init__(
        self,
        box: List[float],
        confidence: float,
        class_id: int,
        class_name: str
    ):
        self.box = [float(v) for v in box]  # [x1, y1, x2, y2]
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_name = str(class_name)

    @property
    def width(self) -> float:
        """Returns the width of the bounding box."""
        return max(0.0, self.box[2] - self.box[0])

    @property
    def height(self) -> float:
        """Returns the height of the bounding box."""
        return max(0.0, self.box[3] - self.box[1])

    @property
    def area(self) -> float:
        """Returns the area of the bounding box in pixels."""
        return self.width * self.height

    @property
    def center(self) -> List[float]:
        """Returns the [cx, cy] centroid coordinates."""
        return [
            (self.box[0] + self.box[2]) / 2.0,
            (self.box[1] + self.box[3]) / 2.0
        ]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "box": [round(v, 2) for v in self.box],
            "confidence": round(self.confidence, 4),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "center": [round(c, 2) for c in self.center],
            "area": round(self.area, 2)
        }

    def __repr__(self) -> str:
        return (
            f"DetectionResult(class='{self.class_name}', id={self.class_id}, "
            f"conf={self.confidence:.2f}, box={self.to_dict()['box']})"
        )


class YOLODetector:
    """
    Manages YOLO model lifecycle, weight caching, device allocation, and inference.
    Lazy-loads model weights on explicit request to avoid unwanted downloads during import.
    """

    DEFAULT_MODEL = "yolo11n.pt"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        confidence_threshold: float = 0.25,
        device: Optional[str] = None,
        weights_dir: Optional[Union[str, Path]] = None
    ):
        self.model_name = model_name
        self.confidence_threshold = max(0.0, min(1.0, float(confidence_threshold)))
        self.device = self._resolve_device(device)
        self.weights_dir = Path(weights_dir) if weights_dir else Path(__file__).resolve().parent.parent / "models"
        self._model = None
        self._class_names: Dict[int, str] = {}
        self._actual_weights_path: Optional[str] = None

    def _resolve_device(self, requested_device: Optional[str]) -> str:
        """Selects computation device safely based on hardware availability."""
        if requested_device is None or requested_device.lower() in ("auto", ""):
            return "cuda" if torch.cuda.is_available() else "cpu"

        clean_device = requested_device.lower().strip()
        if clean_device.startswith("cuda"):
            if not torch.cuda.is_available():
                # Fallback to CPU if CUDA was explicitly requested but unavailable
                return "cpu"
            return clean_device
        return "cpu"

    @property
    def is_loaded(self) -> bool:
        """Returns True if the model weights are loaded and ready for inference."""
        return self._model is not None

    def _resolve_weights_path(self) -> str:
        """
        Determines where weights should be loaded from or saved to.
        Checks if model exists locally in models/ directory first.
        """
        model_p = Path(self.model_name)
        if model_p.exists():
            return str(model_p.resolve())

        # Check inside self.weights_dir
        in_weights_dir = self.weights_dir / self.model_name
        if in_weights_dir.exists():
            return str(in_weights_dir.resolve())

        # If not already present in models/, direct ultralytics to download into models/
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        return str(in_weights_dir.resolve())

    def load_model(self) -> None:
        """
        Loads the Ultralytics YOLO model on demand.
        Downloads pretrained weights only on first invocation if not cached.
        """
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
                f"Failed to load YOLO model '{self.model_name}' (target path: {target_path}): {err}"
            ) from err

    def detect(self, frame: np.ndarray) -> List[DetectionResult]:
        """
        Runs object detection on a single BGR image frame.

        Args:
            frame: np.ndarray containing BGR image data.

        Returns:
            List[DetectionResult]: list of detected bounding boxes and metadata.
        """
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
            raise ValueError("Invalid frame: input must be a non-empty numpy.ndarray")

        try:
            results = self._model.predict(
                source=frame,
                conf=self.confidence_threshold,
                device=self.device,
                verbose=False
            )
        except Exception as err:
            raise RuntimeError(f"YOLO inference failed: {err}") from err

        detections: List[DetectionResult] = []
        if not results:
            return detections

        first_res = results[0]
        boxes = getattr(first_res, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return detections

        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy().tolist()
            conf = float(box.conf[0].cpu().item())
            cls_id = int(box.cls[0].cpu().item())
            cls_name = self._class_names.get(cls_id, f"class_{cls_id}")
            detections.append(DetectionResult(xyxy, conf, cls_id, cls_name))

        return detections

    def get_class_names(self) -> Dict[int, str]:
        """Returns mapping of class IDs to class names."""
        return self._class_names

    def get_model_info(self) -> Dict[str, Any]:
        """Returns diagnostic metadata about the current model configuration."""
        return {
            "model_name": self.model_name,
            "weights_path": self._actual_weights_path,
            "is_loaded": self.is_loaded,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "num_classes": len(self._class_names)
        }
