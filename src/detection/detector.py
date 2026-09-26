"""
Detection module wrapping YOLO models for person & drowning localization.
"""

from typing import Any, Dict, List, Optional
import numpy as np


class DrowningDetector:
    def __init__(self, model_path: str = "yolo11n.pt", conf_thresh: float = 0.35, device: str = "cpu"):
        self.model_path = model_path
        self.conf_thresh = conf_thresh
        self.device = device
        self.model = None

    def load_model(self):
        """Lazy loads the YOLO model from Ultralytics."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(self.model_path)
        except ImportError:
            raise ImportError("Please install ultralytics (`pip install ultralytics`) to use DrowningDetector.")

    def detect_frame(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Runs object detection on a single video frame.
        Returns: list of detections with bbox [x1, y1, x2, y2], confidence, class_id.
        """
        if self.model is None:
            self.load_model()
        results = self.model.predict(source=frame, conf=self.conf_thresh, device=self.device, verbose=False)
        detections = []
        if len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for i in range(len(boxes)):
                xyxy = boxes.xyxy[i].cpu().numpy().tolist()
                conf = float(boxes.conf[i].cpu().numpy())
                cls_id = int(boxes.cls[i].cpu().numpy())
                detections.append({
                    "bbox": xyxy,
                    "confidence": conf,
                    "class_id": cls_id,
                    "class_name": self.model.names.get(cls_id, str(cls_id))
                })
        return detections
