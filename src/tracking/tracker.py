"""
Multi-Object Tracking (MOT) wrapper using ByteTrack to associate persons across frames.
"""

from typing import Any, Dict, List
import numpy as np


class DrowningTracker:
    def __init__(
        self,
        track_thresh: float = 0.25,
        match_thresh: float = 0.8,
        track_buffer: int = 30,
        fps: int = 30,
    ):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.fps = fps
        self._tracker = None

    def _init_tracker(self):
        try:
            import supervision as sv
            self._tracker = sv.ByteTrack(
                track_activation_threshold=self.track_thresh,
                minimum_matching_threshold=self.match_thresh,
                lost_track_buffer=self.track_buffer,
                frame_rate=self.fps,
            )
        except ImportError:
            raise ImportError(
                "Supervision library is required for ByteTrack. Run `.venv/bin/pip install supervision`."
            )

    def update(self, detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Updates tracking with detections from current frame.
        detections: list of dicts with keys: bbox [x1, y1, x2, y2], confidence, class_id
        Returns: list of tracked objects with track_id included.
        """
        if self._tracker is None:
            self._init_tracker()

        import supervision as sv

        if len(detections) == 0:
            sv_detections = sv.Detections.empty()
            tracked = self._tracker.update_with_detections(sv_detections)
            return []

        xyxy = np.array([d["bbox"] for d in detections], dtype=float)
        confidence = np.array([d["confidence"] for d in detections], dtype=float)
        class_id = np.array([d["class_id"] for d in detections], dtype=int)

        sv_detections = sv.Detections(
            xyxy=xyxy,
            confidence=confidence,
            class_id=class_id,
        )

        tracked_sv = self._tracker.update_with_detections(sv_detections)
        tracked_results = []

        if tracked_sv.tracker_id is not None:
            for i, tid in enumerate(tracked_sv.tracker_id):
                tracked_results.append({
                    "track_id": int(tid),
                    "bbox": tracked_sv.xyxy[i].tolist(),
                    "confidence": float(tracked_sv.confidence[i]) if tracked_sv.confidence is not None else 1.0,
                    "class_id": int(tracked_sv.class_id[i]) if tracked_sv.class_id is not None else 0,
                })

        return tracked_results
