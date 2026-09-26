"""
Temporal filter module to eliminate transient false positives in drowning detection.
"""

from collections import deque
from typing import Dict, List, Optional


class TrackHistory:
    def __init__(self, maxlen: int = 60):
        self.scores: deque = deque(maxlen=maxlen)
        self.bboxes: deque = deque(maxlen=maxlen)
        self.timestamps: deque = deque(maxlen=maxlen)
        self.is_alerting: bool = False
        self.alert_trigger_time: Optional[float] = None


class SpatioTemporalFilter:
    """
    Maintains temporal state for each tracked person ID and computes
    whether an alert should be triggered or suppressed as a transient false alarm.
    """
    def __init__(
        self,
        persistence_thresh_frames: int = 15,
        window_size: int = 30,
        positive_ratio_thresh: float = 0.7,
        fps: float = 30.0
    ):
        self.persistence_thresh_frames = persistence_thresh_frames
        self.window_size = window_size
        self.positive_ratio_thresh = positive_ratio_thresh
        self.fps = fps
        self.tracks: Dict[int, TrackHistory] = {}

    def update_track(self, track_id: int, drowning_score: float, bbox: List[float], timestamp: float) -> bool:
        """
        Updates tracking history for track_id and returns True if ALERT condition is met.
        Suppresses single-frame or momentary spikes (False Positives).
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackHistory(maxlen=self.window_size * 2)

        history = self.tracks[track_id]
        history.scores.append(drowning_score)
        history.bboxes.append(bbox)
        history.timestamps.append(timestamp)

        # Check temporal criteria within recent window
        recent_scores = list(history.scores)[-self.window_size:]
        positive_count = sum(1 for s in recent_scores if s > 0.5)
        ratio = positive_count / len(recent_scores) if recent_scores else 0.0

        # Trigger alert only if sustained distress is observed
        if ratio >= self.positive_ratio_thresh and len(recent_scores) >= self.persistence_thresh_frames:
            if not history.is_alerting:
                history.is_alerting = True
                history.alert_trigger_time = timestamp
            return True
        else:
            history.is_alerting = False
            return False
