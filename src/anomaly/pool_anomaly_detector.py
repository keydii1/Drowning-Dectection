"""
Pool Anomaly Detector Module.
Detects abnormal behaviors in swimming pools:
1. Active Distress (Violent in-place thrashing, head bobbing, vertical posture).
2. Passive Drowning (Prolonged motionlessness, submerging, bounding box area decay).
3. Transient Disturbances (Brief splashing, diving, playful behavior filtered out).
"""

from collections import deque
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


class SwimmerState(str, Enum):
    NORMAL = "NORMAL_SWIMMING"
    ACTIVE_DISTRESS = "ACTIVE_DISTRESS"
    PASSIVE_SUBMERGED = "PASSIVE_DROWNING"
    TRANSIENT_DISTURBANCE = "TRANSIENT_SPLASH"


class TrackKinematics:
    """
    Maintains spatial-temporal trajectory and kinematic features for a tracked person.
    """
    def __init__(self, track_id: int, maxlen: int = 90):
        self.track_id = track_id
        self.maxlen = maxlen
        self.centroids: deque = deque(maxlen=maxlen)        # (cx, cy)
        self.bboxes: deque = deque(maxlen=maxlen)           # [x1, y1, x2, y2]
        self.areas: deque = deque(maxlen=maxlen)            # w * h
        self.aspect_ratios: deque = deque(maxlen=maxlen)    # w / h
        self.confidences: deque = deque(maxlen=maxlen)      # YOLO drowning/struggling conf
        self.timestamps: deque = deque(maxlen=maxlen)       # timestamp in sec
        
        # State tracking
        self.current_state: SwimmerState = SwimmerState.NORMAL
        self.anomaly_score: float = 0.0
        self.distress_start_time: Optional[float] = None
        self.is_alarm: bool = False
        self.motionless_frames: int = 0

    def update(self, bbox: List[float], conf: float, timestamp: float):
        x1, y1, x2, y2 = bbox
        w = max(1.0, x2 - x1)
        h = max(1.0, y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        area = w * h
        ar = w / h

        self.centroids.append((cx, cy))
        self.bboxes.append(bbox)
        self.areas.append(area)
        self.aspect_ratios.append(ar)
        self.confidences.append(conf)
        self.timestamps.append(timestamp)


class PoolAnomalyDetector:
    """
    Multi-Feature Spatio-Temporal Anomaly Detector for Swimming Pools.
    Combines:
    - Trajectory Kinematics (Mobility vs Path Inefficiency)
    - Vertical Oscillation (Head Bobbing)
    - Bounding Box Aspect Ratio (Verticality check)
    - Area Decay (Submergence indicator)
    - Temporal Integration (Suppresses transient spikes)
    """
    def __init__(
        self,
        fps: float = 30.0,
        window_size: int = 30,             # 1.0s window
        persistence_sec: float = 0.75,      # Time needed to confirm drowning alarm
        motionless_thresh_sec: float = 3.0, # Time of zero movement to trigger passive drowning
        alarm_threshold: float = 0.65,      # Integrated anomaly score threshold
    ):
        self.fps = fps
        self.window_size = window_size
        self.persistence_frames = int(persistence_sec * fps)
        self.motionless_frames_thresh = int(motionless_thresh_sec * fps)
        self.alarm_threshold = alarm_threshold
        self.tracks: Dict[int, TrackKinematics] = {}

    def compute_kinematic_features(self, history: TrackKinematics) -> Dict[str, float]:
        """
        Calculates physical movement metrics over the sliding window.
        """
        if len(history.centroids) < 5:
            return {
                "velocity": 0.0,
                "inefficiency": 1.0,
                "vertical_oscillation": 0.0,
                "aspect_ratio": 1.0,
                "area_decay": 0.0,
            }

        k = min(self.window_size, len(history.centroids))
        recent_pts = list(history.centroids)[-k:]
        recent_areas = list(history.areas)[-k:]
        recent_ars = list(history.aspect_ratios)[-k:]

        # 1. Path Distance vs Net Displacement (Inefficiency Ratio)
        step_distances = [
            np.hypot(recent_pts[i][0] - recent_pts[i-1][0], recent_pts[i][1] - recent_pts[i-1][1])
            for i in range(1, len(recent_pts))
        ]
        total_path = sum(step_distances)
        net_disp = np.hypot(recent_pts[-1][0] - recent_pts[0][0], recent_pts[-1][1] - recent_pts[0][1])
        velocity = total_path / (k / self.fps) if k > 0 else 0.0

        # Inefficiency: High when moving a lot in place (thrashing/panic), low when swimming smoothly
        inefficiency = (total_path / (net_disp + 1e-4)) if net_disp > 5.0 else (total_path / 5.0)

        # 2. Vertical Oscillation (Head Bobbing)
        y_coords = [p[1] for p in recent_pts]
        vertical_oscillation = float(np.std(y_coords))

        # 3. Mean Aspect Ratio (Swimmers width > height AR > 1; Drowning upright AR < 0.8)
        mean_ar = float(np.mean(recent_ars))

        # 4. Area Decay (Shrinking bounding box indicating sinking)
        area_decay = float((recent_areas[0] - recent_areas[-1]) / (recent_areas[0] + 1e-4))

        return {
            "velocity": round(velocity, 2),
            "inefficiency": round(inefficiency, 2),
            "vertical_oscillation": round(vertical_oscillation, 2),
            "aspect_ratio": round(mean_ar, 2),
            "area_decay": round(area_decay, 3),
        }

    def evaluate_track(
        self,
        track_id: int,
        bbox: List[float],
        yolo_conf: float,
        timestamp: float,
    ) -> Tuple[SwimmerState, float, bool]:
        """
        Updates tracking and evaluates anomaly state.
        Returns: (SwimmerState, AnomalyScore, is_alarm)
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackKinematics(track_id=track_id)

        history = self.tracks[track_id]
        history.update(bbox, yolo_conf, timestamp)

        features = self.compute_kinematic_features(history)
        vel = features["velocity"]
        ineff = features["inefficiency"]
        vert_std = features["vertical_oscillation"]
        ar = features["aspect_ratio"]
        decay = features["area_decay"]

        # Check for Motionlessness (Passive drowning condition)
        if vel < 3.0:
            history.motionless_frames += 1
        else:
            history.motionless_frames = max(0, history.motionless_frames - 2)

        # Multi-factor Anomaly Score formulation:
        # Score components:
        # s_yolo: YOLO classification confidence (0 to 1)
        # s_posture: Upright / sinking posture (ar < 0.8 => 1.0, ar > 1.2 => 0.0)
        # s_bobbing: Vertical standard deviation
        # s_thrash: High inefficiency with moderate/high velocity
        
        s_yolo = yolo_conf
        s_posture = np.clip((1.0 - ar) / 0.5, 0.0, 1.0)
        s_bobbing = np.clip(vert_std / 12.0, 0.0, 1.0)
        s_thrash = np.clip((ineff - 2.0) / 4.0, 0.0, 1.0) if vel > 10.0 else 0.0
        s_passive = np.clip(history.motionless_frames / self.motionless_frames_thresh, 0.0, 1.0)

        # Composite anomaly score
        active_score = 0.35 * s_yolo + 0.25 * s_posture + 0.20 * s_bobbing + 0.20 * s_thrash
        anomaly_score = max(active_score, s_passive)

        # Smooth anomaly score with history (Exponential Moving Average)
        history.anomaly_score = 0.7 * history.anomaly_score + 0.3 * anomaly_score

        # State determination
        if history.motionless_frames >= self.motionless_frames_thresh:
            current_state = SwimmerState.PASSIVE_SUBMERGED
        elif history.anomaly_score >= self.alarm_threshold:
            current_state = SwimmerState.ACTIVE_DISTRESS
        elif s_yolo > 0.4 and (vel > 30.0 or ineff > 5.0):
            current_state = SwimmerState.TRANSIENT_DISTURBANCE
        else:
            current_state = SwimmerState.NORMAL

        history.current_state = current_state

        # Check temporal alarm condition (must persist to avoid transient false positives)
        recent_confs = list(history.confidences)[-self.persistence_frames:]
        high_risk_count = sum(1 for c in recent_confs if c > 0.45 or history.anomaly_score >= self.alarm_threshold)
        persistent = (high_risk_count >= int(0.7 * self.persistence_frames)) if len(recent_confs) >= self.persistence_frames else False

        if (current_state in (SwimmerState.ACTIVE_DISTRESS, SwimmerState.PASSIVE_SUBMERGED)) and persistent:
            history.is_alarm = True
            if history.distress_start_time is None:
                history.distress_start_time = timestamp
        else:
            history.is_alarm = False
            history.distress_start_time = None

        return history.current_state, round(history.anomaly_score, 3), history.is_alarm
