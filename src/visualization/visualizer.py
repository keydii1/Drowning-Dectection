"""
Visualization module to render bounding boxes, track IDs, statuses, anomaly telemetry, and alert banners on frames.
"""

from typing import Any, Dict, List
import cv2
import numpy as np


class DrowningVisualizer:
    def __init__(self):
        self.color_normal = (50, 205, 50)       # Emerald Green (BGR)
        self.color_splash = (0, 215, 255)       # Amber / Yellow (BGR)
        self.color_active_alert = (0, 0, 255)   # Vivid Red (BGR)
        self.color_passive_alert = (180, 0, 180)# Purple / Magenta (BGR)

    def draw_frame(
        self,
        frame: np.ndarray,
        tracked_objects: List[Dict[str, Any]],
        any_alert: bool = False,
        fps_text: str = "",
        mode_text: str = "Mode 6: Spatio-Temporal Anomaly",
    ) -> np.ndarray:
        """
        Draws bounding boxes, tracking labels, telemetry dashboard, and top banner alert.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # 1. Top Surveillance Banner
        if any_alert:
            cv2.rectangle(annotated, (0, 0), (w, 55), (0, 0, 210), -1)
            cv2.putText(
                annotated,
                "🚨 CRITICAL POOL ALARM: DROWNING / DISTRESS DETECTED!",
                (20, 38),
                cv2.FONT_HERSHEY_DUPLEX,
                0.85,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.rectangle(annotated, (0, 0), (w, 40), (30, 30, 30), -1)
            cv2.putText(
                annotated,
                f"POOL SURVEILLANCE ACTIVE | {mode_text} | {fps_text}",
                (15, 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 120),
                2,
                cv2.LINE_AA,
            )

        # 2. Bottom Telemetry Bar
        total_swimmers = len(tracked_objects)
        alert_count = sum(1 for o in tracked_objects if o.get("is_alert", False))
        cv2.rectangle(annotated, (0, h - 30), (w, h), (20, 20, 20), -1)
        telemetry_str = f"Swimmers Tracked: {total_swimmers} | Active Alerts: {alert_count} | Pool Safety Status: {'NORMAL' if alert_count == 0 else 'HAZARD DETECTED'}"
        cv2.putText(
            annotated,
            telemetry_str,
            (15, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200) if alert_count == 0 else (0, 100, 255),
            1,
            cv2.LINE_AA,
        )

        # 3. Draw Each Tracked Person & Anomaly Badge
        for obj in tracked_objects:
            bbox = [int(v) for v in obj["bbox"]]
            x1, y1, x2, y2 = bbox
            track_id = obj.get("track_id", 0)
            is_alert = obj.get("is_alert", False)
            state = obj.get("swimmer_state", "NORMAL_SWIMMING")
            score = obj.get("anomaly_score", obj.get("confidence", 0.0))

            # Select badge color and label
            if is_alert:
                if "PASSIVE" in state:
                    color = self.color_passive_alert
                    state_label = "PASSIVE DROWNING"
                else:
                    color = self.color_active_alert
                    state_label = "ACTIVE DISTRESS"
                thickness = 3
            elif "SPLASH" in state or "TRANSIENT" in state:
                color = self.color_splash
                state_label = "SPLASHING (PLAY)"
                thickness = 2
            else:
                color = self.color_normal
                state_label = "SWIMMING"
                thickness = 2

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Trajectory / Centroid dot
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(annotated, (cx, cy), 4, color, -1)

            # Badge text
            badge_text = f"ID {track_id} | {state_label} [{score:.2f}]"
            (text_w, text_h), baseline = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            badge_y1 = max(0, y1 - text_h - 8)
            cv2.rectangle(
                annotated,
                (x1, badge_y1),
                (x1 + text_w + 8, badge_y1 + text_h + 6),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                badge_text,
                (x1 + 4, badge_y1 + text_h + 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255) if is_alert else (0, 0, 0),
                1,
                cv2.LINE_AA,
            )

        return annotated
