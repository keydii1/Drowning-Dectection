"""
Visualization module to render bounding boxes, track IDs, statuses, and alert banners on frames.
"""

from typing import Any, Dict, List
import cv2
import numpy as np


class DrowningVisualizer:
    def __init__(self):
        self.color_normal = (0, 255, 0)      # Green (BGR)
        self.color_alert = (0, 0, 255)       # Red (BGR)
        self.color_warning = (0, 165, 255)   # Orange (BGR)

    def draw_frame(
        self,
        frame: np.ndarray,
        tracked_objects: List[Dict[str, Any]],
        any_alert: bool = False,
        fps_text: str = "",
    ) -> np.ndarray:
        """
        Draws bounding boxes, tracking labels, and top banner alert if needed.
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw top banner if alert is triggered
        if any_alert:
            cv2.rectangle(annotated, (0, 0), (w, 60), (0, 0, 200), -1)
            cv2.putText(
                annotated,
                "🚨 CRITICAL ALERT: DROWNING DETECTED!",
                (20, 42),
                cv2.FONT_HERSHEY_DUPLEX,
                1.0,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        else:
            cv2.rectangle(annotated, (0, 0), (w, 35), (40, 40, 40), -1)
            cv2.putText(
                annotated,
                f"SURVEILLANCE ACTIVE | {fps_text}",
                (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

        # Draw each tracked person
        for obj in tracked_objects:
            bbox = [int(v) for v in obj["bbox"]]
            x1, y1, x2, y2 = bbox
            track_id = obj.get("track_id", 0)
            is_alert = obj.get("is_alert", False)
            score = obj.get("confidence", 0.0)

            color = self.color_alert if is_alert else self.color_normal
            thickness = 3 if is_alert else 2

            # Bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Label badge
            status_text = "DROWNING" if is_alert else "NORMAL"
            label = f"ID:{track_id} | {status_text} ({score:.2f})"

            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            badge_y1 = max(0, y1 - text_h - 10)
            cv2.rectangle(
                annotated,
                (x1, badge_y1),
                (x1 + text_w + 10, badge_y1 + text_h + 8),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 5, badge_y1 + text_h + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return annotated
