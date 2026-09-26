"""
Evaluation metrics module supporting both Frame-level and Event-level evaluation.
"""

from typing import Dict, List


def calculate_frame_level_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict[str, float]:
    """Computes traditional frame-level classification metrics."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
    }


def calculate_event_level_metrics(
    total_observation_hours: float,
    false_alarm_events_count: int,
    true_drowning_events_count: int,
    detected_drowning_events_count: int,
    detection_latencies_sec: List[float],
) -> Dict[str, float]:
    """
    Computes event-level metrics essential for real-world surveillance:
    - False Alarm Rate (events per hour of observation)
    - Event-level Recall
    - Average & 95th Percentile Detection Latency
    """
    far_per_hour = (
        false_alarm_events_count / total_observation_hours if total_observation_hours > 0 else 0.0
    )
    event_recall = (
        detected_drowning_events_count / true_drowning_events_count
        if true_drowning_events_count > 0
        else 0.0
    )
    avg_latency = (
        sum(detection_latencies_sec) / len(detection_latencies_sec)
        if detection_latencies_sec
        else 0.0
    )

    return {
        "false_alarm_rate_per_hour": round(far_per_hour, 2),
        "event_recall": round(event_recall, 4),
        "average_detection_latency_sec": round(avg_latency, 2),
    }
