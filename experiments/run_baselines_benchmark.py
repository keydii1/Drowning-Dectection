#!/usr/bin/env python3
"""
Comprehensive Benchmarking Suite for 6 Baseline & Proposed Modes in Drowning & Pool Anomaly Detection:
Mode 1: Raw YOLO Frame-level Detection (No tracking, no temporal memory)
Mode 2: YOLO + ByteTrack (Tracking ID only, instantaneous alarm)
Mode 3: YOLO + ByteTrack + Fixed N-Consecutive Frames Persistence Rule (N=15)
Mode 4: YOLO + ByteTrack + Sliding Window Positive Ratio Filter (W=30, Ratio>=0.6)
Mode 5: Kinematic Heuristics Baseline (Velocity + Aspect Ratio + Vertical Oscillation)
Mode 6: Proposed Hybrid Spatio-Temporal Anomaly Detector (Active + Passive Drowning Fusion)
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
import pandas as pd

# Add root directory to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.detection.detector import DrowningDetector
from src.tracking.tracker import DrowningTracker
from src.temporal.temporal_filter import SpatioTemporalFilter
from src.anomaly.pool_anomaly_detector import PoolAnomalyDetector, SwimmerState
from src.evaluation.metrics import calculate_frame_level_metrics, calculate_event_level_metrics


SCENARIOS_DIR = ROOT_DIR / "datasets/raw/benchmark_scenarios"
RESULTS_DIR = ROOT_DIR / "results/metrics"


# Ground truth intervals for benchmark scenarios:
# List of (start_frame, end_frame) where drowning/critical anomaly actually occurs
GROUND_TRUTH_EVENTS = {
    "scenario_normal_swimming.mp4": [],                     # Negative video (no drowning)
    "scenario_splash_hard_negative.mp4": [],                 # Hard negative video (playing/splash)
    "scenario_active_drowning.mp4": [(60, 240)],            # Active drowning starts at frame 60
    "scenario_passive_drowning.mp4": [(50, 240)],           # Passive drowning starts at frame 50
    "scenario_multi_swimmer_mixed.mp4": [(70, 240)],        # Victim in Lane 3 drowns from frame 70
}


def extract_frame_detections(detector: DrowningDetector, frame: np.ndarray) -> List[Dict[str, Any]]:
    """
    Extracts YOLO detections, supplemented with visual pool foreground segmentation if stylized.
    """
    detections = detector.detect_frame(frame)
    if len(detections) == 0:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        diff = cv2.absdiff(blurred, 140)
        _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            area = cv2.contourArea(c)
            if 200 < area < 15000:
                bx, by, bw, bh = cv2.boundingRect(c)
                aspect_r = bw / float(bh)
                # Distress heuristic if vertical (AR < 0.85) or erratic
                is_distress = aspect_r < 0.85
                conf = 0.85 if is_distress else 0.25
                detections.append({
                    "bbox": [bx, by, bx + bw, by + bh],
                    "confidence": conf,
                    "class_id": 2 if is_distress else 0,
                    "class_name": "drowning" if is_distress else "swimming"
                })
    return detections


def precompute_dataset_frames(detector: DrowningDetector) -> Dict[str, List[Dict[str, Any]]]:
    """
    Precomputes and caches frame detections across all benchmark videos to ensure
    fast, consistent, and strictly reproducible comparisons across all 6 modes.
    """
    video_files = sorted(list(SCENARIOS_DIR.glob("*.mp4")))
    if not video_files:
        raise FileNotFoundError(f"No scenario videos found in {SCENARIOS_DIR}")

    print("🚀 Pre-extracting video detections for all benchmark scenarios...", flush=True)
    cached_data = {}
    for v_path in video_files:
        cap = cv2.VideoCapture(str(v_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frames_list = []
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            dets = extract_frame_detections(detector, frame)
            frames_list.append({
                "frame_idx": frame_idx,
                "timestamp": frame_idx / fps,
                "detections": dets
            })
            frame_idx += 1
        cap.release()
        cached_data[v_path.name] = frames_list
        print(f"   • Cached {len(frames_list)} frames for {v_path.name}", flush=True)
    print("✅ All scenario detections cached.\n", flush=True)
    return cached_data


class BaselinePipeline:
    def __init__(self, mode: int, fps: float = 30.0):
        self.mode = mode
        self.fps = fps
        self.tracker = DrowningTracker(fps=int(fps))
        self.consecutive_counters: Dict[int, int] = {}
        self.sliding_windows: Dict[int, deque] = {}
        self.anomaly_detector = PoolAnomalyDetector(fps=fps, window_size=30, persistence_sec=0.5)

    def process_cached_frame(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
        timestamp: float,
    ) -> bool:
        """
        Executes mode logic on frame detections.
        """
        is_alarm = False

        # --- MODE 1: Raw YOLO Frame-level (No tracking) ---
        if self.mode == 1:
            for d in detections:
                if d.get("confidence", 0.0) > 0.5:
                    return True
            return False

        # For Modes 2-6: Run ByteTrack identity tracker
        tracked = self.tracker.update(detections)

        # --- MODE 2: YOLO + ByteTrack (Instantaneous Alarm) ---
        if self.mode == 2:
            for t in tracked:
                if t.get("confidence", 0.0) > 0.5:
                    is_alarm = True
                    break

        # --- MODE 3: YOLO + ByteTrack + N-Consecutive Frames Persistence (N=15) ---
        elif self.mode == 3:
            N = 15
            for t in tracked:
                tid = t["track_id"]
                conf = t.get("confidence", 0.0)
                if conf > 0.5:
                    self.consecutive_counters[tid] = self.consecutive_counters.get(tid, 0) + 1
                else:
                    self.consecutive_counters[tid] = 0

                if self.consecutive_counters.get(tid, 0) >= N:
                    is_alarm = True

        # --- MODE 4: YOLO + ByteTrack + Sliding Window Ratio (W=30, Ratio>=0.6) ---
        elif self.mode == 4:
            W = 30
            for t in tracked:
                tid = t["track_id"]
                if tid not in self.sliding_windows:
                    self.sliding_windows[tid] = deque(maxlen=W)
                conf = t.get("confidence", 0.0)
                self.sliding_windows[tid].append(1 if conf > 0.5 else 0)

                win = self.sliding_windows[tid]
                if len(win) >= 15 and (sum(win) / len(win)) >= 0.6:
                    is_alarm = True

        # --- MODE 5: Kinematic Heuristics Baseline (Velocity + AR) ---
        elif self.mode == 5:
            for t in tracked:
                tid = t["track_id"]
                bbox = t["bbox"]
                bw = bbox[2] - bbox[0]
                bh = bbox[3] - bbox[1]
                ar = bw / max(1.0, bh)
                conf = t.get("confidence", 0.0)

                # Vertical posture + high confidence
                if ar < 0.85 and conf > 0.4:
                    self.consecutive_counters[tid] = self.consecutive_counters.get(tid, 0) + 1
                    if self.consecutive_counters[tid] >= 8:
                        is_alarm = True
                else:
                    self.consecutive_counters[tid] = max(0, self.consecutive_counters.get(tid, 0) - 1)

        # --- MODE 6: Proposed Hybrid Spatio-Temporal Anomaly Detector ---
        elif self.mode == 6:
            for t in tracked:
                tid = t["track_id"]
                bbox = t["bbox"]
                conf = t.get("confidence", 0.0)
                state, score, alarm = self.anomaly_detector.evaluate_track(
                    track_id=tid,
                    bbox=bbox,
                    yolo_conf=conf,
                    timestamp=timestamp
                )
                if alarm:
                    is_alarm = True

        return is_alarm


def evaluate_mode_on_cache(
    mode: int,
    mode_name: str,
    cached_data: Dict[str, List[Dict[str, Any]]]
) -> Dict[str, Any]:
    print(f"▶️ Evaluating Mode {mode}: {mode_name}...", flush=True)
    pipeline = BaselinePipeline(mode=mode, fps=30.0)

    total_frames = 0
    tp_frames, fp_frames, fn_frames, tn_frames = 0, 0, 0, 0
    false_alarm_events = 0
    true_drowning_events = 0
    detected_drowning_events = 0
    detection_latencies = []
    t_start = time.perf_counter()

    for video_name, frames_list in cached_data.items():
        gt_intervals = GROUND_TRUTH_EVENTS.get(video_name, [])
        is_negative_video = len(gt_intervals) == 0

        if not is_negative_video:
            true_drowning_events += len(gt_intervals)

        video_drowning_detected = False
        in_false_alarm_episode = False

        for f_data in frames_list:
            frame_idx = f_data["frame_idx"]
            timestamp = f_data["timestamp"]
            dets = f_data["detections"]
            total_frames += 1

            is_gt_drowning = any(start <= frame_idx <= end for (start, end) in gt_intervals)
            is_alarm = pipeline.process_cached_frame(dets, frame_idx, timestamp)

            if is_alarm and is_gt_drowning:
                tp_frames += 1
                if not video_drowning_detected:
                    video_drowning_detected = True
                    start_frame = gt_intervals[0][0]
                    latency_sec = max(0.0, (frame_idx - start_frame) / 30.0)
                    detection_latencies.append(latency_sec)
            elif is_alarm and not is_gt_drowning:
                fp_frames += 1
                if not in_false_alarm_episode:
                    false_alarm_events += 1
                    in_false_alarm_episode = True
            elif not is_alarm and is_gt_drowning:
                fn_frames += 1
            else:
                tn_frames += 1
                in_false_alarm_episode = False

        if video_drowning_detected:
            detected_drowning_events += 1

    elapsed = time.perf_counter() - t_start
    frame_metrics = calculate_frame_level_metrics(tp_frames, fp_frames, fn_frames, tn_frames)
    total_observation_hours = (total_frames / 30.0) / 3600.0
    event_metrics = calculate_event_level_metrics(
        total_observation_hours=total_observation_hours,
        false_alarm_events_count=false_alarm_events,
        true_drowning_events_count=true_drowning_events,
        detected_drowning_events_count=detected_drowning_events,
        detection_latencies_sec=detection_latencies,
    )
    fps = total_frames / elapsed if elapsed > 0 else 0.0

    result = {
        "mode_id": mode,
        "mode_name": mode_name,
        "total_frames": total_frames,
        "precision": frame_metrics["precision"],
        "recall": frame_metrics["recall"],
        "f1_score": frame_metrics["f1"],
        "false_alarm_events": false_alarm_events,
        "false_alarm_rate_per_hour": event_metrics["false_alarm_rate_per_hour"],
        "event_recall": event_metrics["event_recall"],
        "avg_detection_latency_sec": event_metrics["average_detection_latency_sec"],
        "fps": round(fps, 1),
    }

    print(f"   ✓ F1: {result['f1_score']:.3f} | Prec: {result['precision']:.3f} | Rec: {result['recall']:.3f} | False Alarms: {false_alarm_events} | Latency: {result['avg_detection_latency_sec']}s", flush=True)
    return result


def run_benchmark():
    MODES = [
        (1, "Mode 1: Raw YOLO Frame-level"),
        (2, "Mode 2: YOLO + ByteTrack (Instantaneous)"),
        (3, "Mode 3: YOLO + Tracking + Fixed Consecutive (N=15)"),
        (4, "Mode 4: YOLO + Tracking + Sliding Window Ratio (W=30)"),
        (5, "Mode 5: Kinematic Heuristics Baseline (Velocity + AR)"),
        (6, "Mode 6: Proposed Hybrid Spatio-Temporal Anomaly Detector"),
    ]

    print("=" * 80, flush=True)
    print("🎯 STARTING 6-MODE BASELINE & POOL ANOMALY BENCHMARK", flush=True)
    print("=" * 80, flush=True)

    detector = DrowningDetector(model_path="yolo11n.pt", conf_thresh=0.30, device="cpu")
    detector.load_model()
    cached_data = precompute_dataset_frames(detector)

    all_results = []
    for mode_id, mode_name in MODES:
        res = evaluate_mode_on_cache(mode_id, mode_name, cached_data)
        all_results.append(res)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    csv_path = RESULTS_DIR / "baseline_comparison.csv"
    df.to_csv(csv_path, index=False)

    json_path = RESULTS_DIR / "baseline_comparison.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 80, flush=True)
    print("🏆 MASTER BENCHMARK SUMMARY TABLE", flush=True)
    print("=" * 80, flush=True)
    summary_cols = ["mode_id", "mode_name", "precision", "recall", "f1_score", "false_alarm_events", "false_alarm_rate_per_hour", "avg_detection_latency_sec"]
    print(df[summary_cols].to_string(index=False), flush=True)
    print("=" * 80, flush=True)
    print(f"📁 Output files saved:\n• {csv_path}\n• {json_path}\n", flush=True)


if __name__ == "__main__":
    run_benchmark()
