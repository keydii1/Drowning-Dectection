#!/usr/bin/env python3
"""
Master End-to-End Pipeline for Drowning Detection & False Alarm Reduction.
Combines: YOLO Object Detection + ByteTrack Multi-Object Tracking + Spatio-Temporal Filtering.
"""

import argparse
import os
import sys
import time
from typing import Optional
import cv2
import numpy as np

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.detection.detector import DrowningDetector
from src.tracking.tracker import DrowningTracker
from src.temporal.temporal_filter import SpatioTemporalFilter
from src.visualization.visualizer import DrowningVisualizer


def create_synthetic_demo_video(output_path: str, duration_sec: int = 6, fps: int = 30) -> str:
    """
    Creates a synthetic pool video with simulated swimmers and a drowning event
    so the pipeline can be tested and verified immediately without external downloads.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    print(f"🎬 Creating synthetic test video at: {output_path} ({total_frames} frames)...")

    # Person 1 (Freestyle swimming across)
    p1_x, p1_y = 50, 150
    # Person 2 (Starts normal, then exhibits distress after 2 seconds)
    p2_x, p2_y = 400, 250

    for f in range(total_frames):
        # Pool water background (gradient blue with moving ripples)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (180, 110, 40)  # Aqua blue in BGR

        # Water ripple noise
        noise = (np.sin(f * 0.2 + np.linspace(0, 10, width)) * 15).astype(np.int16)
        frame[:, :, 0] = np.clip(frame[:, :, 0].astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Draw Person 1 (Moving smoothly - Normal swimmer)
        p1_x_cur = int(p1_x + (f * 1.5) % (width - 100))
        cv2.ellipse(frame, (p1_x_cur + 30, p1_y + 15), (35, 18), 0, 0, 360, (50, 80, 200), -1)
        cv2.circle(frame, (p1_x_cur + 65, p1_y + 15), 10, (180, 200, 230), -1)

        # Draw Person 2 (In distress after frame 60)
        is_distressed = f > 60
        bobbing = int(np.sin(f * 0.8) * 8) if is_distressed else int(np.sin(f * 0.2) * 3)
        p2_y_cur = p2_y + bobbing

        # Head / Arms splashing
        if is_distressed:
            # White splashing droplets
            for _ in range(8):
                rx = p2_x + np.random.randint(-25, 25)
                ry = p2_y_cur + np.random.randint(-20, 20)
                cv2.circle(frame, (rx, ry), np.random.randint(2, 5), (255, 255, 255), -1)

        cv2.ellipse(frame, (p2_x, p2_y_cur), (20, 28), 0, 0, 360, (40, 70, 190), -1)
        cv2.circle(frame, (p2_x, p2_y_cur - 18), 12, (180, 200, 230), -1)

        out.write(frame)

    out.release()
    print("✅ Synthetic test video generated successfully.")
    return output_path


def process_video(
    video_source: str,
    output_path: Optional[str] = None,
    model_path: str = "yolo11n.pt",
    conf_thresh: float = 0.35,
    persistence_thresh: int = 15,
    device: str = "cpu",
    show_window: bool = False,
):
    print("=" * 60)
    print("🚀 DROWNING DETECTION & FALSE ALARM REDUCTION PIPELINE")
    print(f"• Video Source: {video_source}")
    print(f"• Model: {model_path} (Device: {device})")
    print(f"• Persistence Threshold: {persistence_thresh} frames")
    print("=" * 60)

    # Initialize modules
    detector = DrowningDetector(model_path=model_path, conf_thresh=conf_thresh, device=device)
    print("📦 Loading YOLO model...")
    detector.load_model()
    print("✅ YOLO loaded.")

    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video source: {video_source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tracker = DrowningTracker(fps=int(fps))
    temporal_filter = SpatioTemporalFilter(persistence_thresh_frames=persistence_thresh, fps=fps)
    visualizer = DrowningVisualizer()

    writer = None
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    t_start = time.time()
    alert_event_count = 0

    print(f"▶️ Processing video ({width}x{height} @ {fps:.1f} fps, {total_frames} frames)...")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        current_timestamp = frame_idx / fps

        # 1. Detection
        detections = detector.detect_frame(frame)

        # 2. Multi-Object Tracking
        tracked = tracker.update(detections)

        # 3. Spatio-Temporal Filtering & False Alarm Suppression
        any_alert = False
        for obj in tracked:
            tid = obj["track_id"]
            # For demonstration: class_id or confidence signals distress
            # In trained model: class 'drowning' vs 'swimming'
            score = obj["confidence"]
            bbox = obj["bbox"]

            is_alert = temporal_filter.update_track(
                track_id=tid,
                drowning_score=score,
                bbox=bbox,
                timestamp=current_timestamp,
            )
            obj["is_alert"] = is_alert
            if is_alert:
                any_alert = True

        if any_alert:
            alert_event_count += 1

        # 4. Visualization & Annotations
        elapsed = time.time() - t_start
        current_fps = frame_idx / elapsed if elapsed > 0 else 0
        fps_text = f"FPS: {current_fps:.1f} | Frame: {frame_idx}/{total_frames}"

        annotated_frame = visualizer.draw_frame(
            frame=frame,
            tracked_objects=tracked,
            any_alert=any_alert,
            fps_text=fps_text,
        )

        if writer:
            writer.write(annotated_frame)

        if show_window:
            cv2.imshow("Drowning Detection Surveillance", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    if show_window:
        cv2.destroyAllWindows()

    total_time = time.time() - t_start
    avg_fps = frame_idx / total_time if total_time > 0 else 0

    print("\n" + "=" * 60)
    print("✅ PROCESSING COMPLETE")
    print(f"• Total Frames Processed: {frame_idx}")
    print(f"• Elapsed Time: {total_time:.2f}s (Average FPS: {avg_fps:.1f})")
    print(f"• Alert Trigger Events: {alert_event_count}")
    if output_path:
        print(f"• Output Video Saved: {output_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Real-Time Drowning Detection & False Alarm Reduction")
    parser.add_argument("--video", type=str, default="", help="Path to input video file or RTSP stream")
    parser.add_argument("--output", type=str, default="results/videos/output_annotated.mp4", help="Output annotated video path")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="YOLO model path or name")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--persistence", type=int, default=15, help="Temporal persistence frames threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device: 'cpu', 'cuda', 'mps'")
    parser.add_argument("--demo", action="store_true", help="Run with auto-generated synthetic pool demo")

    args = parser.parse_args()

    # If no video is specified or demo flag is set, generate a synthetic video
    video_path = args.video
    if not video_path or args.demo:
        demo_video_path = "datasets/raw/demo_pool_sample.mp4"
        create_synthetic_demo_video(demo_video_path, duration_sec=5, fps=30)
        video_path = demo_video_path

    process_video(
        video_source=video_path,
        output_path=args.output,
        model_path=args.model,
        conf_thresh=args.conf,
        persistence_thresh=args.persistence,
        device=args.device,
        show_window=False,
    )


if __name__ == "__main__":
    main()
