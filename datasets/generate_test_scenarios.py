#!/usr/bin/env python3
"""
Test Scenario Generator for Swimming Pool Behavior & Anomaly Evaluation.
Generates 5 realistic benchmark video scenarios:
1. scenario_normal_swimming.mp4: Smooth swimming laps across lanes (True Negative).
2. scenario_splash_hard_negative.mp4: Playful splashing & active arm waving (Hard Negative).
3. scenario_active_drowning.mp4: Normal swimmer transitions into active panic drowning (True Positive).
4. scenario_passive_drowning.mp4: Swimmer stops moving, sinks, floating motionless (Passive Drowning).
5. scenario_multi_swimmer_mixed.mp4: Complex scene with 3 swimmers (1 normal, 1 playing, 1 drowning).
"""

import os
from pathlib import Path
import cv2
import numpy as np

OUTPUT_DIR = Path("/Users/hohoangson/Documents/SienceResearch/datasets/raw/benchmark_scenarios")


def draw_water_background(frame, width, height, frame_idx):
    """Renders realistic pool water gradient with moving surface ripples."""
    frame[:] = (180, 110, 40)  # Pool aqua blue (BGR)
    # Ripple noise
    ripple = (np.sin(frame_idx * 0.15 + np.linspace(0, 12, width)) * 12).astype(np.int16)
    frame[:, :, 0] = np.clip(frame[:, :, 0].astype(np.int16) + ripple, 0, 255).astype(np.uint8)
    frame[:, :, 1] = np.clip(frame[:, :, 1].astype(np.int16) + ripple // 2, 0, 255).astype(np.uint8)
    
    # Pool lane markers
    lane_y1 = height // 3
    lane_y2 = (2 * height) // 3
    cv2.line(frame, (0, lane_y1), (width, lane_y1), (220, 170, 90), 2)
    cv2.line(frame, (0, lane_y2), (width, lane_y2), (220, 170, 90), 2)


def generate_scenario_1_normal(output_path: str, fps: int = 30, duration_sec: int = 8):
    """Scenario 1: Normal freestyle swimmer moving across the pool."""
    width, height = 640, 480
    total_frames = fps * duration_sec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    p_y = height // 2
    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_water_background(frame, width, height, f)

        # Smooth horizontal progression across pool
        p_x = int(50 + (f * 2.2) % (width - 120))
        subtle_bob = int(np.sin(f * 0.25) * 3)

        # Swimmer horizontal body (width > height)
        cv2.ellipse(frame, (p_x + 35, p_y + subtle_bob), (45, 18), 0, 0, 360, (50, 80, 200), -1)
        cv2.circle(frame, (p_x + 75, p_y + subtle_bob), 11, (180, 200, 230), -1)

        out.write(frame)
    out.release()
    print(f"✅ Generated Scenario 1 (Normal): {output_path}")


def generate_scenario_2_splash_hard_negative(output_path: str, fps: int = 30, duration_sec: int = 8):
    """Scenario 2: Vigorous playful splashing (Hard Negative - tests False Alarm filter)."""
    width, height = 640, 480
    total_frames = fps * duration_sec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    p_x, p_y = width // 2, height // 2
    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_water_background(frame, width, height, f)

        # Swimmer plays in place with brief splash bursts (e.g. frames 60-90)
        is_burst = 60 <= f <= 95
        shift_x = int(p_x + np.sin(f * 0.1) * 30)
        shift_y = int(p_y + np.cos(f * 0.1) * 15)

        cv2.ellipse(frame, (shift_x, shift_y), (35, 25), 0, 0, 360, (55, 95, 210), -1)
        cv2.circle(frame, (shift_x + 25, shift_y - 10), 12, (180, 200, 230), -1)

        # White splash droplets during play burst
        if is_burst:
            for _ in range(12):
                sx = shift_x + np.random.randint(-40, 40)
                sy = shift_y + np.random.randint(-30, 30)
                cv2.circle(frame, (sx, sy), np.random.randint(2, 6), (255, 255, 255), -1)

        out.write(frame)
    out.release()
    print(f"✅ Generated Scenario 2 (Splash Hard Negative): {output_path}")


def generate_scenario_3_active_drowning(output_path: str, fps: int = 30, duration_sec: int = 8):
    """Scenario 3: Swimmer transitions from normal swim to active panic drowning at frame 60."""
    width, height = 640, 480
    total_frames = fps * duration_sec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    start_drowning_frame = 60
    p_x = 100
    p_y = height // 2

    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_water_background(frame, width, height, f)

        if f < start_drowning_frame:
            # Normal horizontal swimming
            p_x += 2
            cv2.ellipse(frame, (p_x, p_y), (40, 18), 0, 0, 360, (50, 80, 200), -1)
            cv2.circle(frame, (p_x + 35, p_y), 11, (180, 200, 230), -1)
        else:
            # Active panic drowning: upright posture (height > width), vertical bobbing, in place thrashing
            bob = int(np.sin((f - start_drowning_frame) * 0.7) * 16)
            jitter_x = np.random.randint(-4, 5)
            drown_x = p_x + jitter_x
            drown_y = p_y + bob

            # Upright vertical body
            cv2.ellipse(frame, (drown_x, drown_y), (18, 38), 0, 0, 360, (40, 70, 190), -1)
            cv2.circle(frame, (drown_x, drown_y - 25), 11, (180, 200, 230), -1)

            # Splashing
            for _ in range(10):
                sx = drown_x + np.random.randint(-30, 30)
                sy = drown_y + np.random.randint(-25, 25)
                cv2.circle(frame, (sx, sy), np.random.randint(2, 5), (255, 255, 255), -1)

        out.write(frame)
    out.release()
    print(f"✅ Generated Scenario 3 (Active Drowning): {output_path}")


def generate_scenario_4_passive_drowning(output_path: str, fps: int = 30, duration_sec: int = 8):
    """Scenario 4: Passive drowning - swimmer stops moving, becomes motionless and submerges."""
    width, height = 640, 480
    total_frames = fps * duration_sec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    p_x = width // 2
    p_y = height // 2

    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_water_background(frame, width, height, f)

        if f < 50:
            # Slow movement initially
            shift_x = int(p_x + np.sin(f * 0.2) * 15)
            shift_y = p_y
            size_w, size_h = 35, 20
        else:
            # Complete motionlessness, sinking downward, shrinking visibility
            sink_offset = min(60, int((f - 50) * 0.4))
            shift_x = p_x
            shift_y = p_y + sink_offset
            decay = max(0.4, 1.0 - (f - 50) * 0.004)
            size_w, size_h = int(35 * decay), int(20 * decay)

        # Body darkening as it submerges
        color = (max(20, 50 - f // 6), max(40, 80 - f // 6), max(80, 180 - f // 4))
        cv2.ellipse(frame, (shift_x, shift_y), (size_w, size_h), 0, 0, 360, color, -1)
        cv2.circle(frame, (shift_x + size_w // 2, shift_y), int(10 * decay) if f >= 50 else 10, (140, 160, 190), -1)

        out.write(frame)
    out.release()
    print(f"✅ Generated Scenario 4 (Passive Drowning): {output_path}")


def generate_scenario_5_multi_swimmer_mixed(output_path: str, fps: int = 30, duration_sec: int = 8):
    """Scenario 5: Multi-swimmer pool with Normal Swimmer, Splasher, and Drowning victim."""
    width, height = 640, 480
    total_frames = fps * duration_sec
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for f in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        draw_water_background(frame, width, height, f)

        # Person 1 (Lane 1): Normal Swimmer moving across
        p1_x = int(60 + (f * 2.0) % (width - 100))
        p1_y = 90
        cv2.ellipse(frame, (p1_x, p1_y), (40, 16), 0, 0, 360, (60, 90, 210), -1)
        cv2.circle(frame, (p1_x + 35, p1_y), 10, (180, 200, 230), -1)

        # Person 2 (Lane 2): Playing / Splashing intermittently
        p2_x = 420 + int(np.sin(f * 0.15) * 20)
        p2_y = 240
        cv2.ellipse(frame, (p2_x, p2_y), (30, 22), 0, 0, 360, (50, 120, 200), -1)
        cv2.circle(frame, (p2_x + 20, p2_y - 8), 11, (180, 200, 230), -1)
        if 40 <= f <= 75:
            for _ in range(8):
                cv2.circle(frame, (p2_x + np.random.randint(-30, 30), p2_y + np.random.randint(-20, 20)), np.random.randint(2, 5), (255, 255, 255), -1)

        # Person 3 (Lane 3): Starts swimming then enters drowning state at frame 70
        p3_y = 390
        if f < 70:
            p3_x = int(120 + f * 1.2)
            cv2.ellipse(frame, (p3_x, p3_y), (38, 16), 0, 0, 360, (40, 70, 190), -1)
            cv2.circle(frame, (p3_x + 32, p3_y), 10, (180, 200, 230), -1)
        else:
            drown_x = 120 + int(70 * 1.2) + np.random.randint(-3, 4)
            drown_bob = int(np.sin((f - 70) * 0.8) * 14)
            cv2.ellipse(frame, (drown_x, p3_y + drown_bob), (16, 35), 0, 0, 360, (30, 60, 180), -1)
            cv2.circle(frame, (drown_x, p3_y + drown_bob - 22), 10, (180, 200, 230), -1)
            for _ in range(10):
                cv2.circle(frame, (drown_x + np.random.randint(-25, 25), p3_y + drown_bob + np.random.randint(-20, 20)), np.random.randint(2, 5), (255, 255, 255), -1)

        out.write(frame)
    out.release()
    print(f"✅ Generated Scenario 5 (Multi-Swimmer Mixed): {output_path}")


def generate_all_scenarios():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    generate_scenario_1_normal(str(OUTPUT_DIR / "scenario_normal_swimming.mp4"))
    generate_scenario_2_splash_hard_negative(str(OUTPUT_DIR / "scenario_splash_hard_negative.mp4"))
    generate_scenario_3_active_drowning(str(OUTPUT_DIR / "scenario_active_drowning.mp4"))
    generate_scenario_4_passive_drowning(str(OUTPUT_DIR / "scenario_passive_drowning.mp4"))
    generate_scenario_5_multi_swimmer_mixed(str(OUTPUT_DIR / "scenario_multi_swimmer_mixed.mp4"))
    print(f"\n🎉 All 5 pool evaluation scenarios generated in: {OUTPUT_DIR}")


if __name__ == "__main__":
    generate_all_scenarios()
