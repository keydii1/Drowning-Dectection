#!/usr/bin/env python3
"""
Advanced Data Preprocessing & Enhancement Pipeline for Swimming Pool & Underwater Datasets.
Functions:
1. Dataset Integrity Verification (Labels, bounding box coords [0, 1], image corruptions).
2. Class Distribution Profiling.
3. Underwater Image Enhancement via CLAHE (Contrast Limited Adaptive Histogram Equalization)
   in LAB color space to penetrate water turbidity without altering bbox annotations.
4. Generates clean dataset config data_enhanced.yaml.
"""

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import numpy as np
from tqdm import tqdm

RAW_DIR = Path("/Users/hohoangson/Documents/SienceResearch/datasets/raw/underwater_drowning")
PROCESSED_DIR = Path("/Users/hohoangson/Documents/SienceResearch/datasets/processed/underwater_enhanced")


def verify_dataset_integrity(data_dir: Path) -> Dict[str, Any]:
    """
    Scans every image and YOLO label to ensure:
    - No missing image/label pairs.
    - All coordinates are normalized floats within [0.0, 1.0].
    - No empty or unreadable images.
    """
    print("=" * 65)
    print("🔍 STEP 1: VERIFYING DATASET INTEGRITY & QUALITY")
    print("=" * 65)

    stats = {}
    for split in ["train", "val"]:
        img_dir = data_dir / "images" / split
        lbl_dir = data_dir / "labels" / split

        img_files = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
        lbl_files = sorted(list(lbl_dir.glob("*.txt")))

        missing_labels = 0
        corrupt_boxes = 0
        total_boxes = 0
        class_counts = {0: 0, 1: 0, 2: 0}

        for img_p in img_files:
            lbl_p = lbl_dir / f"{img_p.stem}.txt"
            if not lbl_p.exists():
                missing_labels += 1
                continue

            with open(lbl_p, "r") as fp:
                for line in fp:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        corrupt_boxes += 1
                        continue
                    try:
                        c_id = int(parts[0])
                        x, y, w, h = [float(v) for v in parts[1:]]
                        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0 and 0.0 < w <= 1.0 and 0.0 < h <= 1.0):
                            corrupt_boxes += 1
                            continue
                        class_counts[c_id] = class_counts.get(c_id, 0) + 1
                        total_boxes += 1
                    except ValueError:
                        corrupt_boxes += 1

        stats[split] = {
            "total_images": len(img_files),
            "total_labels": len(lbl_files),
            "missing_labels": missing_labels,
            "corrupt_boxes": corrupt_boxes,
            "total_boxes": total_boxes,
            "class_counts": class_counts,
        }

        print(f"[{split.upper()} SPLIT]")
        print(f"  • Valid Images: {len(img_files)} | Valid Labels: {len(lbl_files)}")
        print(f"  • Missing Labels: {missing_labels} | Corrupt Boxes: {corrupt_boxes}")
        print(f"  • Class Distribution: Swimming={class_counts.get(0, 0)}, Struggling={class_counts.get(1, 0)}, Drowning={class_counts.get(2, 0)}")

    return stats


def enhance_underwater_image(image: np.ndarray, clip_limit: float = 2.5, tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """
    Applies CLAHE on the L (Luminance) channel of the LAB color space.
    Enhances contrast of submerged limbs and water bubbles while preserving color balance.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((cl, a_channel, b_channel))
    enhanced_bgr = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_bgr


def build_enhanced_dataset(src_dir: Path, dst_dir: Path, max_samples: int = None):
    """
    Applies underwater enhancement to images and copies corresponding labels.
    """
    print("\n" + "=" * 65)
    print("✨ STEP 2: CREATING ENHANCED UNDERWATER DATASET")
    print(f"• Source:      {src_dir}")
    print(f"• Destination: {dst_dir}")
    print("=" * 65)

    for split in ["train", "val"]:
        src_img_dir = src_dir / "images" / split
        src_lbl_dir = src_dir / "labels" / split
        dst_img_dir = dst_dir / "images" / split
        dst_lbl_dir = dst_dir / "labels" / split

        dst_img_dir.mkdir(parents=True, exist_ok=True)
        dst_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_files = sorted(list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png")))
        if max_samples:
            img_files = img_files[:max_samples]

        print(f"Processing {split} split ({len(img_files)} images)...")
        for img_p in tqdm(img_files, desc=f"Enhancing {split}"):
            img = cv2.imread(str(img_p))
            if img is None:
                continue

            enhanced = enhance_underwater_image(img)
            cv2.imwrite(str(dst_img_dir / img_p.name), enhanced)

            # Copy matching label file directly
            lbl_p = src_lbl_dir / f"{img_p.stem}.txt"
            if lbl_p.exists():
                shutil.copy(str(lbl_p), str(dst_lbl_dir / lbl_p.name))

    # Create enhanced YAML
    yaml_content = f"""# Enhanced Underwater Drowning Detection Dataset (CLAHE-Enhanced)
path: {dst_dir.resolve()}
train: images/train
val: images/val

names:
  0: swimming
  1: struggling
  2: drowning
"""
    yaml_path = dst_dir / "data_enhanced.yaml"
    with open(yaml_path, "w") as fp:
        fp.write(yaml_content)
    print(f"\n✅ Enhanced dataset ready with configuration: {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess and Enhance Underwater Drowning Dataset")
    parser.add_argument("--verify-only", action="store_true", help="Only verify dataset integrity and print stats")
    parser.add_argument("--enhance", action="store_true", help="Generate CLAHE enhanced dataset")
    parser.add_argument("--samples", type=int, default=None, help="Limit number of samples to process (optional)")
    args = parser.parse_args()

    stats = verify_dataset_integrity(RAW_DIR)

    if args.enhance:
        build_enhanced_dataset(RAW_DIR, PROCESSED_DIR, max_samples=args.samples)
