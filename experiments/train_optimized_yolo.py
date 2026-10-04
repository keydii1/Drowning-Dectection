#!/usr/bin/env python3
"""
Production-Grade YOLO11 Hyperparameter Optimization & Fine-Tuning Script.
Optimized specifically for the Underwater Drowning Detection Dataset.
Features:
- Apple Silicon MPS Hardware Acceleration
- Cosine Annealing Learning Rate Schedule (cos_lr=True)
- AdamW Optimizer with Weight Decay
- Aquatic-Tuned Data Augmentations (HSV, Mosaic, Mixup, Flip)
- Automated Metric Evaluation & Checkpoint Logging
"""

import argparse
import os
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_YAML = ROOT_DIR / "datasets/raw/underwater_drowning/data.yaml"


def train_optimized(
    model_name: str = "yolo11n.pt",
    epochs: int = 50,
    batch: int = 16,
    imgsz: int = 640,
    patience: int = 15,
    device: str = None,
    experiment_name: str = "yolo11n_drowning_optimized"
):
    if not DATA_YAML.exists():
        print(f"Error: Dataset YAML not found at: {DATA_YAML}")
        sys.exit(1)

    # 1. Device selection
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            print("🚀 Hardware Acceleration: Apple Silicon GPU (MPS) Enabled!")
        elif torch.cuda.is_available():
            device = "cuda"
            print("🚀 Hardware Acceleration: NVIDIA CUDA Enabled!")
        else:
            device = "cpu"
            print("ℹ️ Device: CPU (MPS not available).")

    print("\n" + "=" * 70)
    print("🔥 STARTING PRODUCTION-GRADE YOLO11 FINE-TUNING")
    print(f"• Base Model:        {model_name}")
    print(f"• Dataset YAML:      {DATA_YAML}")
    print(f"• Epochs:            {epochs} (Early Stopping Patience: {patience})")
    print(f"• Batch Size:        {batch}")
    print(f"• Image Resolution:  {imgsz}x{imgsz}")
    print(f"• Optimizer:         AdamW with Cosine Annealing Schedule")
    print(f"• Device:            {device}")
    print("=" * 70 + "\n", flush=True)

    # 2. Load model
    model = YOLO(model_name)

    # 3. Fine-tuning with optimal hyperparameters for aquatic drowning detection
    train_results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        patience=patience,
        device=device,
        # Optimizer & Learning Rate Schedule
        optimizer="AdamW",
        lr0=0.001,             # Initial learning rate
        lrf=0.01,              # Final learning rate (lr0 * lrf)
        cos_lr=True,           # Cosine learning rate scheduler
        warmup_epochs=3.0,     # Warmup period
        weight_decay=0.0005,   # Regularization to prevent overfitting
        # Loss Gains
        box=7.5,
        cls=0.5,
        dfl=1.5,
        # Aquatic-Tuned Augmentations
        hsv_h=0.015,           # Color hue variation for water tint
        hsv_s=0.4,             # Saturation variation
        hsv_v=0.4,             # Value/brightness variation
        degrees=10.0,          # Slight tilt rotation
        translate=0.1,         # Swimmer horizontal/vertical shift
        scale=0.2,             # Scale variation (near vs far swimmers)
        fliplr=0.5,            # Horizontal flip (left-to-right swimming)
        mosaic=0.8,            # Mosaic multi-target augmentation
        mixup=0.1,             # Mixup augmentation
        close_mosaic=10,       # Turn off mosaic for last 10 epochs for fine localization
        # Checkpoints & Project
        project="experiments/trained_models",
        name=experiment_name,
        seed=42,
        exist_ok=True,
        save=True,
        val=True,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 70)
    print("✅ FINE-TUNING COMPLETE!")
    best_weights = Path(f"experiments/trained_models/{experiment_name}/weights/best.pt")
    print(f"🏆 Best Checkpoint Saved at: {best_weights}")
    print("=" * 70)

    # 4. Run full validation benchmark on the best checkpoint
    print("\n📊 RUNNING VALIDATION EVALUATION ON BEST WEIGHTS...", flush=True)
    best_model = YOLO(str(best_weights))
    val_results = best_model.val(data=str(DATA_YAML), imgsz=imgsz, split="val", device=device)

    print("\n" + "=" * 70)
    print("🎯 FINAL VALIDATION METRICS:")
    print(f"• Overall mAP@50:     {val_results.box.map50 * 100:.2f}%")
    print(f"• Overall mAP@50-95:  {val_results.box.map * 100:.2f}%")
    print(f"• Precision:          {val_results.box.mp * 100:.2f}%")
    print(f"• Recall:             {val_results.box.mr * 100:.2f}%")
    print("-" * 70)
    print("Class-wise Performance:")
    for idx, cname in best_model.names.items():
        if idx < len(val_results.box.maps):
            map_cls = val_results.box.maps[idx]
            print(f"  - Class {idx} ({cname:<12}): mAP50-95 = {map_cls * 100:.2f}%")
    print("=" * 70 + "\n")

    return best_weights


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO11 for Peak Drowning Detection Performance")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--device", type=str, default=None, help="Device (mps, cuda, cpu)")
    parser.add_argument("--name", type=str, default="yolo11n_drowning_optimized", help="Experiment name")
    args = parser.parse_args()

    train_optimized(
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        device=args.device,
        experiment_name=args.name,
    )
