#!/usr/bin/env python3
"""
Baseline Training Script for Drowning Detection using Ultralytics YOLO11.
Trained on the Underwater Drowning Detection Dataset (Figshare).
Accelerated by Apple Silicon GPU (MPS) if available.
"""

import argparse
import sys
from pathlib import Path
import torch
from ultralytics import YOLO

DATA_YAML = Path("/Users/hohoangson/Documents/SienceResearch/datasets/raw/underwater_drowning/data.yaml")
DEFAULT_WEIGHTS = "yolo11n.pt"

def train(epochs: int = 50, batch: int = 16, imgsz: int = 640, device: str = None):
    if not DATA_YAML.exists():
        print(f"Error: Dataset configuration file not found at: {DATA_YAML}")
        print("Please ensure download_figshare.py has completed successfully.")
        sys.exit(1)

    # Determine optimal device
    if device is None:
        if torch.backends.mps.is_available():
            device = "mps"
            print("🚀 Using Apple Silicon GPU acceleration (MPS)!")
        elif torch.cuda.is_available():
            device = "cuda"
            print("🚀 Using CUDA GPU acceleration!")
        else:
            device = "cpu"
            print("ℹ️ Using CPU for training.")

    print("=" * 60)
    print("🎯 STARTING BASELINE YOLO11 TRAINING")
    print(f"• Dataset YAML: {DATA_YAML}")
    print(f"• Base Model: {DEFAULT_WEIGHTS}")
    print(f"• Epochs: {epochs}")
    print(f"• Batch Size: {batch}")
    print(f"• Image Size: {imgsz}")
    print(f"• Device: {device}")
    print("=" * 60)

    model = YOLO(DEFAULT_WEIGHTS)

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        project="experiments/baseline",
        name="yolo11n_underwater_baseline",
        seed=42,
        exist_ok=True,
        save=True,
        val=True,
        plots=True,
        verbose=True
    )

    print("\n✅ Training complete!")
    print(f"Best model weights saved at: experiments/baseline/yolo11n_underwater_baseline/weights/best.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Baseline YOLO11 on Underwater Drowning Dataset")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50)")
    parser.add_argument("--batch", type=int, default=16, help="Batch size (default: 16)")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution (default: 640)")
    parser.add_argument("--device", type=str, default=None, help="Device to use (mps, cuda, cpu)")
    args = parser.parse_args()

    train(epochs=args.epochs, batch=args.batch, imgsz=args.imgsz, device=args.device)
