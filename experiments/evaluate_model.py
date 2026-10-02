#!/usr/bin/env python3
"""
Evaluation and Inference Script for Drowning Detection YOLO models.
Generates evaluation metrics (mAP, Precision, Recall) and saves sample detection results.
"""

import argparse
import sys
from pathlib import Path
from ultralytics import YOLO

DATA_YAML = Path("/Users/hohoangson/Documents/SienceResearch/datasets/raw/underwater_drowning/data.yaml")

def evaluate(model_path: str, data_yaml: str = str(DATA_YAML), imgsz: int = 640):
    if not Path(model_path).exists():
        print(f"Error: Model file '{model_path}' not found.")
        sys.exit(1)

    print("=" * 60)
    print("📊 EVALUATING MODEL PERFORMANCE")
    print(f"• Model: {model_path}")
    print(f"• Dataset: {data_yaml}")
    print("=" * 60)

    model = YOLO(model_path)
    metrics = model.val(data=data_yaml, imgsz=imgsz, split="val", plots=True)

    print("\n📈 EVALUATION METRICS SUMMARY:")
    print(f"• mAP50:     {metrics.box.map50:.4f}")
    print(f"• mAP50-95:  {metrics.box.map:.4f}")
    print(f"• Precision: {metrics.box.mp:.4f}")
    print(f"• Recall:    {metrics.box.mr:.4f}")

    print("\nClass-wise mAP50:")
    for i, name in model.names.items():
        if i < len(metrics.box.maps):
            print(f"  - {name:<12}: {metrics.box.maps[i]:.4f}")

def run_sample_inference(model_path: str, source: str, conf: float = 0.35, output_dir: str = "results/predictions"):
    model = YOLO(model_path)
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    results = model.predict(source=source, conf=conf, save=True, project=output_dir, name="predict")
    print(f"Predictions saved to {output_dir}/predict")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO Model on Drowning Detection Dataset")
    parser.add_argument("--model", type=str, default="yolo11n.pt", help="Path to YOLO model checkpoint (.pt)")
    parser.add_argument("--data", type=str, default=str(DATA_YAML), help="Path to data.yaml")
    parser.add_argument("--predict", type=str, default=None, help="Path to image/video or folder for inference")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    args = parser.parse_args()

    if args.predict:
        run_sample_inference(args.model, args.predict, conf=args.conf)
    else:
        evaluate(args.model, data_yaml=args.data)
