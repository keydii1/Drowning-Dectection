import os
import sys
import shutil
import argparse
from pathlib import Path

DEFAULT_WORKSPACE = "drowningdetectiontracking"
DEFAULT_PROJECT = "drowningdetectiontracking"
DEFAULT_VERSION = 1
TARGET_DIR = Path("/Users/hohoangson/Documents/SienceResearch/datasets/raw/roboflow_drowning")

def check_disk_space(required_gb=4.0):
    stat = shutil.disk_usage(TARGET_DIR.parent)
    free_gb = stat.free / (1024 ** 3)
    print(f"Available disk space: {free_gb:.2f} GB (Required: ~{required_gb:.2f} GB)")
    if free_gb < required_gb:
        print(f"\n[!] WARNING: Insufficient disk space!")
        print(f"Roboflow DrowningDetectionTracking dataset has ~9,530 images (~3-5 GB).")
        print(f"Current free space is only {free_gb:.2f} GB.")
        print("Please free up at least 4 GB before running this download.")
        return False
    return True

def download_roboflow(api_key, workspace=DEFAULT_WORKSPACE, project_name=DEFAULT_PROJECT, version=DEFAULT_VERSION):
    if not check_disk_space():
        sys.exit(1)

    try:
        from roboflow import Roboflow
    except ImportError:
        print("Roboflow library is not installed. Run: pip install roboflow")
        sys.exit(1)

    print(f"\nConnecting to Roboflow workspace: {workspace}, project: {project_name}, version: {version}...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project_name)
    version_obj = project.version(version)

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset in YOLOv8/YOLO11 format to {TARGET_DIR}...")
    dataset = version_obj.download("yolov8", location=str(TARGET_DIR))
    print(f"\n=== Roboflow dataset successfully downloaded to: {dataset.location} ===")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download Roboflow Drowning Detection Dataset")
    parser.add_argument("--api-key", help="Roboflow API key (can also be set via ROBOFLOW_API_KEY environment variable)")
    parser.add_argument("--workspace", default=DEFAULT_WORKSPACE, help="Roboflow workspace name")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="Roboflow project name")
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION, help="Roboflow project version")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("=" * 60)
        print("Roboflow Universe requires an API key to download datasets.")
        print("To get your free API key:")
        print("1. Go to https://universe.roboflow.com/drowningdetectiontracking/drowningdetectiontracking")
        print("2. Sign in or create a free account.")
        print("3. Click 'Download Dataset' -> Format: YOLOv8 -> 'Show download code'")
        print("4. Copy the api_key shown in the snippet.")
        print("=" * 60)
        try:
            api_key = input("Enter your Roboflow API key: ").strip()
        except EOFError:
            pass

    if not api_key:
        print("Error: No API key provided. Exiting.")
        sys.exit(1)

    download_roboflow(api_key, args.workspace, args.project, args.version)
