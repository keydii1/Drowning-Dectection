import os
import sys
import shutil
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm

DOWNLOAD_URL = "https://ndownloader.figshare.com/files/56039351"
TARGET_DIR = Path("/Users/hohoangson/Documents/SienceResearch/datasets/raw/underwater_drowning")
ZIP_PATH = Path("/Users/hohoangson/Documents/SienceResearch/datasets/raw/figshare_temp.zip")

def check_disk_space(required_gb=3.5):
    stat = shutil.disk_usage(TARGET_DIR.parent)
    free_gb = stat.free / (1024 ** 3)
    print(f"Available disk space: {free_gb:.2f} GB (Required for download+extract: ~{required_gb:.2f} GB)")
    if free_gb < required_gb:
        print(f"WARNING: Low disk space! Only {free_gb:.2f} GB available.")
        return False
    return True

def download_file(url, dest_path):
    print(f"Downloading from {url} to {dest_path}...")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, stream=True, headers=headers, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as f, tqdm(
        desc=dest_path.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))
    print("Download completed!")

def extract_and_organize(zip_path, dest_dir):
    print(f"Extracting {zip_path} to {dest_dir}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest_dir)
    print("Extraction completed!")
    
    # Remove temporary zip to free up space immediately
    if zip_path.exists():
        print(f"Removing temporary archive {zip_path} to reclaim disk space...")
        zip_path.unlink()
        print("Reclaimed disk space!")

    # Check extracted content
    # The zip might contain a root 'datasets/' folder
    extracted_datasets = dest_dir / "datasets"
    if extracted_datasets.exists():
        for item in extracted_datasets.iterdir():
            target = dest_dir / item.name
            if target.exists():
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink()
            shutil.move(str(item), str(dest_dir))
        extracted_datasets.rmdir()

    # Create YOLO data.yaml
    yaml_content = f"""# Underwater Drowning Detection Dataset (Figshare)
path: {dest_dir.resolve()}
train: images/train
val: images/val

names:
  0: swimming
  1: struggling
  2: drowning
"""
    yaml_file = dest_dir / "data.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
    print(f"Created YOLO data configuration: {yaml_file}")

if __name__ == "__main__":
    if not check_disk_space(required_gb=3.2):
        print("Aborting to prevent disk overflow. Please free up disk space first.")
        sys.exit(1)
    
    try:
        download_file(DOWNLOAD_URL, ZIP_PATH)
        extract_and_organize(ZIP_PATH, TARGET_DIR)
        print("\n=== Figshare dataset is ready for training! ===")
    except Exception as e:
        print(f"Error: {e}")
        if ZIP_PATH.exists():
            ZIP_PATH.unlink()
        sys.exit(1)
