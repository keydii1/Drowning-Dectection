import os
import sys
import shutil
import subprocess
from pathlib import Path

REPO_URL = "https://github.com/Wang-Kaikai/drowning-detection-dataset.git"
TARGET_DIR = Path("/Users/hohoangson/Documents/SienceResearch/datasets/raw/wang_kaikai_drowning")
TEMP_CLONE_DIR = Path("/Users/hohoangson/Documents/SienceResearch/datasets/raw/_temp_wang_repo")

def check_disk_space(required_gb=7.0):
    stat = shutil.disk_usage(TARGET_DIR.parent)
    free_gb = stat.free / (1024 ** 3)
    print(f"Available disk space: {free_gb:.2f} GB (Required: ~{required_gb:.2f} GB)")
    if free_gb < required_gb:
        print(f"\n[!] WARNING: Insufficient disk space!")
        print(f"The Wang-Kaikai dataset contains 8,572 high-resolution images (~5-7 GB).")
        print(f"Current free space is only {free_gb:.2f} GB.")
        print("Please free up at least 7 GB on your drive before running this download.")
        return False
    return True

def run_command(cmd, cwd=None):
    print(f"Executing: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, check=True, text=True, capture_output=True)
    return result.stdout

def download_dataset():
    if not check_disk_space():
        sys.exit(1)

    print("\nStarting optimized sparse clone (downloading ONLY the dataset, ignoring 10GB+ of model weights)...")
    if TEMP_CLONE_DIR.exists():
        shutil.rmtree(TEMP_CLONE_DIR)

    # 1. Clone repository structure without any file blobs
    run_command([
        "git", "clone",
        "--depth", "1",
        "--filter=blob:none",
        "--sparse",
        REPO_URL,
        str(TEMP_CLONE_DIR)
    ])

    # 2. Configure sparse-checkout to ONLY fetch 'self-made dataset'
    print("Configuring sparse checkout for 'self-made dataset' folder...")
    run_command(["git", "sparse-checkout", "set", "self-made dataset"], cwd=TEMP_CLONE_DIR)

    # 3. Move the dataset to destination
    src_data = TEMP_CLONE_DIR / "self-made dataset"
    if not src_data.exists():
        raise RuntimeError("Sparse checkout failed: 'self-made dataset' directory not found.")

    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    for item in src_data.iterdir():
        target_path = TARGET_DIR / item.name
        if target_path.exists():
            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
        shutil.move(str(item), str(TARGET_DIR))

    # 4. Clean up temporary git clone repo to save space
    print("Cleaning up git metadata...")
    shutil.rmtree(TEMP_CLONE_DIR)

    # 5. Create YOLO data.yaml
    yaml_content = f"""# Wang-Kaikai Drowning Detection Dataset
path: {TARGET_DIR.resolve()}
train: images/train
val: images/val

names:
  0: swimming
  1: tread water
  2: drowning
"""
    yaml_file = TARGET_DIR / "data.yaml"
    with open(yaml_file, "w") as f:
        f.write(yaml_content)
    print(f"Created YOLO data configuration: {yaml_file}")
    print("\n=== Wang-Kaikai dataset is ready! ===")

if __name__ == "__main__":
    download_dataset()
