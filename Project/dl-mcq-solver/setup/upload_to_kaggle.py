"""
upload_to_kaggle.py
────────────────────────────────────────────────────────────────────────────
Run ONCE locally after package_wheels.sh.
Uploads:
  1. wheels.tar.gz  → Kaggle Dataset: dl-mcq-wheels
  2. Qwen2-VL-7B model weights → Kaggle Dataset: qwen2-vl-7b-weights

Prerequisites:
    pip install kaggle huggingface_hub
    Set KAGGLE_USERNAME and KAGGLE_KEY in your environment (or ~/.kaggle/kaggle.json)
    Set HF_TOKEN if Qwen2-VL requires gated access
────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import shutil
import subprocess
from pathlib import Path

KAGGLE_USERNAME = os.environ.get("KAGGLE_USERNAME", "YOUR_USERNAME_HERE")
HF_TOKEN        = os.environ.get("HF_TOKEN", None)   # only needed if gated


# ─── 1. Upload wheels ────────────────────────────────────────────────────────

def upload_wheels(wheels_archive: str = "./wheels.tar.gz"):
    dataset_dir = Path("./kaggle_upload_wheels")
    dataset_dir.mkdir(exist_ok=True)

    shutil.copy(wheels_archive, dataset_dir / "wheels.tar.gz")

    metadata = {
        "title": "dl-mcq-wheels",
        "id": f"{KAGGLE_USERNAME}/dl-mcq-wheels",
        "licenses": [{"name": "other"}]
    }
    (dataset_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

    print(">>> Uploading wheels to Kaggle...")
    subprocess.run(["kaggle", "datasets", "create", "-p", str(dataset_dir)], check=True)
    print("✅ Wheels uploaded: https://www.kaggle.com/datasets/{KAGGLE_USERNAME}/dl-mcq-wheels")


# ─── 2. Download Qwen2-VL from HuggingFace and upload to Kaggle ──────────────

def download_and_upload_model():
    from huggingface_hub import snapshot_download

    model_dir = Path("./qwen2_vl_7b")
    model_dir.mkdir(exist_ok=True)

    print(">>> Downloading Qwen2-VL-7B-Instruct from HuggingFace...")
    snapshot_download(
        repo_id="Qwen/Qwen2-VL-7B-Instruct",
        local_dir=str(model_dir),
        token=HF_TOKEN,
        ignore_patterns=["*.msgpack", "flax_model*", "tf_model*", "rust_model*"],
    )

    # Write Kaggle dataset metadata
    metadata = {
        "title": "qwen2-vl-7b-weights",
        "id": f"{KAGGLE_USERNAME}/qwen2-vl-7b-weights",
        "licenses": [{"name": "other"}]
    }
    (model_dir / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2))

    print(">>> Uploading model weights to Kaggle (this may take a while)...")
    subprocess.run(["kaggle", "datasets", "create", "-p", str(model_dir)], check=True)
    print(f"✅ Model uploaded: https://www.kaggle.com/datasets/{KAGGLE_USERNAME}/qwen2-vl-7b-weights")


if __name__ == "__main__":
    if KAGGLE_USERNAME == "YOUR_USERNAME_HERE":
        raise ValueError("Set KAGGLE_USERNAME environment variable first.")

    upload_wheels()
    download_and_upload_model()

    print("\n" + "="*60)
    print("Both datasets are now on Kaggle.")
    print("In your notebook, click 'Add Input' and attach:")
    print(f"  → {KAGGLE_USERNAME}/dl-mcq-wheels")
    print(f"  → {KAGGLE_USERNAME}/qwen2-vl-7b-weights")
    print("="*60)
