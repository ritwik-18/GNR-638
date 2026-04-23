#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Run this ONCE on your local machine (or a Colab with internet).
# It downloads all Python wheels needed for the Kaggle offline environment
# and packages them into a single wheels.tar.gz you upload as a Kaggle Dataset.
#
# Usage:
#   bash package_wheels.sh
#
# Output:
#   ./wheels.tar.gz   (upload this to Kaggle as a private Dataset)
# ─────────────────────────────────────────────────────────────────────────────

set -e

WHEELS_DIR="./wheels"
OUTPUT_ARCHIVE="./wheels.tar.gz"

# Python version must match Kaggle's environment (3.10 as of 2025)
PYTHON_VERSION="3.10"
# CUDA version on L40S Kaggle environment
CUDA_VERSION="cu121"
TORCH_VERSION="2.2.0"

echo ">>> Creating wheels directory..."
rm -rf "$WHEELS_DIR"
mkdir -p "$WHEELS_DIR"

echo ">>> Downloading qwen-vl-utils..."
pip download qwen-vl-utils \
    --dest "$WHEELS_DIR" \
    --no-deps

echo ">>> Downloading transformers (pinned for Qwen2-VL compatibility)..."
pip download "transformers>=4.45.0" \
    --dest "$WHEELS_DIR" \
    --no-deps

echo ">>> Downloading accelerate..."
pip download accelerate \
    --dest "$WHEELS_DIR" \
    --no-deps

echo ">>> Downloading flash-attn (prebuilt binary for CUDA 12.1 + torch 2.2)..."
# flash-attn prebuilt wheel — avoids 20-min compilation
pip download flash-attn \
    --dest "$WHEELS_DIR" \
    --no-deps \
    --index-url https://github.com/Dao-AILab/flash-attention/releases/expanded_assets/v2.5.8

echo ">>> Downloading jupyter_client (for sandbox)..."
pip download jupyter_client ipykernel \
    --dest "$WHEELS_DIR" \
    --no-deps

echo ">>> Packaging into $OUTPUT_ARCHIVE ..."
tar -czf "$OUTPUT_ARCHIVE" -C "$(dirname $WHEELS_DIR)" "$(basename $WHEELS_DIR)"

echo ""
echo "✅ Done! Upload $OUTPUT_ARCHIVE to Kaggle as a private Dataset named: dl-mcq-wheels"
echo "   Then attach it to your notebook at: /kaggle/input/dl-mcq-wheels/wheels.tar.gz"
