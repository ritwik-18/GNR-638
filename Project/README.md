# Submission — Kaggle Notebooks

## Files

```
submission/
├── README.md
├── environment.yml          ← for local testing only
├── requirements.txt         ← for local testing only
├── project_1/
│   └── solution.ipynb       ← Kaggle notebook: map stitching + MCQ
└── project_2/
    └── solution.ipynb       ← Kaggle notebook: DL MCQ from images
```

---

## Kaggle Setup (required before submitting each notebook)

### Step 1 — Attach competition dataset
- Notebook editor → **+ Add data** → select this competition's dataset
- Provides: `patches/` or `images/`, `test.csv`, `sample_submission.csv`
- Mounts at `/kaggle/input/<dataset-slug>/`

### Step 2 — Attach Qwen2-VL-7B model (no internet download!)
- Notebook editor → **Add-ons → Models**
- Search: `Qwen2-VL-7B-Instruct`  (publisher: Qwen-LM)
- Select version: `transformers / 7b-instruct`
- Click **Add** → mounts at `/kaggle/input/qwen2-vl/transformers/7b-instruct/1`

> The notebook auto-discovers the model path via a fallback scan, so even if the mount path differs slightly it will work.

### Step 3 — Notebook settings
| Setting | Value |
|---------|-------|
| Accelerator | GPU T4 x2 (or P100) |
| Internet | **OFF** |
| Persistence | Files only |

### Step 4 — Run
Click **Run All** (or Save & Run All for submission).
`submission.csv` is written to `/kaggle/working/submission.csv` automatically.

---

## Local Testing (optional)

```bash
conda env create -f environment.yml
conda activate map_mcq_env

# Download model weights once (internet required)
python -c "
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
import torch
AutoProcessor.from_pretrained('Qwen/Qwen2-VL-7B-Instruct')
Qwen2VLForConditionalGeneration.from_pretrained('Qwen/Qwen2-VL-7B-Instruct', torch_dtype=torch.bfloat16)
print('Cached to ~/.cache/huggingface')
"

# Run notebooks locally
jupyter nbconvert --to notebook --execute project_1/solution.ipynb --ExecutePreprocessor.timeout=3600
jupyter nbconvert --to notebook --execute project_2/solution.ipynb --ExecutePreprocessor.timeout=3600
```

For local runs, change `MODEL_DIR` in Cell 2 to point to your HuggingFace cache, or set:
```python
MODEL_DIR = Path('~/.cache/huggingface/hub/models--Qwen--Qwen2-VL-7B-Instruct/snapshots/<hash>').expanduser()
```

---

## Algorithm Summary

### Project 1 — Map Stitching
1. Sort patches by number (`patch_0` = fixed top-left anchor)
2. SIFT (2 000 features/patch) + FLANN + Lowe ratio test (0.75)
3. RANSAC homography per pair (5 px threshold, ≥10 inliers)
4. BFS from `patch_0` → accumulate global transforms
5. Warp + weighted-average blend onto canvas
6. Fallbacks: OpenCV Stitcher → grid tiling
7. Qwen2-VL-7B reads the stitched map, answers MCQs

### Project 2 — DL MCQ
1. Load `test.csv`, iterate over `image_name` column
2. For each PNG: structured prompt with chain-of-thought reasoning instruction
3. Qwen2-VL-7B parses the question & options printed in the image
4. Extract last digit 1-5 from model output (5 = skip)

---

## Scoring Strategy
- Predict **5** (skip, 0 points) only when the model produces no valid digit
- Otherwise submit 1-4 directly; VLMs at 7B scale are accurate on DL MCQs

## Citations
```
@article{Qwen2VL,
  title={Qwen2-VL: Enhancing Vision-Language Model's Perception of the World at Any Resolution},
  author={Wang, Peng et al.},
  journal={arXiv:2409.12191},
  year={2024}
}
```
OpenCV SIFT & stitching: Lowe, D.G. (2004). Distinctive image features from scale-invariant keypoints.
