# DL MCQ Solver — Kaggle Submission

## Architecture
```
Image → Qwen2-VL-7B → Is computation question?
                          YES → Generate PyTorch → Sandbox → Match output to option
                          NO  → Direct answer
                       ↓
                    3 attempts (temp=0.7)
                    Entropy-weighted majority vote
                    Per-question timeout guard
```

## Directory Structure
```
dl-mcq-solver/
├── README.md
├── setup/
│   ├── package_wheels.sh       # Run ONCE locally to download wheels
│   └── upload_to_kaggle.py     # Run ONCE locally to upload dataset
├── src/
│   ├── config.py               # All hyperparameters in one place
│   ├── sandbox.py              # Jupyter kernel for PyTorch execution
│   ├── solver.py               # Qwen2-VL inference + sandbox routing
│   ├── voting.py               # Entropy-weighted majority vote
│   └── utils.py                # Wheel installer, time budget, helpers
└── kaggle_notebook.py          # ← SUBMIT THIS (copy cells into Kaggle)
```

## Phase 0 — One-Time Offline Setup (Do Before Submitting)

### Step 1: Package wheels locally
```bash
cd setup/
bash package_wheels.sh          # creates wheels.tar.gz
```

### Step 2: Upload to Kaggle as a private Dataset
```bash
pip install kaggle
python upload_to_kaggle.py      # uploads wheels.tar.gz + model weights
```

### Step 3: In your Kaggle notebook
- Click "Add Input" → attach your `dl-mcq-wheels` dataset
- Click "Add Input" → attach your `qwen2-vl-7b-weights` dataset
- Paste cells from `kaggle_notebook.py`

## Model
- **Qwen2-VL-7B-Instruct** — vision + language, reads code/LaTeX natively
- Loaded in `bfloat16` on the 48GB L40S (no quantization needed)
- ~16GB VRAM usage, well within limit

## Runtime Budget
- 50 questions × 3 attempts × ~10s = ~25 minutes
- Well within the 60-minute limit
