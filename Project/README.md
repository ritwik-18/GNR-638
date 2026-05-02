# GNR Project — Deep Learning MCQ Solver

## Environment
- Python 3.11
- Conda environment name: `gnr_project_env`
- GPU: NVIDIA L40s (48 GB VRAM), CUDA 12.6

## Setup (requires internet)

```bash
bash setup.bash
```

This will:
1. Clone the repository
2. Create the `gnr_project_env` conda environment (Python 3.11)
3. Install PyTorch with CUDA 12.6 support
4. Install all Python dependencies
5. Download the model weights from HuggingFace Hub into `models/`

## Inference (no internet needed)

```bash
conda activate gnr_project_env
python inference.py --test_dir <absolute_path_to_test_dir>
```

- `<absolute_path_to_test_dir>` must contain `test.csv` and `images/` sub-folder.
- `submission.csv` is written to the **current working directory**.

## Project Structure

```
gnr_project/
├── inference.py          # Main inference script
├── setup.bash            # One-time setup script
├── requirements.txt      # Python dependencies
├── README.md             # This file
└── models/
    └── Qwen2.5-VL-7B-Instruct/   # Downloaded by setup.bash
```

## Model
- **Model**: Qwen/Qwen2.5-VL-7B-Instruct (Vision-Language Model)
- **Precision**: bfloat16
- **Attention**: Flash Attention 2 (falls back to SDPA if unavailable)

## Scoring Strategy
- Answers 1–4 are submitted when the model is confident.
- Answer 5 (skip) is used when the model output cannot be parsed, to avoid the −1 hallucination penalty.

## Citation
- Qwen2.5-VL: https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct
- HuggingFace Transformers: https://github.com/huggingface/transformers
