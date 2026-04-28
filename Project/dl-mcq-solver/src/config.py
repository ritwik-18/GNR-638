"""
config.py — Single source of truth for all hyperparameters.
Edit here; everything else imports from CFG.
"""

class CFG:

    # ── Paths (Kaggle runtime) ────────────────────────────────────────────────
    MODEL_PATH        = "/kaggle/input/qwen2-vl-7b-weights"
    WHEELS_ARCHIVE    = "/kaggle/input/dl-mcq-wheels/wheels.tar.gz"
    WHEELS_TEMP_DIR   = "/kaggle/tmp/wheels"
    TEST_CSV          = "/kaggle/input/test.csv"          # adjust to actual path
    IMAGES_DIR        = "/kaggle/input/images"            # adjust to actual path
    SUBMISSION_CSV    = "/kaggle/working/submission.csv"

    # ── Model ─────────────────────────────────────────────────────────────────
    MODEL_DTYPE       = "bfloat16"   # full precision — L40S has 48GB, no need to quantize
    MAX_NEW_TOKENS    = 512

    # ── Inference ─────────────────────────────────────────────────────────────
    ATTEMPTS          = 3            # attempts per question for majority voting
    TEMPERATURE       = 0.7          # slight randomness for diversity between attempts
    TEMPERATURE_GREEDY = 0.0         # used for sandbox-verified questions (deterministic)

    # ── Timing ────────────────────────────────────────────────────────────────
    NOTEBOOK_LIMIT_SECONDS = 3300    # 55 min hard cap (leaves 5-min safety buffer)
    BASE_QUESTION_BUDGET   = 60      # seconds reserved per remaining question
    MAX_QUESTION_BUDGET    = 120     # never spend more than 2 min on one question

    # ── Sandbox ───────────────────────────────────────────────────────────────
    SANDBOX_TIMEOUT   = 15           # seconds; PyTorch forward pass should be instant
    SANDBOX_ENABLED   = True         # set False to disable sandbox for debugging

    # Keywords that trigger sandbox routing (computation questions)
    SANDBOX_KEYWORDS  = [
        "output shape", "output size", "spatial size", "final shape",
        "shape of", "dimension", "parameters", "num parameters",
        "flops", "stride", "padding", "kernel",
        "after conv", "after pool", "after linear",
    ]

    # ── Prompts ───────────────────────────────────────────────────────────────
    DIRECT_SYSTEM_PROMPT = (
        "You are an expert deep learning engineer and researcher. "
        "You will be shown an image containing a multiple-choice question about deep learning. "
        "Read all text, equations, code, and diagrams in the image carefully. "
        "Reason step by step, then output ONLY the single uppercase letter "
        "(A, B, C, or D) of the correct answer. No explanation."
    )

    SANDBOX_SYSTEM_PROMPT = (
        "You are an expert deep learning engineer. "
        "You will be shown an image containing a multiple-choice question about deep learning. "
        "Read the question carefully. "
        "To find the correct answer, write a self-contained Python script using PyTorch "
        "that computes the answer numerically. "
        "For tensor shape questions, instantiate the PyTorch layers and pass a dummy tensor (e.g., torch.zeros) through them rather than calculating manually. "
        "YOU MUST wrap your code in a ```python ... ``` fenced block. "
        "The script must print ONLY the final computed value or shape using print(). "
        "Do NOT print the answer letter. Just print the raw value so it can be matched to an option. "
        "Example format:\n"
        "```python\n"
        "import torch\n"
        "import torch.nn as nn\n"
        "# compute here\n"
        "print(output_shape)\n"
        "```\n"
        "After the code block, write NOTHING else."
    )

    COMPUTE_COT_PROMPT = (
        "You are an expert deep learning engineer. "
        "You will be shown an image containing a multiple-choice question that requires computation "
        "(e.g. tensor shapes, parameter counts, convolution output sizes). "
        "Work through the math step by step, showing each calculation clearly. "
        "Then output the final answer as ONLY the single uppercase letter (A, B, C, or D). "
        "Example: 'After conv: (64+2*1-3)/2+1 = 32. After pool: 32/2 = 16. Answer: A'"
    )

    MATCH_SYSTEM_PROMPT = (
        "You are given the output of a PyTorch computation and a multiple-choice question. "
        "Match the computed output to the correct answer option. "
        "Output ONLY the single uppercase letter (A, B, C, or D). No explanation."
    )