"""
inference.py — Deep Learning MCQ Solver using Qwen3-VL-8B-Instruct
Usage:
    python inference.py --test_dir <absolute_path_to_test_dir>
submission.csv is saved in the CURRENT working directory (not test_dir).
"""

import os
import re
import time
import argparse
import pandas as pd
from PIL import Image
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# =============================================================================
# 0. ARGUMENT PARSING
# =============================================================================
parser = argparse.ArgumentParser(description="DL MCQ Inference — Qwen3-VL-8B-Instruct")
parser.add_argument("--test_dir", type=str, required=True,
                    help="Absolute path to test directory (contains test.csv and images/)")
args = parser.parse_args()

TEST_DIR      = args.test_dir
TEST_CSV_FILE = os.path.join(TEST_DIR, "test.csv")
IMAGE_FOLDER  = os.path.join(TEST_DIR, "images")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR  = os.path.join(SCRIPT_DIR, "models", "Qwen3-VL-8B-Instruct")

assert os.path.isdir(MODEL_DIR),      f"[ERROR] Model dir missing: {MODEL_DIR}"
assert os.path.isfile(TEST_CSV_FILE), f"[ERROR] test.csv missing: {TEST_CSV_FILE}"
assert os.path.isdir(IMAGE_FOLDER),   f"[ERROR] images/ folder missing: {IMAGE_FOLDER}"

# =============================================================================
# 1. BUILD IMAGE NAME → FULL PATH MAP
# =============================================================================
ALLOWED_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}
system_image_map = {}
for file_name in os.listdir(IMAGE_FOLDER):
    stem, ext = os.path.splitext(file_name)
    if ext.lower() in ALLOWED_EXTENSIONS:
        system_image_map[stem] = os.path.join(IMAGE_FOLDER, file_name)

# =============================================================================
# 2. PROMPT
# =============================================================================
USER_PROMPT = (
    "You are given an image containing a deep learning multiple-choice question "
    "with four options labeled A, B, C, and D. Exactly one option is correct.\n\n"
    "Before answering, carefully examine:\n"
    "- All mathematical expressions, including subscripts, exponents, and signs\n"
    "- Any code snippets — pay attention to layer types, argument names, and order\n"
    "- What exactly the question is asking: a shape, a count, a formula, a concept\n\n"
    "Useful formulas for spatial dimension questions only:\n"
    "  Conv2d output          : floor((H + 2p - k) / s) + 1\n"
    "  ConvTranspose2d output : (H - 1) * s - 2p + k + output_padding\n"
    "Apply these only when the question is about convolution output sizes. "
    "For all other topics, reason directly from your deep learning knowledge.\n\n"
    "Think step by step before concluding. "
    "Your response must end with this line and nothing after it:\n"
    "ANSWER: X\n"
    "where X is exactly one of A, B, C, D."
)

# =============================================================================
# 3. LOAD MODEL
# =============================================================================
print('\n[Hardware] Loading Qwen3-VL-8B-Instruct in bfloat16...')
boot_time = time.time()

hardware_config = {
    'torch_dtype': torch.bfloat16,
    'device_map': 'auto',
    'trust_remote_code': True,
    'low_cpu_mem_usage': True,
}

try:
    vision_llm = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, attn_implementation="flash_attention_2", **hardware_config
    )
    print("  ✓ Flash Attention 2 enabled.")
except Exception as err:
    print(f"  [WARN] Flash Attention 2 unavailable ({type(err).__name__}). Using SDPA.")
    vision_llm = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR, attn_implementation="sdpa", **hardware_config
    )

vision_processor = AutoProcessor.from_pretrained(
    MODEL_DIR,
    trust_remote_code=True,
    min_pixels=256 * 28 * 28,
    max_pixels=1024 * 28 * 28,
)

vision_llm.eval()
torch.cuda.empty_cache()

vram_used = torch.cuda.max_memory_allocated() / 1e9
print(f'✓ Ready. Load time: {time.time()-boot_time:.1f}s | Peak VRAM: {vram_used:.1f} GB\n')

# =============================================================================
# 4. INFERENCE
# =============================================================================
def decode_exam_image(image_file_path: str) -> str:
    try:
        source_image = Image.open(image_file_path).convert("RGB")
    except Exception as e:
        print(f"  [ERROR] Cannot open image: {e}")
        return ""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": source_image},
                {"type": "text",  "text": USER_PROMPT},
            ],
        }
    ]

    # Qwen3-VL: processor handles pixel encoding inside apply_chat_template
    inputs = vision_processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(vision_llm.device)

    with torch.inference_mode():
        generated_ids = vision_llm.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,         # greedy — most reliable for MCQ
            repetition_penalty=1.05,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]

    output_text = vision_processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    del inputs, generated_ids, generated_ids_trimmed
    return output_text[0].strip() if output_text else ""


# =============================================================================
# 5. PARSE ANSWER  — A/B/C/D → 1/2/3/4
# =============================================================================
LETTER_TO_NUM = {'A': 1, 'B': 2, 'C': 3, 'D': 4}

def isolate_final_choice(ai_text: str) -> int:
    """
    Extract answer from model output and return 1/2/3/4.
    Returns 5 (skip, 0 pts) only when nothing parseable is found —
    safer than a wrong guess which costs -0.25, or a hallucinated
    value which costs -1.
    """
    if not ai_text:
        return 5

    clean = ai_text.replace('*', '').strip()

    # TIER 1: exact "ANSWER: X" on one of the last 3 non-empty lines
    lines = [l.strip() for l in clean.splitlines() if l.strip()]
    for line in reversed(lines[-3:]):
        m = re.match(r'^ANSWER\s*:\s*([A-D])\s*$', line, re.IGNORECASE)
        if m:
            return LETTER_TO_NUM[m.group(1).upper()]

    # TIER 2: "ANSWER: X" anywhere in text
    t2 = re.search(r'\bANSWER\s*:\s*([A-D])\b', clean, re.IGNORECASE)
    if t2:
        return LETTER_TO_NUM[t2.group(1).upper()]

    # TIER 3: "the correct answer is C" / "option B is correct"
    t3 = re.search(
        r'(?:correct\s+)?(?:answer|option|choice)\s+(?:is\s+)?([A-D])\b',
        clean, re.IGNORECASE
    )
    if t3:
        return LETTER_TO_NUM[t3.group(1).upper()]

    # TIER 4: standalone letter near end — "(B)" or "Option D"
    tail = clean[-200:]
    t4 = re.search(r'\b(?:option\s*)?([A-D])\b', tail, re.IGNORECASE)
    if t4:
        return LETTER_TO_NUM[t4.group(1).upper()]

    # TIER 5: last digit 1-4 (in case model ignored instructions)
    digits = re.findall(r'(?<![0-9])([1-4])(?![0-9])', clean)
    if digits:
        return int(digits[-1])

    # TIER 6: truly unparseable → skip (0 pts, not -1)
    return 5


# =============================================================================
# 6. MAIN LOOP
# =============================================================================
exam_df   = pd.read_csv(TEST_CSV_FILE)
total     = len(exam_df)
records   = []
job_start = time.time()
print(f'Starting inference on {total} questions...\n')

for idx, row in enumerate(exam_df.itertuples(index=False)):
    img_name = str(row.image_name).strip()
    img_id   = str(getattr(row, 'image_id', img_name)).strip()
    img_path = system_image_map.get(img_name)

    print(f"[{idx+1:02d}/{total}] {img_name} ...", end=" ", flush=True)

    img_start = time.time()

    if img_path is None:
        print("IMAGE NOT FOUND — skipped (5)")
        pred = 5
    else:
        raw  = decode_exam_image(img_path)
        pred = isolate_final_choice(raw)
        elapsed = time.time() - img_start
        last_line = raw.splitlines()[-1].strip() if raw else ""
        print(f"→ {pred}  [{last_line}]  ({elapsed:.1f}s)")

    records.append({'id': img_id, 'image_name': img_name, 'option': pred})

    if (idx + 1) % 10 == 0:
        torch.cuda.empty_cache()

# =============================================================================
# 7. SAVE submission.csv IN CURRENT DIRECTORY
# =============================================================================
out_path = os.path.join(os.getcwd(), "submission.csv")
pd.DataFrame(records).to_csv(out_path, index=False)

mins = (time.time() - job_start) / 60
print(f'\n✓ Done in {mins:.1f} min. Saved → {out_path}')
print(pd.DataFrame(records)["option"].value_counts().sort_index().to_dict())
