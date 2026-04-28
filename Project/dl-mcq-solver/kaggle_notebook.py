# =============================================================================
# kaggle_notebook.py
# =============================================================================
# Copy each section between the ── CELL ── markers into separate Kaggle cells.
# This is the ONLY file you submit. Everything in src/ is for local development.
# =============================================================================


# ── CELL 1: Offline Setup ─────────────────────────────────────────────────────
# Installs all packages from your attached Kaggle Dataset (no internet needed)

import os, sys, subprocess, warnings
warnings.simplefilter("ignore")

WHEELS_ARCHIVE = "/kaggle/input/dl-mcq-wheels/wheels.tar.gz"
WHEELS_DIR     = "/kaggle/tmp/wheels"

def install_offline_wheels(archive, temp_dir):
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        print(f"Extracting {archive} ...")
        subprocess.run(["tar", "-xzf", archive, "-C", temp_dir], check=True)

    packages = ["qwen-vl-utils", "flash-attn", "jupyter_client", "ipykernel"]
    print(f"Installing: {packages}")
    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--quiet", "--no-index",
        "--find-links", f"{temp_dir}/wheels",
        *packages,
    ], check=True)
    print("✅ Packages installed.")

install_offline_wheels(WHEELS_ARCHIVE, WHEELS_DIR)


# ── CELL 2: Imports & Config ──────────────────────────────────────────────────

import gc
import re
import time
import math
import queue
import contextlib
from collections import defaultdict
from typing import Optional

import torch
import pandas as pd
from PIL import Image

# ── Paths ──────────────────────────────────────────────────────────────────────
MODEL_PATH     = "/kaggle/input/qwen2-vl-7b-weights"
TEST_CSV       = "/kaggle/input/test.csv"          # adjust if path differs
IMAGES_DIR     = "/kaggle/input/images"            # adjust if path differs
SUBMISSION_CSV = "/kaggle/working/submission.csv"

# ── Hyperparameters ────────────────────────────────────────────────────────────
ATTEMPTS               = 3      # attempts per question
TEMPERATURE            = 0.7    # inference temperature for diversity
TEMPERATURE_GREEDY     = 0.0    # deterministic (code generation, matching)
MAX_NEW_TOKENS         = 512
SANDBOX_TIMEOUT        = 15     # seconds per sandbox execution
NOTEBOOK_LIMIT_SECONDS = 3300   # 55-minute hard cap
BASE_QUESTION_BUDGET   = 60     # seconds reserved per remaining question
MAX_QUESTION_BUDGET    = 120

SANDBOX_KEYWORDS = [
    "output shape", "output size", "spatial size", "final shape",
    "shape of", "dimension", "parameters", "num parameters",
    "flops", "stride", "padding", "kernel",
    "after conv", "after pool", "after linear",
]

# ── Prompts ─────────────────────────────────────────────────────────────────────
DIRECT_SYSTEM = (
    "You are an expert deep learning engineer. "
    "Look at this MCQ image. Read all text, equations, and code carefully. "
    "Output ONLY the single uppercase letter (A, B, C, or D) of the correct answer. No explanation."
)
SANDBOX_SYSTEM = (
    "You are an expert deep learning engineer. "
    "Write a self-contained PyTorch script to compute the answer numerically. "
    "YOU MUST wrap your code in a ```python ... ``` fenced block. "
    "Print ONLY the final value or shape using print(). Do NOT print the answer letter. "
    "Example format:\n"
    "```python\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "# compute here\n"
    "print(output_shape)\n"
    "```\n"
    "After the code block, write NOTHING else."
)
COMPUTE_COT_SYSTEM = (
    "You are an expert deep learning engineer. "
    "The question requires numerical computation (e.g. tensor shapes, parameter counts). "
    "Work through the math step by step, showing each formula and intermediate value clearly. "
    "Then state the final answer as ONLY the single uppercase letter (A, B, C, or D). "
    "Example: 'After conv: (64+2*1-3)/2+1 = 32. After pool: 32/2 = 16. Answer: A'"
)
MATCH_SYSTEM = (
    "You are given the output of a PyTorch computation and a multiple-choice question image. "
    "Match the output to the correct option. Output ONLY the letter A, B, C, or D."
)


# ── CELL 3: Helper Functions ──────────────────────────────────────────────────

def parse_answer(text: str) -> Optional[str]:
    text = text.strip()
    if text in {"A", "B", "C", "D"}:
        return text
    for pat in [
        r"(?:answer|option)\s*(?:is\s*)?[:\*]*\s*([A-D])",
        r"\*\*([A-D])\*\*",
        r"^([A-D])[\.:\)]",
    ]:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    matches = re.findall(r"\b([A-D])\b", text)
    return matches[0].upper() if matches else None


def is_computation_question(text: str) -> bool:
    t = text.lower()
    return any(kw in t for kw in SANDBOX_KEYWORDS)


def extract_code_block(text: str) -> Optional[str]:
    # Strategy 1: standard ```python ... ``` fence
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Strategy 2: generic ``` ... ``` fence (no language tag)
    m = re.search(r"```\s*\n(.*?)```", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        if any(kw in candidate for kw in ["import", "torch", "nn.", "print"]):
            return candidate

    # Strategy 3: fence without newline after backticks
    m = re.search(r"```(?:python)?(.*?)```", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        if any(kw in candidate for kw in ["import", "torch", "nn.", "print"]):
            return candidate

    # Strategy 4: grab lines that look like Python code
    lines = text.strip().split("\n")
    code_lines, in_code = [], False
    for line in lines:
        stripped = line.strip()
        if re.match(r"^(import |from |model\s*=|x\s*=|output|torch\.|nn\.|print\()", stripped):
            in_code = True
        if in_code:
            if stripped and not re.match(
                r"^(import |from |#|model|x|h|out|input|conv|pool|linear|print|torch|nn|shape|size|\w+\s*=)",
                stripped
            ) and len(stripped.split()) > 6 and stripped[0].islower():
                break
            code_lines.append(line)

    if code_lines and any("print" in l for l in code_lines):
        return "\n".join(code_lines).strip()

    return None


def entropy_from_scores(scores: tuple) -> float:
    if not scores:
        return float("inf")
    total = 0.0
    for step_logits in scores:
        log_probs = torch.log_softmax(step_logits[0].float(), dim=-1)
        probs = torch.exp(log_probs)
        top_probs, _ = torch.topk(probs, k=min(32, probs.shape[-1]))
        top_probs = top_probs.cpu()
        total += -(top_probs * torch.log2(top_probs + 1e-12)).sum().item()
    return total / len(scores)


def entropy_weighted_vote(results: list[dict]) -> str:
    weights = defaultdict(float)
    votes   = defaultdict(int)
    for r in results:
        ans, ent = r.get("answer"), r.get("entropy", float("inf"))
        if ans not in {"A", "B", "C", "D"}:
            continue
        weights[ans] += 1.0 / max(ent, 1e-9)
        votes[ans]   += 1

    if not weights:
        print("[vote] ⚠️  All attempts failed — defaulting to A")
        return "A"

    rows = [{"Answer": a, "Votes": votes[a], "Score": round(w, 4)}
            for a, w in sorted(weights.items(), key=lambda x: -x[1])]
    print(pd.DataFrame(rows).to_string(index=False))

    winner = max(weights, key=weights.__getitem__)
    print(f"\n[vote] ✅  Winner: {winner}\n")
    return winner


# ── CELL 4: Sandbox ───────────────────────────────────────────────────────────

import queue as _queue

class Sandbox:
    def __init__(self, timeout=15.0):
        from jupyter_client import KernelManager
        self._timeout = timeout
        self._km = KernelManager()
        self._km.start_kernel()
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=30)
        self.execute("import torch\nimport torch.nn as nn\nimport math\nimport numpy as np\n")

    def execute(self, code, timeout=None):
        eff = timeout or self._timeout
        msg_id = self._client.execute(code, store_history=True, allow_stdin=False)
        out, err = [], []
        start = time.time()
        while True:
            if time.time() - start > eff:
                self._km.interrupt_kernel()
                return "[TIMEOUT]"
            try:
                msg = self._client.get_iopub_msg(timeout=1.0)
            except _queue.Empty:
                continue
            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue
            mt = msg.get("msg_type")
            c  = msg.get("content", {})
            if mt == "stream":
                (out if c.get("name") == "stdout" else err).append(c.get("text", ""))
            elif mt == "error":
                err.append("".join(re.sub(r"\x1b\[[0-9;]*m", "", f) for f in c.get("traceback", [])))
            elif mt in {"execute_result", "display_data"}:
                t = c.get("data", {}).get("text/plain", "")
                if t:
                    out.append(t + "\n")
            elif mt == "status" and c.get("execution_state") == "idle":
                break
        stdout = "".join(out).strip()
        stderr = "".join(err)
        if stderr:
            return f"{stdout}\n[STDERR] {stderr}" if stdout else f"[STDERR] {stderr}"
        return stdout or "[WARN] No output."

    def reset(self):
        self.execute("%reset -f\nimport torch\nimport torch.nn as nn\nimport math\nimport numpy as np\n")

    def close(self):
        with contextlib.suppress(Exception): self._client.stop_channels()
        with contextlib.suppress(Exception): self._km.shutdown_kernel(now=True)

    def __del__(self): self.close()


# ── CELL 5: Load Model ────────────────────────────────────────────────────────

from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

print(f"Loading Qwen2-VL from {MODEL_PATH} ...")
t0 = time.time()

processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True,
)
model.eval()
print(f"✅ Model loaded in {(time.time()-t0):.1f}s | "
      f"VRAM: {torch.cuda.memory_allocated()/1e9:.1f} GB")

sandbox = Sandbox(timeout=SANDBOX_TIMEOUT)
print("✅ Sandbox ready.")


# ── CELL 6: Inference Functions ───────────────────────────────────────────────

def run_vlm(image: Image.Image, system_prompt: str, user_prompt: str,
            temperature: float) -> tuple[str, float]:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text",  "text": user_prompt},
        ]},
    ]
    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text_input], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            output_scores=True,
            return_dict_in_generate=True,
        )

    generated_ids = out.sequences[:, inputs["input_ids"].shape[1]:]
    text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    entropy = entropy_from_scores(out.scores)
    return text, entropy


def solve_direct(image: Image.Image, deadline: float,
                 temperature: float = TEMPERATURE) -> dict:
    if time.time() > deadline:
        return {"answer": None, "entropy": float("inf"), "path": "timeout"}
    raw, ent = run_vlm(image, DIRECT_SYSTEM,
                       "Output ONLY the letter A, B, C, or D of the correct answer.",
                       temperature)
    ans = parse_answer(raw)
    print(f"[direct] {raw!r} → {ans}  (entropy={ent:.3f})")
    return {"answer": ans, "entropy": ent, "path": "direct"}


def solve_compute_cot(image: Image.Image, deadline: float) -> dict:
    """Chain-of-thought fallback for computation questions when sandbox fails."""
    if time.time() > deadline:
        return {"answer": None, "entropy": float("inf"), "path": "timeout"}
    raw, ent = run_vlm(
        image, COMPUTE_COT_SYSTEM,
        "Work through the computation step by step. Show each formula and value. "
        "End with: 'Answer: X' where X is A, B, C, or D.",
        TEMPERATURE_GREEDY,
    )
    print(f"[cot] Raw: {raw!r}")
    ans = parse_answer(raw)
    print(f"[cot] Answer: {ans}  entropy: {ent:.3f}")
    return {"answer": ans, "entropy": ent, "path": "compute_cot"}


def solve_sandbox(image: Image.Image, deadline: float) -> dict:
    if time.time() > deadline:
        return {"answer": None, "entropy": float("inf"), "path": "timeout"}

    # Step 1: generate code
    code_text, ent1 = run_vlm(image, SANDBOX_SYSTEM,
        "Write PyTorch code to compute the answer. "
        "Wrap it in a ```python ... ``` block. Print ONLY the final value.",
        TEMPERATURE_GREEDY)
    code = extract_code_block(code_text)
    if not code:
        print("[sandbox] No code extracted → fallback to CoT")
        return solve_compute_cot(image, deadline)

    print(f"[sandbox] Code:\n{code}")

    # Step 2: execute
    if time.time() > deadline:
        return {"answer": None, "entropy": float("inf"), "path": "timeout"}
    exec_out = sandbox.execute(code, timeout=SANDBOX_TIMEOUT)
    print(f"[sandbox] Output: {exec_out}")

    if "[TIMEOUT]" in exec_out or "[STDERR]" in exec_out:
        print("[sandbox] Execution error → fallback to CoT")
        return solve_compute_cot(image, deadline)

    # Step 3: match output to option
    match_prompt = (
        f"The PyTorch computation printed:\n{exec_out}\n\n"
        "Which option (A, B, C, or D) in the image matches this? Output ONLY the letter."
    )
    match_text, ent2 = run_vlm(image, MATCH_SYSTEM, match_prompt, TEMPERATURE_GREEDY)
    ans = parse_answer(match_text)
    mean_ent = (ent1 + ent2) / 2
    print(f"[sandbox] Answer: {ans}  entropy: {mean_ent:.3f}")
    return {"answer": ans, "entropy": mean_ent, "path": "sandbox"}


def solve_one_attempt(image_path: str, deadline: float,
                      temperature: float = TEMPERATURE) -> dict:
    image = Image.open(image_path).convert("RGB")

    # Quick text extraction to detect question type
    caption, _ = run_vlm(image, "Extract all text from this image. Output only the text.",
                          "Extract text.", 0.0)
    print(f"[solver] Text: {caption[:150]!r}")

    use_sandbox = (
        is_computation_question(caption)
        and time.time() < deadline - SANDBOX_TIMEOUT - 5
    )

    if use_sandbox:
        print("[solver] → SANDBOX path")
        return solve_sandbox(image, deadline)
    else:
        print("[solver] → DIRECT path")
        return solve_direct(image, deadline, temperature)


# ── CELL 7: Main Loop ─────────────────────────────────────────────────────────

test_df     = pd.read_csv(TEST_CSV)
n_questions = len(test_df)
predictions = {}

notebook_start = time.time()
print(f"Processing {n_questions} questions...\n")

for idx, row in test_df.iterrows():
    image_id   = row["image_id"]
    image_path = os.path.join(IMAGES_DIR, row["image_name"])

    # ── Time budget ──────────────────────────────────────────────────────────
    elapsed        = time.time() - notebook_start
    time_left      = NOTEBOOK_LIMIT_SECONDS - elapsed
    remaining_qs   = n_questions - idx
    reserved       = max(0, remaining_qs - 1) * BASE_QUESTION_BUDGET
    budget         = min(MAX_QUESTION_BUDGET, time_left - reserved)
    budget         = max(budget, BASE_QUESTION_BUDGET)
    deadline       = time.time() + budget

    print(f"\n{'='*60}")
    print(f"[{idx+1}/{n_questions}] image_id={image_id}  budget={budget:.0f}s")
    print(f"{'='*60}")

    if not os.path.exists(image_path):
        print(f"[warn] Image not found: {image_path} — skipping")
        predictions[image_id] = "A"
        continue

    # ── Run ATTEMPTS attempts, collect results ────────────────────────────────
    results = []
    for attempt_idx in range(ATTEMPTS):
        if time.time() > deadline - 5:
            print(f"[timer] Out of budget after attempt {attempt_idx} — stopping early")
            break

        print(f"\n── Attempt {attempt_idx + 1}/{ATTEMPTS} ──")
        result = solve_one_attempt(image_path, deadline, temperature=TEMPERATURE)
        results.append(result)
        sandbox.reset()

        # Early stop: if 2/3 agree already, no need for 3rd attempt
        votes = defaultdict(int)
        for r in results:
            if r["answer"] in {"A","B","C","D"}:
                votes[r["answer"]] += 1
        if any(v >= 2 for v in votes.values()):
            print("[vote] Early consensus — skipping remaining attempts")
            break

    # ── Vote ──────────────────────────────────────────────────────────────────
    final_answer = entropy_weighted_vote(results)
    predictions[image_id] = final_answer

    gc.collect()
    torch.cuda.empty_cache()

print("\n✅ All questions processed.")


# ── CELL 8: Save Submission ───────────────────────────────────────────────────

submission = pd.DataFrame([
    {"image_id": img_id, "answer": ans}
    for img_id, ans in predictions.items()
])
submission.to_csv(SUBMISSION_CSV, index=False)

print(f"\nSubmission saved to {SUBMISSION_CSV}")
print(submission["answer"].value_counts().to_string())
print(submission.head(10).to_string(index=False))