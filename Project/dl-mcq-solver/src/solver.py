"""
solver.py — Core inference engine.

Responsibilities:
  1. Load Qwen2-VL-7B-Instruct once, keep it on GPU
  2. For each image:
       a. Detect whether it's a computation question
       b. If YES → sandbox path (generate code → execute → match to option)
       c. If NO  → direct VLM answer
  3. Return answer + entropy for voting

Caller (kaggle_notebook.py) handles:
  - Running this ATTEMPTS times per image
  - Entropy-weighted voting
  - Time budget enforcement
"""

import time
import math
from typing import Optional

import torch
from PIL import Image

from src.config import CFG
from src.utils import parse_answer, is_computation_question, format_elapsed
from src.sandbox import Sandbox, extract_code_block
from src.voting import compute_mean_entropy


# ─── Model Loader ─────────────────────────────────────────────────────────────

def load_model():
    """
    Load Qwen2-VL-7B-Instruct in bfloat16.
    On the 48GB L40S this uses ~16GB VRAM — plenty of headroom.
    Returns (model, processor).
    """
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info

    print(f"[model] Loading Qwen2-VL from {CFG.MODEL_PATH} ...")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(CFG.MODEL_PATH, trust_remote_code=True)

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        CFG.MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    print(f"[model] ✅ Loaded in {format_elapsed(time.time() - t0)}")
    return model, processor


# ─── Single Inference Call ────────────────────────────────────────────────────

def _run_vlm(
    model,
    processor,
    image: Image.Image,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
) -> tuple[str, float]:
    """
    Run one VLM forward pass.
    Returns (raw_text_output, mean_entropy).
    """
    from qwen_vl_utils import process_vision_info

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text",  "text": user_prompt},
            ],
        },
    ]

    text_input = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text_input],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    do_sample = temperature > 0.0

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=CFG.MAX_NEW_TOKENS,
            temperature=temperature if do_sample else None,
            do_sample=do_sample,
            output_scores=True,
            return_dict_in_generate=True,
        )

    # Decode text
    generated_ids = output.sequences[:, inputs["input_ids"].shape[1]:]
    raw_text = processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # Compute entropy from scores
    entropy = _entropy_from_scores(output.scores)

    return raw_text, entropy


def _entropy_from_scores(scores: tuple) -> float:
    """Convert HuggingFace generate() scores to mean entropy (bits)."""
    if not scores:
        return float("inf")

    total_entropy = 0.0
    for step_logits in scores:
        # step_logits: (batch=1, vocab_size)
        log_probs = torch.log_softmax(step_logits[0].float(), dim=-1)
        probs     = torch.exp(log_probs)
        # Only top-32 tokens to keep it fast
        top_probs, _ = torch.topk(probs, k=min(32, probs.shape[-1]))
        top_probs = top_probs.cpu()
        h = -(top_probs * torch.log2(top_probs + 1e-12)).sum().item()
        total_entropy += h

    return total_entropy / len(scores)


# ─── Sandbox Path ─────────────────────────────────────────────────────────────

def _sandbox_answer(
    model,
    processor,
    image: Image.Image,
    sandbox: Sandbox,
    deadline: float,
) -> dict:
    """
    Computation question path:
      1. Ask VLM to generate a PyTorch script
      2. Execute in sandbox
      3. Ask VLM to match the output to the correct option letter
    """
    if time.time() > deadline:
        return {"answer": None, "entropy": float("inf"), "path": "sandbox_timeout"}

    # Step 1: code generation (greedy — we want deterministic code)
    code_text, entropy1 = _run_vlm(
        model, processor, image,
        system_prompt=CFG.SANDBOX_SYSTEM_PROMPT,
        user_prompt=(
            "Write a PyTorch script that computes the numerical answer to the question in this image. "
            "Print ONLY the final value or shape."
        ),
        temperature=CFG.TEMPERATURE_GREEDY,
    )

    code = extract_code_block(code_text)
    if not code:
        print("[sandbox] ⚠️  No code extracted — falling back to direct path")
        return _direct_answer(model, processor, image, deadline, temperature=CFG.TEMPERATURE_GREEDY)

    print(f"[sandbox] Generated code:\n{code}\n")

    # Step 2: execute
    if time.time() > deadline:
        return {"answer": None, "entropy": float("inf"), "path": "sandbox_timeout"}

    exec_output = sandbox.execute(code, timeout=CFG.SANDBOX_TIMEOUT)
    print(f"[sandbox] Output: {exec_output}")

    if "[TIMEOUT]" in exec_output or "[STDERR]" in exec_output:
        print("[sandbox] ⚠️  Execution error — falling back to direct path")
        return _direct_answer(model, processor, image, deadline, temperature=CFG.TEMPERATURE_GREEDY)

    # Step 3: match output to option (greedy — deterministic matching)
    match_prompt = (
        f"The PyTorch script produced this output:\n{exec_output}\n\n"
        "Looking at the answer options in the image, which letter (A, B, C, or D) "
        "matches this output? Output ONLY the letter."
    )
    match_text, entropy2 = _run_vlm(
        model, processor, image,
        system_prompt=CFG.MATCH_SYSTEM_PROMPT,
        user_prompt=match_prompt,
        temperature=CFG.TEMPERATURE_GREEDY,
    )

    answer = parse_answer(match_text)
    mean_entropy = (entropy1 + entropy2) / 2

    print(f"[sandbox] ✅ Answer: {answer}  Entropy: {mean_entropy:.3f}")
    return {"answer": answer, "entropy": mean_entropy, "path": "sandbox"}


# ─── Direct Path ──────────────────────────────────────────────────────────────

def _direct_answer(
    model,
    processor,
    image: Image.Image,
    deadline: float,
    temperature: float = None,
) -> dict:
    """
    Conceptual question path: ask VLM directly for A/B/C/D.
    """
    if time.time() > deadline:
        return {"answer": None, "entropy": float("inf"), "path": "timeout"}

    if temperature is None:
        temperature = CFG.TEMPERATURE

    raw_text, entropy = _run_vlm(
        model, processor, image,
        system_prompt=CFG.DIRECT_SYSTEM_PROMPT,
        user_prompt="What is the correct answer? Output ONLY the letter A, B, C, or D.",
        temperature=temperature,
    )

    answer = parse_answer(raw_text)
    print(f"[direct] Raw: {raw_text!r}  →  Answer: {answer}  Entropy: {entropy:.3f}")
    return {"answer": answer, "entropy": entropy, "path": "direct"}


# ─── Public Entry Point ───────────────────────────────────────────────────────

def solve_one(
    model,
    processor,
    image_path: str,
    sandbox: Optional[Sandbox],
    deadline: float,
    temperature: float = CFG.TEMPERATURE,
) -> dict:
    """
    Solve a single MCQ image. Returns:
        {"answer": "A"|"B"|"C"|"D"|None, "entropy": float, "path": str}

    Called ATTEMPTS times per image by the notebook; results fed into voting.
    """
    image = Image.open(image_path).convert("RGB")

    # Detect question type via a fast text-only caption pass
    # We ask the model for the question text first, then route
    caption_text, _ = _run_vlm(
        model, processor, image,
        system_prompt="Extract all text from this image. Output only the extracted text, nothing else.",
        user_prompt="Extract all text.",
        temperature=0.0,
    )
    print(f"[solver] Extracted text preview: {caption_text[:200]!r}")

    if (
        CFG.SANDBOX_ENABLED
        and sandbox is not None
        and is_computation_question(caption_text, CFG.SANDBOX_KEYWORDS)
        and time.time() < deadline - CFG.SANDBOX_TIMEOUT - 5
    ):
        print("[solver] → Routing to SANDBOX path")
        return _sandbox_answer(model, processor, image, sandbox, deadline)
    else:
        print("[solver] → Routing to DIRECT path")
        return _direct_answer(model, processor, image, deadline, temperature)
