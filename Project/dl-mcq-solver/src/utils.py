"""
utils.py — Offline wheel installer, time budget tracker, answer parser.
"""

import os
import re
import sys
import time
import subprocess
from typing import Optional


# ─── Offline Wheel Installer (adapted from AIMO3) ────────────────────────────

def install_offline_wheels(wheels_archive: str, temp_dir: str) -> None:
    """
    Extract wheels.tar.gz and pip-install from local files.
    Safe to call multiple times — skips extraction if already done.
    """
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir, exist_ok=True)
        print(f"[setup] Extracting {wheels_archive} → {temp_dir} ...")
        subprocess.run(["tar", "-xzf", wheels_archive, "-C", temp_dir], check=True)
        print("[setup] Extraction complete.")

    packages = ["qwen-vl-utils", "flash-attn", "jupyter_client", "ipykernel"]

    print(f"[setup] Installing offline packages: {packages}")
    subprocess.run(
        [
            sys.executable, "-m", "pip", "install",
            "--no-index",
            "--find-links", f"{temp_dir}/wheels",
            *packages,
        ],
        check=True,
    )
    print("[setup] ✅ All packages installed.")


# ─── Time Budget Tracker (adapted from AIMO3) ────────────────────────────────

class TimeBudget:
    """
    Tracks notebook-level time and allocates per-question budgets.

    Usage:
        budget = TimeBudget(total_seconds=3300, n_questions=50)
        for q in questions:
            deadline = budget.next_deadline()
            answer   = solve(q, deadline=deadline)
            budget.mark_done()
    """

    def __init__(self, total_seconds: int, n_questions: int,
                 base_per_question: int = 60, max_per_question: int = 120):
        self._start          = time.time()
        self._total          = total_seconds
        self._remaining_qs   = n_questions
        self._base           = base_per_question
        self._max            = max_per_question

    @property
    def elapsed(self) -> float:
        return time.time() - self._start

    @property
    def time_left(self) -> float:
        return max(0.0, self._total - self.elapsed)

    def next_deadline(self) -> float:
        """Return absolute timestamp (time.time()) for current question deadline."""
        others_reserved = max(0, self._remaining_qs - 1) * self._base
        budget = min(self._max, self.time_left - others_reserved)
        budget = max(budget, self._base)   # never starve the current question
        deadline = time.time() + budget
        print(f"[budget] {self.time_left:.0f}s left | "
              f"{self._remaining_qs} questions remaining | "
              f"this question budget: {budget:.0f}s")
        return deadline

    def mark_done(self) -> None:
        self._remaining_qs = max(0, self._remaining_qs - 1)

    def is_over(self) -> bool:
        return self.time_left <= 0


# ─── Answer Parsing ──────────────────────────────────────────────────────────

VALID_ANSWERS = {"A", "B", "C", "D"}

_LETTER_PATTERN = re.compile(r'\b([A-D])\b')


def parse_answer(text: str) -> Optional[str]:
    """
    Extract the first valid answer letter from model output.
    Returns one of 'A','B','C','D' or None if unparseable.
    """
    text = text.strip()

    # Direct single-letter output (ideal case)
    if text in VALID_ANSWERS:
        return text

    # "The answer is C" / "Answer: B" / "**A**"
    for pattern in [
        r'(?:answer|option)\s*(?:is\s*)?[:\*]*\s*([A-D])',
        r'\*\*([A-D])\*\*',
        r'^([A-D])[\.:\)]',
    ]:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Last resort: first standalone letter found
    matches = _LETTER_PATTERN.findall(text)
    if matches:
        return matches[0].upper()

    return None


# ─── Computation Question Detector ───────────────────────────────────────────

def is_computation_question(question_text: str, keywords: list[str]) -> bool:
    """
    Returns True if the question likely requires numeric computation
    (CNN shape, parameter count, etc.) and should be routed through sandbox.
    """
    q_lower = question_text.lower()
    return any(kw in q_lower for kw in keywords)


# ─── Misc ─────────────────────────────────────────────────────────────────────

def format_elapsed(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"
