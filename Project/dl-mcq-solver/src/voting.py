"""
voting.py — Entropy-weighted majority vote across multiple inference attempts.

Adapted from AIMO3's _select_answer and _compute_mean_entropy methods.

Intuition:
  - Run the model N times on the same image (with temperature > 0)
  - Each attempt produces an answer + token log-probabilities
  - Low entropy = model was very confident → higher weight
  - Aggregate weighted votes → pick the winner
"""

import math
from collections import defaultdict
from typing import Optional

import pandas as pd


# ─── Entropy Computation ─────────────────────────────────────────────────────

def compute_mean_entropy(logprobs_sequence: list[dict]) -> float:
    """
    Compute mean token-level entropy from a list of {token: log_prob} dicts.
    Lower entropy = more confident output.

    Args:
        logprobs_sequence: list of dicts like {"A": -0.1, "B": -2.3, ...}

    Returns:
        Mean entropy in bits. Returns inf if no data.
    """
    if not logprobs_sequence:
        return float("inf")

    total_entropy = 0.0
    count = 0

    for top_logprobs in logprobs_sequence:
        if not isinstance(top_logprobs, dict) or not top_logprobs:
            continue

        token_entropy = 0.0
        for _token, log_prob in top_logprobs.items():
            prob = math.exp(log_prob)
            if prob > 1e-12:
                token_entropy -= prob * math.log2(prob)

        total_entropy += token_entropy
        count += 1

    return total_entropy / count if count > 0 else float("inf")


# ─── Weighted Voter ───────────────────────────────────────────────────────────

def entropy_weighted_vote(results: list[dict]) -> str:
    """
    Select the best answer from multiple attempts using entropy-weighted voting.

    Each result dict must have:
        "answer"  : str  — one of 'A','B','C','D' (or None if failed)
        "entropy" : float — mean entropy of the generation

    Returns:
        The winning answer letter. Falls back to 'A' if all attempts failed.
    """
    answer_weights = defaultdict(float)
    answer_votes   = defaultdict(int)

    for r in results:
        ans     = r.get("answer")
        entropy = r.get("entropy", float("inf"))

        if ans is None or ans not in {"A", "B", "C", "D"}:
            continue

        weight = 1.0 / max(entropy, 1e-9)
        answer_weights[ans] += weight
        answer_votes[ans]   += 1

    if not answer_weights:
        print("[vote] ⚠️  All attempts failed — defaulting to 'A'")
        return "A"

    # Display vote summary (visible in Kaggle output log)
    rows = [
        {"Answer": ans, "Votes": answer_votes[ans], "Score": round(w, 4)}
        for ans, w in sorted(answer_weights.items(), key=lambda x: -x[1])
    ]
    print(pd.DataFrame(rows).to_string(index=False))

    winner = max(answer_weights, key=answer_weights.__getitem__)
    print(f"\n[vote] ✅ Selected: {winner}\n")
    return winner


# ─── Early-Stop Shortcut ─────────────────────────────────────────────────────

def has_consensus(results: list[dict], threshold: int = 3) -> bool:
    """
    Returns True if one answer already has >= threshold votes.
    Used to skip remaining attempts early.
    """
    votes = defaultdict(int)
    for r in results:
        ans = r.get("answer")
        if ans in {"A", "B", "C", "D"}:
            votes[ans] += 1
    return any(v >= threshold for v in votes.values())
