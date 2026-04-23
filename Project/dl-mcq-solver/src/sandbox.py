"""
sandbox.py — Persistent Jupyter kernel for executing PyTorch code.
Adapted from AIMO3Sandbox; simplified for single-worker MCQ use.

The solver uses this to:
  1. Get Qwen2-VL to generate a PyTorch script
  2. Execute it here to get deterministic numeric output
  3. Match that output to one of the 4 answer options
"""

import re
import time
import queue
import contextlib
from typing import Optional


class Sandbox:
    """
    Wraps a persistent Jupyter kernel.
    Survives multiple execute() calls (stateful session).
    Call reset() between questions to clear namespace.
    Call close() when done.
    """

    def __init__(self, timeout: float = 15.0):
        from jupyter_client import KernelManager

        self._timeout = timeout
        self._km      = KernelManager()
        self._km.start_kernel()

        self._client  = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=30)

        # Pre-import the deep learning stack
        self.execute(
            "import torch\n"
            "import torch.nn as nn\n"
            "import math\n"
            "import numpy as np\n"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def execute(self, code: str, timeout: Optional[float] = None) -> str:
        """Run code and return stdout+stderr as a single string."""
        effective_timeout = timeout or self._timeout
        msg_id = self._client.execute(
            code, store_history=True, allow_stdin=False, stop_on_error=False
        )

        stdout_parts = []
        stderr_parts = []
        start = time.time()

        while True:
            if time.time() - start > effective_timeout:
                self._km.interrupt_kernel()
                return "[TIMEOUT] Execution exceeded time limit."

            try:
                msg = self._client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content  = msg.get("content", {})

            if msg_type == "stream":
                target = stdout_parts if content.get("name") == "stdout" else stderr_parts
                target.append(content.get("text", ""))

            elif msg_type == "error":
                cleaned = self._clean_traceback(content.get("traceback", []))
                stderr_parts.append(cleaned)

            elif msg_type in {"execute_result", "display_data"}:
                text = content.get("data", {}).get("text/plain", "")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else text + "\n")

            elif msg_type == "status":
                if content.get("execution_state") == "idle":
                    break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            return f"{stdout.rstrip()}\n[STDERR] {stderr}" if stdout else f"[STDERR] {stderr}"

        return stdout.strip() if stdout.strip() else "[WARN] No output. Use print()."

    def reset(self) -> None:
        """Clear kernel namespace between questions."""
        self.execute(
            "%reset -f\n"
            "import torch\n"
            "import torch.nn as nn\n"
            "import math\n"
            "import numpy as np\n"
        )

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._client.stop_channels()
        with contextlib.suppress(Exception):
            self._km.shutdown_kernel(now=True)
        with contextlib.suppress(Exception):
            self._km.cleanup_resources()

    def __del__(self):
        self.close()

    # ── Internals ─────────────────────────────────────────────────────────────

    @staticmethod
    def _clean_traceback(traceback: list[str]) -> str:
        clean = []
        for frame in traceback:
            frame = re.sub(r"\x1b\[[0-9;]*m", "", frame)  # strip ANSI
            clean.append(frame)
        return "".join(clean)


# ─── Code Extractor ───────────────────────────────────────────────────────────

def extract_code_block(text: str) -> Optional[str]:
    """
    Pull the first ```python ... ``` block out of model output.
    Falls back to raw text if no fences found.
    """
    # Fenced block
    m = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    if m:
        return m.group(1).strip()

    # Indented block heuristic: starts with 'import' or 'model = '
    lines = text.strip().split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        if re.match(r"^(import |from |model |#)", line):
            in_code = True
        if in_code:
            code_lines.append(line)

    return "\n".join(code_lines) if code_lines else None
