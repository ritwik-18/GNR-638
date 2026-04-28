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
    Pull executable Python code out of model output.
    Tries multiple strategies in order of reliability.
    """
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

    # Strategy 3: fence without newline after backticks  ```python<code>```
    m = re.search(r"```(?:python)?(.*?)```", text, re.DOTALL)
    if m:
        candidate = m.group(1).strip()
        if any(kw in candidate for kw in ["import", "torch", "nn.", "print"]):
            return candidate

    # Strategy 4: grab all lines that look like Python code
    lines = text.strip().split("\n")
    code_lines = []
    in_code = False
    for line in lines:
        stripped = line.strip()
        # Triggers that indicate we've entered a code block
        if re.match(r"^(import |from |model\s*=|x\s*=|output|torch\.|nn\.|print\()", stripped):
            in_code = True
        if in_code:
            # Stop if we hit prose (lines with no code-like content)
            if stripped and not re.match(
                r"^(import |from |#|model|x|h|out|input|conv|pool|linear|print|torch|nn|shape|size|\w+\s*=)",
                stripped
            ) and len(stripped.split()) > 6 and stripped[0].islower():
                break
            code_lines.append(line)

    if code_lines and any("print" in l for l in code_lines):
        return "\n".join(code_lines).strip()

    return None