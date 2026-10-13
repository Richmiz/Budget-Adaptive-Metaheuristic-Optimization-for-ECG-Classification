"""Hardware fingerprint and guard.

Budgets are wall-clock caps expressed in T_ref units, so T_ref and every confirmatory run
must come from the same GPU type (protocol §3.9.2). On Colab the assigned GPU can change
between sessions; ``require_gpu`` stops a worker that landed on the wrong type.
"""

from __future__ import annotations

import os
import platform
import sys


def fingerprint() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform(), "host": platform.node(),
            "colab": "COLAB_RELEASE_TAG" in os.environ or "COLAB_GPU" in os.environ}
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda"] = torch.version.cuda
        info["cudnn"] = torch.backends.cudnn.version()
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["gpu"] = props.name
            info["gpu_memory_gb"] = round(props.total_memory / 2**30, 1)
    except ImportError:
        info["torch"] = None
    return info


def require_gpu(expected: str | None) -> dict:
    info = fingerprint()
    if expected:
        actual = info.get("gpu")
        if actual is None or expected.lower() not in actual.lower():
            raise RuntimeError(f"GPU guard: expected '{expected}', found '{actual}'. "
                               "Confirmatory runs must use the frozen GPU type.")
    return info
