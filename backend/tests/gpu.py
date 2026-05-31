"""Helpers for tests that require a provisioned GPU runner."""

from __future__ import annotations

import os

GPU_TEST_SKIP_REASON = "set RUN_GPU_TESTS=1 on a provisioned CUDA runner to run GPU/inference checks"


def cuda_available() -> bool:
    """Return True when torch can see a CUDA device."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def gpu_tests_enabled() -> bool:
    """GPU tests are opt-in so normal pytest stays fast on CUDA hosts."""
    return os.environ.get("RUN_GPU_TESTS") == "1" and cuda_available()
