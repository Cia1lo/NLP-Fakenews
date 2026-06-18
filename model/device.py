from __future__ import annotations

import os
import platform
from typing import Any

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch


def mps_diagnostics() -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "mac_ver": platform.mac_ver()[0],
        "torch_version": torch.__version__,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        "mps_usable": False,
        "mps_error": None,
    }
    if diagnostics["mps_built"]:
        try:
            torch.ones(1, device="mps")
            diagnostics["mps_usable"] = True
        except Exception as exc:
            diagnostics["mps_error"] = f"{type(exc).__name__}: {exc}"
    return diagnostics


def cuda_diagnostics() -> dict[str, Any]:
    diagnostics: dict[str, Any] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if torch.cuda.is_available():
        diagnostics["cuda_device_name"] = torch.cuda.get_device_name(0)
    return diagnostics


def device_report() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        **cuda_diagnostics(),
        **mps_diagnostics(),
    }


def select_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        raise RuntimeError("CUDA was requested but is not available.")
    if name == "mps":
        diagnostics = mps_diagnostics()
        if diagnostics["mps_usable"]:
            return torch.device("mps")
        raise RuntimeError(
            "MPS was requested but is not usable in this Python environment. "
            f"Diagnostics: {diagnostics}"
        )
    if name != "auto":
        raise ValueError(f"Unsupported device: {name}")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if mps_diagnostics()["mps_usable"]:
        return torch.device("mps")
    return torch.device("cpu")
