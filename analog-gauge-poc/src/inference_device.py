"""Validated inference-device selection and accelerator runtime evidence."""

from __future__ import annotations

from typing import Any

import onnxruntime as ort
import torch
from torch import version as torch_version

DEVICE_CHOICES = ("cpu", "mps", "cuda")
ALL_DEVICE_CHOICES = ("all", *DEVICE_CHOICES)


def ensure_device_available(device: str) -> None:
    """Fail clearly before model loading when a requested device is unavailable."""

    if device not in DEVICE_CHOICES:
        raise ValueError(f"Unsupported inference device: {device}")
    if device == "mps" and not (
        torch.backends.mps.is_built() and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is not available")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but PyTorch cannot access a CUDA device. "
            "Install the CUDA-enabled PyTorch build and verify the NVIDIA runtime."
        )


def available_devices(selection: str) -> list[str]:
    """Resolve one requested device or every accelerator available on this host."""

    if selection != "all":
        ensure_device_available(selection)
        return [selection]
    devices = ["cpu"]
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


def synchronize(device: str) -> None:
    """Wait for asynchronous accelerator work before recording elapsed time."""

    if device == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()
    elif device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


def accelerator_fingerprint(selected_device: str) -> dict[str, Any]:
    """Return auditable accelerator details without initializing unavailable devices."""

    cuda_available = torch.cuda.is_available()
    cuda_device_count = torch.cuda.device_count() if cuda_available else 0
    cuda_devices = (
        [torch.cuda.get_device_name(index) for index in range(cuda_device_count)]
        if cuda_available
        else []
    )
    cuda_selected = selected_device == "cuda" and cuda_available
    return {
        "selected_device": selected_device,
        "torch_cuda_build": torch_version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": cuda_device_count,
        "cuda_devices": cuda_devices,
        "cuda_current_device": torch.cuda.current_device()
        if cuda_selected
        else None,
        "cuda_memory_allocated_bytes": torch.cuda.memory_allocated()
        if cuda_selected
        else 0,
        "cuda_memory_reserved_bytes": torch.cuda.memory_reserved()
        if cuda_selected
        else 0,
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
        "onnxruntime_available_providers": ort.get_available_providers(),
        "ocr_execution_policy": "cpu",
    }
