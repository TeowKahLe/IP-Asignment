"""Reusable image-quality evaluation for preprocessing results.

Metrics are calculated with PyIQA on the CPU by default.  BRISQUE, NIQE, and
PIQE are no-reference measures (lower is better); SSIM compares the filtered
image to its standardized input (higher is better).
"""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any

import numpy as np


# Keep downloaded no-reference metric weights inside the project, rather than a
# user-profile cache that may be protected on lab machines.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("TORCH_HOME", str(PROJECT_ROOT / ".cache" / "torch"))


@lru_cache(maxsize=2)
def _load_metrics(device: str) -> dict[str, Any]:
    """Create and cache the PyIQA metric models for a selected device."""
    try:
        import pyiqa
    except ImportError as error:
        raise ImportError(
            "Preprocessing metrics require PyIQA. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from error

    return {
        "brisque": pyiqa.create_metric("brisque", device=device),
        "niqe": pyiqa.create_metric("niqe", device=device),
        "piqe": pyiqa.create_metric("piqe", device=device),
        "ssim": pyiqa.create_metric("ssim", device=device),
    }


def _validate_image_pair(reference_image: np.ndarray, processed_image: np.ndarray) -> None:
    if reference_image is None or processed_image is None:
        raise ValueError("reference_image and processed_image must be OpenCV images")
    if reference_image.size == 0 or processed_image.size == 0:
        raise ValueError("reference_image and processed_image must not be empty")
    if reference_image.shape != processed_image.shape:
        raise ValueError("reference_image and processed_image must have the same shape")
    if reference_image.ndim != 3 or reference_image.shape[2] != 3:
        raise ValueError("images must be 3-channel BGR OpenCV images")


def _bgr_image_to_tensor(image: np.ndarray, device: str) -> Any:
    """Convert a uint8 OpenCV BGR image to a normalized PyTorch RGB tensor."""
    try:
        import torch
    except ImportError as error:
        raise ImportError(
            "Preprocessing metrics require PyTorch. Install dependencies with "
            "`pip install -r requirements.txt`."
        ) from error

    rgb_image = np.ascontiguousarray(image[:, :, ::-1])
    return torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)


def evaluate_preprocessing(
    reference_image: np.ndarray,
    processed_image: np.ndarray,
    processing_time_seconds: float,
    device: str = "cpu",
) -> dict[str, float]:
    """Calculate BRISQUE, NIQE, PIQE, SSIM, and preprocessing time.

    Parameters
    ----------
    reference_image:
        Standardized OpenCV BGR image before the preprocessing method.
    processed_image:
        OpenCV BGR image output from the preprocessing method.
    processing_time_seconds:
        Time measured while applying the preprocessing method itself.
    device:
        PyTorch device, ``"cpu"`` by default for reproducible evaluation.
    """
    _validate_image_pair(reference_image, processed_image)
    if processing_time_seconds < 0:
        raise ValueError("processing_time_seconds cannot be negative")

    metrics = _load_metrics(device)
    reference_tensor = _bgr_image_to_tensor(reference_image, device)
    processed_tensor = _bgr_image_to_tensor(processed_image, device)

    return {
        "brisque": float(metrics["brisque"](processed_tensor).item()),
        "niqe": float(metrics["niqe"](processed_tensor).item()),
        "piqe": float(metrics["piqe"](processed_tensor).item()),
        "ssim": float(metrics["ssim"](processed_tensor, reference_tensor).item()),
        "processing_time_seconds": float(processing_time_seconds),
    }
