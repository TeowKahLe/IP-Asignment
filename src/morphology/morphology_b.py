"""Morphological closing for Pipeline B segmentation masks."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import cv2
import numpy as np


DEFAULT_CLOSING_KERNEL_SIZE = 3


def apply_morphological_closing(
    mask: np.ndarray, kernel_size: int = DEFAULT_CLOSING_KERNEL_SIZE,
    kernel_shape: str = "ellipse",
) -> tuple[np.ndarray, dict[str, Any], float]:
    """Fill small holes and gaps in a binary Otsu mask using closing."""
    if mask is None or mask.size == 0:
        raise ValueError("mask must be a non-empty binary image")

    if mask.ndim != 2 or not set(np.unique(mask).tolist()).issubset({0, 255}):
        raise ValueError("mask must be a 2-D image containing only 0 and 255")

    if not isinstance(kernel_size, int) or kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd integer")

    if kernel_shape.lower() != "ellipse":
        raise ValueError("Pipeline A uses an elliptical structuring element")

    started_at = perf_counter()

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    closed = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    processing_time_seconds = perf_counter() - started_at

    metadata = {
        "technique": "Morphological Processing",
        "algorithm": "Morphological Closing",
        "kernel_size": kernel_size,
        "kernel_shape": "ellipse",
        "processing_time_seconds": processing_time_seconds,
    }

    return closed, metadata, processing_time_seconds


# Backwards-compatible descriptive alias.
morphological_closing = apply_morphological_closing