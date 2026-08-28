"""Pipeline A preprocessing: median-filter noise reduction only.

The functions operate in memory and never overwrite or save the source image.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import cv2
import numpy as np


DEFAULT_MEDIAN_KERNEL_SIZE = 3


def _validate_kernel_size(kernel_size: int) -> int:
    """Return a valid OpenCV median-filter kernel size."""
    if not isinstance(kernel_size, int) or kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer greater than or equal to 3")
    return kernel_size


def median_filter(image: np.ndarray, kernel_size: int = DEFAULT_MEDIAN_KERNEL_SIZE) -> np.ndarray:
    """Reduce isolated noise while preserving edges with an OpenCV median filter."""
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty OpenCV image")
    return cv2.medianBlur(image, _validate_kernel_size(kernel_size))


def preprocess_image(
    image: np.ndarray, kernel_size: int = DEFAULT_MEDIAN_KERNEL_SIZE
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply Pipeline A's median-filter preprocessing and return metadata."""
    started_at = perf_counter()
    filtered = median_filter(image, kernel_size)
    return filtered, {
        "technique": "Noise reduction",
        "algorithm": "Median filter",
        "kernel_size": kernel_size,
        "processing_time_seconds": perf_counter() - started_at,
    }
