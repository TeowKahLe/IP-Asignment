"""Global threshold-based segmentation for Pipeline B.

This module intentionally contains no morphology: that is a separate, later
pipeline stage. The input is the BGR image produced by the locked Wiener
filter in preprocessing.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import cv2
import numpy as np


DEFAULT_GLOBAL_THRESHOLD = 128


def _border_foreground_fraction(mask: np.ndarray) -> float:
    """Return the fraction of foreground pixels along the image border."""
    border = np.concatenate((mask[0], mask[-1], mask[1:-1, 0], mask[1:-1, -1]))
    return float(np.mean(border == 255))


def segment_image(
    image: np.ndarray,
    threshold: int = DEFAULT_GLOBAL_THRESHOLD
) -> tuple[np.ndarray, dict[str, Any], float]:
    """Segment a Wiener-filtered BGR image using global thresholding.

    A fixed global threshold is applied to the grayscale image. Between the
    thresholded image and its inverse, the mask with less foreground touching
    the border is selected so the centrally photographed fruit is consistently
    represented as foreground. This is polarity handling, not threshold
    tuning.
    """
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty BGR image")

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a three-channel BGR image")

    if not isinstance(threshold, (int, float)) or not 0 <= threshold <= 255:
        raise ValueError("threshold must be between 0 and 255")

    started_at = perf_counter()

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    _, binary_mask = cv2.threshold(
        grayscale,
        threshold,
        255,
        cv2.THRESH_BINARY
    )

    inverse_mask = cv2.bitwise_not(binary_mask)

    binary_border = _border_foreground_fraction(binary_mask)
    inverse_border = _border_foreground_fraction(inverse_mask)

    if inverse_border < binary_border:
        mask, polarity = inverse_mask, "inverted"
    else:
        mask, polarity = binary_mask, "binary"

    processing_time_seconds = perf_counter() - started_at

    metadata = {
        "technique": "Threshold-Based Segmentation",
        "algorithm": "Global Thresholding",
        "global_threshold": float(threshold),
        "mask_polarity": polarity,
        "processing_time_seconds": processing_time_seconds,
    }

    return mask, metadata, processing_time_seconds