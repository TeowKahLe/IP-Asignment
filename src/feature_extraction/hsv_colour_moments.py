"""HSV colour-moment features for the Hybrid pipeline."""

from __future__ import annotations

from time import perf_counter
from typing import Iterable

import cv2
import numpy as np


def extract_hsv_colour_moments(roi: np.ndarray, fruit_mask: np.ndarray) -> np.ndarray:
    """Return mean, standard deviation, and skewness for H, S, and V channels."""
    if roi is None or roi.size == 0 or roi.ndim != 3 or roi.shape[2] != 3:
        raise ValueError("roi must be a non-empty three-channel BGR image")
    if fruit_mask is None or fruit_mask.shape != roi.shape[:2]:
        raise ValueError("fruit_mask must match roi height and width")
    pixels = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[fruit_mask > 0].astype(np.float64)
    if not len(pixels):
        raise ValueError("fruit_mask must contain at least one fruit pixel")
    mean = pixels.mean(axis=0)
    standard_deviation = pixels.std(axis=0)
    centered = pixels - mean
    skewness = np.zeros(3, dtype=np.float64)
    valid = standard_deviation > np.finfo(np.float64).eps
    skewness[valid] = (centered[:, valid] ** 3).mean(axis=0) / standard_deviation[valid] ** 3
    return np.column_stack((mean, standard_deviation, skewness)).reshape(-1).astype(np.float32)


def extract_hsv_colour_moments_batch(
    rois: Iterable[np.ndarray], masks: Iterable[np.ndarray]
) -> tuple[np.ndarray, float]:
    """Extract nine HSV colour-moment features per masked fruit ROI."""
    started_at = perf_counter()
    vectors = [extract_hsv_colour_moments(roi, mask) for roi, mask in zip(rois, masks)]
    if not vectors:
        raise ValueError("at least one ROI is required")
    return np.vstack(vectors), perf_counter() - started_at
