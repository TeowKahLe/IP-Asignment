"""RGB colour-histogram feature extraction for Pipeline A."""

from __future__ import annotations

from time import perf_counter
from typing import Iterable

import cv2
import numpy as np


RGB_HISTOGRAM_BINS = (8, 16, 32, 64)
RGB_INTENSITY_RANGE = (0, 256)  # OpenCV's upper bound is exclusive: values 0--255.
HISTOGRAM_NORMALIZATION = "L1"


def extract_rgb_histogram(
    standardized_roi: np.ndarray, fruit_mask: np.ndarray, bins_per_channel: int
) -> np.ndarray:
    """Return a concatenated, L1-normalized R/G/B histogram feature vector.

    ``standardized_roi`` and ``fruit_mask`` are the corresponding outputs of
    ROI standardization. Only non-zero mask pixels contribute, excluding the
    letterbox padding and black background outside the fruit contour. Each
    channel is L1-normalized independently.
    """
    if bins_per_channel not in RGB_HISTOGRAM_BINS:
        raise ValueError(f"bins_per_channel must be one of {RGB_HISTOGRAM_BINS}")
    if standardized_roi is None or standardized_roi.size == 0:
        raise ValueError("standardized_roi must be a non-empty OpenCV image")
    if standardized_roi.ndim != 3 or standardized_roi.shape[2] != 3:
        raise ValueError("standardized_roi must be a three-channel BGR image")
    if fruit_mask is None or fruit_mask.shape != standardized_roi.shape[:2]:
        raise ValueError("fruit_mask must be a single-channel image matching standardized_roi")
    binary_mask = np.where(fruit_mask > 0, 255, 0).astype(np.uint8)
    if not np.any(binary_mask):
        raise ValueError("fruit_mask must contain at least one fruit pixel")

    rgb_image = cv2.cvtColor(standardized_roi, cv2.COLOR_BGR2RGB)
    channel_histograms = []
    for channel_index in range(3):
        histogram = cv2.calcHist(
            [rgb_image], [channel_index], binary_mask, [bins_per_channel], RGB_INTENSITY_RANGE
        )
        histogram = cv2.normalize(histogram, None, alpha=1.0, norm_type=cv2.NORM_L1)
        channel_histograms.append(histogram.ravel())
    return np.concatenate(channel_histograms).astype(np.float32, copy=False)


def extract_rgb_histogram_batch(
    standardized_rois: Iterable[np.ndarray], fruit_masks: Iterable[np.ndarray], bins_per_channel: int
) -> tuple[np.ndarray, float]:
    """Extract one RGB-histogram vector per already-standardized ROI."""
    started_at = perf_counter()
    roi_list, mask_list = list(standardized_rois), list(fruit_masks)
    if len(roi_list) != len(mask_list):
        raise ValueError("standardized_rois and fruit_masks must contain the same number of items")
    features = [
        extract_rgb_histogram(roi, mask, bins_per_channel)
        for roi, mask in zip(roi_list, mask_list)
    ]
    if not features:
        raise ValueError("standardized_rois must contain at least one ROI")
    return np.vstack(features), perf_counter() - started_at
