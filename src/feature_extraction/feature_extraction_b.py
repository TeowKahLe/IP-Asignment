"""Local Binary Pattern (LBP) texture feature extraction for Pipeline B."""

from __future__ import annotations

from time import perf_counter
from typing import Iterable

import cv2
import numpy as np
from skimage.feature import local_binary_pattern


LBP_RADII = (1, 2, 3)
LBP_POINTS = 8
LBP_METHOD = "uniform"
HISTOGRAM_NORMALIZATION = "L1"


def extract_lbp(
    standardized_roi: np.ndarray,
    fruit_mask: np.ndarray,
    radius: int,
    points: int = LBP_POINTS,
) -> np.ndarray:
    """Return an L1-normalized LBP texture histogram feature vector.

    ``standardized_roi`` and ``fruit_mask`` are the corresponding outputs of
    ROI standardization. Only non-zero mask pixels contribute, excluding the
    letterbox padding and black background outside the fruit component.
    """

    if radius not in LBP_RADII:
        raise ValueError(
            f"radius must be one of {LBP_RADII}"
        )

    if points != LBP_POINTS:
        raise ValueError(
            f"points must be fixed at {LBP_POINTS}"
        )

    if standardized_roi is None or standardized_roi.size == 0:
        raise ValueError(
            "standardized_roi must be a non-empty OpenCV image"
        )

    if standardized_roi.ndim != 3 or standardized_roi.shape[2] != 3:
        raise ValueError(
            "standardized_roi must be a three-channel BGR image"
        )

    if fruit_mask is None or fruit_mask.shape != standardized_roi.shape[:2]:
        raise ValueError(
            "fruit_mask must be a single-channel image matching standardized_roi"
        )

    binary_mask = np.where(
        fruit_mask > 0,
        255,
        0
    ).astype(np.uint8)

    if not np.any(binary_mask):
        raise ValueError(
            "fruit_mask must contain at least one fruit pixel"
        )

    # Convert standardized fruit ROI to grayscale
    grayscale = cv2.cvtColor(
        standardized_roi,
        cv2.COLOR_BGR2GRAY
    )

    # Extract Local Binary Pattern
    lbp_image = local_binary_pattern(
        grayscale,
        points,
        radius,
        method=LBP_METHOD
    )

    # Keep only LBP values that belong to the fruit region
    fruit_lbp_values = lbp_image[
        binary_mask > 0
    ]

    # Uniform LBP with P points produces P + 2 possible values
    number_of_bins = points + 2

    histogram, _ = np.histogram(
        fruit_lbp_values,
        bins=np.arange(
            0,
            number_of_bins + 1
        )
    )

    histogram = histogram.astype(
        np.float32
    )

    # L1 histogram normalization
    histogram_sum = histogram.sum()

    if histogram_sum > 0:
        histogram /= histogram_sum

    return histogram


def extract_lbp_batch(
    standardized_rois: Iterable[np.ndarray],
    fruit_masks: Iterable[np.ndarray],
    radius: int,
    points: int = LBP_POINTS,
) -> tuple[np.ndarray, float]:
    """Extract one LBP feature vector per already-standardized ROI."""

    started_at = perf_counter()

    roi_list = list(
        standardized_rois
    )

    mask_list = list(
        fruit_masks
    )

    if len(roi_list) != len(mask_list):
        raise ValueError(
            "standardized_rois and fruit_masks must contain the same number of items"
        )

    features = [
        extract_lbp(
            roi,
            mask,
            radius,
            points
        )
        for roi, mask in zip(
            roi_list,
            mask_list
        )
    ]

    if not features:
        raise ValueError(
            "standardized_rois must contain at least one ROI"
        )

    return (
        np.vstack(features),
        perf_counter() - started_at
    )