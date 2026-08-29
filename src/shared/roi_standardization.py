"""Size-preserving standardization for extracted fruit ROIs."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import cv2
import numpy as np

DEFAULT_ROI_TARGET_SIZE = (512, 512)  # (width, height)
DEFAULT_PADDING_VALUE = (114, 114, 114)  # Neutral gray BGR letterbox padding.


def standardize_roi(
    roi: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_ROI_TARGET_SIZE,
    padding_value: tuple[int, int, int] = DEFAULT_PADDING_VALUE,
) -> tuple[np.ndarray, dict[str, Any], float]:
    """Rescale an ROI proportionally and pad it to the requested size.

    One shared scale factor preserves the fruit's proportions; it is never
    stretched or cropped, and remaining canvas receives neutral padding.
    """
    started_at = perf_counter()
    if roi is None or roi.size == 0:
        raise ValueError("roi must be a non-empty OpenCV image")
    if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError("target_size must be a positive (width, height) tuple")

    target_width, target_height = int(target_size[0]), int(target_size[1])
    original_height, original_width = roi.shape[:2]
    scale_factor = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, min(target_width, round(original_width * scale_factor)))
    resized_height = max(1, min(target_height, round(original_height * scale_factor)))
    interpolation = cv2.INTER_AREA if scale_factor < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(roi, (resized_width, resized_height), interpolation=interpolation)

    horizontal_padding = target_width - resized_width
    vertical_padding = target_height - resized_height
    padding = {
        "top": vertical_padding // 2,
        "bottom": vertical_padding - vertical_padding // 2,
        "left": horizontal_padding // 2,
        "right": horizontal_padding - horizontal_padding // 2,
    }
    standardized = cv2.copyMakeBorder(
        resized, padding["top"], padding["bottom"], padding["left"], padding["right"],
        cv2.BORDER_CONSTANT, value=padding_value,
    )
    metadata: dict[str, Any] = {
        "algorithm": "Size-Preserving Rescaling",
        "original_size": (original_width, original_height),
        "resized_size": (resized_width, resized_height),
        "target_size": (target_width, target_height),
        "scale_factor": scale_factor,
        "padding": padding,
        "interpolation_method": "cv2.INTER_AREA" if scale_factor < 1 else "cv2.INTER_CUBIC",
    }
    return standardized, metadata, perf_counter() - started_at


def standardize_roi_mask(
    roi_mask: np.ndarray,
    target_size: tuple[int, int] = DEFAULT_ROI_TARGET_SIZE,
) -> np.ndarray:
    """Size-preserve a binary ROI mask using nearest-neighbour interpolation.

    The spatial transform matches :func:`standardize_roi`, while zero padding
    and nearest-neighbour interpolation keep the fruit/background mask binary.
    """
    if roi_mask is None or roi_mask.size == 0:
        raise ValueError("roi_mask must be a non-empty image")
    if roi_mask.ndim != 2:
        raise ValueError("roi_mask must be a single-channel binary image")
    if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError("target_size must be a positive (width, height) tuple")

    target_width, target_height = int(target_size[0]), int(target_size[1])
    original_height, original_width = roi_mask.shape
    scale_factor = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, min(target_width, round(original_width * scale_factor)))
    resized_height = max(1, min(target_height, round(original_height * scale_factor)))
    binary_mask = np.where(roi_mask > 0, 255, 0).astype(np.uint8)
    resized = cv2.resize(binary_mask, (resized_width, resized_height), interpolation=cv2.INTER_NEAREST)

    horizontal_padding = target_width - resized_width
    vertical_padding = target_height - resized_height
    return cv2.copyMakeBorder(
        resized,
        vertical_padding // 2,
        vertical_padding - vertical_padding // 2,
        horizontal_padding // 2,
        horizontal_padding - horizontal_padding // 2,
        cv2.BORDER_CONSTANT,
        value=0,
    )
