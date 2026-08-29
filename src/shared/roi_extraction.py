"""Contour-based extraction of individual fruit regions of interest (ROIs)."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import cv2
import numpy as np

def extract_fruit_rois(
    image: np.ndarray,
    segmentation_mask: np.ndarray,
    min_contour_area: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    """Extract one masked ROI per valid external contour.

    The image and mask must have the same height and width. Returned ROIs are
    ordered top-to-bottom, then left-to-right for a stable multiple-fruit order.
    """
    started_at = perf_counter()
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty OpenCV image")
    if segmentation_mask is None or segmentation_mask.size == 0:
        raise ValueError("segmentation_mask must be a non-empty image")
    if image.shape[:2] != segmentation_mask.shape[:2]:
        raise ValueError("image and segmentation_mask must have the same height and width")
    if min_contour_area < 0:
        raise ValueError("min_contour_area must be non-negative")

    mask = segmentation_mask
    if mask.ndim == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    binary_mask = np.where(mask > 0, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected: list[dict[str, Any]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_contour_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        local_mask = np.zeros((height, width), dtype=np.uint8)
        shifted_contour = contour - np.array([[[x, y]]])
        cv2.drawContours(local_mask, [shifted_contour], -1, 255, thickness=cv2.FILLED)
        crop = image[y : y + height, x : x + width].copy()
        masked_roi = cv2.bitwise_and(crop, crop, mask=local_mask)
        moments = cv2.moments(contour)
        centroid = (
            (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
            if moments["m00"] else None
        )
        detected.append({
            "roi": masked_roi,
            "roi_mask": local_mask,
            "contour": contour,
            "bounding_box": (x, y, width, height),
            "area": area,
            "centroid": centroid,
        })

    detected.sort(key=lambda item: (item["bounding_box"][1], item["bounding_box"][0]))
    for index, item in enumerate(detected, start=1):
        item["roi_id"] = index

    metadata = {
        "algorithm": "External Contour Detection",
        "input_size": (image.shape[1], image.shape[0]),
        "valid_contour_count": len(detected),
        "min_contour_area": float(min_contour_area),
    }
    return detected, metadata, perf_counter() - started_at
