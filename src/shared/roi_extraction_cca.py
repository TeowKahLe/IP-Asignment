"""Connected Component Analysis-based extraction of individual fruit regions of interest (ROIs)."""

from __future__ import annotations

from time import perf_counter
from typing import Any

import cv2
import numpy as np


def extract_fruit_rois(
    image: np.ndarray,
    segmentation_mask: np.ndarray,
    min_component_area: float = 0.0,
) -> tuple[list[dict[str, Any]], dict[str, Any], float]:
    """Extract one masked ROI per valid connected component.

    The image and mask must have the same height and width. Returned ROIs are
    ordered top-to-bottom, then left-to-right for a stable multiple-fruit order.
    """

    started_at = perf_counter()

    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty OpenCV image")

    if segmentation_mask is None or segmentation_mask.size == 0:
        raise ValueError("segmentation_mask must be a non-empty image")

    if image.shape[:2] != segmentation_mask.shape[:2]:
        raise ValueError(
            "image and segmentation_mask must have the same height and width"
        )

    if min_component_area < 0:
        raise ValueError("min_component_area must be non-negative")

    mask = segmentation_mask

    if mask.ndim == 3:
        mask = cv2.cvtColor(
            mask,
            cv2.COLOR_BGR2GRAY
        )

    binary_mask = np.where(
        mask > 0,
        255,
        0
    ).astype(np.uint8)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary_mask,
        connectivity=8
    )

    detected: list[dict[str, Any]] = []

    for label in range(1, num_labels):

        area = float(
            stats[label, cv2.CC_STAT_AREA]
        )

        if area < min_component_area:
            continue

        x = int(
            stats[label, cv2.CC_STAT_LEFT]
        )

        y = int(
            stats[label, cv2.CC_STAT_TOP]
        )

        width = int(
            stats[label, cv2.CC_STAT_WIDTH]
        )

        height = int(
            stats[label, cv2.CC_STAT_HEIGHT]
        )

        component_mask = np.where(
            labels == label,
            255,
            0
        ).astype(np.uint8)

        local_mask = component_mask[
            y:y + height,
            x:x + width
        ]

        crop = image[
            y:y + height,
            x:x + width
        ].copy()

        masked_roi = cv2.bitwise_and(
            crop,
            crop,
            mask=local_mask
        )

        centroid = (
            float(centroids[label][0]),
            float(centroids[label][1])
        )

        detected.append({
            "roi": masked_roi,
            "roi_mask": local_mask,
            "component": label,
            "bounding_box": (
                x,
                y,
                width,
                height
            ),
            "area": area,
            "centroid": centroid,
        })

    detected.sort(
        key=lambda item: (
            item["bounding_box"][1],
            item["bounding_box"][0]
        )
    )

    for index, item in enumerate(detected, start=1):
        item["roi_id"] = index

    metadata = {
        "algorithm": "Connected Component Analysis (CCA)",
        "connectivity": 8,
        "input_size": (
            image.shape[1],
            image.shape[0]
        ),
        "valid_component_count": len(detected),
        "min_component_area": float(min_component_area),
    }

    return detected, metadata, perf_counter() - started_at