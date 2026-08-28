"""No-ground-truth proxy metrics for image segmentation evaluation."""

from __future__ import annotations

import cv2
import numpy as np


def _mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract a one-pixel boundary from a binary 0/255 segmentation mask."""
    kernel = np.ones((3, 3), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel) > 0


def evaluate_segmentation(
    image: np.ndarray, mask: np.ndarray, processing_time_seconds: float
) -> dict[str, float]:
    """Evaluate a segmentation without pixel-level ground-truth labels.

    Scores are proxies only: boundary/edge overlap, foreground intensity
    consistency, and foreground-background grayscale contrast.  They must not
    be interpreted as IoU, Dice, pixel accuracy, or ground-truth accuracy.
    """
    if image is None or image.size == 0:
        raise ValueError("image must be non-empty")
    if mask is None or mask.size == 0 or mask.shape[:2] != image.shape[:2]:
        raise ValueError("mask must be non-empty and match image dimensions")
    unique_values = set(np.unique(mask).tolist())
    if not unique_values.issubset({0, 255}):
        raise ValueError("mask must contain only 0 (background) and 255 (foreground)")

    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground = mask == 255
    background = ~foreground
    if not np.any(foreground) or not np.any(background):
        raise ValueError("mask must contain both foreground and background pixels")

    boundary = _mask_boundary(mask)
    edge_map = cv2.Canny(grayscale, 100, 200) > 0
    # A small dilation credits an edge immediately adjacent to the boundary.
    nearby_edges = cv2.dilate(edge_map.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
    boundary_edge_alignment = float(np.mean(nearby_edges[boundary])) if np.any(boundary) else 0.0

    foreground_values = grayscale[foreground].astype(np.float32)
    region_uniformity = float(np.clip(1.0 - foreground_values.std() / 127.5, 0.0, 1.0))
    foreground_background_contrast = float(
        abs(foreground_values.mean() - grayscale[background].astype(np.float32).mean()) / 255.0
    )

    return {
        "boundary_edge_alignment": boundary_edge_alignment,
        "region_uniformity": region_uniformity,
        "foreground_background_contrast": foreground_background_contrast,
        "processing_time_ms": float(processing_time_seconds * 1000),
    }
