"""Initial image standardization using aspect-ratio-preserving letterboxing.

This module deliberately does not save images or modify the source dataset.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any

import cv2
import numpy as np


DEFAULT_TARGET_SIZE = (512, 512)  # (width, height)
PADDING_VALUE = (114, 114, 114)  # Neutral gray BGR letterbox padding.
MIN_VALID_SIZE = 64  # Flag smaller images for review; never delete them automatically.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def scan_dataset(dataset_root: str | Path) -> list[dict[str, Any]]:
    """Return supported images inside the expected fruit-group and split folders.

    Image files outside ``single_fruit``/``multiple_fruit`` and
    ``train``/``validation``/``test`` are ignored. This prevents dataset
    reports or charts stored at the dataset root from being treated as data.
    """
    root = Path(dataset_root)
    if not root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {root}")

    records: list[dict[str, Any]] = []
    for image_path in sorted(root.rglob("*")):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        relative_parts = [part.lower() for part in image_path.relative_to(root).parts]
        fruit_group = next((x for x in ("single_fruit", "multiple_fruit") if x in relative_parts), None)
        split = next((x for x in ("train", "validation", "test") if x in relative_parts), None)
        if fruit_group is None or split is None:
            continue
        records.append({"path": image_path, "fruit_group": fruit_group, "split": split})
    return records


def inspect_image_sizes(image_records: list[dict[str, Any]]) -> dict[str, Any]:
    """Read dimensions and return image-size statistics without changing files."""
    widths: list[int] = []
    heights: list[int] = []
    resolutions: Counter[tuple[int, int]] = Counter()
    unreadable_count = 0
    for record in image_records:
        image = cv2.imread(str(record["path"]), cv2.IMREAD_COLOR)
        if image is None:
            unreadable_count += 1
            continue
        height, width = image.shape[:2]
        widths.append(width)
        heights.append(height)
        resolutions[(width, height)] += 1

    if not widths:
        return {
            "minimum_width": None, "maximum_width": None, "average_width": None,
            "minimum_height": None, "maximum_height": None, "average_height": None,
            "most_common_resolutions": [], "unreadable_image_count": unreadable_count,
            "readable_image_count": 0,
        }
    return {
        "minimum_width": min(widths), "maximum_width": max(widths),
        "average_width": sum(widths) / len(widths),
        "minimum_height": min(heights), "maximum_height": max(heights),
        "average_height": sum(heights) / len(heights),
        "most_common_resolutions": resolutions.most_common(10),
        "unreadable_image_count": unreadable_count, "readable_image_count": len(widths),
    }


def print_dataset_summary(image_records: list[dict[str, Any]]) -> None:
    """Print counts by fruit group and train/validation/test split."""
    group_counts = Counter(record["fruit_group"] for record in image_records)
    split_counts = Counter(record["split"] for record in image_records)
    print(f"Total images: {len(image_records)}")
    print(f"Single-fruit images: {group_counts['single_fruit']}")
    print(f"Multiple-fruit images: {group_counts['multiple_fruit']}")
    print(f"Train images: {split_counts['train']}")
    print(f"Validation images: {split_counts['validation']}")
    print(f"Test images: {split_counts['test']}")
    print(f"Unclassified images: {len(find_unclassified_images(image_records))}")


def find_unclassified_images(image_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return images outside the expected fruit-group or dataset-split folders."""
    return [
        record
        for record in image_records
        if record["fruit_group"] not in ("single_fruit", "multiple_fruit")
        or record["split"] not in ("train", "validation", "test")
    ]


def find_small_images(
    image_records: list[dict[str, Any]], min_valid_size: int = MIN_VALID_SIZE
) -> list[dict[str, Any]]:
    """Return readable images smaller than *min_valid_size* in either dimension.

    This is a reporting check only. It does not remove or alter any image.
    """
    if min_valid_size <= 0:
        raise ValueError("min_valid_size must be positive")

    small_images: list[dict[str, Any]] = []
    for record in image_records:
        image = cv2.imread(str(record["path"]), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        if width < min_valid_size or height < min_valid_size:
            small_images.append({**record, "width": width, "height": height})
    return small_images


def _validate_target_size(target_size: tuple[int, int]) -> tuple[int, int]:
    if len(target_size) != 2 or target_size[0] <= 0 or target_size[1] <= 0:
        raise ValueError("target_size must be a positive (width, height) tuple")
    return int(target_size[0]), int(target_size[1])


def _letterbox_resize_with_metadata(
    image: np.ndarray, target_size: tuple[int, int]
) -> tuple[np.ndarray, dict[str, Any]]:
    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty OpenCV image")
    target_width, target_height = _validate_target_size(target_size)
    original_height, original_width = image.shape[:2]
    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, min(target_width, round(original_width * scale)))
    resized_height = max(1, min(target_height, round(original_height * scale)))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    interpolation_name = "cv2.INTER_AREA" if scale < 1 else "cv2.INTER_CUBIC"
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)

    horizontal_padding = target_width - resized_width
    vertical_padding = target_height - resized_height
    padding = {
        "top": vertical_padding // 2, "bottom": vertical_padding - vertical_padding // 2,
        "left": horizontal_padding // 2, "right": horizontal_padding - horizontal_padding // 2,
    }
    standardized = cv2.copyMakeBorder(
        resized, padding["top"], padding["bottom"], padding["left"], padding["right"],
        cv2.BORDER_CONSTANT, value=PADDING_VALUE,
    )
    return standardized, {
        "original_size": (original_width, original_height),
        "resized_size": (resized_width, resized_height),
        "target_size": (target_width, target_height), "scale_factor": scale,
        "padding": padding, "interpolation_method": interpolation_name,
    }


def letterbox_resize(image: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    """Resize proportionally and pad to target size without cropping or stretching."""
    standardized, _ = _letterbox_resize_with_metadata(image, target_size)
    return standardized


def standardize_image(
    image_path: str | Path | dict[str, Any], target_size: tuple[int, int] = DEFAULT_TARGET_SIZE
) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
    """Load and standardize one image or dataset record without writing it to disk.

    Unreadable images return ``(None, None, metadata)`` so callers can continue.
    """
    started_at = perf_counter()
    path = Path(image_path["path"]) if isinstance(image_path, dict) else Path(image_path)
    original = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if original is None:
        return None, None, {
            "path": str(path),
            "error": "Image could not be read (missing, corrupted, or unsupported).",
            "processing_time_seconds": perf_counter() - started_at,
        }
    standardized, metadata = _letterbox_resize_with_metadata(original, target_size)
    metadata["path"] = str(path)
    metadata["processing_time_seconds"] = perf_counter() - started_at
    return original, standardized, metadata


def get_sample_images(
    image_records: list[dict[str, Any]], single_count: int = 3,
    multiple_count: int = 3, seed: int = 42,
) -> list[dict[str, Any]]:
    """Return reproducible, randomly selected single- and multiple-fruit records."""
    if single_count < 0 or multiple_count < 0:
        raise ValueError("sample counts must be non-negative")
    generator = np.random.default_rng(seed)
    selected: list[dict[str, Any]] = []
    for group, count in (("single_fruit", single_count), ("multiple_fruit", multiple_count)):
        candidates = [record for record in image_records if record["fruit_group"] == group]
        sample_size = min(count, len(candidates))
        if sample_size:
            indices = generator.choice(len(candidates), size=sample_size, replace=False)
            selected.extend(candidates[index] for index in indices)
    return selected
