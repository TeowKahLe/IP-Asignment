"""Pipeline B preprocessing: Wiener-filter image deblurring only.

The functions operate in memory and never overwrite or save the source image.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

import cv2
import numpy as np


DEFAULT_KERNEL_SIZE = 15
DEFAULT_ANGLE = 0.0
DEFAULT_BALANCE = 0.01


def _validate_kernel_size(kernel_size: int) -> int:
    """Return a valid odd kernel size for the motion-blur PSF."""
    if not isinstance(kernel_size, int) or kernel_size < 3 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd integer greater than or equal to 3")
    return kernel_size


def create_motion_psf(
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    angle: float = DEFAULT_ANGLE,
) -> np.ndarray:
    """Create a normalized motion-blur Point Spread Function (PSF)."""
    kernel_size = _validate_kernel_size(kernel_size)

    psf = np.zeros((kernel_size, kernel_size), dtype=np.float32)

    center = kernel_size // 2

    cv2.line(
        psf,
        (0, center),
        (kernel_size - 1, center),
        1,
        1,
    )

    rotation_matrix = cv2.getRotationMatrix2D(
        (center, center),
        angle,
        1.0,
    )

    psf = cv2.warpAffine(
        psf,
        rotation_matrix,
        (kernel_size, kernel_size),
    )

    psf /= np.sum(psf)

    return psf


def _wiener_channel(
    channel: np.ndarray,
    psf: np.ndarray,
    balance: float,
) -> np.ndarray:
    """Apply Wiener deconvolution to one image channel."""

    image_float = channel.astype(np.float32)

    psf_padded = np.zeros_like(image_float, dtype=np.float32)

    h, w = psf.shape
    psf_padded[:h, :w] = psf

    psf_padded = np.roll(psf_padded, -(h // 2), axis=0)
    psf_padded = np.roll(psf_padded, -(w // 2), axis=1)

    image_fft = np.fft.fft2(image_float)
    psf_fft = np.fft.fft2(psf_padded)

    wiener_filter = np.conj(psf_fft) / (
        np.abs(psf_fft) ** 2 + balance
    )

    restored_fft = wiener_filter * image_fft

    restored = np.real(np.fft.ifft2(restored_fft))

    return np.clip(restored, 0, 255).astype(np.uint8)


def wiener_filter(
    image: np.ndarray,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    angle: float = DEFAULT_ANGLE,
    balance: float = DEFAULT_BALANCE,
) -> np.ndarray:
    """Deblur an image using Wiener deconvolution."""

    if image is None or image.size == 0:
        raise ValueError("image must be a non-empty OpenCV image")

    if balance <= 0:
        raise ValueError("balance must be greater than 0")

    psf = create_motion_psf(kernel_size, angle)

    if image.ndim == 2:
        return _wiener_channel(image, psf, balance)

    channels = cv2.split(image)

    restored_channels = [
        _wiener_channel(channel, psf, balance)
        for channel in channels
    ]

    return cv2.merge(restored_channels)


def preprocess_image(
    image: np.ndarray,
    kernel_size: int = DEFAULT_KERNEL_SIZE,
    angle: float = DEFAULT_ANGLE,
    balance: float = DEFAULT_BALANCE,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Apply Pipeline A's Wiener-filter preprocessing and return metadata."""

    started_at = perf_counter()

    filtered = wiener_filter(
        image,
        kernel_size,
        angle,
        balance,
    )

    return filtered, {
        "technique": "Image deblurring",
        "algorithm": "Wiener filter",
        "kernel_size": kernel_size,
        "angle": angle,
        "balance": balance,
        "processing_time_seconds": perf_counter() - started_at,
    }