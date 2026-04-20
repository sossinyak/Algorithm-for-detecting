"""
Difference maps for classical bitemporal change detection.
"""

from __future__ import annotations

import cv2
import numpy as np


def normalize01(values: np.ndarray) -> np.ndarray:
    """Return a float32 map scaled to [0, 1] with degenerate maps set to zero."""
    values = values.astype(np.float32, copy=False)
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if max_val - min_val < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_val) / (max_val - min_val)


def to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR input to grayscale when needed."""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def absolute_difference(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Absolute grayscale difference."""
    return np.abs(img2.astype(np.float32) - img1.astype(np.float32))


def log_ratio(img1: np.ndarray, img2: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Absolute log-ratio map with a small offset for dark pixels."""
    i1 = img1.astype(np.float32) + eps
    i2 = img2.astype(np.float32) + eps
    return np.abs(np.log(i2 / i1))


def color_vector_magnitude(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Change vector magnitude for color images.

    This keeps color-only material/roof changes that can be weakened by a
    pure grayscale conversion. For grayscale inputs it falls back to absolute
    difference.
    """
    if len(img1.shape) != 3 or len(img2.shape) != 3:
        return absolute_difference(to_gray(img1), to_gray(img2))

    diff = img2.astype(np.float32) - img1.astype(np.float32)
    return np.sqrt(np.sum(diff * diff, axis=2))


def _ensure_odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _local_mean_std(image: np.ndarray, window_size: int) -> tuple[np.ndarray, np.ndarray]:
    image = image.astype(np.float32, copy=False)
    window_size = _ensure_odd(window_size)
    mean = cv2.boxFilter(image, cv2.CV_32F, (window_size, window_size), normalize=True)
    mean_sq = cv2.boxFilter(image * image, cv2.CV_32F, (window_size, window_size), normalize=True)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))
    return mean, std


def _madi_channel(channel1: np.ndarray, channel2: np.ndarray, window_size: int, eps: float) -> np.ndarray:
    mean1, std1 = _local_mean_std(channel1, window_size)
    mean2, std2 = _local_mean_std(channel2, window_size)
    z1 = (channel1.astype(np.float32) - mean1) / (std1 + eps)
    z2 = (channel2.astype(np.float32) - mean2) / (std2 + eps)
    return np.abs(z2 - z1)


def modified_adaptive_difference_index(
    img1: np.ndarray,
    img2: np.ndarray,
    window_size: int = 31,
    color_weight: float = 0.35,
    eps: float = 1e-3,
) -> np.ndarray:
    """
    Modified Adaptive Difference Index (MADI).

    MADI compares locally standardized images: each pixel is measured relative
    to its local mean and standard deviation. This suppresses slow illumination
    shifts and highlights changes that are unusual for the local texture.
    """
    gray_madi = _madi_channel(to_gray(img1), to_gray(img2), window_size, eps)
    score = normalize01(gray_madi)

    color_weight = float(np.clip(color_weight, 0.0, 1.0))
    if color_weight > 0.0 and len(img1.shape) == 3 and len(img2.shape) == 3:
        channel_maps = [
            _madi_channel(img1[:, :, channel], img2[:, :, channel], window_size, eps)
            for channel in range(img1.shape[2])
        ]
        color_madi = np.sqrt(np.sum(np.stack(channel_maps, axis=2) ** 2, axis=2))
        score = normalize01((1.0 - color_weight) * score + color_weight * normalize01(color_madi))

    return score.astype(np.float32)


def weighted_combination(
    diff_map: np.ndarray,
    log_map: np.ndarray,
    weight: float = 0.6,
) -> np.ndarray:
    """Fuse normalized absolute-difference and log-ratio maps."""
    weight = float(np.clip(weight, 0.0, 1.0))
    diff_norm = normalize01(diff_map)
    log_norm = normalize01(log_map)
    return normalize01(weight * diff_norm + (1.0 - weight) * log_norm)


def compute_combined_difference(
    img1: np.ndarray,
    img2: np.ndarray,
    weight: float = 0.6,
    color_weight: float = 0.0,
) -> np.ndarray:
    """
    Build a normalized difference score map.

    The base score is a weighted fusion of grayscale absolute difference and
    log-ratio. Optionally, a color-vector magnitude term can be mixed in as a
    classical CVA prior.
    """
    gray1 = to_gray(img1)
    gray2 = to_gray(img2)

    diff_map = absolute_difference(gray1, gray2)
    log_map = log_ratio(gray1, gray2)
    combined = weighted_combination(diff_map, log_map, weight)

    color_weight = float(np.clip(color_weight, 0.0, 1.0))
    if color_weight > 0.0:
        color_map = normalize01(color_vector_magnitude(img1, img2))
        combined = normalize01((1.0 - color_weight) * combined + color_weight * color_map)

    return combined.astype(np.float32)


def compute_difference_map(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = "combined",
    weight: float = 0.6,
    color_weight: float = 0.0,
    madi_window_size: int = 31,
    madi_color_weight: float = 0.35,
) -> np.ndarray:
    """
    Dispatch a named classical difference map.

    method:
    - "combined": absolute difference + log-ratio + optional CVA term.
    - "madi": Modified Adaptive Difference Index.
    - "cva": pure color-vector magnitude.
    """
    method = str(method).lower()
    if method == "madi":
        return modified_adaptive_difference_index(
            img1,
            img2,
            window_size=madi_window_size,
            color_weight=madi_color_weight,
        )
    if method == "cva":
        return normalize01(color_vector_magnitude(img1, img2)).astype(np.float32)
    return compute_combined_difference(
        img1,
        img2,
        weight=weight,
        color_weight=color_weight,
    )
