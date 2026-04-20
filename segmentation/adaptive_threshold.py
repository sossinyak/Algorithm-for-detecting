"""
Thresholding helpers for binary change masks.
"""

import cv2
import numpy as np


def local_adaptive_threshold(
    image: np.ndarray,
    block_size: int = 35,
    C: float = 5,
    method: str = "gaussian",
) -> np.ndarray:
    """
    Apply local adaptive thresholding to a grayscale score map.
    """
    img_uint8 = to_uint8(image)
    if is_degenerate_score_map(img_uint8):
        return np.zeros_like(img_uint8, dtype=np.uint8)

    block_size = ensure_odd(block_size)

    adaptive_method = (
        cv2.ADAPTIVE_THRESH_MEAN_C
        if method == "mean"
        else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    )

    return cv2.adaptiveThreshold(
        img_uint8,
        255,
        adaptive_method,
        cv2.THRESH_BINARY,
        block_size,
        C,
    )


def global_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """
    Apply a global Otsu threshold to a score map.
    """
    img_uint8 = to_uint8(image)
    if is_degenerate_score_map(img_uint8):
        return np.zeros_like(img_uint8, dtype=np.uint8)

    _, binary = cv2.threshold(
        img_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary


def ensure_odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def to_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.copy()
    img = image.astype(np.float32, copy=False)
    if float(np.max(img)) <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def is_degenerate_score_map(image: np.ndarray, min_dynamic_range: int = 1) -> bool:
    return int(np.max(image)) - int(np.min(image)) <= int(min_dynamic_range)


def kimura_adaptive_threshold(
    image: np.ndarray,
    window_size: int = 35,
    k: float = 0.15,
    C: float = 0.0,
) -> np.ndarray:
    """
    Local Kimura-style thresholding for bright change-score maps.

    A pixel is marked as changed if it exceeds local mean plus k times local
    standard deviation. This keeps the threshold adaptive to local texture.
    """
    img_uint8 = to_uint8(image)
    if is_degenerate_score_map(img_uint8):
        return np.zeros_like(img_uint8, dtype=np.uint8)

    window_size = ensure_odd(window_size)

    img = img_uint8.astype(np.float32)
    mean = cv2.boxFilter(img, cv2.CV_32F, (window_size, window_size), normalize=True)
    mean_sq = cv2.boxFilter(img * img, cv2.CV_32F, (window_size, window_size), normalize=True)
    std = np.sqrt(np.maximum(mean_sq - mean * mean, 0.0))

    threshold = mean + float(k) * std + float(C)
    return (img > threshold).astype(np.uint8) * 255
