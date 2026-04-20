"""
CLAHE contrast enhancement.
"""

from __future__ import annotations

import cv2
import numpy as np


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: tuple[int, int] = (16, 16),
) -> np.ndarray:
    """
    Apply CLAHE to grayscale input or to the L channel of a BGR image.
    """
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))

    if len(image.shape) == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        lightness, a_channel, b_channel = cv2.split(lab)
        lightness_eq = clahe.apply(lightness)
        return cv2.cvtColor(cv2.merge([lightness_eq, a_channel, b_channel]), cv2.COLOR_LAB2BGR)

    return clahe.apply(image)


def adaptive_clahe(image: np.ndarray) -> tuple[np.ndarray, tuple[float, tuple[int, int]]]:
    """
    Select CLAHE parameters from global RMS contrast and apply CLAHE.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    mean = np.mean(gray)
    rms_contrast = np.sqrt(np.mean((gray.astype(np.float32) - mean) ** 2))

    if rms_contrast < 30:
        clip_limit = 2.5
        tile_size = (16, 16)
    elif rms_contrast < 50:
        clip_limit = 2.0
        tile_size = (16, 16)
    else:
        clip_limit = 1.5
        tile_size = (32, 32)

    return apply_clahe(image, clip_limit, tile_size), (clip_limit, tile_size)
