"""
Gaussian-pyramid multiscale change maps.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

from change_detection.difference_methods import compute_difference_map


def _normalize01(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if max_val - min_val < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_val) / (max_val - min_val)


def _scaled_odd_window(base_window: int, level: int) -> int:
    window = max(3, int(round(int(base_window) / (2 ** level))))
    return window if window % 2 == 1 else window + 1


def compute_multiscale_difference(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = "combined",
    weight: float = 0.7,
    color_weight: float = 0.0,
    madi_window_size: int = 31,
    madi_color_weight: float = 0.35,
    levels: int = 3,
    scale_weight_decay: float = 0.65,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Compute a difference map on a Gaussian pyramid and fuse all levels.
    """
    levels = max(1, int(levels))
    scale_weight_decay = float(np.clip(scale_weight_decay, 0.1, 1.0))
    base_shape = img1.shape[:2]

    current1 = img1.copy()
    current2 = img2.copy()
    maps: List[np.ndarray] = []
    weights: List[float] = []

    for level in range(levels):
        level_madi_window = _scaled_odd_window(madi_window_size, level)
        diff = compute_difference_map(
            current1,
            current2,
            method=method,
            weight=weight,
            color_weight=color_weight,
            madi_window_size=level_madi_window,
            madi_color_weight=madi_color_weight,
        )
        diff = _normalize01(diff)
        if diff.shape[:2] != base_shape:
            diff = cv2.resize(diff, (base_shape[1], base_shape[0]), interpolation=cv2.INTER_LINEAR)
        maps.append(diff.astype(np.float32))
        weights.append(scale_weight_decay ** level)

        if level < levels - 1 and min(current1.shape[:2]) >= 32:
            current1 = cv2.pyrDown(current1)
            current2 = cv2.pyrDown(current2)
        else:
            break

    weight_arr = np.asarray(weights, dtype=np.float32)
    weight_arr /= np.sum(weight_arr)
    fused = np.zeros(base_shape, dtype=np.float32)
    for scale_weight, diff in zip(weight_arr, maps):
        fused += float(scale_weight) * diff

    return _normalize01(fused), maps
