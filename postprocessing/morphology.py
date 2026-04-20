"""
Morphological postprocessing for binary change masks.
"""

from __future__ import annotations

import cv2
import numpy as np

from postprocessing.area_filter import filter_by_area, filter_by_rectangularity


def _kernel_size(value: int) -> int:
    return max(1, int(value))


def morphological_processing(
    mask: np.ndarray,
    closing_kernel_size: int = 3,
    opening_kernel_size: int = 2,
    min_area: int | None = None,
    auto_area: bool = True,
    area_percentile: float = 85,
    fill_holes: bool = True,
    min_rectangularity: float | None = None,
) -> np.ndarray:
    if mask.dtype != np.uint8:
        mask = np.clip(mask * 255 if np.max(mask) <= 1 else mask, 0, 255).astype(np.uint8)

    closing_kernel_size = _kernel_size(closing_kernel_size)
    opening_kernel_size = _kernel_size(opening_kernel_size)
    closing_kernel = np.ones((closing_kernel_size, closing_kernel_size), np.uint8)
    opening_kernel = np.ones((opening_kernel_size, opening_kernel_size), np.uint8)

    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, closing_kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, opening_kernel)

    result = filter_by_area(opened, min_area=min_area, percentile=area_percentile, auto=auto_area)

    if min_rectangularity is not None:
        result = filter_by_rectangularity(result, min_rectangularity=min_rectangularity)

    if fill_holes:
        # Fill only enclosed holes, not the outer background.
        padded = cv2.copyMakeBorder(result, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        flooded = padded.copy()
        flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
        cv2.floodFill(flooded, flood_mask, (0, 0), 255)
        holes = cv2.bitwise_not(flooded)[1:-1, 1:-1]
        result = cv2.bitwise_or(result, holes)

    return result
