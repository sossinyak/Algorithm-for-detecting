"""
Connected-component filters for binary change masks.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np


def _as_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.uint8:
        return mask.copy()
    values = mask.astype(np.float32, copy=False)
    if float(np.max(values)) <= 1.0:
        values = values * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def get_object_areas(mask: np.ndarray) -> List[float]:
    """Return external connected-component areas in pixels."""
    src = _as_uint8_mask(mask)
    contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [float(cv2.contourArea(contour)) for contour in contours]


def auto_area_threshold(mask: np.ndarray, percentile: float = 85) -> int:
    """
    Estimate a minimum component area from the current mask.

    The heuristic is conservative: if there are only a few components, keep
    medium objects; otherwise keep objects above the requested percentile.
    """
    areas = get_object_areas(mask)
    if not areas:
        return 0
    if len(areas) <= 3:
        return max(50, int(np.median(areas) * 0.5))
    return max(50, int(np.percentile(areas, percentile)))


def filter_by_area(
    mask: np.ndarray,
    min_area: Optional[int] = None,
    percentile: float = 85,
    auto: bool = True,
) -> np.ndarray:
    """Remove connected components smaller than the selected area threshold."""
    src = _as_uint8_mask(mask)

    if auto and min_area is None:
        min_area = auto_area_threshold(src, percentile)
    if min_area is None or min_area <= 0:
        return src

    contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(src)
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)
    return filtered_mask


def filter_by_rectangularity(mask: np.ndarray, min_rectangularity: float = 0.7) -> np.ndarray:
    """Keep components whose contour area fills a rotated minimum-area rectangle."""
    src = _as_uint8_mask(mask)
    contours, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_mask = np.zeros_like(src)

    for contour in contours:
        rect = cv2.minAreaRect(contour)
        rect_width, rect_height = rect[1]
        rect_area = rect_width * rect_height if rect_width > 0 and rect_height > 0 else 0
        contour_area = cv2.contourArea(contour)
        rectangularity = contour_area / rect_area if rect_area > 0 else 0.0

        if rectangularity >= min_rectangularity:
            cv2.drawContours(filtered_mask, [contour], -1, 255, -1)

    return filtered_mask
