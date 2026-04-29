"""Фильтрация бинарной маски по площади связных компонент."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def _as_uint8_mask(mask: np.ndarray) -> np.ndarray:
    """Приводит маску к формату uint8 со значениями 0 и 255."""
    if mask.dtype == np.uint8:
        return mask.copy()
    values = mask.astype(np.float32, copy=False)
    if float(np.max(values)) <= 1.0:
        values = values * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def filter_by_area(mask: np.ndarray, min_area: Optional[int] = None) -> np.ndarray:
    """Удаляет предсказанные компоненты меньше min_area пикселей."""
    src = _as_uint8_mask(mask)
    if min_area is None or min_area <= 0:
        return src

    filtered_mask = np.zeros_like(src)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((src > 127).astype(np.uint8), connectivity=8)
    for label_id in range(1, num_labels):
        if int(stats[label_id, cv2.CC_STAT_AREA]) >= int(min_area):
            filtered_mask[labels == label_id] = 255
    return filtered_mask
