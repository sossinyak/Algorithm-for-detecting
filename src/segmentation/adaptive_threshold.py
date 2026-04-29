"""Пороговая бинаризация score-map."""

import cv2
import numpy as np


def to_uint8(image: np.ndarray) -> np.ndarray:
    """Переводит карту float [0,1] или uint8 [0,255] в uint8."""
    if image.dtype == np.uint8:
        return image.copy()
    img = image.astype(np.float32, copy=False)
    if float(np.max(img)) <= 1.0:
        img = img * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)


def global_otsu_threshold(image: np.ndarray) -> np.ndarray:
    """Бинаризация методом Оцу без дополнительного масштабирования порога."""
    img_uint8 = to_uint8(image)
    _, binary = cv2.threshold(
        img_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    return binary


def global_otsu_threshold_scaled(image: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """
    Бинаризация методом Оцу с коэффициентом к найденному порогу.

    scale < 1 делает маску шире и повышает Recall, scale > 1 делает маску
    строже и обычно повышает Precision.
    """
    img_uint8 = to_uint8(image)
    threshold, _ = cv2.threshold(
        img_uint8,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )
    scaled_threshold = float(np.clip(threshold * float(scale), 0.0, 255.0))
    return (img_uint8 > scaled_threshold).astype(np.uint8) * 255
