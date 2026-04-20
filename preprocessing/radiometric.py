"""
Radiometric normalization for bitemporal optical images.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np


def _ensure_uint8(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.copy()
    return np.clip(image, 0, 255).astype(np.uint8)


def _match_histogram_channel(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    src = source.ravel()
    ref = reference.ravel()

    src_values, src_indices, src_counts = np.unique(src, return_inverse=True, return_counts=True)
    ref_values, ref_counts = np.unique(ref, return_counts=True)

    src_quantiles = np.cumsum(src_counts).astype(np.float64)
    src_quantiles /= src_quantiles[-1]
    ref_quantiles = np.cumsum(ref_counts).astype(np.float64)
    ref_quantiles /= ref_quantiles[-1]

    mapped = np.interp(src_quantiles, ref_quantiles, ref_values)
    return mapped[src_indices].reshape(source.shape).astype(np.uint8)


def histogram_match(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Match the source image histogram to the reference image.
    """
    source = _ensure_uint8(source)
    reference = _ensure_uint8(reference)

    if len(source.shape) == 2:
        return _match_histogram_channel(source, reference)

    channels = [
        _match_histogram_channel(source[:, :, channel], reference[:, :, channel])
        for channel in range(source.shape[2])
    ]
    return cv2.merge(channels)


def _quantile_match_channel(
    source: np.ndarray,
    reference: np.ndarray,
    num_quantiles: int = 64,
) -> np.ndarray:
    quantiles = np.linspace(0.0, 100.0, max(8, int(num_quantiles)))
    src_points = np.percentile(source, quantiles)
    ref_points = np.percentile(reference, quantiles)

    src_points, unique_indices = np.unique(src_points, return_index=True)
    ref_points = ref_points[unique_indices]
    if src_points.size < 2:
        return source.copy()

    matched = np.interp(source.astype(np.float32), src_points, ref_points)
    return np.clip(matched, 0, 255).astype(np.uint8)


def quantile_normalize(
    source: np.ndarray,
    reference: np.ndarray,
    num_quantiles: int = 64,
) -> np.ndarray:
    """
    Robust quantile normalization of source to reference.
    """
    source = _ensure_uint8(source)
    reference = _ensure_uint8(reference)

    if len(source.shape) == 2:
        return _quantile_match_channel(source, reference, num_quantiles)

    channels = [
        _quantile_match_channel(source[:, :, channel], reference[:, :, channel], num_quantiles)
        for channel in range(source.shape[2])
    ]
    return cv2.merge(channels)


def normalize_pair(
    img1: np.ndarray,
    img2: np.ndarray,
    method: str = "quantile",
    channels: str = "bgr",
    num_quantiles: int = 64,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Normalize img2 radiometry to img1 before difference-map construction.
    """
    method = method.lower()
    channels = channels.lower()

    if method in {"none", "off", "disabled"}:
        return img1.copy(), img2.copy()

    if channels in {"gray", "luminance", "y"}:
        if len(img1.shape) == 3 and len(img2.shape) == 3:
            lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
            lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
            ref_light = lab1[:, :, 0]
            src_light = lab2[:, :, 0]
        else:
            ref_light = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1
            src_light = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2

        if method == "histogram":
            matched_light = histogram_match(src_light, ref_light)
        else:
            matched_light = quantile_normalize(src_light, ref_light, num_quantiles=num_quantiles)

        if len(img2.shape) == 3:
            lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)
            lab2[:, :, 0] = matched_light
            img2_norm = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)
        else:
            img2_norm = matched_light
        return img1.copy(), img2_norm

    if method == "histogram":
        return img1.copy(), histogram_match(img2, img1)
    return img1.copy(), quantile_normalize(img2, img1, num_quantiles=num_quantiles)
