"""
Building-support filtering for handcrafted change-detection masks.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np

from change_detection.handcrafted_priors import edge_density_map, shadow_support_map


def _as_uint8_mask(mask: np.ndarray) -> np.ndarray:
    if mask.dtype == np.uint8:
        return mask.copy()
    return (mask > 0).astype(np.uint8) * 255


def _component_mask(shape: Tuple[int, int], contour: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, 255, -1)
    return mask


def _safe_mean(values: np.ndarray, selector: np.ndarray) -> float:
    selected = values[selector > 0]
    if selected.size == 0:
        return 0.0
    return float(np.mean(selected))


def _component_metrics(contour: np.ndarray) -> Dict[str, float]:
    area = float(cv2.contourArea(contour))
    rect = cv2.minAreaRect(contour)
    width, height = rect[1]
    rect_area = float(width * height) if width > 0 and height > 0 else 0.0
    rectangularity = area / rect_area if rect_area > 0 else 0.0

    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = float(w * h) if w > 0 and h > 0 else 0.0
    extent = area / bbox_area if bbox_area > 0 else 0.0
    aspect_ratio = max(w / max(h, 1), h / max(w, 1))

    return {
        "area": area,
        "rectangularity": rectangularity,
        "extent": extent,
        "aspect_ratio": float(aspect_ratio),
    }


def filter_by_building_support(
    mask: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
    stable_vegetation: np.ndarray | None = None,
    edge_density: np.ndarray | None = None,
    shadow_support: np.ndarray | None = None,
    min_rectangularity: float = 0.45,
    min_extent: float = 0.30,
    min_edge_density: float = 0.03,
    min_shadow_support: float = 0.01,
    max_vegetation_overlap: float = 0.30,
    min_support_score: float = 0.45,
    max_aspect_ratio: float = 8.0,
    shadow_ring_kernel: int = 9,
) -> np.ndarray:
    """
    Keep only components that look like constructed objects.

    The decision combines shape regularity, local edge density, nearby shadows,
    and low overlap with stable vegetation.
    """
    src_mask = _as_uint8_mask(mask)
    shape = src_mask.shape[:2]
    if stable_vegetation is None:
        stable_vegetation = np.zeros(shape, dtype=np.uint8)
    else:
        stable_vegetation = _as_uint8_mask(stable_vegetation)

    if edge_density is None:
        edge_density = edge_density_map(img1, img2)
    if shadow_support is None:
        shadow_support = shadow_support_map(img1, img2)

    edge_density = edge_density.astype(np.float32, copy=False)
    shadow_support = shadow_support.astype(np.float32, copy=False)

    contours, _ = cv2.findContours(src_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered = np.zeros_like(src_mask)
    ring_kernel_size = max(3, int(shadow_ring_kernel))
    if ring_kernel_size % 2 == 0:
        ring_kernel_size += 1
    ring_kernel = np.ones((ring_kernel_size, ring_kernel_size), np.uint8)

    for contour in contours:
        metrics = _component_metrics(contour)
        if metrics["area"] <= 0:
            continue
        if metrics["aspect_ratio"] > max_aspect_ratio:
            continue

        component = _component_mask(shape, contour)
        ring = cv2.dilate(component, ring_kernel, iterations=1)
        ring = cv2.subtract(ring, component)

        edge_mean = _safe_mean(edge_density, component)
        shadow_mean = max(
            _safe_mean(shadow_support, ring),
            _safe_mean(shadow_support, component) * 0.5,
        )
        vegetation_overlap = float(np.mean(stable_vegetation[component > 0] > 0))

        if vegetation_overlap > max_vegetation_overlap:
            continue

        rect_score = np.clip(metrics["rectangularity"] / max(min_rectangularity, 1e-6), 0.0, 1.0)
        extent_score = np.clip(metrics["extent"] / max(min_extent, 1e-6), 0.0, 1.0)
        edge_score = np.clip(edge_mean / max(min_edge_density, 1e-6), 0.0, 1.0)
        shadow_score = np.clip(shadow_mean / max(min_shadow_support, 1e-6), 0.0, 1.0)
        veg_score = np.clip(1.0 - vegetation_overlap / max(max_vegetation_overlap, 1e-6), 0.0, 1.0)

        shape_score = 0.65 * rect_score + 0.35 * extent_score
        support_score = (0.45 * shape_score + 0.35 * edge_score + 0.20 * shadow_score) * veg_score

        has_shape_support = (
            metrics["rectangularity"] >= min_rectangularity
            or metrics["extent"] >= min_extent
        )
        has_context_support = edge_mean >= min_edge_density or shadow_mean >= min_shadow_support

        if support_score >= min_support_score and has_shape_support and has_context_support:
            cv2.drawContours(filtered, [contour], -1, 255, -1)

    return filtered
