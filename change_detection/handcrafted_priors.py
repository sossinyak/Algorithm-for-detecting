"""
Handcrafted priors used to suppress vegetation and support building-like changes.
"""

from __future__ import annotations

from typing import Dict, Tuple

import cv2
import numpy as np


def _odd_kernel(size: int) -> int:
    size = max(1, int(size))
    return size if size % 2 == 1 else size + 1


def _to_gray_uint8(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.uint8, copy=False)


def _normalize01(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32, copy=False)
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if max_val - min_val < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_val) / (max_val - min_val)


def _bgr_float_channels(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img = image.astype(np.float32) / 255.0
    b, g, r = cv2.split(img)
    return b, g, r


def excess_green(image: np.ndarray) -> np.ndarray:
    """
    Compute Excess Green from OpenCV BGR input as ExG = 2G - R - B.
    """
    b, g, r = _bgr_float_channels(image)
    return 2.0 * g - r - b


def vegetation_mask_exg(
    image: np.ndarray,
    exg_threshold: float = 0.08,
    min_green: float = 0.18,
    morphology_kernel: int = 3,
) -> np.ndarray:
    """
    Segment green vegetation using the RGB Excess Green feature.
    """
    b, g, r = _bgr_float_channels(image)
    exg = 2.0 * g - r - b
    mask = (
        (exg > float(exg_threshold))
        & (g > float(min_green))
        & (g > r * 1.03)
        & (g > b * 1.03)
    ).astype(np.uint8) * 255

    kernel_size = _odd_kernel(morphology_kernel)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def stable_vegetation_mask(
    img1: np.ndarray,
    img2: np.ndarray,
    exg_threshold: float = 0.08,
    min_green: float = 0.18,
    morphology_kernel: int = 3,
) -> np.ndarray:
    """
    Return vegetation that is present in both dates, which is likely stable clutter.
    """
    veg1 = vegetation_mask_exg(img1, exg_threshold, min_green, morphology_kernel)
    veg2 = vegetation_mask_exg(img2, exg_threshold, min_green, morphology_kernel)
    return cv2.bitwise_and(veg1, veg2)


def edge_density_map(
    img1: np.ndarray,
    img2: np.ndarray,
    kernel_size: int = 9,
) -> np.ndarray:
    """
    Build a local edge-density map from the maximum Sobel response of both dates.
    """
    gray1 = _to_gray_uint8(img1)
    gray2 = _to_gray_uint8(img2)

    def sobel_mag(gray: np.ndarray) -> np.ndarray:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        return cv2.magnitude(gx, gy)

    edges = np.maximum(sobel_mag(gray1), sobel_mag(gray2))
    edges = _normalize01(edges)

    kernel_size = _odd_kernel(kernel_size)
    if kernel_size > 1:
        edges = cv2.GaussianBlur(edges, (kernel_size, kernel_size), 0)

    return _normalize01(edges)


def shadow_support_map(
    img1: np.ndarray,
    img2: np.ndarray,
    percentile: float = 25.0,
    dilation_size: int = 9,
) -> np.ndarray:
    """
    Estimate possible building-shadow support from dark pixels near structures.
    """
    gray1 = _to_gray_uint8(img1)
    gray2 = _to_gray_uint8(img2)
    gray = np.minimum(gray1, gray2)
    threshold = np.percentile(gray, float(percentile))
    shadows = (gray <= threshold).astype(np.uint8) * 255

    kernel_size = _odd_kernel(dilation_size)
    if kernel_size > 1:
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        shadows = cv2.morphologyEx(shadows, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        shadows = cv2.dilate(shadows, kernel, iterations=1)
        shadows = cv2.GaussianBlur(shadows, (kernel_size, kernel_size), 0)

    return shadows.astype(np.float32) / 255.0


def building_prior_map(
    img1: np.ndarray,
    img2: np.ndarray,
    stable_vegetation: np.ndarray | None = None,
    edge_kernel: int = 9,
    shadow_percentile: float = 25.0,
    shadow_dilation: int = 9,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Build a soft prior for constructed objects from vegetation, edges, shadows,
    and roof-like color/material cues.
    """
    if stable_vegetation is None:
        stable_vegetation = np.zeros(img1.shape[:2], dtype=np.uint8)

    non_vegetation = 1.0 - (stable_vegetation.astype(np.float32) / 255.0)
    edge_density = edge_density_map(img1, img2, kernel_size=edge_kernel)
    shadow_support = shadow_support_map(
        img1,
        img2,
        percentile=shadow_percentile,
        dilation_size=shadow_dilation,
    )

    exg_high = np.maximum(excess_green(img1), excess_green(img2))
    low_green = 1.0 - np.clip((exg_high + 0.05) / 0.45, 0.0, 1.0)

    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV) if len(img1.shape) == 3 else None
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV) if len(img2.shape) == 3 else None
    if hsv1 is not None and hsv2 is not None:
        sat = (hsv1[:, :, 1].astype(np.float32) + hsv2[:, :, 1].astype(np.float32)) / 510.0
        val = (hsv1[:, :, 2].astype(np.float32) + hsv2[:, :, 2].astype(np.float32)) / 510.0
        roof_tone = np.clip(0.55 * (1.0 - sat) + 0.45 * val, 0.0, 1.0)
    else:
        roof_tone = np.ones_like(non_vegetation, dtype=np.float32) * 0.5

    prior = (
        0.35 * non_vegetation
        + 0.20 * low_green
        + 0.25 * edge_density
        + 0.12 * shadow_support
        + 0.08 * roof_tone
    )
    prior = np.clip(prior * (0.35 + 0.65 * non_vegetation), 0.0, 1.0)

    maps = {
        "building_prior": prior.astype(np.float32),
        "edge_density": edge_density.astype(np.float32),
        "shadow_support": shadow_support.astype(np.float32),
        "low_green": low_green.astype(np.float32),
    }
    return maps["building_prior"], maps


def apply_handcrafted_priors(
    score_map: np.ndarray,
    img1: np.ndarray,
    img2: np.ndarray,
    use_vegetation_suppression: bool = False,
    exg_threshold: float = 0.08,
    exg_min_green: float = 0.18,
    vegetation_kernel: int = 3,
    vegetation_suppression_factor: float = 0.25,
    use_building_prior: bool = False,
    building_prior_strength: float = 0.45,
    building_edge_kernel: int = 9,
    shadow_percentile: float = 25.0,
    shadow_dilation: int = 9,
    need_building_maps: bool = True,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Apply stable-vegetation suppression and a soft building prior to a score map.
    """
    adjusted = score_map.astype(np.float32, copy=True)
    adjusted = np.clip(adjusted, 0.0, 1.0)

    stable_veg = stable_vegetation_mask(
        img1,
        img2,
        exg_threshold=exg_threshold,
        min_green=exg_min_green,
        morphology_kernel=vegetation_kernel,
    )

    maps: Dict[str, np.ndarray] = {}
    if need_building_maps:
        prior, maps = building_prior_map(
            img1,
            img2,
            stable_vegetation=stable_veg,
            edge_kernel=building_edge_kernel,
            shadow_percentile=shadow_percentile,
            shadow_dilation=shadow_dilation,
        )
    else:
        prior = np.ones_like(adjusted, dtype=np.float32)

    if use_building_prior:
        strength = float(np.clip(building_prior_strength, 0.0, 1.0))
        support = (1.0 - strength) + strength * prior
        adjusted *= support

    if use_vegetation_suppression:
        factor = float(np.clip(vegetation_suppression_factor, 0.0, 1.0))
        adjusted[stable_veg > 0] *= factor

    maps["stable_vegetation"] = stable_veg
    maps["prior_adjusted_map"] = adjusted.astype(np.float32)
    return np.clip(adjusted, 0.0, 1.0), maps
