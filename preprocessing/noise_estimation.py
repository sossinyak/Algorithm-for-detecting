"""
Noise estimation and adaptive Gaussian filtering.
"""

from __future__ import annotations

import cv2
import numpy as np


def estimate_noise(image: np.ndarray) -> float:
    """
    Estimate image noise from the median absolute deviation of the Laplacian.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    gray = gray.astype(np.float32) / 255.0
    laplacian = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    median_abs_dev = np.median(np.abs(laplacian - np.median(laplacian)))
    sigma_noise = median_abs_dev / 0.6745
    return float(np.clip(sigma_noise, 0.01, 0.2))


def adaptive_gaussian_blur(image: np.ndarray, max_sigma: float = 3.0) -> tuple[np.ndarray, float]:
    """
    Apply Gaussian blur with sigma derived from the estimated noise level.
    """
    noise_level = estimate_noise(image)
    sigma = 0.8 + (noise_level - 0.03) * 10.0
    sigma = float(np.clip(sigma, 0.5, max_sigma))

    kernel_size = int(2 * np.ceil(3 * sigma) + 1)
    kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
    filtered = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    return filtered, sigma
