"""
Baseline methods used in the comparison experiment.
"""

from __future__ import annotations

import cv2
import numpy as np


class BaselineDiffOtsu:
    """Absolute grayscale difference followed by Otsu thresholding."""

    def process(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = img1, img2

        diff = np.abs(gray2.astype(np.float32) - gray1.astype(np.float32))
        diff_uint8 = (diff / (diff.max() + 1e-6) * 255).astype(np.uint8)
        _, mask = cv2.threshold(
            diff_uint8,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return mask


class BaselineRatioOtsu:
    """Log-ratio change map followed by Otsu thresholding."""

    def process(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = img1, img2

        eps = 1e-6
        ratio = np.abs(
            np.log((gray2.astype(np.float32) + eps) / (gray1.astype(np.float32) + eps))
        )
        ratio_uint8 = (ratio / (ratio.max() + 1e-6) * 255).astype(np.uint8)
        _, mask = cv2.threshold(
            ratio_uint8,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return mask


class BaselineCVA:
    """Change vector analysis for RGB images."""

    def process(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        if len(img1.shape) != 3 or len(img2.shape) != 3:
            diff = np.abs(img2.astype(np.float32) - img1.astype(np.float32))
            diff_uint8 = (diff / (diff.max() + 1e-6) * 255).astype(np.uint8)
            _, mask = cv2.threshold(
                diff_uint8,
                0,
                255,
                cv2.THRESH_BINARY + cv2.THRESH_OTSU,
            )
            return mask

        diff_r = img2[:, :, 0].astype(np.float32) - img1[:, :, 0].astype(np.float32)
        diff_g = img2[:, :, 1].astype(np.float32) - img1[:, :, 1].astype(np.float32)
        diff_b = img2[:, :, 2].astype(np.float32) - img1[:, :, 2].astype(np.float32)

        magnitude = np.sqrt(diff_r**2 + diff_g**2 + diff_b**2)
        magnitude_uint8 = (magnitude / (magnitude.max() + 1e-6) * 255).astype(np.uint8)
        _, mask = cv2.threshold(
            magnitude_uint8,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        return mask


class BaselineCascade:
    """CLAHE + Gaussian blur + Canny edge difference + Otsu threshold."""

    def process(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        if len(img1.shape) == 3:
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        else:
            gray1, gray2 = img1, img2

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        gray1 = clahe.apply(gray1)
        gray2 = clahe.apply(gray2)

        gray1 = cv2.GaussianBlur(gray1, (5, 5), 1.5)
        gray2 = cv2.GaussianBlur(gray2, (5, 5), 1.5)

        edges1 = cv2.Canny(gray1, 50, 150)
        edges2 = cv2.Canny(gray2, 50, 150)

        change_map = cv2.absdiff(edges1, edges2)
        change_map = cv2.GaussianBlur(change_map, (3, 3), 0)
        _, mask = cv2.threshold(
            change_map,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )

        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        return mask
