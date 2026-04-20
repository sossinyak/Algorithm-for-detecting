"""
Contour-based change maps for classical image processing pipelines.
"""

import cv2
import numpy as np


def _to_gray(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def auto_canny(
    image: np.ndarray,
    sigma: float = 0.33,
    percentile: float = 90.0,
    mode: str = "gradient",
) -> np.ndarray:
    """
    Canny edge detector with automatic thresholds.

    The gradient-percentile mode is more selective for textured satellite
    scenes than the classic intensity-median heuristic.
    """
    gray = cv2.GaussianBlur(_to_gray(image), (3, 3), 0)

    if mode == "median":
        median = np.median(gray)
        lower = int(max(0, (1.0 - sigma) * median))
        upper = int(min(255, (1.0 + sigma) * median))
        return cv2.Canny(gray, lower, upper)

    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(grad_x, grad_y)
    upper = float(np.percentile(magnitude, np.clip(percentile, 50.0, 99.5)))
    upper = max(1.0, upper)
    lower = max(0.0, (1.0 - float(np.clip(sigma, 0.05, 0.95))) * upper)
    return cv2.Canny(gray, lower, upper)


def sobel_edges(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Sobel magnitude map normalized to uint8."""
    gray = _to_gray(image)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(grad_x, grad_y)
    return cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def _edge_change(
    edges1: np.ndarray,
    edges2: np.ndarray,
    threshold: bool = False,
    dilate_iterations: int = 1,
) -> np.ndarray:
    if threshold:
        _, edges1 = cv2.threshold(edges1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, edges2 = cv2.threshold(edges2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    change_edges = cv2.bitwise_xor(edges1, edges2)
    dilate_iterations = max(0, int(dilate_iterations))
    if dilate_iterations == 0:
        return change_edges
    kernel = np.ones((3, 3), np.uint8)
    return cv2.dilate(change_edges, kernel, iterations=dilate_iterations)


def canny_change_detection(
    img1: np.ndarray,
    img2: np.ndarray,
    sigma: float = 0.33,
    percentile: float = 90.0,
    dilate_iterations: int = 1,
) -> np.ndarray:
    edges1 = auto_canny(img1, sigma=sigma, percentile=percentile)
    edges2 = auto_canny(img2, sigma=sigma, percentile=percentile)
    return _edge_change(edges1, edges2, dilate_iterations=dilate_iterations)


def sobel_change_detection(
    img1: np.ndarray,
    img2: np.ndarray,
    ksize: int = 3,
    dilate_iterations: int = 1,
) -> np.ndarray:
    edges1 = sobel_edges(img1, ksize=ksize)
    edges2 = sobel_edges(img2, ksize=ksize)
    return _edge_change(edges1, edges2, threshold=True, dilate_iterations=dilate_iterations)


def edge_based_change_detection(
    img1: np.ndarray,
    img2: np.ndarray,
    sigma: float = 0.33,
    detector: str = "canny",
    canny_percentile: float = 90.0,
    dilate_iterations: int = 1,
) -> np.ndarray:
    """
    Build a contour-change map with Canny, Sobel, or their union.

    detector: "canny", "sobel", or "both".
    """
    detector = detector.lower()
    if detector == "sobel":
        return sobel_change_detection(img1, img2, dilate_iterations=dilate_iterations)
    if detector == "both":
        canny_map = canny_change_detection(
            img1,
            img2,
            sigma=sigma,
            percentile=canny_percentile,
            dilate_iterations=dilate_iterations,
        )
        sobel_map = sobel_change_detection(
            img1,
            img2,
            dilate_iterations=dilate_iterations,
        )
        return cv2.bitwise_or(canny_map, sobel_map)
    return canny_change_detection(
        img1,
        img2,
        sigma=sigma,
        percentile=canny_percentile,
        dilate_iterations=dilate_iterations,
    )
