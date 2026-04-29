"""
Итоговый адаптивный алгоритм обнаружения изменений: PCA + CVA.

Порядок обработки:
1. RGB-снимки переводятся из диапазона 0..255 в диапазон [0, 1].
2. Для каждого пикселя формируется вектор признаков: RGB или RGB-окрестность.
3. PCA обучается на объединенных признаках двух снимков одной пары.
4. Оба снимка проецируются в PCA-пространство.
5. CVA считает длину вектора различий между проекциями.
6. Score-map бинаризуется прямым порогом, подобранным на validation.
7. Итоговая маска очищается медианным и морфологическим фильтрами.
"""

from __future__ import annotations

from typing import Dict, Optional

import cv2
import numpy as np

from postprocessing.area_filter import filter_by_area
from segmentation.adaptive_threshold import global_otsu_threshold, global_otsu_threshold_scaled


def _odd_kernel(value: int, minimum: int = 1) -> int:
    """Приводит размер ядра к нечетному числу, как требуется в OpenCV."""
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def _normalize01(values: np.ndarray) -> np.ndarray:
    """Масштабирует карту значений в [0, 1]."""
    values = values.astype(np.float32, copy=False)
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if max_val - min_val < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - min_val) / (max_val - min_val)


def _to_float01(image: np.ndarray) -> np.ndarray:
    """Переводит uint8-изображение 0..255 в float32 0..1."""
    return image.astype(np.float32) / 255.0


def _zscore_channels(image: np.ndarray) -> np.ndarray:
    """Normalize each channel of one image with its own mean and std."""
    values = _to_float01(image)
    if values.ndim == 2:
        values = values[:, :, None]
    mean = np.mean(values, axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    std = np.std(values, axis=(0, 1), keepdims=True, dtype=np.float64).astype(np.float32)
    return (values - mean) / np.maximum(std, 1e-6)


def _extract_patch_features(image: np.ndarray, patch_size: int) -> np.ndarray:
    """
    Формирует признаки для каждого пикселя.

    При patch_size=1 признак равен RGB-вектору пикселя. При большем окне к признаку
    добавляются RGB-значения соседних пикселей.
    """
    patch_size = _odd_kernel(patch_size, minimum=1)
    if len(image.shape) == 2:
        image = image[:, :, None]

    height, width, channels = image.shape
    if patch_size == 1:
        return image.reshape(height * width, channels).astype(np.float32, copy=False)

    radius = patch_size // 2
    padded = cv2.copyMakeBorder(
        image,
        radius,
        radius,
        radius,
        radius,
        borderType=cv2.BORDER_REFLECT_101,
    )

    features = []
    for row_shift in range(patch_size):
        for col_shift in range(patch_size):
            patch = padded[row_shift : row_shift + height, col_shift : col_shift + width, :]
            features.append(patch.reshape(height * width, channels))

    return np.concatenate(features, axis=1).astype(np.float32, copy=False)


def _fit_pca(
    features1: np.ndarray,
    features2: np.ndarray,
    n_components: int = 3,
    variance_ratio: Optional[float] = None,
    whitening: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Обучает PCA на признаках двух снимков одной пары."""
    samples = np.vstack([features1, features2]).astype(np.float32, copy=False)
    mean = np.mean(samples, axis=0, dtype=np.float64).astype(np.float32)
    centered = samples - mean

    # Для небольшого числа признаков матрица ковариации быстрее полного SVD.
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov.astype(np.float64))
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    if variance_ratio is not None:
        total = float(np.sum(np.maximum(eigvals, 0.0)))
        if total > 0:
            cumulative = np.cumsum(np.maximum(eigvals, 0.0)) / total
            n_components = int(np.searchsorted(cumulative, float(variance_ratio)) + 1)

    n_components = max(1, min(int(n_components), eigvecs.shape[1]))
    components = eigvecs[:, :n_components].astype(np.float32)

    if whitening:
        scale = np.sqrt(np.maximum(eigvals[:n_components], 1e-8)).astype(np.float32)
        components = components / scale[None, :]

    return mean, components, eigvals[:n_components].astype(np.float32)


def _project(features: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    """Проецирует признаки в пространство главных компонент."""
    return (features - mean) @ components


def _fill_binary_holes(mask: np.ndarray) -> np.ndarray:
    """Заполняет внутренние дыры в бинарной маске через flood fill от границы."""
    padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    flooded = padded.copy()
    flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), dtype=np.uint8)
    cv2.floodFill(flooded, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flooded)[1:-1, 1:-1]
    return cv2.bitwise_or(mask, holes)


class AdaptiveChangeDetection:
    """Адаптивный PCA+CVA-алгоритм для пары разновременных RGB-снимков."""

    def __init__(
        self,
        patch_size: int = 1,
        pca_components: int = 3,
        pca_variance_ratio: Optional[float] = None,
        whitening: bool = True,
        threshold_value: Optional[float] = None,
        otsu_scale: float = 0.85,
        median_kernel: int = 3,
        opening_kernel: int = 3,
        closing_kernel: int = 3,
        min_area: Optional[int] = 100,
        fill_holes: bool = False,
    ):
        self.patch_size = _odd_kernel(patch_size, minimum=1)
        self.pca_components = pca_components
        self.pca_variance_ratio = pca_variance_ratio
        self.whitening = bool(whitening)
        self.threshold_value = threshold_value
        self.otsu_scale = float(otsu_scale)
        self.median_kernel = _odd_kernel(median_kernel, minimum=1)
        self.opening_kernel = _odd_kernel(opening_kernel, minimum=1)
        self.closing_kernel = _odd_kernel(closing_kernel, minimum=1)
        self.min_area = min_area
        self.fill_holes = bool(fill_holes)
        self.intermediate_results: Dict[str, object] = {}

    def process(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Возвращает бинарную маску изменений со значениями 0 и 255."""
        self.intermediate_results = {}
        height, width = img1.shape[:2]

        norm1 = _to_float01(img1)
        norm2 = _to_float01(img2)
        features1 = _extract_patch_features(norm1, self.patch_size)
        features2 = _extract_patch_features(norm2, self.patch_size)

        mean, components, eigvals = _fit_pca(
            features1,
            features2,
            n_components=self.pca_components,
            variance_ratio=self.pca_variance_ratio,
            whitening=self.whitening,
        )
        projected1 = _project(features1, mean, components)
        projected2 = _project(features2, mean, components)

        # CVA: больше расстояние между векторами "до" и "после" - вероятнее изменение.
        change_vectors = projected1 - projected2
        magnitude = np.sqrt(np.sum(change_vectors * change_vectors, axis=1))
        change_map = _normalize01(magnitude.reshape(height, width))

        self.intermediate_results["change_map"] = change_map
        self.intermediate_results["pca_eigenvalues"] = eigvals

        mask = self._threshold(change_map)
        self.intermediate_results["threshold_mask"] = mask.copy()
        final_mask = self._postprocess(mask)
        self.intermediate_results["final_mask"] = final_mask
        return final_mask

    def _threshold(self, change_map: np.ndarray) -> np.ndarray:
        """Преобразует score-map в бинарную маску."""
        if self.threshold_value is not None:
            return (change_map >= float(self.threshold_value)).astype(np.uint8) * 255

        score_uint8 = np.clip(change_map * 255.0, 0, 255).astype(np.uint8)
        if abs(self.otsu_scale - 1.0) < 1e-6:
            return global_otsu_threshold(score_uint8)
        return global_otsu_threshold_scaled(score_uint8, scale=self.otsu_scale)

    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        """Очищает бинарную маску от мелкого шума."""
        result = mask.astype(np.uint8, copy=True)

        if self.median_kernel > 1:
            result = cv2.medianBlur(result, self.median_kernel)
        self.intermediate_results["median_mask"] = result.copy()

        if self.opening_kernel > 1:
            kernel = np.ones((self.opening_kernel, self.opening_kernel), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        self.intermediate_results["opening_mask"] = result.copy()

        if self.closing_kernel > 1:
            kernel = np.ones((self.closing_kernel, self.closing_kernel), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        self.intermediate_results["closing_mask"] = result.copy()

        result = filter_by_area(result, min_area=self.min_area)
        self.intermediate_results["area_mask"] = result.copy()

        if self.fill_holes:
            # Заполняем только внутренние пустоты, чтобы маска крупных объектов была стабильнее.
            result = _fill_binary_holes(result)
        self.intermediate_results["filled_mask"] = result.copy()

        return result

    def get_intermediate_results(self) -> Dict[str, object]:
        """Возвращает промежуточные карты для визуализации и ручной проверки."""
        return self.intermediate_results


class ZScorePCACVA:
    """Z-score normalization -> PCA -> CVA -> clipped Otsu -> opening -> closing."""

    def __init__(
        self,
        pca_components: int = 3,
        pca_variance_ratio: Optional[float] = 0.95,
        min_components: int = 2,
        max_components: int = 3,
        clip_percentiles: tuple[float, float] | None = (1.0, 99.0),
        otsu_scale: float = 1.0,
        opening_kernel: int = 3,
        closing_kernel: int = 3,
    ) -> None:
        self.pca_components = int(pca_components)
        self.pca_variance_ratio = pca_variance_ratio
        self.min_components = int(min_components)
        self.max_components = int(max_components)
        self.clip_percentiles = clip_percentiles
        self.otsu_scale = float(otsu_scale)
        self.opening_kernel = _odd_kernel(opening_kernel, minimum=1)
        self.closing_kernel = _odd_kernel(closing_kernel, minimum=1)
        self.intermediate_results: Dict[str, object] = {}

    def process(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        self.intermediate_results = {}
        height, width = img1.shape[:2]

        norm1 = _zscore_channels(img1)
        norm2 = _zscore_channels(img2)
        features1 = norm1.reshape(height * width, -1).astype(np.float32, copy=False)
        features2 = norm2.reshape(height * width, -1).astype(np.float32, copy=False)

        mean, components, eigvals = _fit_pca(
            features1,
            features2,
            n_components=self.pca_components,
            variance_ratio=self.pca_variance_ratio,
            whitening=False,
        )
        components = self._limit_components(components)
        projected1 = _project(features1, mean, components)
        projected2 = _project(features2, mean, components)

        change_vectors = projected1 - projected2
        magnitude = np.sqrt(np.sum(change_vectors * change_vectors, axis=1)).reshape(height, width)
        change_map = _normalize01(magnitude)
        clipped_map = self._clip_change_map(change_map)

        self.intermediate_results["change_map"] = change_map
        self.intermediate_results["clipped_change_map"] = clipped_map
        self.intermediate_results["pca_eigenvalues"] = eigvals
        self.intermediate_results["pca_components_used"] = int(components.shape[1])

        mask = self._threshold(clipped_map)
        self.intermediate_results["threshold_mask"] = mask.copy()
        final_mask = self._postprocess(mask)
        self.intermediate_results["final_mask"] = final_mask
        return final_mask

    def _limit_components(self, components: np.ndarray) -> np.ndarray:
        available = int(components.shape[1])
        target = min(max(available, self.min_components), self.max_components)
        return components[:, :target]

    def _clip_change_map(self, change_map: np.ndarray) -> np.ndarray:
        if self.clip_percentiles is None:
            return change_map
        low_pct, high_pct = self.clip_percentiles
        low, high = np.percentile(change_map, [float(low_pct), float(high_pct)])
        if high - low < 1e-6:
            return change_map
        return _normalize01(np.clip(change_map, low, high))

    def _threshold(self, change_map: np.ndarray) -> np.ndarray:
        score_uint8 = np.clip(change_map * 255.0, 0, 255).astype(np.uint8)
        if abs(self.otsu_scale - 1.0) < 1e-6:
            return global_otsu_threshold(score_uint8)
        return global_otsu_threshold_scaled(score_uint8, scale=self.otsu_scale)

    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        result = mask.astype(np.uint8, copy=True)
        if self.opening_kernel > 1:
            kernel = np.ones((self.opening_kernel, self.opening_kernel), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
        self.intermediate_results["opening_mask"] = result.copy()

        if self.closing_kernel > 1:
            kernel = np.ones((self.closing_kernel, self.closing_kernel), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        self.intermediate_results["closing_mask"] = result.copy()
        return result

    def get_intermediate_results(self) -> Dict[str, object]:
        return self.intermediate_results
