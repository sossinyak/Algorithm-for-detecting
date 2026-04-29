"""Классические и комбинированные методы обнаружения изменений.

Модуль содержит два уровня:
- функции score-map строят непрерывную карту вероятных изменений;
- пайплайны превращают score-map в бинарную маску и выполняют постобработку.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable, Mapping

import cv2
import numpy as np

from postprocessing.area_filter import filter_by_area


ScoreFn = Callable[[np.ndarray, np.ndarray], np.ndarray]


def _to_gray(image: np.ndarray) -> np.ndarray:
    """Переводит изображение OpenCV BGR в оттенки серого."""
    if image.ndim == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _odd_kernel(value: int, minimum: int = 1) -> int:
    """Приводит размер ядра к нечетному числу."""
    value = max(minimum, int(value))
    return value if value % 2 else value + 1


def _normalize_uint8(values: np.ndarray) -> np.ndarray:
    """Масштабирует карту значений в диапазон 0..255."""
    values = values.astype(np.float32, copy=False)
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    if max_val - min_val < 1e-6:
        return np.zeros_like(values, dtype=np.uint8)
    scaled = (values - min_val) / (max_val - min_val) * 255.0
    return np.clip(scaled, 0, 255).astype(np.uint8)


def estimate_noise_sigma(img1: np.ndarray, img2: np.ndarray | None = None) -> float:
    """Оценивает шум по медиане модулей разностей соседних пикселей.

    Оценка основана на робастной MAD-статистике. Если переданы два снимка,
    используется среднее значение по обоим снимкам. Возвращается sigma для
    GaussianBlur в разумном диапазоне 0.5..3.0.
    """

    def _one(image: np.ndarray) -> float:
        gray = _to_gray(image).astype(np.float32)
        diffs = []
        if gray.shape[1] > 1:
            diffs.append(np.diff(gray, axis=1).ravel())
        if gray.shape[0] > 1:
            diffs.append(np.diff(gray, axis=0).ravel())
        if not diffs:
            return 1.0
        values = np.concatenate(diffs)
        mad = np.median(np.abs(values - np.median(values)))
        pixel_sigma = float(mad / 0.6745) if mad > 1e-6 else 1.0
        return float(np.clip(pixel_sigma / 18.0, 0.5, 3.0))

    if img2 is None:
        return _one(img1)
    return float(np.clip((_one(img1) + _one(img2)) / 2.0, 0.5, 3.0))


def _apply_gaussian_auto(image: np.ndarray, sigma: float | str | None, other: np.ndarray | None = None) -> np.ndarray:
    """Сглаживает изображение с фиксированной или автоматически оцененной sigma."""
    if sigma is None or sigma == 0:
        return image
    sigma_value = estimate_noise_sigma(image, other) if sigma == "auto" else float(sigma)
    if sigma_value <= 0:
        return image
    kernel = _odd_kernel(int(round(sigma_value * 6 + 1)), minimum=3)
    return cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma_value, sigmaY=sigma_value)


def _otsu_mask(values: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Бинаризует карту методом Оцу с необязательным масштабом порога."""
    score = _normalize_uint8(values)
    threshold, _ = cv2.threshold(score, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return (score > np.clip(threshold * float(scale), 0.0, 255.0)).astype(np.uint8) * 255


def _triangle_mask(values: np.ndarray) -> np.ndarray:
    """Бинаризует карту методом Triangle."""
    _, mask = cv2.threshold(_normalize_uint8(values), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    return mask


def _kmeans_mask(values: np.ndarray) -> np.ndarray:
    """Делит значения карты на два кластера и выбирает кластер с большим центром."""
    score = _normalize_uint8(values)
    samples = score.reshape(-1, 1).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 60, 0.2)
    _, labels, centers = cv2.kmeans(samples, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
    changed_cluster = int(np.argmax(centers.reshape(-1)))
    return (labels.reshape(score.shape) == changed_cluster).astype(np.uint8) * 255


def _adaptive_mask(values: np.ndarray, block_size: int = 35, c_value: float = -2.0) -> np.ndarray:
    """Локальная адаптивная бинаризация score-map."""
    score = _normalize_uint8(values)
    block_size = _odd_kernel(block_size, minimum=3)
    return cv2.adaptiveThreshold(
        score,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        block_size,
        float(c_value),
    )


def _gray_absdiff(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Абсолютная разность яркости."""
    return np.abs(_to_gray(img2).astype(np.float32) - _to_gray(img1).astype(np.float32))


def _gray_log_ratio(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Модуль логарифмического отношения яркости."""
    gray1 = _to_gray(img1).astype(np.float32) + 1.0
    gray2 = _to_gray(img2).astype(np.float32) + 1.0
    return np.abs(np.log(gray2 / gray1))


def _gray_normalized_diff(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Нормированная разность яркости."""
    gray1 = _to_gray(img1).astype(np.float32)
    gray2 = _to_gray(img2).astype(np.float32)
    return np.abs(gray2 - gray1) / (gray2 + gray1 + 1.0)


def _rgb_cva(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """CVA: длина RGB-вектора разности."""
    if img1.ndim != 3 or img2.ndim != 3:
        return _gray_absdiff(img1, img2)
    diff = img2.astype(np.float32) - img1.astype(np.float32)
    return np.sqrt(np.sum(diff * diff, axis=2))


def _lab_cva(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """CVA в цветовом пространстве LAB."""
    lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB).astype(np.float32)
    lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB).astype(np.float32)
    diff = lab2 - lab1
    return np.sqrt(np.sum(diff * diff, axis=2))


def _hsv_cva(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """CVA в HSV с циклической обработкой hue-канала."""
    hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV).astype(np.float32)
    diff = hsv2 - hsv1
    hue = np.minimum(np.abs(diff[:, :, 0]), 180.0 - np.abs(diff[:, :, 0])) / 180.0 * 255.0
    return np.sqrt(hue * hue + np.sum(diff[:, :, 1:] * diff[:, :, 1:], axis=2))


def apply_clahe_lab(image: np.ndarray, clip_limit: float = 2.0, tile_grid_size: int | tuple[int, int] = 16) -> np.ndarray:
    """Применяет CLAHE к L-каналу LAB-изображения."""
    if image.ndim != 3:
        return image.copy()
    if isinstance(tile_grid_size, int):
        tile_grid_size = (int(tile_grid_size), int(tile_grid_size))
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=tuple(tile_grid_size))
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def make_clahe_cva_score(clip_limit: float = 2.0, tile_grid_size: int = 16) -> ScoreFn:
    """Создает настраиваемую CLAHE-CVA score-функцию."""

    def _score(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        eq1 = apply_clahe_lab(img1, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        eq2 = apply_clahe_lab(img2, clip_limit=clip_limit, tile_grid_size=tile_grid_size)
        return _rgb_cva(eq1, eq2)

    return _score


def _clahe_rgb_cva(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """CLAHE-CVA с параметрами по умолчанию."""
    return make_clahe_cva_score()(img1, img2)


def _local_mean_diff(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Локально усредненная абсолютная разность яркости."""
    return cv2.blur(_gray_absdiff(img1, img2), (9, 9))


def _edge_diff(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Разность Canny-контуров."""
    gray1 = cv2.GaussianBlur(_to_gray(img1), (5, 5), 1.2)
    gray2 = cv2.GaussianBlur(_to_gray(img2), (5, 5), 1.2)
    return cv2.GaussianBlur(cv2.absdiff(cv2.Canny(gray1, 50, 150), cv2.Canny(gray2, 50, 150)), (3, 3), 0)


def _laplacian_diff(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Разность лапласианов как текстурная карта изменений."""
    gray1 = cv2.GaussianBlur(_to_gray(img1), (3, 3), 0)
    gray2 = cv2.GaussianBlur(_to_gray(img2), (3, 3), 0)
    return np.abs(cv2.Laplacian(gray2, cv2.CV_32F, ksize=3) - cv2.Laplacian(gray1, cv2.CV_32F, ksize=3))


def _pca_projected_cva(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """CVA после проекции RGB-пикселей в двумерное PCA-пространство."""
    if img1.ndim != 3 or img2.ndim != 3:
        return _gray_absdiff(img1, img2)
    height, width = img1.shape[:2]
    pixels1 = img1.reshape(-1, 3).astype(np.float32) / 255.0
    pixels2 = img2.reshape(-1, 3).astype(np.float32) / 255.0
    mean, eigenvectors = cv2.PCACompute(np.vstack([pixels1, pixels2]), mean=None, maxComponents=2)
    projected1 = cv2.PCAProject(pixels1, mean, eigenvectors)
    projected2 = cv2.PCAProject(pixels2, mean, eigenvectors)
    diff = projected2 - projected1
    return np.sqrt(np.sum(diff * diff, axis=1)).reshape(height, width)


def classical_score_functions() -> "OrderedDict[str, ScoreFn]":
    """Возвращает 11 базовых score-map для итогового сравнения."""
    return OrderedDict(
        [
            ("AbsDiff", _gray_absdiff),
            ("LogRatio", _gray_log_ratio),
            ("NormDiff", _gray_normalized_diff),
            ("RGB-CVA", _rgb_cva),
            ("LAB-CVA", _lab_cva),
            ("HSV-CVA", _hsv_cva),
            ("CLAHE-CVA", _clahe_rgb_cva),
            ("LocalMeanDiff", _local_mean_diff),
            ("EdgeDiff", _edge_diff),
            ("LaplacianDiff", _laplacian_diff),
            ("PCA-CVA", _pca_projected_cva),
        ]
    )


@dataclass
class TunableClassicalPipeline:
    """Классический детектор с настраиваемым порогом и постобработкой."""

    score_fn: ScoreFn
    threshold: str = "otsu"
    threshold_scale: float = 1.0
    postprocess: str = "area"
    median_kernel: int = 3
    morph_kernel: int = 3
    min_area: int | None = None
    adaptive_block_size: int = 35
    adaptive_c: float = -2.0
    sigma: float | str | None = None

    def process(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        img1p = _apply_gaussian_auto(img1, self.sigma, img2)
        img2p = _apply_gaussian_auto(img2, self.sigma, img1)
        score = self.score_fn(img1p, img2p)
        return self._postprocess(self._threshold(score))

    def _threshold(self, score: np.ndarray) -> np.ndarray:
        if self.threshold == "kmeans":
            return _kmeans_mask(score)
        if self.threshold == "triangle":
            return _triangle_mask(score)
        if self.threshold == "adaptive":
            return _adaptive_mask(score, self.adaptive_block_size, self.adaptive_c)
        return _otsu_mask(score, scale=self.threshold_scale)

    def _postprocess(self, mask: np.ndarray) -> np.ndarray:
        if self.postprocess == "raw":
            return mask.astype(np.uint8, copy=True)
        result = mask.astype(np.uint8, copy=True)
        median_kernel = _odd_kernel(self.median_kernel)
        if median_kernel > 1:
            result = cv2.medianBlur(result, median_kernel)
        morph_kernel = max(1, int(self.morph_kernel))
        if morph_kernel > 1:
            kernel = np.ones((morph_kernel, morph_kernel), np.uint8)
            result = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel)
        if self.postprocess == "area":
            min_area = self.min_area
            if min_area is None:
                min_area = max(16, int(mask.shape[0] * mask.shape[1] * 0.0008))
            result = filter_by_area(result, min_area=int(min_area))
        return result


@dataclass
class CombinedChangePipeline(TunableClassicalPipeline):
    """Взвешенное объединение нескольких карт изменений."""

    weights: Mapping[str, float] = field(default_factory=lambda: {"AbsDiff": 0.35, "LogRatio": 0.25, "RGB-CVA": 0.40})
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 16

    def __init__(
        self,
        weights: Mapping[str, float] | None = None,
        threshold: str = "otsu",
        threshold_scale: float = 1.0,
        postprocess: str = "area",
        median_kernel: int = 3,
        morph_kernel: int = 3,
        min_area: int | None = None,
        adaptive_block_size: int = 35,
        adaptive_c: float = -2.0,
        sigma: float | str | None = "auto",
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 16,
    ) -> None:
        self.weights = weights or {"AbsDiff": 0.35, "LogRatio": 0.25, "RGB-CVA": 0.40}
        self.clahe_clip_limit = float(clahe_clip_limit)
        self.clahe_tile_grid_size = int(clahe_tile_grid_size)
        super().__init__(
            score_fn=self._combined_score,
            threshold=threshold,
            threshold_scale=threshold_scale,
            postprocess=postprocess,
            median_kernel=median_kernel,
            morph_kernel=morph_kernel,
            min_area=min_area,
            adaptive_block_size=adaptive_block_size,
            adaptive_c=adaptive_c,
            sigma=sigma,
        )

    def _combined_score(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        scores = classical_score_functions()
        scores["CLAHE-CVA"] = make_clahe_cva_score(self.clahe_clip_limit, self.clahe_tile_grid_size)
        total = None
        weight_sum = 0.0
        for name, weight in self.weights.items():
            if abs(float(weight)) < 1e-12:
                continue
            score = _normalize_uint8(scores[name](img1, img2)).astype(np.float32) / 255.0
            total = score * float(weight) if total is None else total + score * float(weight)
            weight_sum += abs(float(weight))
        if total is None or weight_sum < 1e-12:
            return np.zeros(img1.shape[:2], dtype=np.float32)
        return total / weight_sum


class EdgeBasedDetector(TunableClassicalPipeline):
    """Полноценный edge-based detector: Canny(T1) XOR Canny(T2) + difference map."""

    def __init__(
        self,
        canny_low: int = 50,
        canny_high: int = 150,
        edge_weight: float = 0.45,
        diff_weight: float = 0.55,
        sigma: float | str | None = "auto",
        threshold: str = "otsu",
        threshold_scale: float = 1.0,
        postprocess: str = "area",
        median_kernel: int = 3,
        morph_kernel: int = 3,
        min_area: int | None = None,
        adaptive_block_size: int = 35,
        adaptive_c: float = -2.0,
    ) -> None:
        self.canny_low = int(canny_low)
        self.canny_high = int(canny_high)
        self.edge_weight = float(edge_weight)
        self.diff_weight = float(diff_weight)
        super().__init__(
            score_fn=self._edge_score,
            threshold=threshold,
            threshold_scale=threshold_scale,
            postprocess=postprocess,
            median_kernel=median_kernel,
            morph_kernel=morph_kernel,
            min_area=min_area,
            adaptive_block_size=adaptive_block_size,
            adaptive_c=adaptive_c,
            sigma=sigma,
        )

    def _edge_score(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        gray1 = _to_gray(img1)
        gray2 = _to_gray(img2)
        edges1 = cv2.Canny(gray1, self.canny_low, self.canny_high)
        edges2 = cv2.Canny(gray2, self.canny_low, self.canny_high)
        edge_xor = cv2.bitwise_xor(edges1, edges2).astype(np.float32) / 255.0
        diff = _normalize_uint8(_gray_absdiff(img1, img2)).astype(np.float32) / 255.0
        score = self.edge_weight * edge_xor + self.diff_weight * diff
        return cv2.GaussianBlur(score, (3, 3), 0)


def build_tunable_classical_method(
    score_name: str,
    threshold: str = "otsu",
    threshold_scale: float = 1.0,
    postprocess: str = "area",
    median_kernel: int = 3,
    morph_kernel: int = 3,
    min_area: int | None = None,
    adaptive_block_size: int = 35,
    adaptive_c: float = -2.0,
    sigma: float | str | None = None,
    clahe_clip_limit: float = 2.0,
    clahe_tile_grid_size: int = 16,
) -> TunableClassicalPipeline:
    """Создает настраиваемый детектор по имени карты изменений."""
    if score_name == "CLAHE-CVA":
        score_fn = make_clahe_cva_score(clahe_clip_limit, clahe_tile_grid_size)
    else:
        scores = classical_score_functions()
        if score_name not in scores:
            raise KeyError(f"Неизвестная карта изменений: {score_name}")
        score_fn = scores[score_name]
    return TunableClassicalPipeline(
        score_fn=score_fn,
        threshold=threshold,
        threshold_scale=threshold_scale,
        postprocess=postprocess,
        median_kernel=median_kernel,
        morph_kernel=morph_kernel,
        min_area=min_area,
        adaptive_block_size=adaptive_block_size,
        adaptive_c=adaptive_c,
        sigma=sigma,
    )


class BaselinePCAKMeans:
    """PCA-CVA с порогом K-means."""

    def __init__(self, postprocess: str = "area", min_area: int | None = None) -> None:
        self.postprocess = postprocess
        self.min_area = min_area

    def process(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        detector = TunableClassicalPipeline(
            score_fn=_pca_projected_cva,
            threshold="kmeans",
            postprocess=self.postprocess,
            min_area=self.min_area,
        )
        return detector.process(img1, img2)


def build_pruned_classical_methods() -> "OrderedDict[str, object]":
    """Возвращает очищенный набор классических методов."""
    methods: OrderedDict[str, object] = OrderedDict()
    for score_name in classical_score_functions():
        methods[score_name] = build_tunable_classical_method(score_name)
    return methods


def build_research_methods() -> "OrderedDict[str, object]":
    """Возвращает расширенный набор для исследовательского протокола."""
    methods = build_pruned_classical_methods()
    methods["Combined-AbsLogCVA"] = CombinedChangePipeline()
    methods["Edge-Canny-XOR"] = EdgeBasedDetector()
    methods["PCA-KMeans"] = BaselinePCAKMeans()
    return methods


def build_classical_methods(method_set: str = "strong") -> "OrderedDict[str, object]":
    """Возвращает набор методов по имени режима."""
    method_set = method_set.lower()
    if method_set in {"strong", "pruned", "tuned", "full"}:
        methods = build_pruned_classical_methods()
        methods["PCA-KMeans"] = BaselinePCAKMeans()
        return methods
    if method_set in {"research", "protocol"}:
        return build_research_methods()
    if method_set in {"core", "fast"}:
        selected = ["AbsDiff", "LogRatio", "RGB-CVA", "LAB-CVA", "EdgeDiff", "PCA-CVA"]
        methods = OrderedDict((name, build_tunable_classical_method(name, postprocess="raw")) for name in selected)
        methods["PCA-KMeans"] = BaselinePCAKMeans(postprocess="raw")
        return methods
    if method_set == "legacy":
        return OrderedDict(
            [
                ("Absolute Difference + Otsu", BaselineDiffOtsu()),
                ("Log Ratio + Otsu", BaselineRatioOtsu()),
                ("CVA + Otsu", BaselineCVA()),
                ("CLAHE + Filter + Canny", BaselineCascade()),
            ]
        )
    raise ValueError(f"Неизвестный method_set: {method_set}")


class BaselineDiffOtsu(TunableClassicalPipeline):
    def __init__(self) -> None:
        super().__init__(_gray_absdiff, threshold="otsu", postprocess="raw")


class BaselineRatioOtsu(TunableClassicalPipeline):
    def __init__(self) -> None:
        super().__init__(_gray_log_ratio, threshold="otsu", postprocess="raw")


class BaselineCVA(TunableClassicalPipeline):
    def __init__(self) -> None:
        super().__init__(_rgb_cva, threshold="otsu", postprocess="raw")


class BaselineCascade(TunableClassicalPipeline):
    def __init__(self) -> None:
        super().__init__(_edge_diff, threshold="otsu", postprocess="area", sigma="auto")
