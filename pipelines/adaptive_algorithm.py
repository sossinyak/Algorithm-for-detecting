"""
Main adaptive classical pipeline for bitemporal change detection.

Processing order:
1. Optional Gaussian denoising.
2. Optional CLAHE contrast normalization.
3. Optional radiometric normalization of the second date to the first date.
4. Difference-map construction, optionally through a Gaussian pyramid.
5. Optional contour-change fusion.
6. Optional handcrafted priors: stable vegetation suppression and building support.
7. Adaptive/global thresholding.
8. Morphology, connected-component filtering, and optional building-support filtering.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from change_detection.difference_methods import compute_difference_map, normalize01
from change_detection.edge_detection import edge_based_change_detection
from change_detection.handcrafted_priors import apply_handcrafted_priors
from change_detection.multiscale import compute_multiscale_difference
from postprocessing.building_support import filter_by_building_support
from postprocessing.morphology import morphological_processing
from preprocessing.contrast import adaptive_clahe, apply_clahe
from preprocessing.noise_estimation import adaptive_gaussian_blur
from preprocessing.radiometric import normalize_pair
from segmentation.adaptive_threshold import (
    global_otsu_threshold,
    kimura_adaptive_threshold,
    local_adaptive_threshold,
)


class AdaptiveChangeDetection:
    """Configurable classical change-detection algorithm."""

    def __init__(
        self,
        diff_weight: float = 0.6,
        color_weight: float = 0.0,
        difference_method: str = "combined",
        madi_window_size: int = 31,
        madi_color_weight: float = 0.35,
        use_filter: bool = True,
        filter_sigma: Optional[float] = None,
        gaussian_max_sigma: float = 3.0,
        use_clahe: bool = True,
        clahe_mode: str = "adaptive",
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: Tuple[int, int] = (16, 16),
        use_radiometric_normalization: bool = False,
        radiometric_method: str = "quantile",
        radiometric_channels: str = "bgr",
        radiometric_quantiles: int = 64,
        use_canny: bool = True,
        canny_sigma: float = 0.33,
        canny_percentile: float = 90.0,
        edge_detector: str = "both",
        edge_weight: float = 0.15,
        edge_dilation_iterations: int = 1,
        fusion_method: str = "weighted",
        use_multiscale: bool = False,
        multiscale_levels: int = 3,
        multiscale_weight_decay: float = 0.65,
        block_size: int = 35,
        threshold_C: float = 5,
        kimura_k: float = 0.15,
        threshold_method: str = "hybrid",
        min_area: Optional[int] = None,
        auto_area: bool = True,
        area_percentile: float = 85,
        closing_kernel: int = 3,
        opening_kernel: int = 2,
        fill_holes: bool = True,
        min_rectangularity: Optional[float] = None,
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
        use_building_filter: bool = False,
        building_min_rectangularity: float = 0.45,
        building_min_extent: float = 0.30,
        building_min_edge_density: float = 0.03,
        building_min_shadow_support: float = 0.01,
        building_max_vegetation_overlap: float = 0.30,
        building_min_support_score: float = 0.45,
        building_max_aspect_ratio: float = 8.0,
        building_shadow_ring_kernel: int = 9,
    ):
        self.diff_weight = float(np.clip(diff_weight, 0.0, 1.0))
        self.color_weight = float(np.clip(color_weight, 0.0, 1.0))
        self.difference_method = difference_method
        self.madi_window_size = madi_window_size
        self.madi_color_weight = float(np.clip(madi_color_weight, 0.0, 1.0))
        self.use_filter = bool(use_filter)
        self.filter_sigma = filter_sigma
        self.gaussian_max_sigma = gaussian_max_sigma
        self.use_clahe = bool(use_clahe)
        self.clahe_mode = clahe_mode
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid_size = tuple(clahe_tile_grid_size)
        self.use_radiometric_normalization = bool(use_radiometric_normalization)
        self.radiometric_method = radiometric_method
        self.radiometric_channels = radiometric_channels
        self.radiometric_quantiles = radiometric_quantiles
        self.use_canny = bool(use_canny)
        self.canny_sigma = canny_sigma
        self.canny_percentile = canny_percentile
        self.edge_detector = edge_detector
        self.edge_weight = float(np.clip(edge_weight, 0.0, 1.0))
        self.edge_dilation_iterations = edge_dilation_iterations
        self.fusion_method = fusion_method
        self.use_multiscale = bool(use_multiscale)
        self.multiscale_levels = multiscale_levels
        self.multiscale_weight_decay = multiscale_weight_decay
        self.block_size = block_size
        self.threshold_C = threshold_C
        self.kimura_k = kimura_k
        self.threshold_method = threshold_method
        self.min_area = min_area
        self.auto_area = auto_area
        self.area_percentile = area_percentile
        self.closing_kernel = closing_kernel
        self.opening_kernel = opening_kernel
        self.fill_holes = fill_holes
        self.min_rectangularity = min_rectangularity
        self.use_vegetation_suppression = bool(use_vegetation_suppression)
        self.exg_threshold = exg_threshold
        self.exg_min_green = exg_min_green
        self.vegetation_kernel = vegetation_kernel
        self.vegetation_suppression_factor = vegetation_suppression_factor
        self.use_building_prior = bool(use_building_prior)
        self.building_prior_strength = building_prior_strength
        self.building_edge_kernel = building_edge_kernel
        self.shadow_percentile = shadow_percentile
        self.shadow_dilation = shadow_dilation
        self.use_building_filter = bool(use_building_filter)
        self.building_min_rectangularity = building_min_rectangularity
        self.building_min_extent = building_min_extent
        self.building_min_edge_density = building_min_edge_density
        self.building_min_shadow_support = building_min_shadow_support
        self.building_max_vegetation_overlap = building_max_vegetation_overlap
        self.building_min_support_score = building_min_support_score
        self.building_max_aspect_ratio = building_max_aspect_ratio
        self.building_shadow_ring_kernel = building_shadow_ring_kernel
        self.intermediate_results: Dict[str, object] = {}

    def process(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        """Run the full pipeline and return a binary mask with values 0/255."""
        self.reset_intermediate_results()

        img1_f, img2_f, sigma_pair = self._filter_pair(img1, img2)
        self.intermediate_results["sigma"] = sigma_pair
        self.intermediate_results["filtered"] = (img1_f, img2_f)

        img1_c, img2_c, clahe_params = self._contrast_pair(img1_f, img2_f)
        self.intermediate_results["clahe_params"] = clahe_params
        self.intermediate_results["clahe"] = (img1_c, img2_c)

        img1_r, img2_r = self._radiometric_pair(img1_c, img2_c)
        self.intermediate_results["radiometric"] = (img1_r, img2_r)

        diff_map = self._difference_map(img1_r, img2_r)
        self.intermediate_results["diff_map"] = diff_map

        combined_map = self._fuse_edges_if_needed(diff_map, img1_r, img2_r)
        self.intermediate_results["combined_map_raw"] = combined_map

        prior_maps: Dict[str, np.ndarray] = {}
        if self.use_vegetation_suppression or self.use_building_prior or self.use_building_filter:
            combined_map, prior_maps = apply_handcrafted_priors(
                combined_map,
                img1,
                img2,
                use_vegetation_suppression=self.use_vegetation_suppression,
                exg_threshold=self.exg_threshold,
                exg_min_green=self.exg_min_green,
                vegetation_kernel=self.vegetation_kernel,
                vegetation_suppression_factor=self.vegetation_suppression_factor,
                use_building_prior=self.use_building_prior,
                building_prior_strength=self.building_prior_strength,
                building_edge_kernel=self.building_edge_kernel,
                shadow_percentile=self.shadow_percentile,
                shadow_dilation=self.shadow_dilation,
                need_building_maps=self.use_building_prior or self.use_building_filter,
            )
            self.intermediate_results.update(prior_maps)

        combined_map = np.clip(combined_map, 0.0, 1.0).astype(np.float32)
        self.intermediate_results["combined_map"] = combined_map

        binary_mask = self._threshold(combined_map)
        binary_mask = self._apply_binary_edge_logic_if_needed(binary_mask)
        self.intermediate_results["binary_mask"] = binary_mask

        final_mask = morphological_processing(
            binary_mask,
            closing_kernel_size=self.closing_kernel,
            opening_kernel_size=self.opening_kernel,
            min_area=self.min_area,
            auto_area=self.auto_area,
            area_percentile=self.area_percentile,
            fill_holes=self.fill_holes,
            min_rectangularity=self.min_rectangularity,
        )

        if self.use_building_filter:
            final_mask = filter_by_building_support(
                final_mask,
                img1,
                img2,
                stable_vegetation=prior_maps.get("stable_vegetation"),
                edge_density=prior_maps.get("edge_density"),
                shadow_support=prior_maps.get("shadow_support"),
                min_rectangularity=self.building_min_rectangularity,
                min_extent=self.building_min_extent,
                min_edge_density=self.building_min_edge_density,
                min_shadow_support=self.building_min_shadow_support,
                max_vegetation_overlap=self.building_max_vegetation_overlap,
                min_support_score=self.building_min_support_score,
                max_aspect_ratio=self.building_max_aspect_ratio,
                shadow_ring_kernel=self.building_shadow_ring_kernel,
            )
            self.intermediate_results["building_filtered_mask"] = final_mask

        self.intermediate_results["final_mask"] = final_mask
        return final_mask

    def _filter_pair(self, img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
        if not self.use_filter:
            return img1.copy(), img2.copy(), (0.0, 0.0)

        if self.filter_sigma is None:
            img1_f, sigma1 = adaptive_gaussian_blur(img1, max_sigma=self.gaussian_max_sigma)
            img2_f, sigma2 = adaptive_gaussian_blur(img2, max_sigma=self.gaussian_max_sigma)
            return img1_f, img2_f, (float(sigma1), float(sigma2))

        sigma = float(self.filter_sigma)
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        return (
            cv2.GaussianBlur(img1, (kernel_size, kernel_size), sigma),
            cv2.GaussianBlur(img2, (kernel_size, kernel_size), sigma),
            (sigma, sigma),
        )

    def _contrast_pair(self, img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray, tuple | None]:
        if not self.use_clahe:
            return img1.copy(), img2.copy(), None

        if self.clahe_mode == "fixed":
            img1_c = apply_clahe(img1, self.clahe_clip_limit, self.clahe_tile_grid_size)
            img2_c = apply_clahe(img2, self.clahe_clip_limit, self.clahe_tile_grid_size)
            return img1_c, img2_c, (
                (self.clahe_clip_limit, self.clahe_tile_grid_size),
                (self.clahe_clip_limit, self.clahe_tile_grid_size),
            )

        img1_c, params1 = adaptive_clahe(img1)
        img2_c, params2 = adaptive_clahe(img2)
        return img1_c, img2_c, (params1, params2)

    def _radiometric_pair(self, img1: np.ndarray, img2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.use_radiometric_normalization:
            return img1, img2

        return normalize_pair(
            img1,
            img2,
            method=self.radiometric_method,
            channels=self.radiometric_channels,
            num_quantiles=self.radiometric_quantiles,
        )

    def _difference_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        if self.use_multiscale:
            diff_map, scale_maps = compute_multiscale_difference(
                img1,
                img2,
                method=self.difference_method,
                weight=self.diff_weight,
                color_weight=self.color_weight,
                madi_window_size=self.madi_window_size,
                madi_color_weight=self.madi_color_weight,
                levels=self.multiscale_levels,
                scale_weight_decay=self.multiscale_weight_decay,
            )
            self.intermediate_results["multiscale_maps"] = scale_maps
            return diff_map

        return compute_difference_map(
            img1,
            img2,
            method=self.difference_method,
            weight=self.diff_weight,
            color_weight=self.color_weight,
            madi_window_size=self.madi_window_size,
            madi_color_weight=self.madi_color_weight,
        )

    def _fuse_edges_if_needed(self, diff_map: np.ndarray, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        diff_norm = normalize01(diff_map)
        fusion_method = str(self.fusion_method).lower()
        if fusion_method == "diff_canny_xor_and":
            self.intermediate_results["edge_map"] = self._canny_xor_map(img1, img2)
            return diff_norm

        needs_edges = self.use_canny and (
            fusion_method == "max" or (fusion_method == "weighted" and self.edge_weight > 0.0)
        )
        if not needs_edges:
            return diff_norm

        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1.copy()
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2.copy()
        edge_map = edge_based_change_detection(
            gray1,
            gray2,
            sigma=self.canny_sigma,
            detector=self.edge_detector,
            canny_percentile=self.canny_percentile,
            dilate_iterations=self.edge_dilation_iterations,
        )
        self.intermediate_results["edge_map"] = edge_map

        edge_norm = edge_map.astype(np.float32) / 255.0
        if fusion_method == "max":
            return np.maximum(diff_norm, edge_norm).astype(np.float32)

        return normalize01((1.0 - self.edge_weight) * diff_norm + self.edge_weight * edge_norm)

    def _canny_xor_map(self, img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY) if len(img1.shape) == 3 else img1.copy()
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2.copy()
        return edge_based_change_detection(
            gray1,
            gray2,
            sigma=self.canny_sigma,
            detector="canny",
            canny_percentile=self.canny_percentile,
            dilate_iterations=self.edge_dilation_iterations,
        )

    def _apply_binary_edge_logic_if_needed(self, binary_mask: np.ndarray) -> np.ndarray:
        if str(self.fusion_method).lower() != "diff_canny_xor_and":
            return binary_mask

        edge_map = self.intermediate_results.get("edge_map")
        if edge_map is None:
            return binary_mask

        edge_mask = (edge_map > 0).astype(np.uint8) * 255
        result = cv2.bitwise_and(binary_mask, edge_mask)
        self.intermediate_results["diff_canny_xor_and_mask"] = result
        return result

    def _threshold(self, score_map: np.ndarray) -> np.ndarray:
        score_uint8 = np.clip(score_map * 255.0, 0, 255).astype(np.uint8)
        method = str(self.threshold_method).lower()

        if method == "otsu":
            return global_otsu_threshold(score_uint8)
        if method == "adaptive":
            return local_adaptive_threshold(
                score_uint8,
                block_size=self.block_size,
                C=self.threshold_C,
            )
        if method == "kimura":
            return kimura_adaptive_threshold(
                score_uint8,
                window_size=self.block_size,
                k=self.kimura_k,
                C=self.threshold_C,
            )

        global_mask = global_otsu_threshold(score_uint8)
        local_mask = local_adaptive_threshold(
            score_uint8,
            block_size=self.block_size,
            C=self.threshold_C,
        )
        return cv2.bitwise_and(global_mask, local_mask)

    def get_intermediate_results(self) -> Dict:
        """Return intermediate maps for visualization/debugging."""
        return self.intermediate_results

    def reset_intermediate_results(self) -> None:
        """Clear intermediate maps before processing another pair."""
        self.intermediate_results = {}

    def get_params(self) -> Dict:
        """Return current algorithm parameters."""
        return {
            "diff_weight": self.diff_weight,
            "color_weight": self.color_weight,
            "difference_method": self.difference_method,
            "madi_window_size": self.madi_window_size,
            "madi_color_weight": self.madi_color_weight,
            "use_filter": self.use_filter,
            "filter_sigma": self.filter_sigma,
            "gaussian_max_sigma": self.gaussian_max_sigma,
            "use_clahe": self.use_clahe,
            "clahe_mode": self.clahe_mode,
            "clahe_clip_limit": self.clahe_clip_limit,
            "clahe_tile_grid_size": self.clahe_tile_grid_size,
            "use_radiometric_normalization": self.use_radiometric_normalization,
            "radiometric_method": self.radiometric_method,
            "radiometric_channels": self.radiometric_channels,
            "radiometric_quantiles": self.radiometric_quantiles,
            "use_canny": self.use_canny,
            "canny_sigma": self.canny_sigma,
            "canny_percentile": self.canny_percentile,
            "edge_detector": self.edge_detector,
            "edge_weight": self.edge_weight,
            "edge_dilation_iterations": self.edge_dilation_iterations,
            "fusion_method": self.fusion_method,
            "use_multiscale": self.use_multiscale,
            "multiscale_levels": self.multiscale_levels,
            "multiscale_weight_decay": self.multiscale_weight_decay,
            "block_size": self.block_size,
            "threshold_C": self.threshold_C,
            "kimura_k": self.kimura_k,
            "threshold_method": self.threshold_method,
            "min_area": self.min_area,
            "auto_area": self.auto_area,
            "area_percentile": self.area_percentile,
            "closing_kernel": self.closing_kernel,
            "opening_kernel": self.opening_kernel,
            "fill_holes": self.fill_holes,
            "min_rectangularity": self.min_rectangularity,
            "use_vegetation_suppression": self.use_vegetation_suppression,
            "exg_threshold": self.exg_threshold,
            "exg_min_green": self.exg_min_green,
            "vegetation_kernel": self.vegetation_kernel,
            "vegetation_suppression_factor": self.vegetation_suppression_factor,
            "use_building_prior": self.use_building_prior,
            "building_prior_strength": self.building_prior_strength,
            "building_edge_kernel": self.building_edge_kernel,
            "shadow_percentile": self.shadow_percentile,
            "shadow_dilation": self.shadow_dilation,
            "use_building_filter": self.use_building_filter,
            "building_min_rectangularity": self.building_min_rectangularity,
            "building_min_extent": self.building_min_extent,
            "building_min_edge_density": self.building_min_edge_density,
            "building_min_shadow_support": self.building_min_shadow_support,
            "building_max_vegetation_overlap": self.building_max_vegetation_overlap,
            "building_min_support_score": self.building_min_support_score,
            "building_max_aspect_ratio": self.building_max_aspect_ratio,
            "building_shadow_ring_kernel": self.building_shadow_ring_kernel,
        }
