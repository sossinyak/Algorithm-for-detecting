from __future__ import annotations

from typing import Dict, List

from utils.data_loader import LEVIRCDLoader


def build_adaptive_params(config: dict) -> Dict[str, object]:
    detection = config.get("detection", {})
    segmentation = config.get("segmentation", {})
    morphology = config.get("postprocessing", {}).get("morphology", {})
    post = config.get("postprocessing", {})
    preprocessing = config.get("preprocessing", {})
    gaussian = preprocessing.get("gaussian", {})
    clahe = preprocessing.get("clahe", {})
    radiometric = preprocessing.get("radiometric", {})
    priors = config.get("priors", {})
    vegetation = priors.get("vegetation", {})
    building = priors.get("building", {})
    building_filter = building.get("filter", {})

    sigma = gaussian.get("sigma")
    filter_sigma = None if sigma in (None, "auto") else float(sigma)

    return {
        "diff_weight": detection.get("difference_weight", 0.7),
        "color_weight": detection.get("color_weight", 0.0),
        "difference_method": detection.get("difference_method", "combined"),
        "madi_window_size": detection.get("madi_window_size", 31),
        "madi_color_weight": detection.get("madi_color_weight", 0.35),
        "use_filter": detection.get("use_filter", True),
        "filter_sigma": filter_sigma,
        "gaussian_max_sigma": gaussian.get("max_sigma", 3.0),
        "use_clahe": clahe.get("enabled", detection.get("use_clahe", True)),
        "clahe_mode": clahe.get("mode", detection.get("clahe_mode", "adaptive")),
        "clahe_clip_limit": clahe.get("clip_limit", 2.0),
        "clahe_tile_grid_size": tuple(clahe.get("tile_grid_size", [16, 16])),
        "use_radiometric_normalization": radiometric.get("enabled", False),
        "radiometric_method": radiometric.get("method", "quantile"),
        "radiometric_channels": radiometric.get("channels", "bgr"),
        "radiometric_quantiles": radiometric.get("num_quantiles", 64),
        "use_canny": detection.get("use_canny", True),
        "canny_percentile": detection.get("canny_percentile", 90.0),
        "edge_detector": detection.get("edge_detector", "both"),
        "edge_weight": detection.get("edge_weight", 0.15),
        "edge_dilation_iterations": detection.get("edge_dilation_iterations", 1),
        "fusion_method": detection.get("fusion_method", "weighted"),
        "use_multiscale": detection.get("use_multiscale", False),
        "multiscale_levels": detection.get("multiscale_levels", 3),
        "multiscale_weight_decay": detection.get("multiscale_weight_decay", 0.65),
        "block_size": segmentation.get("block_size", 35),
        "threshold_C": segmentation.get("C", 12),
        "kimura_k": segmentation.get("kimura_k", 0.15),
        "threshold_method": segmentation.get("method", "hybrid"),
        "closing_kernel": morphology.get("closing_kernel", 5),
        "opening_kernel": morphology.get("opening_kernel", 3),
        "min_area": post.get("min_area"),
        "auto_area": post.get("auto_area", False),
        "fill_holes": post.get("fill_holes", True),
        "min_rectangularity": post.get("min_rectangularity"),
        "use_vegetation_suppression": vegetation.get("enabled", False),
        "exg_threshold": vegetation.get("exg_threshold", 0.08),
        "exg_min_green": vegetation.get("min_green", 0.18),
        "vegetation_kernel": vegetation.get("morphology_kernel", 3),
        "vegetation_suppression_factor": vegetation.get("suppression_factor", 0.25),
        "use_building_prior": building.get("enabled", False),
        "building_prior_strength": building.get("strength", 0.45),
        "building_edge_kernel": building.get("edge_density_kernel", 9),
        "shadow_percentile": building.get("shadow_percentile", 25.0),
        "shadow_dilation": building.get("shadow_dilation", 9),
        "use_building_filter": building_filter.get("enabled", False),
        "building_min_rectangularity": building_filter.get("min_rectangularity", 0.45),
        "building_min_extent": building_filter.get("min_extent", 0.30),
        "building_min_edge_density": building_filter.get("min_edge_density", 0.03),
        "building_min_shadow_support": building_filter.get("min_shadow_support", 0.01),
        "building_max_vegetation_overlap": building_filter.get("max_vegetation_overlap", 0.30),
        "building_min_support_score": building_filter.get("min_support_score", 0.45),
        "building_max_aspect_ratio": building_filter.get("max_aspect_ratio", 8.0),
        "building_shadow_ring_kernel": building_filter.get("shadow_ring_kernel", 9),
    }


def load_configured_pairs(config: dict, split: str = "test", max_pairs: int | None = None) -> List[dict]:
    data_cfg = config.get("data", {})
    loader = LEVIRCDLoader(data_cfg["data_path"], data_cfg["img_size"])
    subset = data_cfg.get("subset", "all")
    return loader.load_subset(split=split, subset=subset, max_pairs=max_pairs)
