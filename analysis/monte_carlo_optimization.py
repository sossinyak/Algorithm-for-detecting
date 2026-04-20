"""
Оптимизация параметров классического адаптивного алгоритма методом Монте-Карло.
"""

from __future__ import annotations

import copy
import os
import time
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from analysis.metrics import calculate_metrics
from pipelines.adaptive_algorithm import AdaptiveChangeDetection
from utils.pipeline_config import build_adaptive_params, load_configured_pairs


def _evaluate(pairs: List[dict], params: dict) -> dict:
    valid_pairs = [pair for pair in pairs if pair.get("label") is not None]
    if not valid_pairs:
        raise ValueError("No labeled image pairs were provided for evaluation.")

    algo = AdaptiveChangeDetection(**params)
    values = {"precision": [], "recall": [], "f1": [], "time_ms": []}

    for pair in valid_pairs:
        start = time.perf_counter()
        pred = algo.process(pair["img_a"], pair["img_b"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics = calculate_metrics(pred, pair["label"])

        for key in ["precision", "recall", "f1"]:
            values[key].append(metrics[key])
        values["time_ms"].append(elapsed_ms)

    return {
        "precision": float(np.mean(values["precision"])),
        "recall": float(np.mean(values["recall"])),
        "f1": float(np.mean(values["f1"])),
        "time_ms": float(np.mean(values["time_ms"])),
        "f1_std": float(np.std(values["f1"])),
    }


def _sample_odd_int(rng: np.random.Generator, bounds: List[int]) -> int:
    low, high = int(bounds[0]), int(bounds[1])
    value = int(rng.integers(low, high + 1))
    if value % 2 == 0:
        value += 1 if value < high else -1
    return max(3, value)


def _sample_from_options(rng: np.random.Generator, options: List[Any]) -> Any:
    return copy.deepcopy(options[int(rng.integers(0, len(options)))])


def _format_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return list(value)
    return value


def _sample_params(base: dict, search_cfg: dict, rng: np.random.Generator) -> dict:
    params = dict(base)

    use_filter = bool(_sample_from_options(rng, search_cfg.get("use_filter_options", [True, False])))
    params["use_filter"] = use_filter
    if use_filter:
        include_auto = search_cfg.get("include_auto_sigma", True)
        auto_probability = float(search_cfg.get("auto_sigma_probability", 0.15))
        sigma_range = search_cfg.get("filter_sigma_range", [0.5, 3.0])
        if include_auto and rng.random() < auto_probability:
            params["filter_sigma"] = None
        else:
            params["filter_sigma"] = float(rng.uniform(float(sigma_range[0]), float(sigma_range[1])))
    else:
        params["filter_sigma"] = base.get("filter_sigma")

    use_clahe = bool(_sample_from_options(rng, search_cfg.get("use_clahe_options", [True, False])))
    params["use_clahe"] = use_clahe
    if use_clahe:
        params["clahe_mode"] = _sample_from_options(
            rng,
            search_cfg.get("clahe_modes", ["adaptive", "fixed"]),
        )
        clip_range = search_cfg.get("clahe_clip_range", [1.0, 3.0])
        params["clahe_clip_limit"] = float(
            rng.uniform(float(clip_range[0]), float(clip_range[1]))
        )
        tile_options = search_cfg.get("clahe_tile_options", [[8, 8], [16, 16], [32, 32]])
        params["clahe_tile_grid_size"] = tuple(_sample_from_options(rng, tile_options))
    else:
        params["clahe_mode"] = base.get("clahe_mode", "adaptive")

    use_radiometric = bool(
        _sample_from_options(rng, search_cfg.get("radiometric_options", [True, False]))
    )
    params["use_radiometric_normalization"] = use_radiometric
    if use_radiometric:
        params["radiometric_method"] = _sample_from_options(
            rng,
            search_cfg.get("radiometric_methods", ["quantile", "histogram"]),
        )
        params["radiometric_channels"] = _sample_from_options(
            rng,
            search_cfg.get("radiometric_channels", ["bgr", "luminance"]),
        )
        params["radiometric_quantiles"] = int(
            _sample_from_options(rng, search_cfg.get("radiometric_quantile_options", [32, 64, 128]))
        )
    else:
        params["radiometric_method"] = base.get("radiometric_method", "quantile")
        params["radiometric_channels"] = base.get("radiometric_channels", "bgr")
        params["radiometric_quantiles"] = base.get("radiometric_quantiles", 64)

    diff_range = search_cfg.get("diff_weight_range", [0.3, 0.8])
    params["diff_weight"] = float(rng.uniform(float(diff_range[0]), float(diff_range[1])))
    color_range = search_cfg.get("color_weight_range", [0.0, 1.0])
    params["color_weight"] = float(rng.uniform(float(color_range[0]), float(color_range[1])))
    params["difference_method"] = _sample_from_options(
        rng,
        search_cfg.get("difference_methods", ["combined", "madi"]),
    )
    params["madi_window_size"] = _sample_odd_int(
        rng,
        search_cfg.get("madi_window_range", [15, 55]),
    )
    madi_color_range = search_cfg.get("madi_color_weight_range", [0.0, 0.75])
    params["madi_color_weight"] = float(
        rng.uniform(float(madi_color_range[0]), float(madi_color_range[1]))
    )

    use_canny = bool(_sample_from_options(rng, search_cfg.get("use_canny_options", [True, False])))
    params["use_canny"] = use_canny
    if use_canny:
        params["edge_detector"] = _sample_from_options(
            rng,
            search_cfg.get("edge_detectors", ["canny", "sobel", "both"]),
        )
        percentile_range = search_cfg.get("canny_percentile_range", [85.0, 97.0])
        params["canny_percentile"] = float(
            rng.uniform(float(percentile_range[0]), float(percentile_range[1]))
        )
        params["edge_dilation_iterations"] = int(
            _sample_from_options(rng, search_cfg.get("edge_dilation_options", [0, 1]))
        )
        edge_range = search_cfg.get("edge_weight_range", [0.0, 0.4])
        params["edge_weight"] = float(rng.uniform(float(edge_range[0]), float(edge_range[1])))
        params["fusion_method"] = _sample_from_options(
            rng,
            search_cfg.get("fusion_methods", ["weighted", "max", "diff_only"]),
        )
    else:
        params["edge_weight"] = 0.0
        params["fusion_method"] = "diff_only"
        params["edge_detector"] = base.get("edge_detector", "canny")
        params["canny_percentile"] = base.get("canny_percentile", 90.0)
        params["edge_dilation_iterations"] = base.get("edge_dilation_iterations", 1)

    use_multiscale = bool(_sample_from_options(rng, search_cfg.get("multiscale_options", [True, False])))
    params["use_multiscale"] = use_multiscale
    if use_multiscale:
        params["multiscale_levels"] = int(
            _sample_from_options(rng, search_cfg.get("multiscale_level_options", [2, 3]))
        )
        decay_range = search_cfg.get("multiscale_weight_decay_range", [0.45, 0.85])
        params["multiscale_weight_decay"] = float(
            rng.uniform(float(decay_range[0]), float(decay_range[1]))
        )
    else:
        params["multiscale_levels"] = 1
        params["multiscale_weight_decay"] = base.get("multiscale_weight_decay", 0.65)

    params["threshold_method"] = _sample_from_options(
        rng,
        search_cfg.get("threshold_methods", ["otsu", "adaptive", "hybrid", "kimura"]),
    )
    params["block_size"] = _sample_odd_int(rng, search_cfg.get("block_size_range", [15, 55]))
    c_range = search_cfg.get("threshold_C_range", [4, 20])
    params["threshold_C"] = float(rng.integers(int(c_range[0]), int(c_range[1]) + 1))
    kimura_range = search_cfg.get("kimura_k_range", [0.05, 0.35])
    params["kimura_k"] = float(rng.uniform(float(kimura_range[0]), float(kimura_range[1])))

    closing_options = search_cfg.get("closing_kernel_options", [3, 5, 7])
    opening_options = search_cfg.get("opening_kernel_options", [1, 3, 5])
    params["closing_kernel"] = int(_sample_from_options(rng, closing_options))
    params["opening_kernel"] = int(_sample_from_options(rng, opening_options))
    if params["opening_kernel"] > params["closing_kernel"]:
        params["opening_kernel"] = params["closing_kernel"]

    area_range = search_cfg.get("min_area_range", [0, 200])
    params["min_area"] = int(rng.integers(int(area_range[0]), int(area_range[1]) + 1))
    params["auto_area"] = bool(_sample_from_options(rng, search_cfg.get("auto_area_options", [False])))
    params["fill_holes"] = bool(
        _sample_from_options(rng, search_cfg.get("fill_holes_options", [True, False]))
    )
    params["min_rectangularity"] = None

    params["use_vegetation_suppression"] = bool(
        _sample_from_options(rng, search_cfg.get("vegetation_options", [True, False]))
    )
    veg_range = search_cfg.get("vegetation_suppression_range", [0.15, 0.65])
    params["vegetation_suppression_factor"] = float(
        rng.uniform(float(veg_range[0]), float(veg_range[1]))
    )

    params["use_building_prior"] = bool(
        _sample_from_options(rng, search_cfg.get("building_prior_options", [True, False]))
    )
    prior_range = search_cfg.get("building_prior_strength_range", [0.15, 0.55])
    params["building_prior_strength"] = float(
        rng.uniform(float(prior_range[0]), float(prior_range[1]))
    )

    params["use_building_filter"] = bool(
        _sample_from_options(rng, search_cfg.get("building_filter_options", [True, False]))
    )
    rect_range = search_cfg.get("building_min_rectangularity_range", [0.25, 0.55])
    extent_range = search_cfg.get("building_min_extent_range", [0.15, 0.35])
    edge_density_range = search_cfg.get("building_min_edge_density_range", [0.01, 0.05])
    shadow_support_range = search_cfg.get("building_min_shadow_support_range", [0.002, 0.02])
    veg_overlap_range = search_cfg.get("building_max_vegetation_overlap_range", [0.25, 0.60])
    support_score_range = search_cfg.get("building_min_support_score_range", [0.20, 0.55])
    params["building_min_rectangularity"] = float(
        rng.uniform(float(rect_range[0]), float(rect_range[1]))
    )
    params["building_min_extent"] = float(
        rng.uniform(float(extent_range[0]), float(extent_range[1]))
    )
    params["building_min_edge_density"] = float(
        rng.uniform(float(edge_density_range[0]), float(edge_density_range[1]))
    )
    params["building_min_shadow_support"] = float(
        rng.uniform(float(shadow_support_range[0]), float(shadow_support_range[1]))
    )
    params["building_max_vegetation_overlap"] = float(
        rng.uniform(float(veg_overlap_range[0]), float(veg_overlap_range[1]))
    )
    params["building_min_support_score"] = float(
        rng.uniform(float(support_score_range[0]), float(support_score_range[1]))
    )

    return params


def _params_to_record(params: dict) -> dict:
    record = {}
    for key, value in params.items():
        record[key] = _format_value(value)
    record["filter_sigma_mode"] = "auto" if params.get("filter_sigma") is None else "fixed"
    return record


def _update_config_with_params(config: dict, params: dict) -> dict:
    tuned = copy.deepcopy(config)

    tuned.setdefault("preprocessing", {}).setdefault("gaussian", {})
    tuned["preprocessing"]["gaussian"]["sigma"] = (
        "auto" if params.get("filter_sigma") is None else float(params["filter_sigma"])
    )
    tuned["preprocessing"]["gaussian"]["max_sigma"] = float(params.get("gaussian_max_sigma", 3.0))

    tuned["preprocessing"].setdefault("clahe", {})
    tuned["preprocessing"]["clahe"]["enabled"] = bool(params.get("use_clahe", True))
    tuned["preprocessing"]["clahe"]["mode"] = params.get("clahe_mode", "adaptive")
    tuned["preprocessing"]["clahe"]["clip_limit"] = float(params.get("clahe_clip_limit", 2.0))
    tuned["preprocessing"]["clahe"]["tile_grid_size"] = list(
        params.get("clahe_tile_grid_size", (16, 16))
    )

    tuned["preprocessing"].setdefault("radiometric", {})
    tuned["preprocessing"]["radiometric"]["enabled"] = bool(
        params.get("use_radiometric_normalization", False)
    )
    tuned["preprocessing"]["radiometric"]["method"] = params.get("radiometric_method", "quantile")
    tuned["preprocessing"]["radiometric"]["channels"] = params.get("radiometric_channels", "bgr")
    tuned["preprocessing"]["radiometric"]["num_quantiles"] = int(
        params.get("radiometric_quantiles", 64)
    )

    tuned.setdefault("detection", {})
    tuned["detection"]["difference_weight"] = float(params.get("diff_weight", 0.7))
    tuned["detection"]["color_weight"] = float(params.get("color_weight", 0.0))
    tuned["detection"]["difference_method"] = params.get("difference_method", "combined")
    tuned["detection"]["madi_window_size"] = int(params.get("madi_window_size", 31))
    tuned["detection"]["madi_color_weight"] = float(params.get("madi_color_weight", 0.35))
    tuned["detection"]["use_filter"] = bool(params.get("use_filter", True))
    tuned["detection"].pop("use_clahe", None)
    tuned["detection"]["use_canny"] = bool(params.get("use_canny", True))
    tuned["detection"]["edge_detector"] = params.get("edge_detector", "both")
    tuned["detection"]["canny_percentile"] = float(params.get("canny_percentile", 90.0))
    tuned["detection"]["edge_weight"] = float(params.get("edge_weight", 0.15))
    tuned["detection"]["edge_dilation_iterations"] = int(
        params.get("edge_dilation_iterations", 1)
    )
    tuned["detection"]["fusion_method"] = params.get("fusion_method", "weighted")
    tuned["detection"]["use_multiscale"] = bool(params.get("use_multiscale", False))
    tuned["detection"]["multiscale_levels"] = int(params.get("multiscale_levels", 1))
    tuned["detection"]["multiscale_weight_decay"] = float(
        params.get("multiscale_weight_decay", 0.65)
    )

    tuned.setdefault("segmentation", {})
    tuned["segmentation"]["method"] = params.get("threshold_method", "hybrid")
    tuned["segmentation"]["block_size"] = int(params.get("block_size", 35))
    tuned["segmentation"]["C"] = int(params.get("threshold_C", 12))
    tuned["segmentation"]["kimura_k"] = float(params.get("kimura_k", 0.15))

    tuned.setdefault("postprocessing", {})
    tuned["postprocessing"]["min_area"] = int(params.get("min_area", 100))
    tuned["postprocessing"]["auto_area"] = bool(params.get("auto_area", False))
    tuned["postprocessing"]["fill_holes"] = bool(params.get("fill_holes", True))
    tuned["postprocessing"]["min_rectangularity"] = params.get("min_rectangularity")
    tuned["postprocessing"].setdefault("morphology", {})
    tuned["postprocessing"]["morphology"]["closing_kernel"] = int(
        params.get("closing_kernel", 5)
    )
    tuned["postprocessing"]["morphology"]["opening_kernel"] = int(
        params.get("opening_kernel", 3)
    )

    tuned.setdefault("priors", {}).setdefault("vegetation", {})
    tuned["priors"]["vegetation"]["enabled"] = bool(
        params.get("use_vegetation_suppression", False)
    )
    tuned["priors"]["vegetation"]["suppression_factor"] = float(
        params.get("vegetation_suppression_factor", 0.25)
    )

    tuned["priors"].setdefault("building", {})
    tuned["priors"]["building"]["enabled"] = bool(params.get("use_building_prior", False))
    tuned["priors"]["building"]["strength"] = float(params.get("building_prior_strength", 0.45))
    tuned["priors"]["building"].setdefault("filter", {})
    tuned["priors"]["building"]["filter"]["enabled"] = bool(
        params.get("use_building_filter", False)
    )
    tuned["priors"]["building"]["filter"]["min_rectangularity"] = float(
        params.get("building_min_rectangularity", 0.45)
    )
    tuned["priors"]["building"]["filter"]["min_extent"] = float(
        params.get("building_min_extent", 0.30)
    )
    tuned["priors"]["building"]["filter"]["min_edge_density"] = float(
        params.get("building_min_edge_density", 0.03)
    )
    tuned["priors"]["building"]["filter"]["min_shadow_support"] = float(
        params.get("building_min_shadow_support", 0.01)
    )
    tuned["priors"]["building"]["filter"]["max_vegetation_overlap"] = float(
        params.get("building_max_vegetation_overlap", 0.30)
    )
    tuned["priors"]["building"]["filter"]["min_support_score"] = float(
        params.get("building_min_support_score", 0.45)
    )
    tuned["priors"]["building"]["filter"]["max_aspect_ratio"] = float(
        params.get("building_max_aspect_ratio", 8.0)
    )
    tuned["priors"]["building"]["filter"]["shadow_ring_kernel"] = int(
        params.get("building_shadow_ring_kernel", 9)
    )

    return tuned


def run_monte_carlo_optimization(config: dict) -> dict:
    experiment_cfg = config.get("experiments", {})
    monte_cfg = experiment_cfg.get("monte_carlo", {})
    search_cfg = monte_cfg.get("search_space", {})

    optimization_split = monte_cfg.get("optimization_split", "val")
    fallback_split = monte_cfg.get("fallback_split", "train")
    evaluation_split = monte_cfg.get("evaluation_split", "test")
    max_samples = experiment_cfg.get("max_samples") or monte_cfg.get("max_samples", 120)
    evaluation_max_samples = monte_cfg.get("evaluation_max_samples")
    num_trials = int(monte_cfg.get("num_trials", 60))
    top_k = int(monte_cfg.get("top_k", 10))

    print(f"Загрузка выборки для оптимизации: {optimization_split}")
    optimization_pairs = load_configured_pairs(
        config,
        split=optimization_split,
        max_pairs=max_samples,
    )
    if len(optimization_pairs) == 0 and fallback_split and fallback_split != optimization_split:
        print(
            f"Выборка '{optimization_split}' пуста. "
            f"Используется запасная выборка '{fallback_split}'."
        )
        optimization_split = fallback_split
        optimization_pairs = load_configured_pairs(
            config,
            split=optimization_split,
            max_pairs=max_samples,
        )
    print(f"Загружено пар для оптимизации: {len(optimization_pairs)}")
    if len(optimization_pairs) == 0:
        raise ValueError("Для оптимизации Монте-Карло не удалось загрузить пары изображений.")

    print(f"Загрузка контрольной выборки: {evaluation_split}")
    evaluation_pairs = load_configured_pairs(
        config,
        split=evaluation_split,
        max_pairs=evaluation_max_samples,
    )
    print(f"Загружено пар для контрольной оценки: {len(evaluation_pairs)}")
    if len(evaluation_pairs) == 0:
        raise ValueError("Для контрольной оценки Монте-Карло не удалось загрузить пары изображений.")

    base_params = build_adaptive_params(config)
    rng = np.random.default_rng(int(monte_cfg.get("seed", config.get("seed", 42))))

    rows = []
    base_metrics = _evaluate(optimization_pairs, base_params)
    rows.append(
        {
            "trial": 0,
            "source": "base_config",
            **base_metrics,
            **_params_to_record(base_params),
        }
    )

    print("\nОПТИМИЗАЦИЯ МЕТОДОМ МОНТЕ-КАРЛО")
    for trial in tqdm(range(1, num_trials + 1), desc="Испытания Монте-Карло"):
        params = _sample_params(base_params, search_cfg, rng)
        metrics = _evaluate(optimization_pairs, params)
        rows.append(
            {
                "trial": trial,
                "source": "random_search",
                **metrics,
                **_params_to_record(params),
            }
        )

    trials_df = pd.DataFrame(rows).sort_values(
        ["f1", "precision", "time_ms"],
        ascending=[False, False, True],
    )

    best_row = trials_df.iloc[0].to_dict()
    best_params = {
        key: base_params[key]
        for key in base_params
    }
    for key in best_params:
        if key in best_row:
            value = best_row[key]
            if key == "clahe_tile_grid_size" and isinstance(value, list):
                value = tuple(value)
            if key in {
                "use_filter",
                "use_clahe",
                "use_radiometric_normalization",
                "use_canny",
                "use_multiscale",
                "auto_area",
                "fill_holes",
                "use_vegetation_suppression",
                "use_building_prior",
                "use_building_filter",
            }:
                value = bool(value)
            if key in {
                "radiometric_quantiles",
                "multiscale_levels",
                "block_size",
                "closing_kernel",
                "opening_kernel",
                "min_area",
                "madi_window_size",
                "edge_dilation_iterations",
                "building_shadow_ring_kernel",
            }:
                value = int(value)
            if key in {
                "diff_weight",
                "color_weight",
                "madi_color_weight",
                "canny_percentile",
                "gaussian_max_sigma",
                "clahe_clip_limit",
                "edge_weight",
                "multiscale_weight_decay",
                "threshold_C",
                "kimura_k",
                "vegetation_suppression_factor",
                "building_prior_strength",
                "building_min_rectangularity",
                "building_min_extent",
                "building_min_edge_density",
                "building_min_shadow_support",
                "building_max_vegetation_overlap",
                "building_min_support_score",
                "building_max_aspect_ratio",
            }:
                value = float(value)
            best_params[key] = value

    if best_row.get("filter_sigma_mode") == "auto":
        best_params["filter_sigma"] = None

    base_test = _evaluate(evaluation_pairs, base_params)
    best_test = _evaluate(evaluation_pairs, best_params)

    summary_df = pd.DataFrame(
        [
            {"config": "base", "split": optimization_split, **base_metrics},
            {"config": "best", "split": optimization_split, **_evaluate(optimization_pairs, best_params)},
            {"config": "base", "split": evaluation_split, **base_test},
            {"config": "best", "split": evaluation_split, **best_test},
        ]
    )

    os.makedirs("results", exist_ok=True)
    trials_df.to_csv("results/monte_carlo_trials.csv", index=False, encoding="utf-8-sig")
    trials_df.head(top_k).to_csv("results/monte_carlo_top.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv("results/monte_carlo_summary.csv", index=False, encoding="utf-8-sig")

    best_config = _update_config_with_params(config, best_params)
    with open("results/monte_carlo_best_config.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(best_config, file, sort_keys=False, allow_unicode=True)

    best_payload = {
        "optimization_split": optimization_split,
        "evaluation_split": evaluation_split,
        "best_trial": int(best_row["trial"]),
        "base_optimization_f1": float(base_metrics["f1"]),
        "best_optimization_f1": float(summary_df.query("config == 'best' and split == @optimization_split")["f1"].iloc[0]),
        "base_evaluation_f1": float(base_test["f1"]),
        "best_evaluation_f1": float(best_test["f1"]),
        "best_params": {key: _format_value(value) for key, value in best_params.items()},
    }
    with open("results/monte_carlo_best.yaml", "w", encoding="utf-8") as file:
        yaml.safe_dump(best_payload, file, sort_keys=False, allow_unicode=True)

    print("\nЛучшие испытания на оптимизационной выборке:")
    print(
        trials_df.head(top_k)[
            ["trial", "f1", "precision", "recall", "time_ms"]
        ].to_string(index=False)
    )
    print("\nСводка на контрольной выборке:")
    print(summary_df.to_string(index=False))

    return {
        "trials": trials_df,
        "summary": summary_df,
        "best_params": best_params,
        "best_payload": best_payload,
    }
