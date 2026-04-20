"""
Ablation study, который оценивает вклад адаптивных улучшений в итоговый F1.
"""

from __future__ import annotations

import os
import time
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.metrics import calculate_metrics
from pipelines.adaptive_algorithm import AdaptiveChangeDetection
from utils.pipeline_config import build_adaptive_params, load_configured_pairs


def _base_params(config: dict) -> dict:
    return build_adaptive_params(config)


def _minimal_classical_params(base: dict) -> dict:
    params = dict(base)
    params.update(
        {
            "use_filter": False,
            "use_clahe": False,
            "use_radiometric_normalization": False,
            "use_canny": False,
            "edge_weight": 0.0,
            "fusion_method": "diff_only",
            "use_multiscale": False,
            "threshold_method": "otsu",
            "closing_kernel": 1,
            "opening_kernel": 1,
            "min_area": 0,
            "auto_area": False,
            "fill_holes": False,
            "min_rectangularity": None,
            "use_vegetation_suppression": False,
            "use_building_prior": False,
            "use_building_filter": False,
        }
    )
    return params


def _cumulative_improvement_steps(base: dict) -> List[Tuple[str, str, dict]]:
    steps: List[Tuple[str, str, dict]] = []
    current = _minimal_classical_params(base)
    steps.append(("База", "Карта различий + глобальная бинаризация Оцу", dict(current)))

    if base.get("use_filter"):
        current.update({"use_filter": True, "filter_sigma": base.get("filter_sigma")})
        steps.append(("+ адаптивное сглаживание", "Гауссов фильтр с фиксированной/адаптивной sigma", dict(current)))

    if base.get("use_clahe"):
        current.update(
            {
                "use_clahe": True,
                "clahe_mode": base.get("clahe_mode"),
                "clahe_clip_limit": base.get("clahe_clip_limit"),
                "clahe_tile_grid_size": base.get("clahe_tile_grid_size"),
            }
        )
        steps.append(("+ CLAHE", "Локальное контрастирование", dict(current)))

    if base.get("use_radiometric_normalization"):
        current.update(
            {
                "use_radiometric_normalization": True,
                "radiometric_method": base.get("radiometric_method"),
                "radiometric_channels": base.get("radiometric_channels"),
                "radiometric_quantiles": base.get("radiometric_quantiles"),
            }
        )
        steps.append(("+ радиометрическая нормализация", "Согласование яркостей двух дат", dict(current)))

    if base.get("use_multiscale"):
        current.update(
            {
                "use_multiscale": True,
                "multiscale_levels": base.get("multiscale_levels"),
                "multiscale_weight_decay": base.get("multiscale_weight_decay"),
            }
        )
        steps.append(("+ многоуровневый анализ", "Gaussian pyramid для нескольких масштабов", dict(current)))

    if base.get("use_canny"):
        current.update(
            {
                "use_canny": True,
                "edge_detector": base.get("edge_detector"),
                "canny_percentile": base.get("canny_percentile"),
                "edge_weight": base.get("edge_weight"),
                "edge_dilation_iterations": base.get("edge_dilation_iterations"),
                "fusion_method": base.get("fusion_method"),
            }
        )
        steps.append(("+ контурная карта", "Интеграция Canny/Sobel с картой различий", dict(current)))

    if base.get("use_vegetation_suppression"):
        current.update(
            {
                "use_vegetation_suppression": True,
                "exg_threshold": base.get("exg_threshold"),
                "exg_min_green": base.get("exg_min_green"),
                "vegetation_kernel": base.get("vegetation_kernel"),
                "vegetation_suppression_factor": base.get("vegetation_suppression_factor"),
            }
        )
        steps.append(("+ подавление растительности", "Excess Green для стабильной растительности", dict(current)))

    if base.get("use_building_prior"):
        current.update(
            {
                "use_building_prior": True,
                "building_prior_strength": base.get("building_prior_strength"),
                "building_edge_kernel": base.get("building_edge_kernel"),
                "shadow_percentile": base.get("shadow_percentile"),
                "shadow_dilation": base.get("shadow_dilation"),
            }
        )
        steps.append(("+ prior застройки", "Handcrafted-поддержка зданий по RGB/контурам/теням", dict(current)))

    current.update(
        {
            "threshold_method": base.get("threshold_method"),
            "block_size": base.get("block_size"),
            "threshold_C": base.get("threshold_C"),
            "kimura_k": base.get("kimura_k"),
        }
    )
    steps.append(("+ адаптивная бинаризация", "Гибрид Оцу и локального порога", dict(current)))

    current.update(
        {
            "closing_kernel": base.get("closing_kernel"),
            "opening_kernel": base.get("opening_kernel"),
            "min_area": base.get("min_area"),
            "auto_area": base.get("auto_area"),
            "fill_holes": base.get("fill_holes"),
            "min_rectangularity": base.get("min_rectangularity"),
        }
    )
    steps.append(("+ морфология и CCA", "Замыкание/размыкание и фильтр связных компонент", dict(current)))

    if base.get("use_building_filter"):
        current.update(
            {
                "use_building_filter": True,
                "building_min_rectangularity": base.get("building_min_rectangularity"),
                "building_min_extent": base.get("building_min_extent"),
                "building_min_edge_density": base.get("building_min_edge_density"),
                "building_min_shadow_support": base.get("building_min_shadow_support"),
                "building_max_vegetation_overlap": base.get("building_max_vegetation_overlap"),
                "building_min_support_score": base.get("building_min_support_score"),
                "building_max_aspect_ratio": base.get("building_max_aspect_ratio"),
                "building_shadow_ring_kernel": base.get("building_shadow_ring_kernel"),
            }
        )
        steps.append(("+ building-support фильтр", "Фильтр компонентов по форме, контурам, теням и растительности", dict(current)))

    steps.append(("Итоговый алгоритм", "Полная конфигурация из config.yaml", dict(base)))
    return steps


def _plot_contributions(summary: pd.DataFrame, save_path: str) -> None:
    plot_df = summary[~summary["method"].isin(["База", "Итоговый алгоритм"])].copy()
    if plot_df.empty:
        return

    colors = ["#2E7D32" if value >= 0 else "#C62828" for value in plot_df["delta_f1"]]
    fig, ax = plt.subplots(figsize=(11, 5.8))
    bars = ax.bar(plot_df["method"], plot_df["delta_f1"], color=colors)
    ax.axhline(0, color="#333333", linewidth=1)
    ax.set_title("Вклад адаптивных улучшений в F1-score")
    ax.set_ylabel("Прирост F1 относительно предыдущего шага")
    ax.set_xlabel("Добавленное улучшение")
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{height:+.4f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4 if height >= 0 else -14),
            textcoords="offset points",
            ha="center",
            va="bottom" if height >= 0 else "top",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_ablation_study(config: dict) -> dict:
    experiment_cfg = config.get("experiments", {})
    ablation_cfg = experiment_cfg.get("ablation", {})
    max_samples = experiment_cfg.get("max_samples") or ablation_cfg.get("max_samples", 200)

    pairs = load_configured_pairs(config, split="test", max_pairs=max_samples)
    pairs = [pair for pair in pairs if pair.get("label") is not None]
    if not pairs:
        raise ValueError("В тестовой выборке нет пар с эталонными масками label.")

    steps = _cumulative_improvement_steps(_base_params(config))

    rows = []
    per_sample = []
    previous_f1 = None

    print("\nABLATION STUDY: ВКЛАД АДАПТИВНЫХ УЛУЧШЕНИЙ")
    for index, (name, description, params) in enumerate(steps):
        print(f"Шаг {index}: {name} — {description}")
        method = AdaptiveChangeDetection(**params)
        values = {"precision": [], "recall": [], "f1": [], "time_ms": []}

        for pair in tqdm(pairs, desc=name):
            start = time.perf_counter()
            pred = method.process(pair["img_a"], pair["img_b"])
            elapsed_ms = (time.perf_counter() - start) * 1000
            metrics = calculate_metrics(pred, pair["label"])

            row = {
                "stage": index,
                "method": name,
                "description": description,
                "sample": pair["name"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "time_ms": elapsed_ms,
            }
            per_sample.append(row)
            for key in values:
                values[key].append(row[key])

        mean_f1 = float(np.mean(values["f1"]))
        delta_f1 = 0.0 if previous_f1 is None else mean_f1 - previous_f1
        previous_f1 = mean_f1
        rows.append(
            {
                "stage": index,
                "method": name,
                "description": description,
                "precision": float(np.mean(values["precision"])),
                "recall": float(np.mean(values["recall"])),
                "f1": mean_f1,
                "delta_f1": delta_f1,
                "delta_f1_percent": 0.0 if mean_f1 == 0.0 else 100.0 * delta_f1 / max(mean_f1 - delta_f1, 1e-6),
                "time_ms": float(np.mean(values["time_ms"])),
                "f1_std": float(np.std(values["f1"])),
            }
        )

    summary = pd.DataFrame(rows)
    samples = pd.DataFrame(per_sample)
    os.makedirs("results", exist_ok=True)
    summary.to_csv("results/ablation_study.csv", index=False, encoding="utf-8-sig")
    samples.to_csv("results/ablation_study_per_sample.csv", index=False, encoding="utf-8-sig")
    _plot_contributions(summary, "results/ablation_contribution_plot.png")

    print("\nСводка вклада улучшений:")
    print(summary[["stage", "method", "f1", "delta_f1", "precision", "recall", "time_ms"]].to_string(index=False))
    print("График сохранен: results/ablation_contribution_plot.png")
    return {"summary": summary, "per_sample": samples}
