"""
Эксперимент сравнения базовых методов и старого пиксельного адаптивного алгоритма.
"""

from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.metrics import calculate_metrics
from pipelines.adaptive_algorithm import AdaptiveChangeDetection
from pipelines.baseline_methods import (
    BaselineCascade,
    BaselineCVA,
    BaselineDiffOtsu,
    BaselineRatioOtsu,
)
from utils.pipeline_config import build_adaptive_params, load_configured_pairs
from visualization.plot_results import plot_comparison


def run_comparison_experiment(config: dict) -> dict:
    experiment_cfg = config.get("experiments", {})
    max_samples = experiment_cfg.get("max_samples")
    no_plot = experiment_cfg.get("no_plot", False)

    print("Загрузка данных LEVIR-CD...")
    test_pairs = load_configured_pairs(config, split="test", max_pairs=max_samples)
    print(f"Загружено пар изображений: {len(test_pairs)}")
    labeled_pairs = [pair for pair in test_pairs if pair.get("label") is not None]
    if not labeled_pairs:
        raise ValueError("В тестовой выборке нет пар с эталонными масками label.")

    methods = {
        "Абс. разность + Оцу": BaselineDiffOtsu(),
        "Лог-отношение + Оцу": BaselineRatioOtsu(),
        "CVA + Оцу": BaselineCVA(),
        "Каскад CLAHE+фильтр+Кэнни": BaselineCascade(),
        "Пиксельный адаптивный алгоритм": AdaptiveChangeDetection(**build_adaptive_params(config)),
    }

    results = {
        name: {"f1": [], "precision": [], "recall": [], "time": []}
        for name in methods
    }

    print("\n" + "=" * 60)
    print("ЭКСПЕРИМЕНТ СРАВНЕНИЯ")
    print("=" * 60)

    for pair in tqdm(labeled_pairs, desc="Пары"):
        img_a = pair["img_a"]
        img_b = pair["img_b"]
        label = pair["label"]

        for name, method in methods.items():
            start_time = time.perf_counter()
            pred_mask = method.process(img_a, img_b)
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            metrics = calculate_metrics(pred_mask, label)
            results[name]["f1"].append(metrics["f1"])
            results[name]["precision"].append(metrics["precision"])
            results[name]["recall"].append(metrics["recall"])
            results[name]["time"].append(elapsed_ms)

    summary = {}
    for name in methods:
        summary[name] = {
            "precision": float(np.mean(results[name]["precision"])),
            "recall": float(np.mean(results[name]["recall"])),
            "f1": float(np.mean(results[name]["f1"])),
            "f1_std": float(np.std(results[name]["f1"])),
            "time_ms": float(np.mean(results[name]["time"])),
        }

    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ")
    print("=" * 60)

    df = pd.DataFrame(summary).T
    df.index.name = "method"
    print(df.round(4))

    os.makedirs("results", exist_ok=True)
    df.to_csv("results/comparison_results.csv", encoding="utf-8-sig")

    if not no_plot:
        plot_comparison(summary, save_path="results/comparison_plot.png")

    return summary
