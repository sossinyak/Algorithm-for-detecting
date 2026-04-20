"""
Исследование параметров классического адаптивного алгоритма.
"""

from __future__ import annotations

import os
import time
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from analysis.metrics import calculate_metrics
from pipelines.adaptive_algorithm import AdaptiveChangeDetection
from utils.pipeline_config import build_adaptive_params, load_configured_pairs


def _base_params(config: dict) -> dict:
    return build_adaptive_params(config)


def _evaluate(pairs: List[dict], params: dict) -> dict:
    algo = AdaptiveChangeDetection(**params)
    values = {"precision": [], "recall": [], "f1": [], "time_ms": []}

    for pair in pairs:
        start = time.perf_counter()
        pred = algo.process(pair["img_a"], pair["img_b"])
        elapsed_ms = (time.perf_counter() - start) * 1000
        metrics = calculate_metrics(pred, pair["label"])

        for key in ["precision", "recall", "f1"]:
            values[key].append(metrics[key])
        values["time_ms"].append(elapsed_ms)

    return {key: float(np.mean(val)) for key, val in values.items()}


def _plot_parameter_results(df: pd.DataFrame, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for parameter, group in df.groupby("parameter"):
        x = group["value"].astype(str)
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        ax1.plot(x, group["f1"], marker="o", label="F1")
        ax1.set_xlabel(parameter)
        ax1.set_ylabel("Качество")
        ax1.grid(True, linestyle="--", alpha=0.35)
        ax1.legend(loc="upper left")

        ax2 = ax1.twinx()
        ax2.plot(x, group["time_ms"], marker="^", color="#C00000", label="Время, мс")
        ax2.set_ylabel("Время, мс")
        ax2.legend(loc="upper right")
        plt.title(f"Исследование параметра: {parameter}")
        plt.tight_layout()
        safe_name = parameter.replace("/", "_").replace(" ", "_")
        plt.savefig(os.path.join(out_dir, f"parameter_study_{safe_name}.png"), dpi=200)
        plt.close()


def run_parameter_study(config: dict) -> dict:
    experiment_cfg = config.get("experiments", {})
    study_cfg = experiment_cfg.get("parameter_study", {})
    max_samples = experiment_cfg.get("max_samples") or study_cfg.get("max_samples", 120)

    pairs = [
        pair
        for pair in load_configured_pairs(config, split="test", max_pairs=max_samples)
        if pair.get("label") is not None
    ]
    if not pairs:
        raise ValueError("В тестовой выборке нет пар с эталонными масками label.")

    base = _base_params(config)
    studies = {
        "filter_sigma": study_cfg.get("sigma_values", [None, 0.5, 1.0, 1.5, 2.0]),
        "clahe_clip_limit": study_cfg.get("cliplimit_values", [1.0, 1.5, 2.0, 2.5, 3.0]),
        "diff_weight": study_cfg.get("weight_values", [0.3, 0.5, 0.7, 0.9]),
        "color_weight": study_cfg.get("color_weight_values", [0.0, 0.1, 0.2, 0.3]),
        "edge_weight": study_cfg.get("edge_weight_values", [0.0, 0.1, 0.15, 0.25, 0.4]),
        "edge_detector": study_cfg.get("edge_detector_values", ["canny", "sobel", "both"]),
        "block_size": study_cfg.get("block_size_values", [15, 25, 35, 45, 55]),
        "threshold_C": study_cfg.get("c_values", [4, 8, 12, 16, 20]),
    }

    rows = []
    for parameter, values in studies.items():
        print(f"\nИсследование параметра: {parameter}")
        for value in tqdm(values, desc=parameter):
            params = dict(base)
            params[parameter] = value
            if parameter == "clahe_clip_limit":
                params["clahe_mode"] = "fixed"
            if parameter == "filter_sigma" and value == "auto":
                params[parameter] = None
            result = _evaluate(pairs, params)
            rows.append({"parameter": parameter, "value": str(value), **result})

    df = pd.DataFrame(rows)
    os.makedirs("results", exist_ok=True)
    df.to_csv("results/parameter_study.csv", index=False, encoding="utf-8-sig")
    _plot_parameter_results(df, "results/parameter_study_plots")

    best_rows = (
        df.sort_values(["parameter", "f1"], ascending=[True, False]).groupby("parameter").head(1)
    )
    best_rows.to_csv("results/parameter_study_best.csv", index=False, encoding="utf-8-sig")
    print("\nЛучшие значения по F1:")
    print(best_rows[["parameter", "value", "f1", "time_ms"]].to_string(index=False))

    return {"parameter_study": df, "best": best_rows}
