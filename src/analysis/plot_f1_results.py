"""Построение сравнительных графиков F1-score по методам."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CLASSICAL_11_METHODS = [
    "AbsDiff",
    "LogRatio",
    "NormDiff",
    "RGB-CVA",
    "LAB-CVA",
    "HSV-CVA",
    "CLAHE-CVA",
    "LocalMeanDiff",
    "EdgeDiff",
    "LaplacianDiff",
    "PCA-CVA",
]


def _prepare_classical_11(summary: pd.DataFrame) -> pd.DataFrame:
    """Оставляет только 11 классических методов в фиксированном порядке."""
    filtered = summary[summary["method"].isin(CLASSICAL_11_METHODS)].copy()
    filtered["method"] = pd.Categorical(filtered["method"], categories=CLASSICAL_11_METHODS, ordered=True)
    return filtered.sort_values(["dataset", "method"]).reset_index(drop=True)


def _plot_grouped_bars(summary: pd.DataFrame, methods: list[str], datasets: list[str], title: str, output_path: Path) -> None:
    """Строит grouped bar chart для выбранных методов и датасетов."""
    plot_df = summary[summary["dataset"].isin(datasets)].pivot(index="method", columns="dataset", values="f1")
    plot_df = plot_df.reindex(methods)
    if plot_df.empty:
        return

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(max(12, len(methods) * 0.72), 6.8))
    x = np.arange(len(plot_df.index))
    width = 0.72 / max(len(datasets), 1)
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    for index, dataset in enumerate(datasets):
        if dataset not in plot_df.columns:
            continue
        values = plot_df[dataset].values
        offset = (index - (len(datasets) - 1) / 2) * width
        bars = ax.bar(x + offset, values, width, label=dataset, color=colors[index % len(colors)])
        for bar, value in zip(bars, values):
            if not np.isnan(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.006,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    rotation=90,
                )

    ax.set_title(title)
    ax.set_ylabel("F1-score")
    ax.set_xlabel("Метод")
    max_value = float(np.nanmax(plot_df.values)) if np.isfinite(plot_df.values).any() else 1.0
    ax.set_ylim(0, min(1.08, max(1.0, max_value * 1.16)))
    ax.set_xticks(x)
    ax.set_xticklabels(plot_df.index, rotation=35, ha="right")
    ax.legend(title="Датасет")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_dataset_ranking(summary: pd.DataFrame, dataset: str, output_path: Path) -> None:
    """Строит отдельный рейтинг всех методов для одного датасета."""
    dataset_df = summary[summary["dataset"] == dataset].sort_values("f1", ascending=False).copy()
    if dataset_df.empty:
        return
    fig, ax = plt.subplots(figsize=(12, max(5, len(dataset_df) * 0.36)))
    colors = ["#16a34a" if idx == 0 else "#4b5563" for idx in range(len(dataset_df))]
    ax.barh(dataset_df["method"], dataset_df["f1"], color=colors)
    ax.invert_yaxis()
    ax.set_xlim(0, min(1.05, max(1.0, float(dataset_df["f1"].max()) * 1.15)))
    ax.set_xlabel("F1-score")
    ax.set_title(f"Рейтинг методов по F1: {dataset}")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    for index, value in enumerate(dataset_df["f1"]):
        ax.text(value + 0.006, index, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_f1_plots(summary_csv: Path, output_dir: Path) -> dict[str, str]:
    """Создает графики F1 и возвращает пути к артефактам."""
    summary = pd.read_csv(summary_csv)
    output_dir.mkdir(parents=True, exist_ok=True)

    classical = _prepare_classical_11(summary)
    classical_summary_csv = output_dir / "f1_11_methods_summary.csv"
    classical.to_csv(classical_summary_csv, index=False, encoding="utf-8-sig")

    artifacts: dict[str, str] = {"classical_summary_csv": str(classical_summary_csv.resolve())}
    synthetic_path = output_dir / "f1_11_methods_synthetic.png"
    real_path = output_dir / "f1_11_methods_real_datasets.png"
    _plot_grouped_bars(
        classical,
        CLASSICAL_11_METHODS,
        ["synthetic-lab"],
        "F1-score 11 классических методов на synthetic-lab",
        synthetic_path,
    )
    _plot_grouped_bars(
        classical,
        CLASSICAL_11_METHODS,
        ["JL1-CD", "LEVIR-CD-filtred"],
        "F1-score 11 классических методов на JL1-CD и LEVIR-CD-filtred",
        real_path,
    )
    artifacts["classical_synthetic_png"] = str(synthetic_path.resolve())
    artifacts["classical_real_png"] = str(real_path.resolve())

    all_methods = sorted(summary["method"].unique().tolist())
    all_methods_path = output_dir / "f1_all_methods_all_datasets.png"
    _plot_grouped_bars(
        summary,
        all_methods,
        sorted(summary["dataset"].unique().tolist()),
        "F1-score всех методов по датасетам",
        all_methods_path,
    )
    artifacts["all_methods_png"] = str(all_methods_path.resolve())

    for dataset in sorted(summary["dataset"].unique()):
        dataset_path = output_dir / f"f1_ranking_{dataset}.png"
        _plot_dataset_ranking(summary, dataset, dataset_path)
        artifacts[f"ranking_{dataset}"] = str(dataset_path.resolve())

    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Построить графики F1-score по методам.")
    parser.add_argument("--summary-csv", type=Path, default=Path("results/parameter_study/all_parameter_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/final_plots"))
    args = parser.parse_args()

    print(json.dumps(build_f1_plots(args.summary_csv, args.output_dir), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
