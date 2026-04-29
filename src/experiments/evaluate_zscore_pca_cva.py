"""Оценка отдельного пайплайна ZScorePCACVA на доступных датасетах.

Скрипт нужен для проверки идеи:
z-score по каналам -> PCA -> CVA -> percentile clipping -> Otsu -> morphology.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.metrics import calculate_metrics
from pipelines.adaptive_pipeline import ZScorePCACVA
from utils.data_loader import LEVIRCDLoader


def _has_split(dataset_path: Path, split: str) -> bool:
    """Проверяет наличие структуры split/A, split/B, split/label."""
    return all((dataset_path / split / folder).is_dir() for folder in ("A", "B", "label"))


def _load_pairs(dataset_path: Path, split: str, max_samples: int | None, seed: int) -> list[dict]:
    """Загружает размеченные пары и при необходимости берет воспроизводимую подвыборку."""
    pairs = [pair for pair in LEVIRCDLoader(str(dataset_path)).load_split(split=split) if pair.get("label") is not None]
    if max_samples is not None and max_samples > 0 and len(pairs) > max_samples:
        rng = random.Random(seed)
        indices = list(range(len(pairs)))
        rng.shuffle(indices)
        pairs = [pairs[index] for index in sorted(indices[:max_samples])]
    return pairs


def _summarize(rows: list[dict]) -> dict:
    """Собирает micro-метрики по суммарной матрице ошибок."""
    tp = sum(int(row["tp"]) for row in rows)
    tn = sum(int(row["tn"]) for row in rows)
    fp = sum(int(row["fp"]) for row in rows)
    fn = sum(int(row["fn"]) for row in rows)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "f1_std": float(np.std([row["f1"] for row in rows])) if rows else 0.0,
        "time_ms": float(np.mean([row["time_ms"] for row in rows])) if rows else 0.0,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "samples": int(len(rows)),
    }


def evaluate_dataset(dataset_path: Path, split: str, args: argparse.Namespace) -> tuple[dict, list[dict]]:
    """Считает метрики ZScorePCACVA на одном датасете."""
    pairs = _load_pairs(dataset_path, split=split, max_samples=args.max_samples, seed=args.seed)
    method = ZScorePCACVA(
        pca_components=args.pca_components,
        pca_variance_ratio=args.variance_ratio,
        min_components=args.min_components,
        max_components=args.max_components,
        clip_percentiles=(args.clip_low, args.clip_high),
        otsu_scale=args.otsu_scale,
        opening_kernel=args.opening_kernel,
        closing_kernel=args.closing_kernel,
    )

    rows = []
    for pair in pairs:
        start = time.perf_counter()
        prediction = method.process(pair["img_a"], pair["img_b"])
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics = calculate_metrics(prediction, pair["label"])
        rows.append(
            {
                "dataset": dataset_path.name,
                "split": split,
                "patch_name": pair["name"],
                "method": "ZScorePCACVA",
                "time_ms": elapsed_ms,
                "pred_positive_fraction": float(np.mean(prediction > 127)),
                "gt_positive_fraction": float(np.mean(pair["label"] > 127)),
                **metrics,
                "pca_components_used": method.get_intermediate_results().get("pca_components_used"),
            }
        )

    summary = {
        "dataset": dataset_path.name,
        "split": split,
        "method": "ZScorePCACVA",
        **_summarize(rows),
    }
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценить ZScorePCACVA на всех датасетах с выбранным split.")
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--results-dir", type=Path, default=Path("results/zscore_pca_cva"))
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pca-components", type=int, default=3)
    parser.add_argument("--variance-ratio", type=float, default=0.95)
    parser.add_argument("--min-components", type=int, default=2)
    parser.add_argument("--max-components", type=int, default=3)
    parser.add_argument("--clip-low", type=float, default=1.0)
    parser.add_argument("--clip-high", type=float, default=99.0)
    parser.add_argument("--otsu-scale", type=float, default=1.0)
    parser.add_argument("--opening-kernel", type=int, default=3)
    parser.add_argument("--closing-kernel", type=int, default=3)
    args = parser.parse_args()

    dataset_paths = [path for path in sorted(args.data_root.iterdir()) if path.is_dir() and _has_split(path, args.split)]
    if not dataset_paths:
        raise RuntimeError(f"В {args.data_root.resolve()} нет датасетов со split={args.split!r}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    per_patch_rows = []
    for dataset_path in dataset_paths:
        summary, rows = evaluate_dataset(dataset_path, split=args.split, args=args)
        summaries.append(summary)
        per_patch_rows.extend(rows)

    summary_path = args.results_dir / "zscore_pca_cva_summary.csv"
    per_patch_path = args.results_dir / "zscore_pca_cva_per_patch.csv"
    pd.DataFrame(summaries).to_csv(summary_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(per_patch_rows).to_csv(per_patch_path, index=False, encoding="utf-8-sig")
    print(
        json.dumps(
            {
                "summary_csv": str(summary_path.resolve()),
                "per_patch_csv": str(per_patch_path.resolve()),
                "summary": summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
