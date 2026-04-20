from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from analysis.metrics import calculate_metrics
from pipelines.adaptive_algorithm import AdaptiveChangeDetection
from utils.pipeline_config import build_adaptive_params, load_configured_pairs


def load_config(path: str | Path = ROOT / "config.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_algorithm(config: dict) -> AdaptiveChangeDetection:
    return AdaptiveChangeDetection(**build_adaptive_params(config))


def to_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def overlay_mask(image: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (255, 80, 80)) -> np.ndarray:
    base = to_rgb(image).copy()
    binary = mask > 127
    if not np.any(binary):
        return base

    overlay = base.copy()
    overlay[binary] = color
    return cv2.addWeighted(base, 0.65, overlay, 0.35, 0)


def describe_errors(pred_mask: np.ndarray, label: np.ndarray, score_map: np.ndarray) -> dict:
    pred = pred_mask > 127
    true = label > 127
    fp = pred & ~true
    fn = ~pred & true

    pred_pixels = int(pred.sum())
    true_pixels = int(true.sum())

    fp_ratio = float(fp.sum() / max(pred_pixels, 1))
    fn_ratio = float(fn.sum() / max(true_pixels, 1))
    fp_mean = float(score_map[fp].mean()) if np.any(fp) else 0.0
    fn_mean = float(score_map[fn].mean()) if np.any(fn) else 0.0

    if fp_ratio >= 0.7 and fn_ratio <= 0.4:
        error_type = "много ложных срабатываний: вероятны тени, дороги или сезонная текстура"
    elif fn_ratio >= 0.7 and fp_ratio <= 0.4:
        error_type = "много пропусков: слабый контраст или слишком жесткая бинаризация"
    elif fp_ratio >= 0.5 and fn_ratio >= 0.5:
        error_type = "смешанные ошибки границ и семантики"
    else:
        error_type = "ошибки относительно сбалансированы"

    return {
        "fp_ratio_in_prediction": fp_ratio,
        "fn_ratio_in_label": fn_ratio,
        "fp_diff_mean": fp_mean,
        "fn_diff_mean": fn_mean,
        "error_type": error_type,
    }


def save_report_figure(
    pair: dict,
    prediction: np.ndarray,
    label: np.ndarray,
    intermediate: dict,
    metrics: dict,
    output_path: Path,
) -> None:
    diff_map = intermediate["diff_map"]
    edge_map = intermediate.get("edge_map")
    binary_mask = intermediate["binary_mask"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    fig.suptitle(
        f"{pair['name']} | F1={metrics['f1']:.3f}, "
        f"Precision={metrics['precision']:.3f}, Recall={metrics['recall']:.3f}",
        fontsize=12,
    )

    panels = [
        ("T1", to_rgb(pair["img_a"]), "image"),
        ("T2", to_rgb(pair["img_b"]), "image"),
        ("Эталонная маска", label, "mask"),
        ("Предсказание поверх T2", overlay_mask(pair["img_b"], prediction), "image"),
        ("Карта различий", diff_map, "heat"),
        ("Карта контуров", edge_map if edge_map is not None else np.zeros_like(label), "mask"),
        ("Бинарная маска", binary_mask, "mask"),
        ("Итоговая маска", prediction, "mask"),
    ]

    for ax, (title, data, kind) in zip(axes.flat, panels):
        if kind == "image":
            ax.imshow(data)
        elif kind == "heat":
            ax.imshow(data, cmap="magma")
        else:
            ax.imshow(data, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_visual_report(
    data_path: str,
    split: str = "test",
    max_samples: int = 5,
    output_dir: str | Path = ROOT / "results" / "visual_report",
    config: dict | None = None,
) -> pd.DataFrame:
    if config is None:
        config = load_config()

    config = dict(config)
    config.setdefault("data", {})["data_path"] = data_path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = [
        pair
        for pair in load_configured_pairs(config, split=split, max_pairs=max_samples)
        if pair.get("label") is not None
    ]
    if not pairs:
        raise ValueError("В выбранной выборке нет пар с эталонными масками label.")

    algorithm = build_algorithm(config)

    rows = []
    for index, pair in enumerate(pairs, start=1):
        algorithm.reset_intermediate_results()
        prediction = algorithm.process(pair["img_a"], pair["img_b"])
        intermediate = algorithm.get_intermediate_results()
        metrics = calculate_metrics(prediction, pair["label"])
        score_map = (intermediate["combined_map"] * 255).astype(np.uint8)
        error_info = describe_errors(prediction, pair["label"], score_map)

        save_report_figure(
            pair=pair,
            prediction=prediction,
            label=pair["label"],
            intermediate=intermediate,
            metrics=metrics,
            output_path=output_dir / f"visual_report_{index:02d}_{pair['name']}",
        )

        rows.append(
            {
                "sample": pair["name"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                **error_info,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "visual_report_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Сформировать визуальный отчет для классического алгоритма обнаружения изменений.")
    parser.add_argument("--data_path", default="./data/LEVIR-CD-filtered")
    parser.add_argument("--split", default="test")
    parser.add_argument("--max_samples", type=int, default=5)
    parser.add_argument("--config", default=str(ROOT / "config.yaml"))
    parser.add_argument("--output_dir", default=str(ROOT / "results" / "visual_report"))
    args = parser.parse_args()

    config = load_config(args.config)
    build_visual_report(
        data_path=args.data_path,
        split=args.split,
        max_samples=args.max_samples,
        output_dir=args.output_dir,
        config=config,
    )


if __name__ == "__main__":
    main()
