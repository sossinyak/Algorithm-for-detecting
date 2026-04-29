"""Единый запуск исследовательского протокола."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd

SRC_ROOT = Path(__file__).resolve().parent / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from analysis.experiment_logger import ExperimentLogger


def _run(command: list[str], logger: ExperimentLogger | None = None, step: str | None = None) -> None:
    """Печатает команду и запускает ее как отдельный этап протокола."""
    print(json.dumps({"run": command}, ensure_ascii=False), flush=True)
    started = time.perf_counter()
    if logger:
        logger.log_event("stage_started", {"step": step, "command": command})
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        if logger:
            logger.log_event(
                "stage_failed",
                {
                    "step": step,
                    "command": command,
                    "returncode": error.returncode,
                    "duration_sec": round(time.perf_counter() - started, 3),
                },
            )
        raise
    if logger:
        logger.log_metrics({"duration_sec": round(time.perf_counter() - started, 3)}, step=step)
        logger.log_event(
            "stage_finished",
            {"step": step, "command": command, "duration_sec": round(time.perf_counter() - started, 3)},
        )


def _best_methods(summary_csv: Path) -> list[dict]:
    """Возвращает лучший метод для каждого датасета по F1."""
    if not summary_csv.exists():
        return []
    df = pd.read_csv(summary_csv)
    if df.empty:
        return []
    best = df.sort_values(["dataset", "f1", "precision"], ascending=[True, False, False]).groupby("dataset").head(1)
    return best[["dataset", "method"]].to_dict("records")


def main() -> None:
    parser = argparse.ArgumentParser(description="Запустить полный исследовательский протокол.")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--split-root", type=Path, default=Path("data"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--max-tune-samples", type=int, default=24)
    parser.add_argument("--max-eval-samples", type=int, default=80)
    parser.add_argument("--monte-carlo-trials", type=int, default=48)
    parser.add_argument("--error-samples", type=int, default=30)
    parser.add_argument("--stage-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-split", action="store_true")
    args = parser.parse_args()

    logger = ExperimentLogger(
        args.results_dir / "runs",
        "research_protocol",
        tags={"stage": "run_all_experiments", "results_dir": args.results_dir},
    )
    logger.log_params({"args": vars(args)})

    python = sys.executable
    try:
        if not args.skip_split:
            split_command = [
                python,
                str(SRC_ROOT / "tools" / "data" / "split_datasets.py"),
                "--data-root",
                str(args.data_root),
                "--output-root",
                str(args.split_root),
                "--seed",
                str(args.seed),
            ]
            if args.split_root.resolve() == args.data_root.resolve():
                split_command.append("--in-place")
            _run(split_command, logger=logger, step="split_datasets")

        study_dir = args.results_dir / "parameter_study"
        _run(
            [
                python,
                str(SRC_ROOT / "analysis" / "parameter_analyzer.py"),
                "--config",
                str(args.config),
                "--data-root",
                str(args.split_root),
                "--results-dir",
                str(study_dir),
                "--max-tune-samples",
                str(args.max_tune_samples),
                "--max-eval-samples",
                str(args.max_eval_samples),
                "--monte-carlo-trials",
                str(args.monte_carlo_trials),
                "--seed",
                str(args.seed),
            ],
            logger=logger,
            step="parameter_study",
        )

        pair_metrics_csv = study_dir / "all_pair_metrics.csv"
        summary_csv = study_dir / "all_parameter_summary.csv"
        best_methods = _best_methods(summary_csv)
        logger.log_metrics({"best_methods": len(best_methods)}, step="select_best_methods")
        for item in best_methods:
            output_csv = args.results_dir / f"statistical_validation_{item['dataset']}.csv"
            _run(
                [
                    python,
                    str(SRC_ROOT / "analysis" / "statistical_validation.py"),
                    "--pair-metrics-csv",
                    str(pair_metrics_csv),
                    "--baseline",
                    "AbsDiff",
                    "--candidate",
                    item["method"],
                    "--dataset",
                    item["dataset"],
                    "--output-csv",
                    str(output_csv),
                    "--seed",
                    str(args.seed),
                ],
                logger=logger,
                step=f"statistical_validation:{item['dataset']}",
            )
            logger.log_artifact(output_csv)

        error_dir = args.results_dir / "error_analysis"
        for item in best_methods:
            dataset_path = args.split_root / item["dataset"]
            if not dataset_path.exists():
                continue
            _run(
                [
                    python,
                    str(SRC_ROOT / "analysis" / "error_analysis.py"),
                    "--data-path",
                    str(dataset_path),
                    "--method",
                    item["method"],
                    "--config",
                    str(args.config),
                    "--split",
                    "test",
                    "--max-samples",
                    str(args.error_samples),
                    "--results-dir",
                    str(error_dir),
                ],
                logger=logger,
                step=f"error_analysis:{item['dataset']}",
            )

        final_plots_dir = args.results_dir / "final_plots"
        _run(
            [
                python,
                str(SRC_ROOT / "analysis" / "plot_f1_results.py"),
                "--summary-csv",
                str(summary_csv),
                "--output-dir",
                str(final_plots_dir),
            ],
            logger=logger,
            step="plot_f1_results",
        )

        stage_root = args.results_dir / "stage_visualization"
        for item in best_methods:
            dataset_path = args.split_root / item["dataset"]
            if not dataset_path.exists():
                continue
            _run(
                [
                    python,
                    str(SRC_ROOT / "tools" / "visualization" / "visualize_adaptive_stages.py"),
                    "--config",
                    str(args.config),
                    "--data_path",
                    str(dataset_path),
                    "--split",
                    "test",
                    "--selection",
                    "representative",
                    "--count",
                    str(args.stage_samples),
                    "--output-dir",
                    str(stage_root / item["dataset"]),
                ],
                logger=logger,
                step=f"stage_visualization:{item['dataset']}",
            )

        report_path = args.results_dir / "research_report.html"
        _run(
            [
                python,
                str(SRC_ROOT / "analysis" / "generate_report.py"),
                "--results-dir",
                str(args.results_dir),
                "--output-html",
                str(report_path),
            ],
            logger=logger,
            step="generate_report",
        )
        logger.log_artifacts([summary_csv, pair_metrics_csv, study_dir / "best_params.yaml", final_plots_dir, stage_root, report_path])
        logger.finish()
    except Exception as error:
        logger.finish(status="failed", error=str(error))
        raise


if __name__ == "__main__":
    main()
