"""
Парные статистические тесты для сравнения методов по F1-score.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def _cohens_d_paired(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    std = np.std(diff, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(diff) / std)


def run_statistical_tests(
    per_sample_path: str = "results/ablation_study_per_sample.csv",
    out_path: str = "results/statistical_tests.csv",
    reference_method: str = "Итоговый алгоритм",
) -> pd.DataFrame:
    if not os.path.exists(per_sample_path):
        raise FileNotFoundError(
            f"Не найден файл {per_sample_path}. Сначала запустите ablation study."
        )

    df = pd.read_csv(per_sample_path)
    reference = df[df["method"] == reference_method][["sample", "f1"]].rename(
        columns={"f1": "f1_ref"}
    )
    if reference.empty:
        raise ValueError(f"В {per_sample_path} не найден reference_method='{reference_method}'.")

    rows = []
    for method in sorted(df["method"].unique()):
        if method == reference_method:
            continue
        cur = df[df["method"] == method][["sample", "f1"]].rename(
            columns={"f1": "f1_method"}
        )
        merged = reference.merge(cur, on="sample", how="inner")
        if len(merged) < 2:
            continue

        f1_ref = merged["f1_ref"].to_numpy()
        f1_method = merged["f1_method"].to_numpy()

        try:
            f1_stat, f1_p = wilcoxon(f1_ref, f1_method, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            f1_stat, f1_p = 0.0, 1.0

        rows.append({
            "reference": reference_method,
            "method": method,
            "n": len(merged),
            "mean_f1_reference": float(np.mean(f1_ref)),
            "mean_f1_method": float(np.mean(f1_method)),
            "delta_f1": float(np.mean(f1_ref - f1_method)),
            "wilcoxon_f1_stat": float(f1_stat),
            "wilcoxon_f1_p": float(f1_p),
            "paired_cohens_d_f1": _cohens_d_paired(f1_ref, f1_method),
        })

    result = pd.DataFrame(rows).sort_values("wilcoxon_f1_p")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    return result


if __name__ == "__main__":
    run_statistical_tests()
