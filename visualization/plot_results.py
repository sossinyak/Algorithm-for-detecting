"""
Построение графиков для сводок экспериментов.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(results: dict, save_path: str | None = None) -> None:
    """
    Построить сравнение методов по F1-score.
    """
    methods = list(results.keys())
    f1_scores = [results[m]["f1"] for m in methods]

    x = np.arange(len(methods))

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(x, f1_scores, width=0.55, label="F1-score", color="steelblue")

    ax.set_ylabel("Значение F1-score")
    ax.set_title("Сравнение методов обнаружения изменений")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)

    for rect in bars:
        height = rect.get_height()
        ax.annotate(
            f"{height:.3f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
