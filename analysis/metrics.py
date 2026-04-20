"""
Пиксельные метрики для бинарных масок изменений.
"""

from __future__ import annotations

import numpy as np


def calculate_metrics(pred_mask: np.ndarray, true_mask: np.ndarray) -> dict:
    """Рассчитать precision, recall, F1, accuracy и элементы матрицы ошибок."""
    pred = (pred_mask > 127).astype(np.uint8)
    true = true_mask.astype(np.uint8) if true_mask.max() <= 1 else (true_mask > 127).astype(np.uint8)

    tp = np.sum((pred == 1) & (true == 1))
    tn = np.sum((pred == 0) & (true == 0))
    fp = np.sum((pred == 1) & (true == 0))
    fn = np.sum((pred == 0) & (true == 1))

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
