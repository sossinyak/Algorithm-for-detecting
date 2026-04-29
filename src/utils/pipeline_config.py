"""Преобразование YAML-конфига в параметры объектов программы."""

from __future__ import annotations

from typing import Dict

from utils.data_loader import LEVIRCDLoader


def build_adaptive_params(config: dict) -> Dict[str, object]:
    """Собирает параметры итогового адаптивного алгоритма."""
    adaptive_cfg = config.get("adaptive_algorithm", {})
    threshold = adaptive_cfg.get("threshold", {})
    post = adaptive_cfg.get("postprocessing", {})

    return {
        "patch_size": adaptive_cfg.get("patch_size", 1),
        "pca_components": adaptive_cfg.get("components", 3),
        "pca_variance_ratio": adaptive_cfg.get("variance_ratio"),
        "whitening": adaptive_cfg.get("whitening", True),
        "threshold_value": threshold.get("value"),
        "otsu_scale": threshold.get("otsu_scale", 0.85),
        "median_kernel": post.get("median_kernel", 3),
        "opening_kernel": post.get("opening_kernel", 3),
        "closing_kernel": post.get("closing_kernel", 3),
        "min_area": post.get("min_area", 100),
        "fill_holes": post.get("fill_holes", False),
    }


def load_configured_pairs(config: dict, split: str = "test", max_pairs: int | None = None) -> list[dict]:
    """Загружает пары снимков по пути из config.yaml."""
    data_cfg = config.get("data", {})
    loader = LEVIRCDLoader(data_cfg.get("data_path", "./data/LEVIR-CD-filtred"))
    return loader.load_split(split=split, max_pairs=max_pairs)
