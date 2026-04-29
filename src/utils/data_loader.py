"""
Загрузка пар изображений формата LEVIR-CD.

Ожидаемая структура:

dataset_root/
    train|val|test/
        A/      снимки до изменения
        B/      снимки после изменения
        label/  эталонные маски изменений
"""

from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np

from utils.image_io import read_image


class LEVIRCDLoader:
    """Читает пары снимков и соответствующие эталонные маски."""

    def __init__(self, data_path: str):
        self.data_path = Path(data_path)

    def load_pair(
        self,
        a_path: Path,
        b_path: Path,
        label_path: Path | None = None,
    ) -> tuple[np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        """Загружает одну пару A/B и соответствующую маску label."""
        img_a = read_image(a_path, cv2.IMREAD_COLOR)
        img_b = read_image(b_path, cv2.IMREAD_COLOR)
        if img_a is None or img_b is None:
            return None, None, None

        label = None
        if label_path is not None and label_path.exists():
            label = read_image(label_path, cv2.IMREAD_GRAYSCALE)

        if img_b.shape[:2] != img_a.shape[:2]:
            img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_LINEAR)
        if label is not None and label.shape[:2] != img_a.shape[:2]:
            label = cv2.resize(label, (img_a.shape[1], img_a.shape[0]), interpolation=cv2.INTER_NEAREST)

        return img_a, img_b, label

    def load_split(self, split: str = "test", max_pairs: int | None = None) -> list[dict]:
        """Возвращает список пар для выбранной части датасета."""
        split_path = self.data_path / split
        a_dir = split_path / "A"
        b_dir = split_path / "B"
        label_dir = split_path / "label"

        if not a_dir.is_dir() or not b_dir.is_dir():
            return []

        a_files = sorted(path.name for path in a_dir.glob("*.png"))
        if max_pairs is not None:
            a_files = a_files[:max_pairs]

        pairs: list[dict] = []
        for fname in a_files:
            a_path = a_dir / fname
            b_path = b_dir / fname
            label_path = label_dir / fname if label_dir.is_dir() else None

            img_a, img_b, label = self.load_pair(a_path, b_path, label_path)
            if img_a is None or img_b is None:
                continue

            pairs.append(
                {
                    "img_a": img_a,
                    "img_b": img_b,
                    "label": label,
                    "name": fname,
                }
            )

        return pairs
