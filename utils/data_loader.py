"""
Data loading helpers for LEVIR-like change-detection datasets.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


class LEVIRCDLoader:
    """
    Loader for datasets with the following split structure:

    dataset_root/
        train|val|test/
            A/
            B/
            label/
    """

    def __init__(self, data_path: str, img_size: int = 256):
        self.data_path = data_path
        self.img_size = img_size

    def load_pair(
        self,
        a_path: str,
        b_path: str,
        label_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        img_a = cv2.imread(a_path)
        img_b = cv2.imread(b_path)

        if img_a is None or img_b is None:
            raise ValueError(f"Failed to load input images: {a_path}, {b_path}")

        target_size = (self.img_size, self.img_size)
        if img_a.shape[:2] != target_size:
            img_a = cv2.resize(img_a, target_size, interpolation=cv2.INTER_LINEAR)
        if img_b.shape[:2] != target_size:
            img_b = cv2.resize(img_b, target_size, interpolation=cv2.INTER_LINEAR)

        label = None
        if label_path and os.path.exists(label_path):
            label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
            if label is None:
                raise ValueError(f"Failed to load label: {label_path}")
            if label.shape[:2] != target_size:
                # Use nearest-neighbor to keep the mask binary after resize.
                label = cv2.resize(label, target_size, interpolation=cv2.INTER_NEAREST)

        return img_a, img_b, label

    def load_split(self, split: str = "test", max_pairs: Optional[int] = None) -> List[Dict]:
        split_path = os.path.join(self.data_path, split)
        a_dir = os.path.join(split_path, "A")
        b_dir = os.path.join(split_path, "B")
        label_dir = os.path.join(split_path, "label")

        if not os.path.isdir(a_dir) or not os.path.isdir(b_dir):
            return []

        a_files = sorted(f for f in os.listdir(a_dir) if f.endswith(".png"))
        if max_pairs is not None:
            a_files = a_files[:max_pairs]

        pairs: List[Dict] = []
        for fname in a_files:
            a_path = os.path.join(a_dir, fname)
            b_path = os.path.join(b_dir, fname)
            label_path = os.path.join(label_dir, fname) if os.path.exists(label_dir) else None

            img_a, img_b, label = self.load_pair(a_path, b_path, label_path)
            pairs.append(
                {
                    "img_a": img_a,
                    "img_b": img_b,
                    "label": label,
                    "name": fname,
                }
            )

        return pairs

    def get_industrial_subset(self, split: str = "test", max_pairs: Optional[int] = None) -> List[Dict]:
        split_path = os.path.join(self.data_path, split)
        a_dir = os.path.join(split_path, "A")
        b_dir = os.path.join(split_path, "B")
        label_dir = os.path.join(split_path, "label")

        if not os.path.isdir(a_dir) or not os.path.isdir(b_dir):
            return []

        a_files = sorted(f for f in os.listdir(a_dir) if f.endswith(".png"))
        industrial_keywords = ["industrial", "factory", "warehouse", "construction"]

        selected_files = [
            fname for fname in a_files if any(keyword in fname.lower() for keyword in industrial_keywords)
        ]

        if len(selected_files) == 0:
            print("Industrial subset was not detected by filename heuristic. Falling back to all samples.")
            selected_files = a_files

        if max_pairs is not None:
            selected_files = selected_files[:max_pairs]

        pairs: List[Dict] = []
        for fname in selected_files:
            a_path = os.path.join(a_dir, fname)
            b_path = os.path.join(b_dir, fname)
            label_path = os.path.join(label_dir, fname) if os.path.exists(label_dir) else None
            img_a, img_b, label = self.load_pair(a_path, b_path, label_path)
            pairs.append(
                {
                    "img_a": img_a,
                    "img_b": img_b,
                    "label": label,
                    "name": fname,
                }
            )

        return pairs

    def load_subset(
        self,
        split: str = "test",
        subset: str = "all",
        max_pairs: Optional[int] = None,
    ) -> List[Dict]:
        if subset.lower() == "industrial":
            return self.get_industrial_subset(split=split, max_pairs=max_pairs)
        return self.load_split(split=split, max_pairs=max_pairs)
