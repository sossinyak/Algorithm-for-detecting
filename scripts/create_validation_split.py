"""
Create a reproducible validation split from the train partition of a
LEVIR-like dataset.
"""

from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np


SPLITS = ("train", "val", "test")
SUBDIRS = ("A", "B", "label")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a validation split from train for a LEVIR-like dataset."
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="./data/LEVIR-CD",
        help="Path to the source dataset root.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Destination dataset root. Defaults to <data_path>_with_val.",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Share of train samples to place into validation.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned split statistics without copying files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_path if it already exists.",
    )
    return parser.parse_args()


def ensure_structure(root: Path) -> None:
    for split in SPLITS:
        for subdir in SUBDIRS:
            path = root / split / subdir
            if not path.exists():
                raise FileNotFoundError(f"Missing required directory: {path}")


def list_png_names(directory: Path) -> List[str]:
    return sorted(path.name for path in directory.iterdir() if path.suffix.lower() == ".png")


def assert_partition_consistency(root: Path, split: str) -> List[str]:
    lists = {subdir: list_png_names(root / split / subdir) for subdir in SUBDIRS}
    names = lists["A"]
    for subdir, cur in lists.items():
        if cur != names:
            raise ValueError(
                f"Inconsistent file lists in {root / split}: directory '{subdir}' does not match 'A'."
            )
    return names


def count_change_pixels(mask_path: Path) -> int:
    data = mask_path.read_bytes()
    mask = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Failed to read mask: {mask_path}")
    return int((mask > 127).sum())


def bucketize(change_pixels: int) -> str:
    if change_pixels == 0:
        return "empty"
    if change_pixels < 100:
        return "tiny"
    if change_pixels < 1000:
        return "small"
    if change_pixels < 5000:
        return "medium"
    return "large"


def build_train_metadata(dataset_root: Path) -> List[Dict[str, object]]:
    names = assert_partition_consistency(dataset_root, "train")
    metadata: List[Dict[str, object]] = []
    for name in names:
        change_pixels = count_change_pixels(dataset_root / "train" / "label" / name)
        metadata.append(
            {
                "name": name,
                "change_pixels": change_pixels,
                "bucket": bucketize(change_pixels),
            }
        )
    return metadata


def choose_validation_samples(
    items: List[Dict[str, object]], val_ratio: float, seed: int
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    if not 0.0 < val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    rng = random.Random(seed)
    grouped: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for item in items:
        grouped[str(item["bucket"])].append(item)

    train_items: List[Dict[str, object]] = []
    val_items: List[Dict[str, object]] = []

    for bucket, bucket_items in sorted(grouped.items()):
        rng.shuffle(bucket_items)
        proposed = round(len(bucket_items) * val_ratio)
        if len(bucket_items) > 1:
            val_count = min(max(1, proposed), len(bucket_items) - 1)
        else:
            val_count = 0
        val_items.extend(bucket_items[:val_count])
        train_items.extend(bucket_items[val_count:])

    train_items.sort(key=lambda item: str(item["name"]))
    val_items.sort(key=lambda item: str(item["name"]))
    return train_items, val_items


def summarize(items: Iterable[Dict[str, object]]) -> Dict[str, object]:
    rows = list(items)
    bucket_counts = Counter(str(row["bucket"]) for row in rows)
    return {
        "count": len(rows),
        "bucket_counts": dict(sorted(bucket_counts.items())),
    }


def print_summary(title: str, summary: Dict[str, object]) -> None:
    print(title)
    print(f"  count: {summary['count']}")
    for bucket, count in summary["bucket_counts"].items():
        print(f"  {bucket}: {count}")


def resolve_output_path(data_path: Path, output_path: str | None) -> Path:
    if output_path:
        return Path(output_path)
    return data_path.parent / f"{data_path.name}_with_val"


def validate_source_val_is_empty(dataset_root: Path) -> None:
    existing = sum(len(list((dataset_root / "val" / subdir).glob("*.png"))) for subdir in SUBDIRS)
    if existing > 0:
        raise ValueError(
            f"Source val split is not empty ({existing} files found). "
            "This script assumes val should be created from train."
        )


def prepare_output_root(output_root: Path, overwrite: bool) -> None:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output path already exists: {output_root}. "
                "Use --overwrite to recreate it."
            )
        shutil.rmtree(output_root)
    for split in SPLITS:
        for subdir in SUBDIRS:
            (output_root / split / subdir).mkdir(parents=True, exist_ok=True)


def copy_partition(
    source_root: Path,
    output_root: Path,
    source_split: str,
    target_split: str,
    names: Iterable[str],
) -> None:
    for name in names:
        for subdir in SUBDIRS:
            src = source_root / source_split / subdir / name
            dst = output_root / target_split / subdir / name
            shutil.copy2(src, dst)


def write_manifest(
    output_root: Path,
    train_items: List[Dict[str, object]],
    val_items: List[Dict[str, object]],
    source_data_path: Path,
    val_ratio: float,
    seed: int,
) -> None:
    manifest_path = output_root / "split_manifest.csv"
    rows = []
    for subset, items in (("train", train_items), ("val", val_items)):
        for item in items:
            rows.append(
                {
                    "subset": subset,
                    "filename": item["name"],
                    "change_pixels": item["change_pixels"],
                    "bucket": item["bucket"],
                    "source_data_path": str(source_data_path),
                    "val_ratio": val_ratio,
                    "seed": seed,
                }
            )

    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "subset",
                "filename",
                "change_pixels",
                "bucket",
                "source_data_path",
                "val_ratio",
                "seed",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    data_path = Path(args.data_path).resolve()
    output_path = resolve_output_path(data_path, args.output_path).resolve()

    ensure_structure(data_path)
    validate_source_val_is_empty(data_path)

    metadata = build_train_metadata(data_path)
    train_items, val_items = choose_validation_samples(metadata, args.val_ratio, args.seed)
    test_names = assert_partition_consistency(data_path, "test")

    print("=" * 72)
    print("CREATE VALIDATION SPLIT")
    print("=" * 72)
    print(f"source: {data_path}")
    print(f"output: {output_path}")
    print(f"val_ratio: {args.val_ratio}")
    print(f"seed: {args.seed}")
    print(f"dry_run: {args.dry_run}")
    print("-" * 72)
    print_summary("full train summary:", summarize(metadata))
    print_summary("new train summary:", summarize(train_items))
    print_summary("new val summary:", summarize(val_items))
    print(f"test count (copied unchanged): {len(test_names)}")
    print("=" * 72)

    if args.dry_run:
        print("Dry-run complete. No files were copied.")
        return

    prepare_output_root(output_path, args.overwrite)
    copy_partition(
        data_path, output_path, "train", "train", [str(item["name"]) for item in train_items]
    )
    copy_partition(
        data_path, output_path, "train", "val", [str(item["name"]) for item in val_items]
    )
    copy_partition(data_path, output_path, "test", "test", test_names)

    # Destination train already contains only train_items.
    write_manifest(output_path, train_items, val_items, data_path, args.val_ratio, args.seed)

    print(f"Split created successfully: {output_path}")
    print(f"Manifest: {output_path / 'split_manifest.csv'}")


if __name__ == "__main__":
    main()
