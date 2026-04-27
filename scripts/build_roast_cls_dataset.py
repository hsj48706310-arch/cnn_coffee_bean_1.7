from __future__ import annotations

import argparse
import csv
import os
import random
import stat
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import yaml


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a roasting-stage classification dataset from YOLO detection labels"
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets") / "Coffee Defect.v1-coffee-bean_yolo26n.yolo26",
        help="Path to YOLO dataset root (contains train/valid/test and data.yaml)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts") / "roast_cls_dataset",
        help="Output root for classification dataset",
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        default=["roasted-beans", "under_roast"],
        help="Class names to include in classification dataset",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=0.05,
        help="Padding ratio added around each bbox (relative to max(box_w, box_h))",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=16,
        help="Skip crops smaller than this pixel size",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove output directory before generating dataset",
    )
    parser.add_argument(
        "--max-label-files-per-split",
        type=int,
        default=0,
        help="Process at most this many label files per split (0 means all)",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=500,
        help="Print progress every N label files",
    )
    parser.add_argument(
        "--shuffle-label-files",
        action="store_true",
        help="Shuffle label files before applying max-label-files-per-split",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffled sampling",
    )
    return parser


def load_class_map(data_yaml: Path) -> dict[str, int]:
    payload = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = payload.get("names", [])
    if isinstance(names, dict):
        names = [names[idx] for idx in sorted(names)]
    return {name: idx for idx, name in enumerate(names)}


def yolo_to_xyxy(xc: float, yc: float, bw: float, bh: float, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = int((xc - bw / 2.0) * w)
    y1 = int((yc - bh / 2.0) * h)
    x2 = int((xc + bw / 2.0) * w)
    y2 = int((yc + bh / 2.0) * h)
    return x1, y1, x2, y2


def clamp_bbox(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[int, int, int, int]:
    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(1, min(x2, w))
    y2 = max(1, min(y2, h))
    return x1, y1, x2, y2


def image_path_for_label(labels_dir: Path, images_dir: Path, label_file: Path) -> Path | None:
    stem = label_file.stem
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = images_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def remove_dir_with_retry(path: Path, retries: int = 3) -> None:
    def onerror(func, value, exc_info):
        if not os.access(value, os.W_OK):
            os.chmod(value, stat.S_IWUSR)
            func(value)
            return
        raise exc_info[1]

    for attempt in range(1, retries + 1):
        try:
            shutil.rmtree(path, onerror=onerror)
            return
        except OSError:
            if attempt == retries:
                raise
            time.sleep(0.3)


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()

    data_yaml = dataset_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found: {data_yaml}")

    class_map = load_class_map(data_yaml)
    missing = [name for name in args.classes if name not in class_map]
    if missing:
        raise ValueError(f"Class names not found in data.yaml: {missing}")

    target_ids = {class_map[name]: name for name in args.classes}

    if output_root.exists() and args.overwrite:
        try:
            remove_dir_with_retry(output_root)
        except OSError as e:
            raise RuntimeError(
                f"Failed to clear output directory '{output_root}'. "
                "Another process may still be writing files there. "
                "Stop running dataset builders and retry."
            ) from e

    split_map = {"train": "train", "valid": "val", "test": "test"}
    generated_counts: dict[str, Counter[str]] = defaultdict(Counter)
    manifest_rows: list[list[str]] = []

    print(f"[INFO] dataset_root={dataset_root}")
    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] target_classes={args.classes}")

    for src_split, dst_split in split_map.items():
        labels_dir = dataset_root / src_split / "labels"
        images_dir = dataset_root / src_split / "images"

        if not labels_dir.exists() or not images_dir.exists():
            continue

        for class_name in args.classes:
            (output_root / dst_split / class_name).mkdir(parents=True, exist_ok=True)

        label_files = sorted(labels_dir.glob("*.txt"))
        if args.shuffle_label_files:
            rng = random.Random(args.seed)
            rng.shuffle(label_files)
        if args.max_label_files_per_split > 0:
            label_files = label_files[: args.max_label_files_per_split]

        print(f"[INFO] split={src_split} label_files={len(label_files)}")

        for file_idx, label_file in enumerate(label_files, start=1):
            if args.progress_every > 0 and file_idx % args.progress_every == 0:
                current_total = sum(generated_counts[dst_split].values())
                print(f"[INFO] split={dst_split} processed={file_idx}/{len(label_files)} crops={current_total}")

            lines = label_file.read_text(encoding="utf-8", errors="ignore").splitlines()
            target_lines: list[tuple[int, list[float], int]] = []
            for ann_idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                cls_id = int(float(parts[0]))
                if cls_id not in target_ids:
                    continue
                target_lines.append((cls_id, list(map(float, parts[1:])), ann_idx))

            if not target_lines:
                continue

            image_path = image_path_for_label(labels_dir, images_dir, label_file)
            if image_path is None:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            h, w = image.shape[:2]
            written_per_image = 0

            for cls_id, coords, ann_idx in target_lines:
                xc, yc, bw, bh = coords
                x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, w, h)

                pad = int(max((x2 - x1), (y2 - y1)) * args.padding)
                x1, y1, x2, y2 = clamp_bbox(x1 - pad, y1 - pad, x2 + pad, y2 + pad, w, h)

                if (x2 - x1) < args.min_size or (y2 - y1) < args.min_size:
                    continue

                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                class_name = target_ids[cls_id]
                out_dir = output_root / dst_split / class_name
                out_name = f"{image_path.stem}_{ann_idx:03d}_{written_per_image:02d}.jpg"
                out_path = out_dir / out_name

                # Ensure uniqueness in case of repeated names.
                suffix = 1
                while out_path.exists():
                    out_name = f"{image_path.stem}_{ann_idx:03d}_{written_per_image:02d}_{suffix}.jpg"
                    out_path = out_dir / out_name
                    suffix += 1

                ok = cv2.imwrite(str(out_path), crop)
                if not ok:
                    continue

                generated_counts[dst_split][class_name] += 1
                manifest_rows.append(
                    [
                        dst_split,
                        class_name,
                        out_path.relative_to(output_root).as_posix(),
                        image_path.relative_to(dataset_root).as_posix(),
                        label_file.relative_to(dataset_root).as_posix(),
                        str(x1),
                        str(y1),
                        str(x2),
                        str(y2),
                    ]
                )
                written_per_image += 1

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "split",
                "class_name",
                "crop_path",
                "source_image",
                "source_label",
                "x1",
                "y1",
                "x2",
                "y2",
            ]
        )
        writer.writerows(manifest_rows)

    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] manifest={manifest_path}")
    for split_name in ("train", "val", "test"):
        split_total = sum(generated_counts[split_name].values())
        print(f"[INFO] split={split_name} total_crops={split_total}")
        for class_name in args.classes:
            print(f"  - {class_name}: {generated_counts[split_name][class_name]}")


if __name__ == "__main__":
    main()
