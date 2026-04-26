from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class ClassSpec:
    class_id: int
    class_name: str


CLASS_SPECS = [
    ClassSpec(class_id=3, class_name="roasted_beans"),
    ClassSpec(class_id=5, class_name="under_roast"),
]

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build a roast-level classification dataset from YOLO detection labels."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("datasets") / "Coffee Defect.v1-coffee-bean_yolo26n.yolo26",
        help="Path to source YOLO detection dataset root.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("datasets") / "roast_cls",
        help="Path to output classification dataset root.",
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=16,
        help="Minimum crop width/height in pixels. Smaller boxes are skipped.",
    )
    return parser


def find_image(images_dir: Path, stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = images_dir / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def xywhn_to_xyxy(
    x_center: float,
    y_center: float,
    width: float,
    height: float,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    x1 = int((x_center - width / 2.0) * image_width)
    y1 = int((y_center - height / 2.0) * image_height)
    x2 = int((x_center + width / 2.0) * image_width)
    y2 = int((y_center + height / 2.0) * image_height)

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width))
    y2 = max(0, min(y2, image_height))
    return x1, y1, x2, y2


def main() -> None:
    args = build_parser().parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    class_lookup = {spec.class_id: spec.class_name for spec in CLASS_SPECS}

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")

    summary: dict[str, dict[str, int]] = {}

    for split in ("train", "valid", "test"):
        images_dir = dataset_root / split / "images"
        labels_dir = dataset_root / split / "labels"

        if not images_dir.exists() or not labels_dir.exists():
            raise FileNotFoundError(f"Missing split directories for '{split}'")

        split_counts = {spec.class_name: 0 for spec in CLASS_SPECS}
        summary[split] = split_counts

        for label_path in sorted(labels_dir.glob("*.txt")):
            image_path = find_image(images_dir, label_path.stem)
            if image_path is None:
                continue

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            image_height, image_width = image.shape[:2]
            lines = label_path.read_text(encoding="utf-8").splitlines()

            for object_index, line in enumerate(lines):
                parts = line.split()
                if len(parts) != 5:
                    continue

                class_id = int(parts[0])
                if class_id not in class_lookup:
                    continue

                x_center, y_center, width, height = map(float, parts[1:])
                x1, y1, x2, y2 = xywhn_to_xyxy(
                    x_center,
                    y_center,
                    width,
                    height,
                    image_width,
                    image_height,
                )

                if (x2 - x1) < args.min_size or (y2 - y1) < args.min_size:
                    continue

                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue

                class_name = class_lookup[class_id]
                class_dir = output_root / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)

                output_name = f"{label_path.stem}__obj{object_index}.jpg"
                output_path = class_dir / output_name
                cv2.imwrite(str(output_path), crop)
                split_counts[class_name] += 1

    print(f"[INFO] source={dataset_root}")
    print(f"[INFO] output={output_root}")
    for split, counts in summary.items():
        count_parts = ", ".join([f"{name}={value}" for name, value in counts.items()])
        print(f"[INFO] {split}: {count_parts}")


if __name__ == "__main__":
    main()
