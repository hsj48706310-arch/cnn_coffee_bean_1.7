from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a roast-level image classification model")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("datasets") / "roast_cls",
        help="Path to classification dataset root containing train/valid/test subfolders",
    )
    parser.add_argument("--model", default="yolov8n-cls.pt", help="Ultralytics classification model")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=224, help="Input image size")
    parser.add_argument("--batch", type=int, default=64, help="Batch size")
    parser.add_argument("--device", default="cpu", help="CUDA device index or 'cpu'")
    parser.add_argument(
        "--artifacts-root",
        type=Path,
        default=Path("artifacts") / "cls",
        help="Root output directory for classification runs",
    )
    parser.add_argument("--project", type=Path, default=None, help="Optional custom output project path")
    parser.add_argument("--name", default="roast_cls_v8n", help="Run name")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run folder")
    return parser


def validate_dataset_root(dataset_root: Path) -> None:
    for split in ("train", "valid"):
        split_dir = dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Missing required split folder: {split_dir}")


def main() -> None:
    args = build_parser().parse_args()
    data_path = args.data.resolve()
    validate_dataset_root(data_path)

    artifacts_root = args.artifacts_root.resolve()
    artifacts_root.mkdir(parents=True, exist_ok=True)
    project_path = args.project.resolve() if args.project else artifacts_root

    print(f"[INFO] task=classification")
    print(f"[INFO] data={data_path}")
    print(f"[INFO] project={project_path}")
    print(f"[INFO] run_name={args.name}")

    model = YOLO(args.model)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(project_path),
        name=args.name,
        exist_ok=args.exist_ok,
    )


if __name__ == "__main__":
    main()
