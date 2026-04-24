from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def build_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Train YOLO26n on local Roboflow dataset")
	parser.add_argument(
		"--data",
		type=Path,
		default=Path("datasets") / "Coffee Defect.v1-coffee-bean_yolo26n.yolo26" / "data.yaml",
		help="Path to data.yaml",
	)
	parser.add_argument("--model", default="yolo26n.pt", help="Model checkpoint name or path")
	parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
	parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
	parser.add_argument("--batch", type=int, default=16, help="Batch size")
	parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
	parser.add_argument(
		"--artifacts-root",
		type=Path,
		default=Path("artifacts"),
		help="Root directory for train/predict/export outputs",
	)
	parser.add_argument("--project", type=Path, default=None, help="Output project directory")
	parser.add_argument("--name", default="coffee_yolo26n", help="Run name")
	parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing run folder")
	return parser


def main() -> None:
	args = build_parser().parse_args()
	data_path = args.data.resolve()

	if not data_path.exists():
		raise FileNotFoundError(f"data.yaml not found: {data_path}")

	artifacts_root = args.artifacts_root.resolve()
	(artifacts_root / "train").mkdir(parents=True, exist_ok=True)
	(artifacts_root / "predict").mkdir(parents=True, exist_ok=True)
	(artifacts_root / "export").mkdir(parents=True, exist_ok=True)
	project_path = args.project.resolve() if args.project else artifacts_root / "train"

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


