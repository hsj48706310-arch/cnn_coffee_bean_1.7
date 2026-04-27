cnn_coffee_bean_1.7

Train setup (YOLO26n + Roboflow export)

1) Extract dataset zip into:

datasets/Coffee Defect.v1-coffee-bean_yolo26n.yolo26

2) Sync dependencies in this project folder:

uv sync --active

3) Run training:

uv run --active main.py

Useful options:

uv run --active main.py --epochs 200 --batch 8 --imgsz 640
uv run --active main.py --device cpu

If your installed Ultralytics build does not provide yolo26n weights,
pass a local checkpoint path with:

--model path/to/yolo26n.pt

If you see this warning:

warning: `VIRTUAL_ENV=...` does not match the project environment path `.venv`

run with --active as above from this folder, or deactivate the other virtual environment first.

Output layout

artifacts/train  : training runs and weights
artifacts/predict: prediction outputs
artifacts/export : exported model files

Quick run scripts

PowerShell -ExecutionPolicy Bypass -File scripts/train_gpu.ps1
PowerShell -ExecutionPolicy Bypass -File scripts/train_cpu.ps1

Roasting classifier training (EfficientNetV2-S)

PowerShell -ExecutionPolicy Bypass -File scripts/train_roast_gpu.ps1
PowerShell -ExecutionPolicy Bypass -File scripts/train_roast_cpu.ps1

Recommended concurrent training setup (single RTX 4060 Laptop GPU)

1) Run defect detection (YOLO26n) on GPU with a smaller batch to reduce VRAM spikes.
2) Run roasting classification on CPU, or run both on GPU only with reduced batch sizes.

Example concurrent commands:

PowerShell -ExecutionPolicy Bypass -File scripts/train_gpu.ps1
PowerShell -ExecutionPolicy Bypass -File scripts/train_roast_cpu.ps1

Roasting classification dataset (crop generation)

Build a separate classification dataset for roasting-stage labels
(`roasted-beans`, `under_roast`) from YOLO annotations:

c:/miniproject1.7/.venv/Scripts/python.exe scripts/build_roast_cls_dataset.py --overwrite

Quick sanity check (small subset):

c:/miniproject1.7/.venv/Scripts/python.exe scripts/build_roast_cls_dataset.py --overwrite --max-label-files-per-split 1000

Representative subset sampling (recommended for quick checks):

c:/miniproject1.7/.venv/Scripts/python.exe scripts/build_roast_cls_dataset.py --overwrite --max-label-files-per-split 1000 --shuffle-label-files

Output:

artifacts/roast_cls_dataset/train/<class_name>
artifacts/roast_cls_dataset/val/<class_name>
artifacts/roast_cls_dataset/test/<class_name>
artifacts/roast_cls_dataset/manifest.csv