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

Roast classification (separate model)

1) Build classification crops from detection labels (roasted-beans vs under_roast):

uv run --active prepare_roast_cls_dataset.py

2) Train a dedicated classifier:

uv run --active train_roast_cls.py --device cpu --epochs 20 --imgsz 224 --batch 64

Quick scripts:

PowerShell -ExecutionPolicy Bypass -File scripts/train_roast_cls_cpu.ps1
PowerShell -ExecutionPolicy Bypass -File scripts/train_roast_cls_gpu.ps1

Unified Web UI (detection + roast classification)

Run one app with two tabs:

PowerShell -ExecutionPolicy Bypass -File scripts/run_web_ui.ps1

Or run directly:

uv run --active web_app.py

Default model paths used by the app:
- Detection: artifacts/train/coffee_yolo26n_final/weights/best.pt
- Classification: artifacts/cls/roast_cls_v8n_cpu/weights/best.pt