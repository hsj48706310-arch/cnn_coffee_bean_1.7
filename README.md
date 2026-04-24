cnn_coffee_bean_1.7

Train setup (YOLO26n + Roboflow export)

1) Extract dataset zip into:

datasets/Coffee Defect.v1-coffee-bean_yolo26n.yolo26

2) Run training:

c:/miniproject1.7/.venv/Scripts/python.exe main.py

Useful options:

c:/miniproject1.7/.venv/Scripts/python.exe main.py --epochs 200 --batch 8 --imgsz 640
c:/miniproject1.7/.venv/Scripts/python.exe main.py --device cpu

If your installed Ultralytics build does not provide yolo26n weights,
pass a local checkpoint path with:

--model path/to/yolo26n.pt

Output layout

artifacts/train  : training runs and weights
artifacts/predict: prediction outputs
artifacts/export : exported model files

Quick run scripts

PowerShell -ExecutionPolicy Bypass -File scripts/train_gpu.ps1
PowerShell -ExecutionPolicy Bypass -File scripts/train_cpu.ps1