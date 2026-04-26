$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
	uv run --active main.py --epochs 100 --batch 16 --imgsz 640 --device 0 --name coffee_yolo26n_gpu
}
finally {
	Pop-Location
}
