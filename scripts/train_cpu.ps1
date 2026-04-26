$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
	uv run --active main.py --epochs 30 --batch 8 --imgsz 640 --device cpu --name coffee_yolo26n_cpu
}
finally {
	Pop-Location
}
