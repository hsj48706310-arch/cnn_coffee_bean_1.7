$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location $projectRoot
try {
    uv run --active prepare_roast_cls_dataset.py
    uv run --active train_roast_cls.py --device 0 --epochs 20 --imgsz 224 --batch 128 --name roast_cls_v8n_gpu
}
finally {
    Pop-Location
}
