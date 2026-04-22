"""ONNX export + onnxruntime 속도 측정.
사용:
    python -m src.export_onnx --config configs/default.yaml
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path
import numpy as np
import torch

from src.utils.config import load_config
from src.model import CoffeeClassifier
from src.evaluate import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    task, classes = cfg["task"], cfg["classes"]
    img_size = cfg["data"]["img_size"]
    ckpt_path = args.ckpt or f"{cfg['paths']['ckpt_dir']}/best_{task}.pth"
    out_path = args.out or f"{cfg['paths']['ckpt_dir']}/best_{task}.onnx"

    _, state = load_checkpoint(ckpt_path, "cpu")
    model = CoffeeClassifier(
        backbone=cfg["model"]["name"], n_classes=len(classes), pretrained=False,
        dropout=cfg["model"]["dropout"], hidden=cfg["model"]["hidden"],
    )
    model.load_state_dict(state); model.eval()

    dummy = torch.randn(1, 3, img_size, img_size)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, out_path,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"[OK] exported {out_path}")

    # 속도 비교
    import onnxruntime as ort
    sess = ort.InferenceSession(out_path, providers=["CPUExecutionProvider"])
    x = np.random.randn(1, 3, img_size, img_size).astype(np.float32)

    # warm-up
    for _ in range(5):
        sess.run(None, {"input": x})
        with torch.no_grad():
            model(torch.from_numpy(x))

    N = 30
    t0 = time.perf_counter()
    for _ in range(N):
        sess.run(None, {"input": x})
    onnx_ms = (time.perf_counter() - t0) * 1000 / N

    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            model(torch.from_numpy(x))
    pt_ms = (time.perf_counter() - t0) * 1000 / N

    print(f"[BENCH] PyTorch CPU: {pt_ms:6.2f} ms/img  ({1000/pt_ms:5.1f} FPS)")
    print(f"[BENCH] ONNX    CPU: {onnx_ms:6.2f} ms/img  ({1000/onnx_ms:5.1f} FPS)")


if __name__ == "__main__":
    main()
