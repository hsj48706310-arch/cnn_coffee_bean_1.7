"""단일 이미지 추론.
사용:
    python -m src.infer --image path/to/img.jpg --config configs/default.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
from PIL import Image

from src.utils.config import load_config
from src.dataset import get_transforms
from src.model import CoffeeClassifier
from src.evaluate import load_checkpoint


def predict(image_path: str, cfg: dict, ckpt_path: str | None = None,
            device: str | None = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    task, classes = cfg["task"], cfg["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in c2i.items()}

    ckpt_path = ckpt_path or f"{cfg['paths']['ckpt_dir']}/best_{task}.pth"
    _, state = load_checkpoint(ckpt_path, device)

    model = CoffeeClassifier(
        backbone=cfg["model"]["name"], n_classes=len(classes), pretrained=False,
        dropout=cfg["model"]["dropout"], hidden=cfg["model"]["hidden"],
    ).to(device)
    model.load_state_dict(state); model.eval()

    tf = get_transforms(task=task, train=False, img_size=cfg["data"]["img_size"])
    img = np.array(Image.open(image_path).convert("RGB"))
    x = tf(image=img)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
    top = int(probs.argmax())
    return {
        "label": idx_to_class[top],
        "prob": float(probs[top]),
        "all_probs": {idx_to_class[i]: float(probs[i]) for i in range(len(classes))},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = predict(args.image, cfg, args.ckpt)
    print(out)


if __name__ == "__main__":
    main()
