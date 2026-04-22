"""
테스트셋 평가 + Confusion Matrix 저장.

사용:
    python -m src.evaluate --config configs/default.yaml
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.dataset import SingleTaskDataset, get_transforms
from src.model import CoffeeClassifier


def load_checkpoint(path: str | Path, device: str):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and \
            "model_state_dict" in ckpt else ckpt
    return ckpt, state


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out_dir", default="docs")
    args = ap.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task = cfg["task"]
    classes = cfg["classes"]
    c2i = {c: i for i, c in enumerate(classes)}
    idx_to_class = {i: c for c, i in c2i.items()}

    ckpt_path = args.ckpt or f"{cfg['paths']['ckpt_dir']}/best_{task}.pth"
    ckpt, state = load_checkpoint(ckpt_path, device)

    model = CoffeeClassifier(
        backbone=cfg["model"]["name"], n_classes=len(classes),
        pretrained=False, dropout=cfg["model"]["dropout"],
        hidden=cfg["model"]["hidden"],
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    test_ds = SingleTaskDataset(
        csv_path=cfg["data"]["test_csv"], task=task, class_to_idx=c2i,
        transform=get_transforms(task=task, train=False,
                                 img_size=cfg["data"]["img_size"]),
    )
    loader = DataLoader(test_ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=cfg["train"]["num_workers"])

    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            ps.append(model(x).argmax(1).cpu())
            ys.append(y)
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"\n[TEST] acc={acc:.4f}  macro_f1={f1:.4f}\n")
    print(classification_report(y_true, y_pred,
                                target_names=[idx_to_class[i] for i in range(len(classes))],
                                digits=4))

    cm = confusion_matrix(y_true, y_pred)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes)
    plt.xlabel("predicted"); plt.ylabel("true")
    plt.title(f"Confusion Matrix ({task})  acc={acc:.3f}  f1={f1:.3f}")
    plt.tight_layout()
    cm_path = out / f"confusion_matrix_{task}.png"
    plt.savefig(cm_path, dpi=120)
    print(f"[OK] saved {cm_path}")


if __name__ == "__main__":
    main()
