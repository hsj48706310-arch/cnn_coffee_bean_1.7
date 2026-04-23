"""Run test eval for both tasks and dump metrics to docs/test_metrics.json."""
from __future__ import annotations
import json
from pathlib import Path

import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from torch.utils.data import DataLoader

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.dataset import SingleTaskDataset, get_transforms
from src.model import CoffeeClassifier


def eval_one(cfg_path: str, ckpt_path: str) -> dict:
    cfg = load_config(cfg_path)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    task = cfg["task"]
    classes = cfg["classes"]
    c2i = {c: i for i, c in enumerate(classes)}

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    backbone = (ckpt.get("config") or cfg)["model"]["name"]

    model = CoffeeClassifier(
        backbone=backbone, n_classes=len(classes),
        pretrained=False, dropout=cfg["model"]["dropout"],
        hidden=cfg["model"]["hidden"],
    ).to(device)
    model.load_state_dict(state)
    model.eval()

    ds = SingleTaskDataset(
        csv_path=cfg["data"]["test_csv"], task=task, class_to_idx=c2i,
        transform=get_transforms(task=task, train=False,
                                 img_size=cfg["data"]["img_size"]),
    )
    loader = DataLoader(ds, batch_size=cfg["train"]["batch_size"],
                        shuffle=False, num_workers=cfg["train"]["num_workers"])

    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps.append(model(x.to(device)).argmax(1).cpu())
            ys.append(y)
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro"))
    report = classification_report(
        y_true, y_pred, target_names=classes, digits=4, output_dict=True,
        zero_division=0,
    )
    return {
        "task": task,
        "n_classes": len(classes),
        "n_test": int(len(y_true)),
        "test_acc": acc,
        "test_macro_f1": f1,
        "ckpt_val_macro_f1": float(ckpt.get("val_macro_f1", -1)),
        "ckpt_epoch": int(ckpt.get("epoch", -1)),
        "classes": classes,
        "per_class_f1": {c: float(report[c]["f1-score"]) for c in classes},
    }


def main() -> None:
    out = {
        "roast": eval_one("configs/default.yaml", "checkpoints/best_roast.pth"),
        "defect": eval_one("configs/defect.yaml", "checkpoints/best_defect.pth"),
    }
    Path("docs").mkdir(exist_ok=True)
    Path("docs/test_metrics.json").write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
