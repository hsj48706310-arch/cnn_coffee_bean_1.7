"""
학습 스크립트.

사용:
    python -m src.train --config configs/default.yaml
    python -m src.train --config configs/default.yaml --override model.name=resnet18
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from src.utils.config import load_config
from src.utils.seed import set_seed
from src.utils.logger import TBLogger
from src.dataset import SingleTaskDataset, get_transforms, compute_class_weights
from src.model import CoffeeClassifier
from src.losses import cross_entropy_loss


def parse_overrides(items: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for it in items:
        k, v = it.split("=", 1)
        try:
            v = eval(v, {"__builtins__": {}}, {})
        except Exception:
            pass
        out[k] = v
    return out


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    for k, v in overrides.items():
        cur = cfg
        keys = k.split(".")
        for kk in keys[:-1]:
            cur = cur.setdefault(kk, {})
        cur[keys[-1]] = v
    return cfg


def make_loader(csv: str, task: str, classes: list[str], img_size: int,
                train: bool, batch_size: int, num_workers: int):
    c2i = {c: i for i, c in enumerate(classes)}
    ds = SingleTaskDataset(
        csv_path=csv, task=task, class_to_idx=c2i,
        transform=get_transforms(task=task, train=train, img_size=img_size),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=train,
                      num_workers=num_workers, pin_memory=False), c2i


def evaluate(model, loader, device) -> dict:
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            ps.append(logits.argmax(1).cpu())
            ys.append(y)
    y_true = torch.cat(ys).numpy()
    y_pred = torch.cat(ps).numpy()
    return {
        "acc": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, parse_overrides(args.override))
    set_seed(cfg["seed"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(cfg["train"].get("amp", False)) and device == "cuda"
    print(f"[i] device={device}  amp={use_amp}")

    task = cfg["task"]
    classes = cfg["classes"]
    img_size = cfg["data"]["img_size"]

    train_loader, c2i = make_loader(
        cfg["data"]["train_csv"], task, classes, img_size,
        train=True, batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )
    val_loader, _ = make_loader(
        cfg["data"]["val_csv"], task, classes, img_size,
        train=False, batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    model = CoffeeClassifier(
        backbone=cfg["model"]["name"],
        n_classes=len(classes),
        pretrained=cfg["model"]["pretrained"],
        dropout=cfg["model"]["dropout"],
        hidden=cfg["model"]["hidden"],
    ).to(device)

    cls_w = compute_class_weights(cfg["data"]["train_csv"], task, c2i).to(device)
    criterion = cross_entropy_loss(weight=cls_w,
                                   label_smoothing=cfg["train"]["label_smoothing"])

    optimizer = AdamW(model.parameters(), lr=cfg["train"]["lr"],
                      weight_decay=cfg["train"]["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg["train"]["epochs"])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    logger = TBLogger(cfg["paths"]["log_dir"])
    ckpt_dir = Path(cfg["paths"]["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    warmup = cfg["train"].get("warmup_epochs", 0)
    if warmup > 0:
        model.freeze_backbone(True)
        print(f"[i] backbone frozen for {warmup} warm-up epoch(s)")

    best_f1 = -1.0
    bad = 0
    for epoch in range(1, cfg["train"]["epochs"] + 1):
        if epoch == warmup + 1 and warmup > 0:
            model.freeze_backbone(False)
            print(f"[i] backbone unfrozen at epoch {epoch}")

        model.train()
        running = 0.0
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                loss = criterion(model(x), y)
            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward(); optimizer.step()
            running += loss.item() * x.size(0)
        scheduler.step()

        train_loss = running / len(train_loader.dataset)
        metrics = evaluate(model, val_loader, device)
        print(f"[E{epoch:02d}] loss={train_loss:.4f}  "
              f"val_acc={metrics['acc']:.4f}  val_f1={metrics['macro_f1']:.4f}")
        logger.log({"train/loss": train_loss,
                    "val/acc": metrics["acc"],
                    "val/macro_f1": metrics["macro_f1"],
                    "lr": optimizer.param_groups[0]["lr"]}, epoch)

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]
            bad = 0
            ckpt_path = ckpt_dir / f"best_{task}.pth"
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": c2i,
                "config": cfg,
                "epoch": epoch,
                "val_macro_f1": best_f1,
            }, ckpt_path)
            print(f"   ↳ saved {ckpt_path} (f1={best_f1:.4f})")
        else:
            bad += 1
            if bad >= cfg["train"]["early_stop_patience"]:
                print(f"[i] early stopping at epoch {epoch}")
                break

    logger.close()
    print(f"[DONE] best val_macro_f1 = {best_f1:.4f}")


if __name__ == "__main__":
    main()
