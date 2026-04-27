from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import EfficientNet_V2_S_Weights, efficientnet_v2_s

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class EpochResult:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train EfficientNetV2-S for roasting-stage classification")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("artifacts") / "roast_cls_dataset",
        help="Classification dataset root containing train/val[/test] folders",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts") / "roast_cls_train",
        help="Directory to store checkpoints and logs",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
) -> tuple[float, float]:
    train_mode = optimizer is not None
    model.train(train_mode)

    loss_sum = 0.0
    acc_sum = 0.0
    sample_count = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if train_mode:
            optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        if train_mode:
            loss.backward()
            optimizer.step()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        acc_sum += accuracy(logits, labels) * batch_size
        sample_count += batch_size

    return loss_sum / sample_count, acc_sum / sample_count


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    if args.device != str(device):
        print("[WARN] CUDA is not available. Falling back to CPU.")

    data_root = args.data_root.resolve()
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(f"Missing train/val folders under: {data_root}")

    weights = EfficientNet_V2_S_Weights.DEFAULT
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = efficientnet_v2_s(weights=weights)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(train_ds.classes))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] device={device}")
    print(f"[INFO] classes={train_ds.classes}")
    print(f"[INFO] train_samples={len(train_ds)} val_samples={len(val_ds)}")

    history: list[EpochResult] = []
    best_val_acc = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, device, optimizer=None)

        result = EpochResult(epoch, train_loss, train_acc, val_loss, val_acc)
        history.append(result)

        print(
            f"[EPOCH {epoch:03d}] "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "class_to_idx": train_ds.class_to_idx,
                    "val_acc": val_acc,
                    "img_size": args.img_size,
                },
                output_dir / "best.pt",
            )
            print(f"[INFO] Saved best checkpoint at epoch {epoch} (val_acc={val_acc:.4f})")

    with (output_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump([r.__dict__ for r in history], f, ensure_ascii=True, indent=2)

    print(f"[INFO] Training complete. best_val_acc={best_val_acc:.4f}")
    print(f"[INFO] Output: {output_dir}")


if __name__ == "__main__":
    main()
