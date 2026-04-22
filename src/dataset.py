"""Single-task dataset + Albumentations transforms (Kaggle 전용)."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_transforms(task: str = "roast", train: bool = True, img_size: int = 224):
    """task별 증강. roast=색 보존, defect=색 변화 OK + erasing 금지."""
    if not train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(),
            ToTensorV2(),
        ])

    if task == "roast":
        return A.Compose([
            A.RandomResizedCrop(size=(img_size, img_size), scale=(0.85, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=20, p=0.5),
            A.RandomBrightnessContrast(0.1, 0.1, p=0.5),
            A.Normalize(),
            ToTensorV2(),
        ])

    # defect
    return A.Compose([
        A.RandomResizedCrop(size=(img_size, img_size), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=30, p=0.5),
        A.ColorJitter(0.2, 0.2, 0.2, 0.05, p=0.5),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ])


class SingleTaskDataset(Dataset):
    """CSV에 'path' 와 '<task>_label' 컬럼이 있어야 함."""

    def __init__(
        self,
        csv_path: str | Path,
        task: str = "roast",
        class_to_idx: dict[str, int] | None = None,
        transform=None,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.label_col = f"{task}_label"
        if self.label_col not in self.df.columns:
            raise KeyError(f"CSV에 '{self.label_col}' 컬럼이 없습니다.")
        self.classes = sorted(self.df[self.label_col].unique().tolist())
        self.class_to_idx = class_to_idx or {c: i for i, c in enumerate(self.classes)}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = np.array(Image.open(row["path"]).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)["image"]
        y = self.class_to_idx[row[self.label_col]]
        return img, torch.tensor(y, dtype=torch.long)


def compute_class_weights(csv_path: str | Path, task: str,
                          class_to_idx: dict[str, int]) -> torch.Tensor:
    """클래스 불균형 보정용 weight."""
    df = pd.read_csv(csv_path)
    counts = df[f"{task}_label"].value_counts().to_dict()
    total = sum(counts.values())
    n_cls = len(class_to_idx)
    weights = torch.ones(n_cls, dtype=torch.float)
    for cls, idx in class_to_idx.items():
        c = counts.get(cls, 0)
        weights[idx] = total / (n_cls * c) if c > 0 else 0.0
    return weights
