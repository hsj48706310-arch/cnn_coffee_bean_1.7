"""Single-head classifier (timm backbone + MLP head)."""
from __future__ import annotations
import torch.nn as nn
import timm


class CoffeeClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b0",
        n_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3,
        hidden: int = 256,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat = self.backbone.num_features
        self.head = nn.Sequential(
            nn.Linear(feat, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def freeze_backbone(self, freeze: bool = True) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = not freeze
