"""Loss functions. MVP는 CE + LabelSmoothing. Focal은 옵션."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy_loss(weight: torch.Tensor | None = None,
                       label_smoothing: float = 0.0) -> nn.Module:
    return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)


class FocalLoss(nn.Module):
    """클래스 불균형이 매우 심할 때만 사용."""
    def __init__(self, alpha: torch.Tensor | None = None, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, target, weight=self.alpha, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()
