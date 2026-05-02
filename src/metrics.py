from __future__ import annotations
from typing import Dict
import torch


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    topk = torch.topk(logits, k=k, dim=1).indices
    correct = topk.eq(targets.view(-1, 1)).any(dim=1)
    return correct.float().mean().item()


def classification_stats(logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    return {
        "accuracy": accuracy(logits, targets),
        "top3_accuracy": topk_accuracy(logits, targets, k=min(3, logits.shape[1])),
    }


class RunningAverage:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.total = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n

    @property
    def value(self) -> float:
        if self.count == 0:
            return 0.0
        return self.total / self.count