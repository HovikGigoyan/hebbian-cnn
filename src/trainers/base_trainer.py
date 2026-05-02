from __future__ import annotations
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
from src.metrics import RunningAverage, accuracy
from src.utils import save_json
import torch
import torch.nn as nn


class BaseTrainer:
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: Path,
    ) -> None:
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.metrics_dir = self.output_dir / "metrics"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)

        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
        }

        self.best_val_acc = -1.0

    def save_checkpoint(self, name: str = "best.pt") -> None:
        path = self.checkpoints_dir / name
        torch.save(self.model.state_dict(), path)

    def save_history(self, name: str = "history.json") -> None:
        path = self.metrics_dir / name
        save_json(path, self.history)

    def evaluate(
        self,
        dataloader: torch.utils.data.DataLoader,
        criterion: nn.Module,
    ) -> Dict[str, float]:
        self.model.eval()
        loss_meter = RunningAverage()
        acc_meter = RunningAverage()

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Eval", leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                logits = self.model(images)
                loss = criterion(logits, labels)
                batch_acc = accuracy(logits, labels)
                loss_meter.update(loss.item(), images.size(0))
                acc_meter.update(batch_acc, images.size(0))

        return {
            "loss": loss_meter.value,
            "accuracy": acc_meter.value,
        }