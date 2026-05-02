from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
from src.metrics import RunningAverage, accuracy
from src.trainers.base_trainer import BaseTrainer
import torch
import torch.nn as nn


class SupervisedTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: Path,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> None:
        super().__init__(model, device, output_dir)
        self.optimizer = optimizer
        self.criterion = criterion

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.model.train()
        loss_meter = RunningAverage()
        acc_meter = RunningAverage()

        for images, labels in tqdm(dataloader, desc="Train", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.optimizer.step()
            batch_acc = accuracy(logits, labels)
            loss_meter.update(loss.item(), images.size(0))
            acc_meter.update(batch_acc, images.size(0))

        return {
            "loss": loss_meter.value,
            "accuracy": acc_meter.value,
        }

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int,
    ) -> None:
        for epoch in range(1, epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, self.criterion)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            print(
                f"[Epoch {epoch}/{epochs}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.save_checkpoint("best_supervised.pt")

        self.save_history("history_supervised.json")