from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
from src.metrics import RunningAverage, accuracy
from src.trainers.base_trainer import BaseTrainer
import torch
import torch.nn as nn


class HybridTrainer(BaseTrainer):
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

    def _hebbian_layers(self) -> list[nn.Module]:
        layers = []
        if hasattr(self.model, "conv1") and hasattr(self.model.conv1, "hebbian_update"):
            layers.append(self.model.conv1)
        if hasattr(self.model, "conv2") and hasattr(self.model.conv2, "hebbian_update"):
            layers.append(self.model.conv2)
        return layers

    def pretrain_hebbian_features(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int,
    ) -> None:
        self.model.train()
        hebb_layers = self._hebbian_layers()
        if not hebb_layers:
            return

        for epoch in range(1, epochs + 1):
            for images, _ in tqdm(dataloader, desc=f"Hybrid Hebbian pretrain {epoch}/{epochs}", leave=False):
                images = images.to(self.device)
                _ = self.model.forward_features(images)

                for layer in hebb_layers:
                    layer.hebbian_update()

            print(f"[Hybrid Hebbian pretrain epoch {epoch}/{epochs}] completed.")

    def freeze_hebbian_layers(self) -> None:
        for layer in self._hebbian_layers():
            for p in layer.parameters():
                p.requires_grad = False

    def train_epoch(self, dataloader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.model.train()
        loss_meter = RunningAverage()
        acc_meter = RunningAverage()

        for images, labels in tqdm(dataloader, desc="Hybrid train", leave=False):
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
        hebb_epochs: int,
        classifier_epochs: int,
        ckpt_name: str = "best_hybrid.pt",
        history_name: str = "history_hybrid.json",
    ) -> None:
        self.pretrain_hebbian_features(train_loader, hebb_epochs)
        self.freeze_hebbian_layers()

        for epoch in range(1, classifier_epochs + 1):
            train_metrics = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, self.criterion)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            print(
                f"[Hybrid epoch {epoch}/{classifier_epochs}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.save_checkpoint(ckpt_name)

        self.save_history(history_name)