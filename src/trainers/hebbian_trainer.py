from __future__ import annotations
from pathlib import Path
from tqdm import tqdm
from src.metrics import RunningAverage, accuracy
from src.trainers.base_trainer import BaseTrainer
import torch
import torch.nn as nn


class HebbianTrainer(BaseTrainer):
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        output_dir: Path,
        classifier_optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> None:
        super().__init__(model, device, output_dir)
        self.classifier_optimizer = classifier_optimizer
        self.criterion = criterion

    def pretrain_hebbian_features(
        self,
        dataloader: torch.utils.data.DataLoader,
        epochs: int,
    ) -> None:
        self.model.train()

        for epoch in range(1, epochs + 1):
            for images, _ in tqdm(dataloader, desc=f"Hebbian pretrain {epoch}/{epochs}", leave=False):
                images = images.to(self.device)
                _ = self.model.forward_features(images)

                if hasattr(self.model, "conv1") and hasattr(self.model.conv1, "hebbian_update"):
                    self.model.conv1.hebbian_update()

                if hasattr(self.model, "conv2") and hasattr(self.model.conv2, "hebbian_update"):
                    self.model.conv2.hebbian_update()

            print(f"[Hebbian pretrain epoch {epoch}/{epochs}] completed.")

    def freeze_hebbian_features(self) -> None:
        if hasattr(self.model, "conv1"):
            for p in self.model.conv1.parameters():
                p.requires_grad = False

        if hasattr(self.model, "conv2"):
            for p in self.model.conv2.parameters():
                p.requires_grad = False

    def train_classifier_epoch(self, dataloader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.model.train()
        loss_meter = RunningAverage()
        acc_meter = RunningAverage()

        for images, labels in tqdm(dataloader, desc="Classifier train", leave=False):
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.classifier_optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, labels)
            loss.backward()
            self.classifier_optimizer.step()
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
    ) -> None:
        self.pretrain_hebbian_features(train_loader, hebb_epochs)
        self.freeze_hebbian_features()

        for epoch in range(1, classifier_epochs + 1):
            train_metrics = self.train_classifier_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, self.criterion)
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            print(
                f"[Classifier epoch {epoch}/{classifier_epochs}] "
                f"train_loss={train_metrics['loss']:.4f} "
                f"train_acc={train_metrics['accuracy']:.4f} "
                f"val_loss={val_metrics['loss']:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f}"
            )

            if val_metrics["accuracy"] > self.best_val_acc:
                self.best_val_acc = val_metrics["accuracy"]
                self.save_checkpoint("best_full_hebb.pt")

        self.save_history("history_full_hebb.json")