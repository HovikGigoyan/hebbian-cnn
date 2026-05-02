from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import torch
import matplotlib.pyplot as plt
import torchvision.utils as vutils


def plot_learning_curves(history: Dict[str, List[float]], save_path: str | Path, title: str) -> None:
    save_path = Path(save_path)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss")
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def denormalize_images(images: torch.Tensor, dataset_name: str) -> torch.Tensor:
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        mean = torch.tensor([0.4914, 0.4822, 0.4465], device=images.device).view(1, 3, 1, 1)
        std = torch.tensor([0.2470, 0.2435, 0.2616], device=images.device).view(1, 3, 1, 1)
        return images * std + mean

    if dataset_name == "fashion_mnist":
        mean = torch.tensor([0.2860], device=images.device).view(1, 1, 1, 1)
        std = torch.tensor([0.3530], device=images.device).view(1, 1, 1, 1)
        return images * std + mean

    return images


def save_prediction_grid(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor,
    class_names: List[str],
    dataset_name: str,
    save_path: str | Path,
    max_images: int = 16,
) -> None:
    save_path = Path(save_path)

    images = images[:max_images].detach().cpu()
    true_labels = true_labels[:max_images].detach().cpu()
    pred_labels = pred_labels[:max_images].detach().cpu()

    images = denormalize_images(images, dataset_name).clamp(0, 1)

    rows = 4
    cols = min(4, len(images))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 10))
    axes = axes.flatten()

    for idx in range(len(axes)):
        axes[idx].axis("off")

    for i in range(min(len(images), len(axes))):
        img = images[i]
        if img.shape[0] == 1:
            axes[i].imshow(img.squeeze(0), cmap="gray")
        else:
            axes[i].imshow(img.permute(1, 2, 0))

        true_name = class_names[int(true_labels[i])]
        pred_name = class_names[int(pred_labels[i])]
        axes[i].set_title(f"T: {true_name}\nP: {pred_name}", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()


def save_first_layer_filters(
    weight: torch.Tensor,
    save_path: str | Path,
    title: str = "First-layer filters",
) -> None:
    save_path = Path(save_path)
    w = weight.detach().cpu().clone()
    w_min = w.view(w.size(0), -1).min(dim=1)[0].view(-1, 1, 1, 1)
    w_max = w.view(w.size(0), -1).max(dim=1)[0].view(-1, 1, 1, 1)
    w = (w - w_min) / (w_max - w_min + 1e-8)

    grid = vutils.make_grid(w, nrow=8, padding=1)

    plt.figure(figsize=(10, 8))

    if grid.shape[0] == 1:
        plt.imshow(grid.squeeze(0), cmap="gray")
    else:
        plt.imshow(grid.permute(1, 2, 0))

    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()