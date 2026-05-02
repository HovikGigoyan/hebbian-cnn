from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import datasets
from src.config import DataConfig
from src.transforms import build_train_transform, build_eval_transform
import torch


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_shape: Tuple[int, int, int]
    num_classes: int
    class_names: list[str]


def _subset_dataset(dataset: Dataset, fraction: float) -> Dataset:
    if fraction >= 1.0:
        return dataset

    total_size = len(dataset)
    subset_size = max(1, int(total_size * fraction))
    indices = torch.randperm(total_size)[:subset_size].tolist()
    return Subset(dataset, indices)


def build_datasets(cfg: DataConfig, num_workers: int = 0) -> DatasetBundle:
    dataset_name = cfg.dataset.lower()

    train_transform = build_train_transform(
        dataset_name,
        normalize=cfg.normalize,
        augmentation=cfg.train_augmentation,
    )
    eval_transform = build_eval_transform(
        dataset_name,
        normalize=cfg.normalize,
    )

    if dataset_name == "cifar10":
        full_train_ds = datasets.CIFAR10(
            root=cfg.root,
            train=True,
            transform=train_transform,
            download=True,
        )

        full_train_eval_ds = datasets.CIFAR10(
            root=cfg.root,
            train=True,
            transform=eval_transform,
            download=True,
        )

        test_ds = datasets.CIFAR10(
            root=cfg.root,
            train=False,
            transform=eval_transform,
            download=True,
        )

        input_shape = (3, 32, 32)
        num_classes = 10
        class_names = list(test_ds.classes)

    elif dataset_name == "fashion_mnist":
        full_train_ds = datasets.FashionMNIST(
            root=cfg.root,
            train=True,
            transform=train_transform,
            download=True,
        )

        full_train_eval_ds = datasets.FashionMNIST(
            root=cfg.root,
            train=True,
            transform=eval_transform,
            download=True,
        )

        test_ds = datasets.FashionMNIST(
            root=cfg.root,
            train=False,
            transform=eval_transform,
            download=True,
        )

        input_shape = (1, 28, 28)
        num_classes = 10
        class_names = [
            "T-shirt/top",
            "Trouser",
            "Pullover",
            "Dress",
            "Coat",
            "Sandal",
            "Shirt",
            "Sneaker",
            "Bag",
            "Ankle boot",
        ]
    else:
        raise ValueError(f"Unsupported dataset: {cfg.dataset}")

    full_train_ds = _subset_dataset(full_train_ds, cfg.use_subset_fraction)
    full_train_eval_ds = _subset_dataset(full_train_eval_ds, cfg.use_subset_fraction)

    val_size = max(1, int(len(full_train_ds) * cfg.val_split))
    train_size = len(full_train_ds) - val_size

    generator = torch.Generator().manual_seed(42)
    train_indices, val_indices = random_split(
        range(len(full_train_ds)),
        [train_size, val_size],
        generator=generator,
    )

    train_ds = Subset(full_train_ds, train_indices.indices)
    val_ds = Subset(full_train_eval_ds, val_indices.indices)
    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.eval_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_shape=input_shape,
        num_classes=num_classes,
        class_names=class_names,
    )