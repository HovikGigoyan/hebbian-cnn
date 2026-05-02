from __future__ import annotations
from torchvision import transforms


def build_train_transform(dataset_name: str, normalize: bool = True, augmentation: bool = True):
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        ops = []
        if augmentation:
            ops.extend([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
            ])

        ops.append(transforms.ToTensor())

        if normalize:
            ops.append(transforms.Normalize(mean, std))

        return transforms.Compose(ops)

    if dataset_name == "fashion_mnist":
        mean = (0.2860,)
        std = (0.3530,)
        ops = [transforms.ToTensor()]

        if normalize:
            ops.append(transforms.Normalize(mean, std))

        return transforms.Compose(ops)

    raise ValueError(f"Unsupported dataset: {dataset_name}")


def build_eval_transform(dataset_name: str, normalize: bool = True):
    dataset_name = dataset_name.lower()

    if dataset_name == "cifar10":
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        ops = [transforms.ToTensor()]

        if normalize:
            ops.append(transforms.Normalize(mean, std))

        return transforms.Compose(ops)

    if dataset_name == "fashion_mnist":
        mean = (0.2860,)
        std = (0.3530,)
        ops = [transforms.ToTensor()]

        if normalize:
            ops.append(transforms.Normalize(mean, std))

        return transforms.Compose(ops)

    raise ValueError(f"Unsupported dataset: {dataset_name}")