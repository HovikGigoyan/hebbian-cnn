import argparse
import copy
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from models import SupervisedCNN, HybridCNN


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_stratified_split(dataset, labeled_size, unlabeled_size, val_ratio, seed):
    rng = np.random.default_rng(seed)
    targets = np.array(dataset.targets)

    classes = np.unique(targets)
    num_classes = len(classes)

    labeled_per_class = labeled_size // num_classes
    unlabeled_per_class = unlabeled_size // num_classes

    labeled_indices = []
    unlabeled_indices = []

    for cls in classes:
        cls_indices = np.where(targets == cls)[0]
        rng.shuffle(cls_indices)

        labeled_cls = cls_indices[:labeled_per_class]
        unlabeled_cls = cls_indices[
            labeled_per_class:labeled_per_class + unlabeled_per_class
        ]

        labeled_indices.extend(labeled_cls.tolist())
        unlabeled_indices.extend(unlabeled_cls.tolist())

    rng.shuffle(labeled_indices)
    rng.shuffle(unlabeled_indices)

    val_size = int(len(labeled_indices) * val_ratio)
    val_indices = labeled_indices[:val_size]
    train_indices = labeled_indices[val_size:]

    return train_indices, val_indices, unlabeled_indices


def evaluate(model, loader, device):
    model.eval()

    criterion = nn.CrossEntropyLoss()

    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    accuracy = correct / total
    avg_loss = total_loss / total

    return accuracy, avg_loss


def train_supervised(model, train_loader, val_loader, device, epochs, lr):
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    best_state = None
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()

        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            train_correct += (outputs.argmax(dim=1) == labels).sum().item()
            train_total += labels.size(0)

        val_acc, val_loss = evaluate(model, val_loader, device)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss / train_total:.4f} | "
            f"train_acc={train_correct / train_total:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_acc


def pretrain_hebbian_first_layer(model, unlabeled_loader, device, epochs):
    model.to(device)
    model.train()

    if hasattr(model.conv1, "hebbian_enabled"):
        model.conv1.hebbian_enabled = True

    w_before = model.conv1.weight.detach().clone()

    for epoch in range(1, epochs + 1):
        for images, _ in unlabeled_loader:
            images = images.to(device)

            with torch.no_grad():
                _ = model.conv1(images)

        diff = (model.conv1.weight.detach() - w_before).abs().mean().item()

        print(
            f"Hebbian pretraining epoch {epoch:02d}/{epochs} finished | "
            f"mean_weight_change={diff:.8f}"
        )


def prepare_frozen_hebbian(model):
    if hasattr(model.conv1, "hebbian_enabled"):
        model.conv1.hebbian_enabled = False

    for param in model.conv1.parameters():
        param.requires_grad = False

    print("Hebbian first layer frozen")
    print("conv1 hebbian_enabled:", model.conv1.hebbian_enabled)
    print("conv1 weight requires_grad:", model.conv1.weight.requires_grad)


def run_experiment(args, labeled_size):
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"\nDevice: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)),
    ])

    train_dataset = datasets.FashionMNIST(
        root=args.data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=args.data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    train_indices, val_indices, unlabeled_indices = make_stratified_split(
        dataset=train_dataset,
        labeled_size=labeled_size,
        unlabeled_size=args.unlabeled_size,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    val_loader = DataLoader(
        Subset(train_dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    unlabeled_loader = DataLoader(
        Subset(train_dataset, unlabeled_indices),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    print("\nSplit:")
    print(f"unlabeled for M2 Hebbian pretraining: {len(unlabeled_indices)}")
    print(f"labeled train: {len(train_indices)}")
    print(f"labeled validation: {len(val_indices)}")
    print(f"test: {len(test_dataset)}")

    print("\n=== M0: supervised CNN on Fashion-MNIST ===")

    m0 = SupervisedCNN(
        in_channels=1,
        num_classes=10,
        input_size=(28, 28),
    ).to(device)

    m0_val_acc = train_supervised(
        model=m0,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.supervised_epochs,
        lr=args.lr,
    )

    m0_test_acc, m0_test_loss = evaluate(
        m0,
        test_loader,
        device,
    )

    print(f"M0 best val accuracy: {m0_val_acc:.4f}")
    print(f"M0 test accuracy: {m0_test_acc:.4f}")

    print("\n=== M2: Hebbian pretraining + supervised training ===")

    m2 = HybridCNN(
        in_channels=1,
        num_classes=10,
        input_size=(28, 28),
        hebb_lr=args.hebb_lr,
    ).to(device)

    pretrain_hebbian_first_layer(
        model=m2,
        unlabeled_loader=unlabeled_loader,
        device=device,
        epochs=args.hebb_epochs,
    )

    prepare_frozen_hebbian(m2)

    m2_val_acc = train_supervised(
        model=m2,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.supervised_epochs,
        lr=args.lr,
    )

    m2_test_acc, m2_test_loss = evaluate(
        m2,
        test_loader,
        device,
    )

    print(f"M2 best val accuracy: {m2_val_acc:.4f}")
    print(f"M2 test accuracy: {m2_test_acc:.4f}")
    print(f"M0 test accuracy: {m0_test_acc:.4f}")
    print(f"M0 test loss: {m0_test_loss:.4f}")
    print(f"M2 test accuracy: {m2_test_acc:.4f}")
    print(f"M2 test loss: {m2_test_loss:.4f}")

    return {
        "dataset": "Fashion-MNIST",
        "image_size": "28x28",
        "labeled_size": labeled_size,
        "unlabeled_size": len(unlabeled_indices),
        "labeled_train": len(train_indices),
        "labeled_val": len(val_indices),
        "m0_val_acc": m0_val_acc,
        "m0_test_acc": m0_test_acc,
        "m0_test_loss": m0_test_loss,
        "m2_test_loss": m2_test_loss,
        "m2_val_acc": m2_val_acc,
        "m2_test_acc": m2_test_acc,
        "hebb_epochs": args.hebb_epochs,
        "supervised_epochs": args.supervised_epochs,
        "hebb_lr": args.hebb_lr,
        "lr": args.lr,
        "seed": args.seed,
    }


def save_results(results, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(results[0].keys())
    write_header = not output_path.exists()

    with output_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if write_header:
            writer.writeheader()

        writer.writerows(results)

    print(f"\nResults saved to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--output", type=str, default="results/fashion_m0_m2_semisupervised.csv")

    parser.add_argument("--labeled-sizes", type=int, nargs="+", default=[10000])
    parser.add_argument("--unlabeled-size", type=int, default=50000)
    parser.add_argument("--val-ratio", type=float, default=0.2)

    parser.add_argument("--hebb-epochs", type=int, default=10)
    parser.add_argument("--supervised-epochs", type=int, default=15)

    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--test-batch-size", type=int, default=128)

    parser.add_argument("--hebb-lr", type=float, default=1e-5)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    all_results = []

    for labeled_size in args.labeled_sizes:
        print("\n" + "=" * 70)
        print(
            f"Fashion-MNIST experiment | "
            f"labeled_size={labeled_size}, "
            f"unlabeled_size={args.unlabeled_size}"
        )
        print("=" * 70)

        result = run_experiment(args, labeled_size)
        all_results.append(result)

    save_results(all_results, args.output)


if __name__ == "__main__":
    main()