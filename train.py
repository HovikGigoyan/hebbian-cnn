from pathlib import Path
from src.config import load_config, resolve_device
from src.datasets import build_datasets
from src.models import SupervisedCNN, FullHebbCNN, HybridCNN, PartialHybridCNN
from src.trainers.hebbian_trainer import HebbianTrainer
from src.trainers.hybrid_trainer import HybridTrainer
from src.trainers.supervised_trainer import SupervisedTrainer
from src.utils import prepare_output_dirs, set_seed
import torch
import torch.nn as nn


def build_model(cfg, bundle):
    input_h = bundle.input_shape[1]
    input_w = bundle.input_shape[2]

    common_kwargs = dict(
        in_channels=bundle.input_shape[0],
        num_classes=bundle.num_classes,
        input_size=(input_h, input_w),
        conv1_channels=cfg.model.conv1_channels,
        conv2_channels=cfg.model.conv2_channels,
        kernel_size=cfg.model.kernel_size,
        padding=cfg.model.padding,
    )

    mode = cfg.hebbian.mode

    if mode == "supervised_baseline":
        return SupervisedCNN(**common_kwargs)

    if mode == "full_hebb":
        return FullHebbCNN(
            **common_kwargs,
            hebb_lr=cfg.hebbian.lr,
        )

    if mode == "hybrid":
        return HybridCNN(
            **common_kwargs,
            hebb_lr=cfg.hebbian.lr,
        )

    if mode == "partial_hybrid":
        return PartialHybridCNN(
            **common_kwargs,
            hebb_lr=cfg.hebbian.lr,
        )

    raise ValueError(f"Unknown mode: {mode}")


def main() -> None:
    cfg = load_config("configs/base.yaml")
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    out_dirs = prepare_output_dirs(cfg.output_dir)

    bundle = build_datasets(cfg.data, num_workers=cfg.num_workers)
    model = build_model(cfg, bundle).to(device)

    criterion = nn.CrossEntropyLoss()
    mode = cfg.hebbian.mode

    print(f"Running mode: {mode}")
    print(f"Device: {device}")
    print(f"Dataset: {cfg.data.dataset}")

    if mode == "supervised_baseline":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

        trainer = SupervisedTrainer(
            model=model,
            device=device,
            output_dir=Path(cfg.output_dir),
            optimizer=optimizer,
            criterion=criterion,
        )

        trainer.fit(
            train_loader=bundle.train_loader,
            val_loader=bundle.val_loader,
            epochs=cfg.train.epochs,
        )


    elif mode == "full_hebb":

        classifier_params = list(model.feature_norm.parameters()) + list(model.classifier.parameters())

        optimizer = torch.optim.Adam(

            classifier_params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,

        )

        trainer = HebbianTrainer(
            model=model,
            device=device,
            output_dir=Path(cfg.output_dir),
            classifier_optimizer=optimizer,
            criterion=criterion,

        )

        trainer.fit(
            train_loader=bundle.train_loader,
            val_loader=bundle.val_loader,
            hebb_epochs=max(1, cfg.train.epochs // 2),
            classifier_epochs=cfg.train.epochs,
        )



    elif mode == "hybrid":

        trainable_params = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(
            trainable_params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,

        )

        trainer = HybridTrainer(
            model=model,
            device=device,
            output_dir=Path(cfg.output_dir),
            optimizer=optimizer,
            criterion=criterion,
        )

        trainer.fit(
            train_loader=bundle.train_loader,
            val_loader=bundle.val_loader,
            hebb_epochs=max(1, cfg.train.epochs // 2),
            classifier_epochs=cfg.train.epochs,
            ckpt_name="best_hybrid.pt",
            history_name="history_hybrid.json",
        )


    elif mode == "partial_hybrid":
        trainable_params = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(
            trainable_params,
            lr=cfg.train.lr,
            weight_decay=cfg.train.weight_decay,
        )

        trainer = HybridTrainer(
            model=model,
            device=device,
            output_dir=Path(cfg.output_dir),
            optimizer=optimizer,
            criterion=criterion,
        )

        trainer.fit(
            train_loader=bundle.train_loader,
            val_loader=bundle.val_loader,
            hebb_epochs=max(1, cfg.train.epochs // 2),
            classifier_epochs=cfg.train.epochs,
            ckpt_name="best_partial_hybrid.pt",
            history_name="history_partial_hybrid.json",
        )


    else:
        raise ValueError(f"Unsupported mode: {mode}")


if __name__ == "__main__":
    main()