from src.config import load_config, resolve_device
from src.datasets import build_datasets
from src.models import SupervisedCNN, FullHebbCNN, HybridCNN, PartialHybridCNN
from src.utils import prepare_output_dirs, save_json, set_seed
from src.visualization import (
    plot_learning_curves,
    save_first_layer_filters,
    save_prediction_grid,
)
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
        return FullHebbCNN(**common_kwargs, hebb_lr=cfg.hebbian.lr)

    if mode == "hybrid":
        return HybridCNN(**common_kwargs, hebb_lr=cfg.hebbian.lr)

    if mode == "partial_hybrid":
        return PartialHybridCNN(**common_kwargs, hebb_lr=cfg.hebbian.lr)

    raise ValueError(f"Unknown mode: {mode}")


def get_checkpoint_name(mode: str) -> str:
    mapping = {
        "supervised_baseline": "best_supervised.pt",
        "full_hebb": "best_full_hebb.pt",
        "hybrid": "best_hybrid.pt",
        "partial_hybrid": "best_partial_hybrid.pt",
    }
    return mapping[mode]


def get_history_name(mode: str) -> str:
    mapping = {
        "supervised_baseline": "history_supervised.json",
        "full_hebb": "history_full_hebb.json",
        "hybrid": "history_hybrid.json",
        "partial_hybrid": "history_partial_hybrid.json",
    }
    return mapping[mode]


def evaluate_model(model, dataloader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    saved_images = None
    saved_labels = None
    saved_preds = None

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (preds == labels).sum().item()
            total += images.size(0)

            if saved_images is None:
                saved_images = images[:16].detach().cpu()
                saved_labels = labels[:16].detach().cpu()
                saved_preds = preds[:16].detach().cpu()

    return {
        "loss": total_loss / total,
        "accuracy": total_correct / total,
        "images": saved_images,
        "labels": saved_labels,
        "preds": saved_preds,
    }


def main() -> None:
    cfg = load_config("configs/base.yaml")
    device = resolve_device(cfg.device)
    set_seed(cfg.seed)
    out_dirs = prepare_output_dirs(cfg.output_dir)

    bundle = build_datasets(cfg.data, num_workers=cfg.num_workers)
    model = build_model(cfg, bundle).to(device)

    mode = cfg.hebbian.mode
    ckpt_name = get_checkpoint_name(mode)
    ckpt_path = out_dirs["checkpoints"] / ckpt_name

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    criterion = nn.CrossEntropyLoss()

    result = evaluate_model(model, bundle.test_loader, device, criterion)

    metrics_payload = {
        "mode": mode,
        "dataset": cfg.data.dataset,
        "test_loss": result["loss"],
        "test_accuracy": result["accuracy"],
    }
    save_json(out_dirs["metrics"] / f"test_{mode}.json", metrics_payload)

    print("=== Test results ===")
    print(metrics_payload)

    save_prediction_grid(
        images=result["images"],
        true_labels=result["labels"],
        pred_labels=result["preds"],
        class_names=bundle.class_names,
        dataset_name=cfg.data.dataset,
        save_path=out_dirs["predictions"] / f"{mode}_predictions.png",
    )

    # save filters
    if hasattr(model, "conv1") and hasattr(model.conv1, "weight"):
        save_first_layer_filters(
            model.conv1.weight,
            out_dirs["filters"] / f"{mode}_conv1_filters.png",
            title=f"{mode} - conv1 filters",
        )

    # save learning curves if history exists
    history_path = out_dirs["metrics"] / get_history_name(mode)
    if history_path.exists():
        import json
        with open(history_path, "r", encoding="utf-8") as f:
            history = json.load(f)

        plot_learning_curves(
            history=history,
            save_path=out_dirs["plots"] / f"{mode}_learning_curves.png",
            title=f"{mode} learning curves",
        )


if __name__ == "__main__":
    main()