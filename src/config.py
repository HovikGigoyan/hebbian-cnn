from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List
import torch
import yaml


@dataclass
class DataConfig:
    root: str
    dataset: str
    batch_size: int
    eval_batch_size: int
    val_split: float
    use_subset_fraction: float
    normalize: bool
    train_augmentation: bool


@dataclass
class ModelConfig:
    in_channels: int
    num_classes: int
    conv1_channels: int
    conv2_channels: int
    kernel_size: int
    padding: int
    pooling: str


@dataclass
class TrainConfig:
    epochs: int
    lr: float
    weight_decay: float
    optimizer: str


@dataclass
class HebbianConfig:
    enabled: bool
    rule: str
    lr: float
    normalize_filters: bool
    per_batch_update: bool
    use_relu_output: bool
    partial_layers: List[str]
    mode: str


@dataclass
class AppConfig:
    seed: int
    device: str
    num_workers: int
    output_dir: str
    data: DataConfig
    model: ModelConfig
    train: TrainConfig
    hebbian: HebbianConfig


def _read_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(path: str | Path) -> AppConfig:
    raw = _read_yaml(path)

    return AppConfig(
        seed=raw["seed"],
        device=raw["device"],
        num_workers=raw["num_workers"],
        output_dir=raw["output_dir"],
        data=DataConfig(**raw["data"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        hebbian=HebbianConfig(**raw["hebbian"]),
    )


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    return torch.device(device_name)