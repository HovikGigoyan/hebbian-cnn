from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import numpy as np
import torch
import json
import random


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_output_dirs(base_dir: str | Path) -> Dict[str, Path]:
    base = ensure_dir(base_dir)

    dirs = {
        "base": base,
        "checkpoints": ensure_dir(base / "checkpoints"),
        "metrics": ensure_dir(base / "metrics"),
        "plots": ensure_dir(base / "plots"),
        "filters": ensure_dir(base / "filters"),
        "predictions": ensure_dir(base / "predictions"),
    }
    return dirs


def save_json(path: str | Path, payload: Dict[str, Any]) -> None:
    path = Path(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def count_parameters(model: torch.nn.Module, only_trainable: bool = False) -> int:
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return type(batch)(to_device(x, device) for x in batch)
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    return batch