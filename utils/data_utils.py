from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_yaml_file(config_path: Optional[str] = None) -> dict:
    path = Path(config_path) if config_path is not None else PROJECT_ROOT / "config.yml"
    with open(path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_dataset_dir(dataset_name: str, data_root: str = "dataset") -> Path:
    return PROJECT_ROOT / data_root / dataset_name


def resolve_artifact_dir(artifact_root: str = "artifacts") -> Path:
    artifact_dir = PROJECT_ROOT / artifact_root
    artifact_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir


def read_metadata(dataset_name: str, data_root: str = "dataset") -> dict:
    metadata_path = resolve_dataset_dir(dataset_name, data_root) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found at {metadata_path}. "
            f"Run prepare_kdd99_federated.py first."
        )
    with open(metadata_path, "r", encoding="utf-8") as file:
        return json.load(file)


def infer_input_dim(dataset_name: str, data_root: str = "dataset") -> int:
    metadata = read_metadata(dataset_name, data_root)
    return int(metadata["input_dim"])


def _read_npz(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Client shard not found: {path}")
    data = np.load(path, allow_pickle=False)
    x = data["x"].astype(np.float32)
    y = data["y"].astype(np.int64) if "y" in data.files else np.full((x.shape[0],), -1, dtype=np.int64)
    return x, y


def read_npz_file(npz_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    return _read_npz(Path(npz_path))


def _natural_shard_sort_key(path: Path):
    stem = path.stem
    try:
        return (0, int(stem))
    except ValueError:
        return (1, stem)


def list_client_shard_paths(
    dataset_name: str,
    is_train: bool = True,
    data_root: str = "dataset",
) -> list[Path]:
    split = "train" if is_train else "test"
    split_dir = resolve_dataset_dir(dataset_name, data_root) / split
    shard_paths = sorted(split_dir.glob("*.npz"), key=_natural_shard_sort_key)
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found under {split_dir}")
    return shard_paths


def resolve_client_shard_path(
    dataset_name: str,
    client_id: int,
    is_train: bool = True,
    data_root: str = "dataset",
) -> Path:
    split = "train" if is_train else "test"
    split_dir = resolve_dataset_dir(dataset_name, data_root) / split
    exact_path = split_dir / f"{client_id}.npz"
    if exact_path.exists():
        return exact_path

    shard_paths = list_client_shard_paths(dataset_name, is_train=is_train, data_root=data_root)
    mapped_index = client_id % len(shard_paths)
    return shard_paths[mapped_index]


def read_client_data(
    dataset_name: str,
    idx: int,
    is_train: bool = True,
    data_root: str = "dataset",
) -> TensorDataset:
    npz_path = resolve_client_shard_path(dataset_name, idx, is_train=is_train, data_root=data_root)
    x, y = _read_npz(npz_path)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    return TensorDataset(x_tensor, y_tensor)


def tensor_dataset_from_npz(npz_path: str | Path) -> TensorDataset:
    x, y = read_npz_file(npz_path)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.int64)
    return TensorDataset(x_tensor, y_tensor)


def compute_threshold(
    scores: np.ndarray,
    method: str = "std",
    std_factor: float = 3.0,
    quantile: float = 0.99,
) -> float:
    if scores.size == 0:
        raise ValueError("Cannot compute threshold from an empty score array.")

    method = method.lower()
    if method == "std":
        return float(scores.mean() + std_factor * scores.std(ddof=0))
    if method == "quantile":
        return float(np.quantile(scores, quantile))
    raise ValueError(f"Unsupported threshold method: {method}")
