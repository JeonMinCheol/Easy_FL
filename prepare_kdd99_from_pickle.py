from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from prepare_kdd99_federated import maybe_update_config, save_shards


NORMAL_CANDIDATES = ["normal.", "normal"]


def _to_numpy(x):
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=np.float32), list(x.columns)
    if isinstance(x, np.ndarray):
        return x.astype(np.float32), [f"f{i}" for i in range(x.shape[1])]
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D features, got shape={arr.shape}")
    return arr, [f"f{i}" for i in range(arr.shape[1])]


def _to_vector(y):
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = np.asarray(y).reshape(-1)
    else:
        y = np.asarray(y).reshape(-1)
    return y.astype(np.int64)


def infer_normal_index(label_encoder, user_normal_label: str | None = None) -> tuple[int, str]:
    classes = [str(c) for c in label_encoder.classes_]

    search_order = []
    if user_normal_label is not None:
        search_order.append(user_normal_label)
    search_order.extend(NORMAL_CANDIDATES)

    for candidate in search_order:
        if candidate in classes:
            return classes.index(candidate), candidate

    raise ValueError(
        "Could not find the normal label inside LabelEncoder classes. "
        f"Available classes: {classes}. Pass --normal_label explicitly."
    )


def binarize_labels(y: np.ndarray, normal_index: int) -> np.ndarray:
    return (y != normal_index).astype(np.int64)


def apply_train_contamination(
    x_train: np.ndarray,
    y_train_bin: np.ndarray,
    train_anomaly_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    normal_idx = np.where(y_train_bin == 0)[0]
    anomaly_idx = np.where(y_train_bin == 1)[0]

    if train_anomaly_fraction <= 0.0 or anomaly_idx.size == 0:
        keep_idx = normal_idx
    else:
        target_anomaly_count = int(
            len(normal_idx) * train_anomaly_fraction / max(1e-12, 1.0 - train_anomaly_fraction)
        )
        target_anomaly_count = min(target_anomaly_count, len(anomaly_idx))
        sampled_anomaly_idx = rng.choice(anomaly_idx, size=target_anomaly_count, replace=False)
        keep_idx = np.concatenate([normal_idx, sampled_anomaly_idx], axis=0)
        rng.shuffle(keep_idx)

    return x_train[keep_idx], y_train_bin[keep_idx]


def main():
    parser = argparse.ArgumentParser(
        description="Create Easy_FL client shards directly from preprocessed_data_full.pkl"
    )
    parser.add_argument(
        "--pickle_path",
        type=str,
        required=True,
        help="Path to preprocessed_data_full.pkl created by Lab1-preprocess_data.py",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="dataset/KDD99",
        help="Output directory for train/test client shards.",
    )
    parser.add_argument("--num_clients", type=int, default=3, help="Number of federated clients.")
    parser.add_argument(
        "--train_anomaly_fraction",
        type=float,
        default=0.0,
        help="Fraction of anomalies kept in local training shards after binarization.",
    )
    parser.add_argument(
        "--normal_label",
        type=str,
        default=None,
        help="Optional explicit normal label string inside LabelEncoder classes, e.g. 'normal.'.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--config_path", type=str, default="config.yml", help="Config file to update.")
    parser.add_argument("--dataset_name", type=str, default="KDD99", help="Dataset name used by FL code.")
    args = parser.parse_args()

    pickle_path = Path(args.pickle_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config_path).resolve()

    with open(pickle_path, "rb") as file:
        data = pickle.load(file)

    required_keys = {"x_train", "y_train", "x_test", "y_test", "le"}
    missing = required_keys.difference(data.keys())
    if missing:
        raise KeyError(f"Missing keys in pickle file: {sorted(missing)}")

    x_train_raw, feature_columns = _to_numpy(data["x_train"])
    x_test_raw, _ = _to_numpy(data["x_test"])
    y_train_raw = _to_vector(data["y_train"])
    y_test_raw = _to_vector(data["y_test"])
    label_encoder = data["le"]

    normal_index, normal_label = infer_normal_index(label_encoder, args.normal_label)
    y_train_bin = binarize_labels(y_train_raw, normal_index)
    y_test_bin = binarize_labels(y_test_raw, normal_index)

    scaler = MinMaxScaler()
    scaler.fit(x_train_raw)
    x_train_scaled = scaler.transform(x_train_raw).astype(np.float32)
    x_test_scaled = scaler.transform(x_test_raw).astype(np.float32)

    x_train_runtime, y_train_runtime = apply_train_contamination(
        x_train_scaled,
        y_train_bin,
        train_anomaly_fraction=args.train_anomaly_fraction,
        seed=args.seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    shard_info = save_shards(
        x_train=x_train_runtime,
        y_train=y_train_runtime,
        x_test=x_test_scaled,
        y_test=y_test_bin,
        output_dir=output_dir,
        num_clients=args.num_clients,
        seed=args.seed,
    )

    preprocessing_obj = {
        "feature_columns": feature_columns,
        "scaler": scaler,
        "source_pickle": str(pickle_path),
        "normal_label": normal_label,
        "normal_index": int(normal_index),
        "original_classes": [str(c) for c in label_encoder.classes_],
    }
    with open(output_dir / "preprocessing.pkl", "wb") as file:
        pickle.dump(preprocessing_obj, file)

    metadata = {
        "dataset_name": args.dataset_name,
        "source_pickle": str(pickle_path),
        "input_dim": int(x_train_runtime.shape[1]),
        "num_clients": int(args.num_clients),
        "train_total": int(len(x_train_runtime)),
        "test_total": int(len(x_test_scaled)),
        "train_anomaly_fraction": float(args.train_anomaly_fraction),
        "train_anomaly_count": int(y_train_runtime.sum()),
        "test_anomaly_count": int(y_test_bin.sum()),
        "normal_label": normal_label,
        "normal_index": int(normal_index),
        "feature_count": int(x_train_runtime.shape[1]),
        **shard_info,
    }

    with open(output_dir / "metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    maybe_update_config(
        config_path=config_path,
        dataset_name=args.dataset_name,
        data_root=Path(args.output_dir).parts[0] if len(Path(args.output_dir).parts) > 1 else "dataset",
        num_clients=args.num_clients,
    )

    print("=== Federated shard generation from pickle complete ===")
    print(f"Source pickle: {pickle_path}")
    print(f"Detected normal label: {normal_label} (index={normal_index})")
    print(f"Output directory: {output_dir}")
    print(f"Input dimension: {x_train_runtime.shape[1]}")
    print(f"Train samples used for AE training: {len(x_train_runtime)}")
    print(f"Test samples: {len(x_test_scaled)}")
    print(f"Train anomalies kept: {int(y_train_runtime.sum())}")
    print(f"Test anomalies: {int(y_test_bin.sum())}")
    print(f"Per-client train sizes: {shard_info['train_sizes']}")
    print(f"Per-client test sizes: {shard_info['test_sizes']}")
    print(f"Per-client test anomalies: {shard_info['test_anomaly_counts']}")


if __name__ == "__main__":
    main()
