\
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder

KDD99_COLUMNS = [
    "duration",
    "protocol_type",
    "service",
    "flag",
    "src_bytes",
    "dst_bytes",
    "land",
    "wrong_fragment",
    "urgent",
    "hot",
    "num_failed_logins",
    "logged_in",
    "num_compromised",
    "root_shell",
    "su_attempted",
    "num_root",
    "num_file_creations",
    "num_shells",
    "num_access_files",
    "num_outbound_cmds",
    "is_host_login",
    "is_guest_login",
    "count",
    "srv_count",
    "serror_rate",
    "srv_serror_rate",
    "rerror_rate",
    "srv_rerror_rate",
    "same_srv_rate",
    "diff_srv_rate",
    "srv_diff_host_rate",
    "dst_host_count",
    "dst_host_srv_count",
    "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate",
    "dst_host_srv_serror_rate",
    "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate",
]

CATEGORICAL_COLUMNS = ["protocol_type", "service", "flag"]
NORMAL_LABELS = {"normal", "normal."}


def read_kdd_dataframe(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, header=None)

    if df.shape[1] in (42, 43):
        columns = KDD99_COLUMNS + ["label"]
        if df.shape[1] == 43:
            columns += ["difficulty"]

        first_row = [str(v).strip().lower() for v in df.iloc[0].tolist()[:4]]
        if first_row == ["duration", "protocol_type", "service", "flag"]:
            df = df.iloc[1:].reset_index(drop=True)

        df.columns = columns
    else:
        df = pd.read_csv(csv_path)
        if "label" not in df.columns:
            raise ValueError(
                f"Could not infer label column from {csv_path}. "
                "Expected KDD99/NSL-KDD format with 42 or 43 columns, or a CSV with a 'label' column."
            )

    df["label"] = df["label"].astype(str)
    numeric_cols = [c for c in KDD99_COLUMNS if c not in CATEGORICAL_COLUMNS and c in df.columns]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df


def binary_labels(label_series: pd.Series) -> np.ndarray:
    labels = label_series.astype(str).str.strip().str.lower()
    return (~labels.isin(NORMAL_LABELS)).astype(np.int64).to_numpy()


def sample_train_with_contamination(
    train_df: pd.DataFrame,
    train_anomaly_fraction: float,
    seed: int,
) -> pd.DataFrame:
    y_bin = binary_labels(train_df["label"])
    normal_df = train_df[y_bin == 0].copy()
    anomaly_df = train_df[y_bin == 1].copy()

    if train_anomaly_fraction <= 0 or anomaly_df.empty:
        return normal_df.reset_index(drop=True)

    target_anomaly_count = int(len(normal_df) * train_anomaly_fraction / max(1e-12, 1.0 - train_anomaly_fraction))
    target_anomaly_count = min(target_anomaly_count, len(anomaly_df))

    anomaly_sample = anomaly_df.sample(n=target_anomaly_count, random_state=seed)
    merged = pd.concat([normal_df, anomaly_sample], axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return merged


def build_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    test_size: float,
    seed: int,
    train_anomaly_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      fit_reference_df: dataframe used to fit one-hot encoder + scaler
      train_runtime_df: dataframe used for autoencoder local training
      test_runtime_df: dataframe used for local evaluation
    """
    if test_df is not None:
        fit_reference_df = train_df.copy().reset_index(drop=True)
        train_runtime_df = sample_train_with_contamination(train_df, train_anomaly_fraction, seed)
        test_runtime_df = test_df.copy().reset_index(drop=True)
        return fit_reference_df, train_runtime_df, test_runtime_df

    y_bin = binary_labels(train_df["label"])
    normal_df = train_df[y_bin == 0].copy().reset_index(drop=True)
    anomaly_df = train_df[y_bin == 1].copy().reset_index(drop=True)

    normal_train, normal_test = train_test_split(
        normal_df,
        test_size=test_size,
        random_state=seed,
        shuffle=True,
    )

    fit_reference_df = train_df.copy().reset_index(drop=True)
    train_runtime_df = sample_train_with_contamination(normal_train, train_anomaly_fraction, seed)

    test_parts = [normal_test]
    if not anomaly_df.empty:
        test_parts.append(anomaly_df)
    test_runtime_df = pd.concat(test_parts, axis=0).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return fit_reference_df, train_runtime_df, test_runtime_df


def make_preprocessor(feature_columns):
    categorical = [col for col in CATEGORICAL_COLUMNS if col in feature_columns]
    numeric = [col for col in feature_columns if col not in categorical]

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformer = ColumnTransformer(
        transformers=[
            ("cat", encoder, categorical),
            ("num", "passthrough", numeric),
        ],
        remainder="drop",
    )
    return transformer, categorical, numeric


def stratified_partition_indices(y: np.ndarray, num_clients: int, seed: int):
    rng = np.random.default_rng(seed)
    bucket = [[] for _ in range(num_clients)]

    for cls in np.unique(y):
        cls_indices = np.where(y == cls)[0]
        rng.shuffle(cls_indices)
        splits = np.array_split(cls_indices, num_clients)
        for client_id, split in enumerate(splits):
            bucket[client_id].extend(split.tolist())

    for client_id in range(num_clients):
        rng.shuffle(bucket[client_id])

    return [np.array(indices, dtype=np.int64) for indices in bucket]


def random_partition_indices(n_samples: int, num_clients: int, seed: int):
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    return [split.astype(np.int64) for split in np.array_split(indices, num_clients)]


def save_shards(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    num_clients: int,
    seed: int,
) -> dict:
    train_dir = output_dir / "train"
    test_dir = output_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    train_splits = random_partition_indices(len(x_train), num_clients, seed)
    test_splits = stratified_partition_indices(y_test, num_clients, seed)

    train_sizes = {}
    test_sizes = {}
    test_anomaly_counts = {}

    for client_id in range(num_clients):
        tr_idx = train_splits[client_id]
        te_idx = test_splits[client_id]

        np.savez_compressed(
            train_dir / f"{client_id}.npz",
            x=x_train[tr_idx].astype(np.float32),
            y=y_train[tr_idx].astype(np.int64),
        )
        np.savez_compressed(
            test_dir / f"{client_id}.npz",
            x=x_test[te_idx].astype(np.float32),
            y=y_test[te_idx].astype(np.int64),
        )

        train_sizes[str(client_id)] = int(len(tr_idx))
        test_sizes[str(client_id)] = int(len(te_idx))
        test_anomaly_counts[str(client_id)] = int(y_test[te_idx].sum())

    return {
        "train_sizes": train_sizes,
        "test_sizes": test_sizes,
        "test_anomaly_counts": test_anomaly_counts,
    }


def maybe_update_config(config_path: Path, dataset_name: str, data_root: str, num_clients: int):
    if not config_path.exists():
        return

    with open(config_path, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    config.setdefault("Dataset", {})
    config["Dataset"]["name"] = dataset_name
    config["Dataset"]["data_root"] = data_root

    config.setdefault("Server", {})
    config["Server"]["number_of_clients"] = int(num_clients)

    with open(config_path, "w", encoding="utf-8") as file:
        yaml.safe_dump(config, file, sort_keys=False, allow_unicode=True)


def main():
    parser = argparse.ArgumentParser(description="Prepare KDD99/NSL-KDD data for federated autoencoder training.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to raw KDD99/NSL-KDD training CSV/TXT file.")
    parser.add_argument("--test_csv", type=str, default=None, help="Optional path to raw KDD99/NSL-KDD test CSV/TXT file.")
    parser.add_argument("--output_dir", type=str, default="dataset/KDD99", help="Output directory for federated shards.")
    parser.add_argument("--num_clients", type=int, default=3, help="Number of federated clients.")
    parser.add_argument("--test_size", type=float, default=0.2, help="Used only when --test_csv is not provided.")
    parser.add_argument(
        "--train_anomaly_fraction",
        type=float,
        default=0.0,
        help="Fraction of anomalies kept in the training set. 0.0 means normal-only training.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--config_path", type=str, default="config.yml", help="Config file to update.")
    parser.add_argument("--dataset_name", type=str, default="KDD99", help="Dataset name used by the FL code.")
    args = parser.parse_args()

    train_path = Path(args.train_csv).resolve()
    test_path = Path(args.test_csv).resolve() if args.test_csv else None
    output_dir = Path(args.output_dir).resolve()
    config_path = Path(args.config_path).resolve()

    raw_train_df = read_kdd_dataframe(train_path)
    raw_test_df = read_kdd_dataframe(test_path) if test_path else None

    fit_reference_df, train_runtime_df, test_runtime_df = build_train_test(
        train_df=raw_train_df,
        test_df=raw_test_df,
        test_size=args.test_size,
        seed=args.seed,
        train_anomaly_fraction=args.train_anomaly_fraction,
    )

    feature_columns = [col for col in fit_reference_df.columns if col not in {"label", "difficulty"}]
    preprocessor, categorical_cols, numeric_cols = make_preprocessor(feature_columns)

    x_fit = preprocessor.fit_transform(fit_reference_df[feature_columns])
    x_train = preprocessor.transform(train_runtime_df[feature_columns])
    x_test = preprocessor.transform(test_runtime_df[feature_columns])

    scaler = MinMaxScaler()
    scaler.fit(x_fit)
    x_train = scaler.transform(x_train).astype(np.float32)
    x_test = scaler.transform(x_test).astype(np.float32)

    y_train = binary_labels(train_runtime_df["label"])
    y_test = binary_labels(test_runtime_df["label"])

    output_dir.mkdir(parents=True, exist_ok=True)

    shard_info = save_shards(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        output_dir=output_dir,
        num_clients=args.num_clients,
        seed=args.seed,
    )

    preprocessing_obj = {
        "feature_columns": feature_columns,
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
        "preprocessor": preprocessor,
        "scaler": scaler,
    }
    with open(output_dir / "preprocessing.pkl", "wb") as file:
        pickle.dump(preprocessing_obj, file)

    metadata = {
        "dataset_name": args.dataset_name,
        "input_dim": int(x_train.shape[1]),
        "num_clients": int(args.num_clients),
        "train_total": int(len(x_train)),
        "test_total": int(len(x_test)),
        "train_anomaly_fraction": float(args.train_anomaly_fraction),
        "train_anomaly_count": int(y_train.sum()),
        "test_anomaly_count": int(y_test.sum()),
        "categorical_columns": categorical_cols,
        "numeric_columns": numeric_cols,
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

    print("=== Federated KDD99 preprocessing complete ===")
    print(f"Output directory: {output_dir}")
    print(f"Input dimension after one-hot encoding: {x_train.shape[1]}")
    print(f"Train samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Train anomalies kept: {int(y_train.sum())}")
    print(f"Test anomalies: {int(y_test.sum())}")
    print(f"Per-client train sizes: {shard_info['train_sizes']}")
    print(f"Per-client test sizes: {shard_info['test_sizes']}")
    print(f"Per-client test anomalies: {shard_info['test_anomaly_counts']}")


if __name__ == "__main__":
    main()
