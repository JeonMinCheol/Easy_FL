from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from utils.data_utils import compute_threshold, load_yaml_file, read_client_data, read_npz_file


def build_dataloader_from_npz(npz_path: str | Path, batch_size: int = 512) -> DataLoader:
    x, y = read_npz_file(npz_path)
    dataset = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.int64),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)


def collect_scores(model, dataloader, device: str) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            recon = model(x)
            scores = torch.mean((recon - x) ** 2, dim=1)
            all_scores.append(scores.detach().cpu().numpy())
            all_labels.append(y.detach().cpu().numpy())

    score_array = np.concatenate(all_scores, axis=0) if all_scores else np.array([], dtype=np.float32)
    label_array = np.concatenate(all_labels, axis=0) if all_labels else np.array([], dtype=np.int64)
    return score_array, label_array


def main() -> None:
    parser = argparse.ArgumentParser(description="Run anomaly detection with a saved FL autoencoder checkpoint.")
    parser.add_argument("--checkpoint_path", required=True, help="Path to a saved .pt checkpoint")
    parser.add_argument("--client_id", type=int, default=None, help="Client id for using local shard data")
    parser.add_argument("--split", choices=["train", "test"], default="test")
    parser.add_argument("--npz_path", default=None, help="Optional path to a specific .npz file to evaluate")
    parser.add_argument("--threshold", type=float, default=None, help="Optional manual threshold override")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--config_path", default=None)
    args = parser.parse_args()

    config = load_yaml_file(args.config_path)
    dataset_cfg = config["Dataset"]
    data_root = dataset_cfg.get("data_root", "dataset")
    dataset_name = dataset_cfg["name"]
    threshold_method = dataset_cfg.get("threshold_method", "std")
    threshold_std_factor = float(dataset_cfg.get("threshold_std_factor", 3.0))
    threshold_quantile = float(dataset_cfg.get("threshold_quantile", 0.99))

    checkpoint = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    input_dim = int(checkpoint["input_dim"])
    model_cfg = checkpoint.get("model_config", config["Model"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(model_cfg, input_dim=input_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)

    if args.npz_path:
        eval_loader = build_dataloader_from_npz(args.npz_path, batch_size=args.batch_size)
        eval_name = args.npz_path
    else:
        if args.client_id is None:
            raise ValueError("Either --npz_path or --client_id must be provided.")
        is_train = args.split == "train"
        dataset = read_client_data(dataset_name, args.client_id, is_train=is_train, data_root=data_root)
        eval_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        eval_name = f"{dataset_name}:{args.split}:client_{args.client_id}"

    if args.threshold is not None:
        threshold = float(args.threshold)
    elif "threshold" in checkpoint:
        threshold = float(checkpoint["threshold"])
    else:
        if args.client_id is None:
            raise ValueError("Threshold missing in checkpoint. Provide --threshold or --client_id.")
        train_dataset = read_client_data(dataset_name, args.client_id, is_train=True, data_root=data_root)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)
        train_scores, train_labels = collect_scores(model, train_loader, device)
        normal_mask = train_labels == 0
        ref_scores = train_scores[normal_mask] if normal_mask.any() else train_scores
        threshold = compute_threshold(
            ref_scores,
            method=threshold_method,
            std_factor=threshold_std_factor,
            quantile=threshold_quantile,
        )

    scores, labels = collect_scores(model, eval_loader, device)
    preds = (scores > threshold).astype(np.int64)
    loss = float(np.mean(scores)) if scores.size else float("nan")

    print(f"checkpoint={args.checkpoint_path}")
    print(f"eval_data={eval_name}")
    print(f"threshold={threshold:.6f}")
    print(f"loss={loss:.6f}")

    if labels.size and np.all(labels >= 0):
        acc = float(np.mean(preds == labels))
        print(f"acc={acc:.4f}")
    else:
        print("acc=nan (labels not available)")

    print(f"predicted_anomalies={int(np.sum(preds == 1))}/{len(preds)}")


if __name__ == "__main__":
    main()
