from __future__ import annotations

import copy
import os
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from utils.data_utils import (
    compute_threshold,
    infer_input_dim,
    read_client_data,
    resolve_artifact_dir,
    resolve_client_shard_path,
    seed_everything,
    tensor_dataset_from_npz,
)


class Client:
    """Federated client for KDD99 anomaly detection with an autoencoder."""

    def __init__(self, config: dict):
        server_configs = config["Server"]
        client_configs = config["Client"]
        dataset_configs = config["Dataset"]
        model_configs = config["Model"]

        seed_everything(int(server_configs.get("seed", 42)))

        self.config = copy.deepcopy(config)
        self.model_config = copy.deepcopy(model_configs)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[Client] device={self.device}")

        self.server_ip = server_configs["ip"]
        self.server_port = int(server_configs["port"])

        self.dataset = dataset_configs["name"]
        self.data_root = dataset_configs.get("data_root", "dataset")
        self.threshold_method = dataset_configs.get("threshold_method", "std")
        self.threshold_std_factor = float(dataset_configs.get("threshold_std_factor", 3.0))
        self.threshold_quantile = float(dataset_configs.get("threshold_quantile", 0.99))
        self.artifact_root = dataset_configs.get("artifact_root", "artifacts")

        self.timeout = int(client_configs["timeout"])
        self.batch_size = int(client_configs["batch_size"])
        self.learning_rate = float(client_configs["learning_rate"])
        self.local_epochs = int(client_configs["local_epochs"])

        self.input_dim = infer_input_dim(self.dataset, self.data_root)
        self.loss_fn = nn.MSELoss()
        self.model = build_model(model_configs, input_dim=self.input_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.id: int | None = None
        self.train_shard_path: Path | None = None
        self.test_shard_path: Path | None = None

    def assign_client_id(self, client_id: int) -> None:
        self.id = int(client_id)
        self.train_shard_path = resolve_client_shard_path(
            self.dataset,
            self.id,
            is_train=True,
            data_root=self.data_root,
        )
        self.test_shard_path = resolve_client_shard_path(
            self.dataset,
            self.id,
            is_train=False,
            data_root=self.data_root,
        )
        print(f"[Client {self.id}] train shard -> {self.train_shard_path}")
        print(f"[Client {self.id}] test shard  -> {self.test_shard_path}")

    def train(self) -> None:
        trainloader = self.load_train_data()
        self.model.train()

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            sample_count = 0

            for x, _ in trainloader:
                x = x.to(self.device, non_blocking=True)
                reconstruction = self.model(x)
                loss = self.loss_fn(reconstruction, x)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                batch_size = x.size(0)
                epoch_loss += loss.item() * batch_size
                sample_count += batch_size

            avg_loss = epoch_loss / max(sample_count, 1)
            print(f"[Client {self.id}] epoch {epoch + 1}/{self.local_epochs} | train_loss={avg_loss:.6f}")

    def getModelParams(self):
        return {k: v.detach().cpu() for k, v in self.model.state_dict().items()}

    def setModelParams(self, model_params):
        self.model.load_state_dict(model_params, strict=True)

    def load_train_data(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True, data_root=self.data_root)
        return DataLoader(
            train_data,
            batch_size=batch_size,
            drop_last=False,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),
            num_workers=0,
        )

    def load_test_data(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, data_root=self.data_root)
        return DataLoader(
            test_data,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=0,
        )

    def load_npz_data(self, npz_path: str | Path, batch_size=None):
        batch_size = batch_size or self.batch_size
        dataset = tensor_dataset_from_npz(npz_path)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            drop_last=False,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            num_workers=0,
        )

    def _collect_scores(self, dataloader) -> tuple[np.ndarray, np.ndarray]:
        self.model.eval()

        all_scores = []
        all_labels = []

        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device, non_blocking=True)
                reconstruction = self.model(x)
                per_sample_mse = torch.mean((reconstruction - x) ** 2, dim=1)

                all_scores.append(per_sample_mse.detach().cpu().numpy())
                all_labels.append(y.detach().cpu().numpy())

        scores = np.concatenate(all_scores, axis=0) if all_scores else np.array([], dtype=np.float32)
        labels = np.concatenate(all_labels, axis=0) if all_labels else np.array([], dtype=np.int64)
        return scores, labels

    def _compute_threshold_from_dataloader(self, dataloader) -> float:
        train_scores, train_labels = self._collect_scores(dataloader)
        normal_mask = train_labels == 0
        reference_scores = train_scores[normal_mask] if normal_mask.any() else train_scores

        return compute_threshold(
            reference_scores,
            method=self.threshold_method,
            std_factor=self.threshold_std_factor,
            quantile=self.threshold_quantile,
        )

    def _compute_local_threshold(self) -> float:
        return self._compute_threshold_from_dataloader(self.load_train_data())

    def train_metrics(self) -> Dict[str, float]:
        trainloader = self.load_train_data()
        train_scores, _ = self._collect_scores(trainloader)
        return {
            "loss": float(np.mean(train_scores)) if train_scores.size else float("nan"),
        }

    def _evaluate_with_threshold(self, dataloader, threshold: float) -> Dict[str, float]:
        scores, y_true = self._collect_scores(dataloader)
        y_pred = (scores > threshold).astype(np.int64)
        loss = float(np.mean(scores)) if scores.size else float("nan")
        accuracy = float(np.mean(y_pred == y_true)) if y_true.size and np.all(y_true >= 0) else float("nan")

        return {
            "loss": loss,
            "accuracy": accuracy,
            "threshold": float(threshold),
            "num_samples": int(scores.size),
            "num_pred_anomaly": int(np.sum(y_pred == 1)),
        }

    def test_metrics(self) -> Dict[str, float]:
        threshold = self._compute_local_threshold()
        return self._evaluate_with_threshold(self.load_test_data(), threshold)

    def evaluate_npz(self, npz_path: str | Path, threshold: float | None = None) -> Dict[str, float]:
        threshold = float(threshold) if threshold is not None else self._compute_local_threshold()
        dataloader = self.load_npz_data(npz_path)
        metrics = self._evaluate_with_threshold(dataloader, threshold)
        metrics["npz_path"] = str(npz_path)
        return metrics

    def save_local_artifact(self, round_idx: int | None = None, tag: str = "latest") -> Path:
        artifact_dir = resolve_artifact_dir(self.artifact_root) / "clients"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        threshold = self._compute_local_threshold()
        payload = {
            "kind": "client_autoencoder_checkpoint",
            "client_id": self.id,
            "round_idx": round_idx,
            "dataset": self.dataset,
            "data_root": self.data_root,
            "input_dim": self.input_dim,
            "model_config": self.model_config,
            "model_state_dict": self.getModelParams(),
            "threshold": float(threshold),
            "threshold_method": self.threshold_method,
            "threshold_std_factor": self.threshold_std_factor,
            "threshold_quantile": self.threshold_quantile,
            "train_shard_path": str(self.train_shard_path) if self.train_shard_path else None,
            "test_shard_path": str(self.test_shard_path) if self.test_shard_path else None,
        }
        path = artifact_dir / f"client_{self.id}_{tag}.pt"
        torch.save(payload, path)
        return path
