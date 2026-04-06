from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import torch
import requests

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import build_model
from utils.data_utils import infer_input_dim, resolve_artifact_dir, seed_everything

def get_external_ip():
    try:
        response = requests.get('https://api.ipify.org')
        external_ip = response.text
        return external_ip
    except requests.RequestException as e:
        print(f"외부 IP 주소를 가져오는 중 오류가 발생했습니다: {e}")
        return None
    
class Server:
    def __init__(self, configs: dict):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        server_configs = configs["Server"]
        dataset_configs = configs["Dataset"]
        model_configs = configs["Model"]

        seed_everything(int(server_configs.get("seed", 42)))

        self.config = copy.deepcopy(configs)
        self.model_config = copy.deepcopy(model_configs)
        self.ip = get_external_ip()
        self.port = int(server_configs["port"])
        self.timeout = int(server_configs["timeout"])
        self.global_rounds = int(server_configs["rounds"])
        self.join_ratio = float(server_configs["join_ratio"])
        self.num_clients = int(server_configs["number_of_clients"])
        self.num_join_clients = max(1, int(round(self.num_clients * self.join_ratio)))

        self.dataset = dataset_configs["name"]
        self.data_root = dataset_configs.get("data_root", "dataset")
        self.artifact_root = dataset_configs.get("artifact_root", "artifacts")
        self.input_dim = infer_input_dim(self.dataset, self.data_root)

        self.global_model = copy.deepcopy(build_model(model_configs, input_dim=self.input_dim)).to(self.device)
        self.current_round = 0
        self.client_sockets = []
        self.uploaded_weights = []

    def getModelParams(self):
        return {k: v.detach().cpu() for k, v in self.global_model.state_dict().items()}

    def fedavg(self):
        if not self.uploaded_weights:
            print("[Server] No uploaded weights received. Skipping FedAvg.")
            return self.getModelParams()

        new_state = {}
        first_state = self.uploaded_weights[0]

        for key in first_state.keys():
            avg_tensor = sum(client_state[key] for client_state in self.uploaded_weights) / len(self.uploaded_weights)
            new_state[key] = avg_tensor

        self.global_model.load_state_dict(new_state, strict=True)
        return new_state

    def save_global_checkpoint(self, round_idx: int | None = None, tag: str = "latest") -> Path:
        artifact_dir = resolve_artifact_dir(self.artifact_root) / "server"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        path = artifact_dir / f"global_model_{tag}.pt"
        payload = {
            "kind": "server_global_autoencoder_checkpoint",
            "round_idx": round_idx,
            "dataset": self.dataset,
            "data_root": self.data_root,
            "input_dim": self.input_dim,
            "model_config": self.model_config,
            "model_state_dict": self.getModelParams(),
            "port": self.port,
        }
        torch.save(payload, path)
        return path
