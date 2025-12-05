import torch
import copy
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import MLP

class Server(object):
    def __init__(self, configs):
        self.aggregation_times = []
        self.aggregation_memories = []

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        server_configs = configs['Server']
        dataset_configs = configs['Dataset']
        model_configs = configs['Model']

        self.ip = server_configs["ip"]
        self.port = server_configs["port"]
        self.timeout = server_configs["timeout"]
        self.global_rounds = server_configs["rounds"]
        self.join_ratio = server_configs["join_ratio"]
        self.num_clients = server_configs["number_of_clients"]
        self.num_join_clients =  max(1, int(self.num_clients * self.join_ratio))
        self.global_model = copy.deepcopy(MLP(num_classes=dataset_configs["num_classes"], in_features=model_configs["in_features"], hidden_dim=model_configs["hidden_dim"])).to(self.device)
        self.current_round = 0
        self.client_sockets = []
        self.uploaded_weights = []

    def getModelParams(self):
        return self.global_model.state_dict()

    def fedavg(self):
        new_state = {}

        # 첫 클라이언트 기준으로 키 생성
        for key in self.uploaded_weights[0].keys():
            # 모든 클라이언트 파라미터 평균
            new_state[key] = sum(w[key] for w in self.uploaded_weights) / len(self.uploaded_weights)
        
        self.global_model.load_state_dict(new_state)
        return new_state