import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.data_utils import read_client_data
from models import MLP

class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, config):
        np.random.seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(self.device)

        server_configs = config['Server']
        client_configs = config['Client']
        dataset_configs = config['Dataset']
        model_configs = config['Model']

        self.server_ip = server_configs['ip']
        self.server_port = server_configs['port']

        self.dataset = dataset_configs['name']
        self.num_classes = dataset_configs['num_classes']

        self.timeout = client_configs['timeout']
        self.batch_size = client_configs['batch_size']
        self.learning_rate = client_configs['learning_rate']
        self.local_epochs = client_configs['local_epochs']

        self.loss = nn.CrossEntropyLoss()
        self.model = MLP(num_classes=dataset_configs["num_classes"], in_features=model_configs["in_features"], hidden_dim=model_configs["hidden_dim"]).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        for step in range(self.local_epochs):
            print(f"step {step+1}/{self.local_epochs} started.")
            for i, (x, y) in enumerate(trainloader):
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def getModelParams(self):
        return self.model.state_dict()
    
    def setModelParams(self, model_params):
        self.model.load_state_dict(model_params)

    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        return DataLoader(train_data, batch_size, drop_last=False, shuffle=False, pin_memory=True, num_workers=0)

    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True, pin_memory=True, num_workers=0)

    def train_metrics(self):
        trainloader = self.load_train_data()
        self.model.to(self.device, non_blocking=True)
        self.model.eval()

        train_correct = 0
        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device, non_blocking=True)
                else:
                    x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)

                output = self.model(x)   
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]
                train_correct += (torch.sum(torch.argmax(output, dim=1) == y)).detach()
        train_correct = train_correct.item() if isinstance(train_correct, torch.Tensor) else train_correct

        return train_correct / train_num, losses

    def test_metrics(self):
        # 1) 테스트 로더 준비
        testloader = self.load_test_data()
        self.model.to(self.device, non_blocking=True)
        self.model.eval()

        # 2) 결과 저장용
        test_correct = 0
        test_num     = 0

        with torch.no_grad():
            for x, y in testloader:
                if isinstance(x, list):
                    x = x[0]
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                output = self.model(x)   

                # (d) accuracy 집계
                test_correct += (output.argmax(dim=1) == y).sum().detach()
                test_num     += y.size(0)

        test_correct = test_correct.item() if isinstance(test_correct, torch.Tensor) else test_correct
        return test_correct / test_num
