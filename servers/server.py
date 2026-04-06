from __future__ import annotations

import os
import random
import socket
import sys
import time

import serverbase

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.communication import recv_object, send_object
from utils.data_utils import load_yaml_file

DATA = load_yaml_file()
server_instance = serverbase.Server(DATA)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
server_socket.bind(("0.0.0.0", server_instance.port))
print(f"configs.yml 파일의 아이피를 수정해주세요. ip: {server_instance.ip}")
server_socket.listen(server_instance.num_clients)

print(f"Server listening on {server_instance.ip}:{server_instance.port}")

while len(server_instance.client_sockets) < server_instance.num_clients:
    client_socket, addr = server_socket.accept()
    server_instance.client_sockets.append(client_socket)
    print(f"[Server] Client connected from {addr}. 현재 연결 수: {len(server_instance.client_sockets)}")

for idx, cs in enumerate(server_instance.client_sockets):
    cs.sendall(str(idx).encode())
    print(f"[Server] Assigned ID {idx} to client socket {cs.getpeername()}")

while server_instance.current_round < server_instance.global_rounds:
    round_idx = server_instance.current_round + 1
    print(f"\n[Server] Starting Round {round_idx}/{server_instance.global_rounds}")

    num_sel = min(server_instance.num_join_clients, len(server_instance.client_sockets))
    selected_clients = random.sample(server_instance.client_sockets, num_sel)
    sel_addrs = [cs.getpeername() for cs in selected_clients]
    print(f"[Server] Selected {len(selected_clients)} clients: {sel_addrs}")

    current_params = server_instance.getModelParams()
    for cs in server_instance.client_sockets:
        try:
            send_object(cs, current_params)
            time.sleep(0.01)
        except Exception as exc:
            print(f"[Server] Error sending params to {cs.getpeername()}: {exc}")

    for cs in server_instance.client_sockets:
        try:
            cs.sendall(b"1" if cs in selected_clients else b"0")
            time.sleep(0.005)
        except Exception as exc:
            print(f"[Server] Error sending signal to {cs.getpeername()}: {exc}")

    server_instance.uploaded_weights = []
    for cs in selected_clients:
        try:
            print(f"[Server] Waiting for upload from {cs.getpeername()} ...")
            weights = recv_object(cs)
            if weights is None:
                print(f"[Server] Warning: received None from {cs.getpeername()}")
            else:
                server_instance.uploaded_weights.append(weights)
                print(f"[Server] Received weights from {cs.getpeername()}")
        except Exception as exc:
            print(f"[Server] Error receiving from {cs.getpeername()}: {exc}")
        time.sleep(0.005)

    print("[Server] Running FedAvg...")
    server_instance.fedavg()
    saved_path = server_instance.save_global_checkpoint(round_idx=round_idx, tag="latest")
    print(f"[Server] checkpoint saved -> {saved_path}")

    new_params = server_instance.getModelParams()
    for cs in server_instance.client_sockets:
        try:
            send_object(cs, new_params)
            time.sleep(0.01)
        except Exception as exc:
            print(f"[Server] Error sending updated params to {cs.getpeername()}: {exc}")

    server_instance.current_round += 1
    print(f"[Server] Round {server_instance.current_round} 완료")

for cs in server_instance.client_sockets:
    try:
        cs.sendall(b"-1")
        time.sleep(0.005)
        cs.close()
    except Exception:
        pass

server_socket.close()
print("[Server] Training finished, server closed.")
