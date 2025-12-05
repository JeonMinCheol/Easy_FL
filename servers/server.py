import socket
import random
import serverbase
from utils.data_utils import load_yaml_file

import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.communication import send_object, recv_object

# -------------------------------
# 서버 설정
# -------------------------------
DATA = load_yaml_file()
server_instance = serverbase.Server(DATA)

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((server_instance.ip, server_instance.port))
server_socket.listen(server_instance.num_clients)

print(f"Server listening on {server_instance.ip}:{server_instance.port}")

# -------------------------------
# 클라이언트 연결
# -------------------------------
while len(server_instance.client_sockets) < server_instance.num_clients:
    client_socket, addr = server_socket.accept()
    server_instance.client_sockets.append(client_socket)
    print(f"[Server] Client connected from {addr}. 현재 연결 수: {len(server_instance.client_sockets)}")

# 클라이언트 ID 전송
for idx, cs in enumerate(server_instance.client_sockets):
    cs.sendall(str(idx).encode())
    print(f"[Server] Assigned ID {idx} to client socket {cs.getpeername()}")

# -------------------------------
# 라운드 반복
# -------------------------------
while server_instance.current_round < server_instance.global_rounds:
    print(f"\n[Server] Starting Round {server_instance.current_round}")

    # 선택 클라이언트 결정
    if server_instance.current_round == 0:
        selected_clients = server_instance.client_sockets
    else:
        # num_join_clients may be > available clients; clamp it
        num_sel = random.randint(1, max(1, min(server_instance.num_join_clients, len(server_instance.client_sockets))))
        selected_clients = random.sample(server_instance.client_sockets, num_sel)

    sel_addrs = [cs.getpeername() for cs in selected_clients]
    print(f"[Server] Selected {len(selected_clients)} clients for this round: {sel_addrs}")

    # 0) (선택) 약간 쉬기 — 안정화
    time.sleep(0.05)

    # 1) 모든 클라이언트에게 현재 global params 전송 (동기화)
    current_params = server_instance.getModelParams()
    for cs in server_instance.client_sockets:
        try:
            print(f"[Server] Sending global params to {cs.getpeername()}")
            send_object(cs, current_params)
            time.sleep(0.02)
        except Exception as e:
            print(f"[Server] Error sending params to {cs.getpeername()}: {e}")

    # 2) 각 클라이언트에게 학습 여부 신호 전송 (selected -> '1', others -> '0')
    for cs in server_instance.client_sockets:
        try:
            if cs in selected_clients:
                cs.sendall(b'1')
            else:
                cs.sendall(b'0')
            time.sleep(0.01)
        except Exception as e:
            print(f"[Server] Error sending signal to {cs.getpeername()}: {e}")

    # 3) selected_clients로부터 파라미터 수신 (동기적으로)
    server_instance.uploaded_weights = []
    for cs in selected_clients:
        try:
            print(f"[Server] Waiting for upload from {cs.getpeername()} ...")
            w = recv_object(cs)
            if w is None:
                print(f"[Server] Warning: received None from {cs.getpeername()}")
            else:
                server_instance.uploaded_weights.append(w)
                print(f"[Server] Received weights from {cs.getpeername()}")
        except Exception as e:
            print(f"[Server] Error receiving from {cs.getpeername()}: {e}")

        # 소량의 휴지기 — 서버가 너무 빨리 다음 recv로 넘어가지 않도록
        time.sleep(0.01)

    # 4) FedAvg 적용
    print("[Server] FedAvg 실행 중...")
    try:
        server_instance.fedavg()
    except Exception as e:
        print(f"[Server] FedAvg error: {e}")

    # 5) 모든 클라이언트에게 업데이트된 global params 전송
    new_params = server_instance.getModelParams()
    for cs in server_instance.client_sockets:
        try:
            print(f"[Server] Sending updated params to {cs.getpeername()}")
            send_object(cs, new_params)
            time.sleep(0.02)
        except Exception as e:
            print(f"[Server] Error sending updated params to {cs.getpeername()}: {e}")

    server_instance.current_round += 1
    print(f"[Server] Round {server_instance.current_round} 완료")

# 종료 신호 전송
for cs in server_instance.client_sockets:
    try:
        cs.sendall(b'-1')
        time.sleep(0.01)
        cs.close()
    except Exception:
        pass

server_socket.close()
print("[Server] Training finished, server closed.")
