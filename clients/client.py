from __future__ import annotations

import os
import socket
import sys
import time
from pathlib import Path

import yaml

import clientbase

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.communication import recv_object, send_object


def load_yaml_file():
    current_file = Path(__file__).resolve()
    file_path = current_file.parent.parent / "config.yml"
    with open(file_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


DATA = load_yaml_file()


def _print_test_metrics(client_instance, metrics, tag: str) -> None:
    print(
        f"[Client {client_instance.id}] {tag} | "
        f"loss={metrics['loss']:.6f} | "
        f"acc={metrics['accuracy']:.4f}"
    )


def start_client():
    client_instance = clientbase.Client(config=DATA)
    round_idx = 0

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            print(f"서버 {client_instance.server_ip}:{client_instance.server_port}에 연결 시도...")
            client_socket.connect((client_instance.server_ip, client_instance.server_port))

            client_id_raw = client_socket.recv(1024).decode()
            client_instance.assign_client_id(int(client_id_raw))
            print(f"[Client] 연결 완료. 클라이언트 ID: {client_instance.id}")

            client_socket.settimeout(client_instance.timeout)

            while True:
                print("[Client] Waiting for global params from server...")
                new_params = recv_object(client_socket)
                if new_params is None:
                    print("[Client] 서버로부터 파라미터 수신 실패(혹은 연결 종료). 종료합니다.")
                    break

                round_idx += 1
                client_instance.setModelParams(new_params)
                print("[Client] Global params updated from server.")

                signal = client_socket.recv(1)
                if not signal:
                    print("[Client] 서버로부터 신호 수신 실패. 연결 종료.")
                    break

                if signal == b"1":
                    print("[Client] 학습 시작 신호 수신.")
                    start_time = time.time()

                    client_instance.train()

                    learning_time = time.time() - start_time
                    print(f"[Client {client_instance.id}] local training finished in {learning_time:.2f}s.")

                    train_metrics = client_instance.train_metrics()
                    print(f"[Client {client_instance.id}] train_loss={train_metrics['loss']:.6f}")

                    send_object(client_socket, client_instance.getModelParams())
                    time.sleep(0.02)

                    updated = recv_object(client_socket)
                    if updated is None:
                        print("[Client] Updated params not received. 종료.")
                        break

                    client_instance.setModelParams(updated)
                    print("[Client] Updated params set after upload.")

                    metrics = client_instance.test_metrics()
                    _print_test_metrics(client_instance, metrics, tag="post-aggregation eval")
                    saved_path = client_instance.save_local_artifact(round_idx=round_idx, tag="latest")
                    print(f"[Client {client_instance.id}] checkpoint saved -> {saved_path}")

                elif signal == b"0":
                    print("[Client] 이번 라운드 PASS (학습하지 않음). 기다리는 중...")
                    updated = recv_object(client_socket)
                    if updated is None:
                        print("[Client] Updated params not received. 종료.")
                        break

                    client_instance.setModelParams(updated)
                    print("[Client] Updated params set (pass).")

                    metrics = client_instance.test_metrics()
                    _print_test_metrics(client_instance, metrics, tag="pass-round eval")
                    saved_path = client_instance.save_local_artifact(round_idx=round_idx, tag="latest")
                    print(f"[Client {client_instance.id}] checkpoint saved -> {saved_path}")

                elif signal == b"-1":
                    print("[Client] 종료 신호 수신")
                    break
                else:
                    print(f"[Client] 알 수 없는 신호 수신: {signal}")

        except Exception as exc:
            print(f"클라이언트 오류 발생: {exc}")


if __name__ == "__main__":
    start_client()
