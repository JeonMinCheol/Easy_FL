import socket
import yaml
from pathlib import Path
import clientbase
import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.communication import send_object, recv_object

# -------------------------------
# 설정 로드
# -------------------------------
def load_yaml_file():
    current_file = Path(__file__).resolve()
    file_path = current_file.parent.parent / 'config.yml'
    try:
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
            return data
    except Exception as e:
        print("YAML load error:", e)
        return None

DATA = load_yaml_file()
if not DATA:
    exit()

# -------------------------------
# 클라이언트
# -------------------------------
def start_client():
    client_instance = clientbase.Client(config=DATA)

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        try:
            print(f"서버 {client_instance.server_ip}:{client_instance.server_port}에 연결 시도...")
            client_socket.connect((client_instance.server_ip, client_instance.server_port))

            # 클라이언트 ID 읽기
            client_id_raw = client_socket.recv(1024).decode()
            client_instance.id = int(client_id_raw)
            print(f"[Client] 연결 완료. 클라이언트 ID: {client_instance.id}")

            client_socket.settimeout(client_instance.timeout)

            while True:
                # 1) 서버가 보낸 global params 수신 (blocking, 길이-프리픽스 기반)
                print("[Client] Waiting for global params from server...")
                new_params = recv_object(client_socket)
                if new_params is None:
                    print("[Client] 서버로부터 파라미터 수신 실패(혹은 연결 종료). 종료합니다.")
                    break

                client_instance.setModelParams(new_params)
                print("[Client] Global params updated from server.")

                # 2) 서버가 보낼 학습 여부 신호(1바이트) 수신
                signal = client_socket.recv(1)
                if not signal:
                    print("[Client] 서버로부터 신호 수신 실패. 연결 종료.")
                    break

                if signal == b'1':
                    print("[Client] 학습 시작 신호 수신.")

                    start_time = time.time()
                    client_instance.train()
                    learning_time = time.time() - start_time
                    print(f"Client finished training after {learning_time:.2f}s.")
                    
                    # (선택) 학습 지표 출력 — 기존 출력 유지
                    try:
                        train_metrics = client_instance.train_metrics()
                        print(f"[Client] 학습 정확도: {train_metrics[0]*100:.2f}%, 손실: {train_metrics[1]:.4f}")
                    except Exception:
                        pass

                    # 모델 전송 (업로드)
                    send_object(client_socket, client_instance.getModelParams())
                    # 전송 직후 소량 슬립으로 안정화
                    time.sleep(0.02)

                    # 서버로부터 업데이트된 params 수신 (post-aggregation)
                    updated = recv_object(client_socket)
                    if updated is None:
                        print("[Client] Updated params not received. 종료.")
                        break
                    client_instance.setModelParams(updated)
                    print("[Client] Updated params set after upload.")

                    # (선택) 테스트 지표 출력 — 기존 출력 유지
                    try:
                        test_metrics = client_instance.test_metrics()
                        print(f"[Client] 테스트 정확도: {test_metrics*100:.2f}%")
                    except Exception:
                        pass

                elif signal == b'0':
                    # 이번 라운드는 선택되지 않음 — server already sent global params above,
                    # then server will wait for others and finally send updated params to all,
                    # so just wait for the updated params here.
                    print("[Client] 이번 라운드 PASS (학습하지 않음). 기다리는 중...")
                    updated = recv_object(client_socket)
                    if updated is None:
                        print("[Client] Updated params not received. 종료.")
                        break
                    client_instance.setModelParams(updated)
                    print("[Client] Updated params set (pass).")

                elif signal == b'-1':
                    print("[Client] 종료 신호 수신")
                    break

                else:
                    print(f"[Client] 알 수 없는 신호 수신: {signal}")

        except Exception as e:
            print(f"클라이언트 오류 발생: {e}")

if __name__ == '__main__':
    start_client()
