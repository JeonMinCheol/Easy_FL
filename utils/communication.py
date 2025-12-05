# --- 공용 send/recv 함수 ---
import pickle
import struct

def send_object(sock, obj):
    payload = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    length = struct.pack('Q', len(payload))  # 8바이트 길이
    sock.sendall(length)
    sock.sendall(payload)

def recv_object(sock):
    # 길이 먼저 받기
    raw_len = sock.recv(8)
    if not raw_len:
        return None
    total_len = struct.unpack('Q', raw_len)[0]

    # 정확히 total_len 만큼 받기
    received = b''
    while len(received) < total_len:
        packet = sock.recv(4096)
        if not packet:
            break
        received += packet

    return pickle.loads(received)