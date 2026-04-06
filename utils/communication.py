import io
import pickle
import struct

import torch


def send_object(sock, obj):
    buffer = io.BytesIO()
    torch.save(obj, buffer)
    payload = buffer.getvalue()
    length = struct.pack('Q', len(payload))
    sock.sendall(length)
    sock.sendall(payload)


def _recv_exact(sock, num_bytes: int):
    chunks = []
    received = 0
    while received < num_bytes:
        chunk = sock.recv(num_bytes - received)
        if not chunk:
            return None
        chunks.append(chunk)
        received += len(chunk)
    return b''.join(chunks)


def recv_object(sock):
    raw_len = _recv_exact(sock, 8)
    if not raw_len:
        return None

    total_len = struct.unpack('Q', raw_len)[0]
    payload = _recv_exact(sock, total_len)
    if payload is None:
        return None

    buffer = io.BytesIO(payload)
    return torch.load(buffer, map_location='cpu', weights_only=False)
