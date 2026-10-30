"""Bounded, versioned local JSON RPC; no model, ROS or web dependencies."""
from __future__ import annotations
import json
import socket
import struct

VERSION = 1
MAX_PACKET = 2 * 1024 * 1024
COMMANDS = frozenset({'enable', 'pause', 'record', 'save', 'discard', 'recover',
                      'stop', 'mode', 'intervention', 'critical', 'next_subtask'})


def receive(sock):
    def exact(count):
        chunks = bytearray()
        while len(chunks) < count:
            chunk = sock.recv(count - len(chunks))
            if not chunk:
                raise ConnectionError('local connection closed')
            chunks.extend(chunk)
        return bytes(chunks)
    length = struct.unpack('!I', exact(4))[0]
    if not 0 < length <= MAX_PACKET:
        raise ValueError('local packet too large')
    value = json.loads(exact(length))
    if not isinstance(value, dict) or value.get('version') != VERSION:
        raise ValueError('unsupported local protocol')
    return value


def send(sock, value):
    payload = json.dumps({'version': VERSION, **value}, allow_nan=False, separators=(',', ':')).encode()
    if len(payload) > MAX_PACKET:
        raise ValueError('local packet too large')
    sock.sendall(struct.pack('!I', len(payload)) + payload)


def request(path, value, timeout=2.):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        sock.connect(str(path))
        send(sock, value)
        result = receive(sock)
    if result.get('error'):
        raise RuntimeError(result['error'])
    return result
