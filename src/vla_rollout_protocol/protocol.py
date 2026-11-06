from __future__ import annotations

import json
import socket
import struct
from typing import Any

import numpy as np


PROTOCOL_VERSION = 2
MAGIC = b"VLARLT02"
PREFIX = struct.Struct("!8sIQ")
DEFAULT_MAX_HEADER_BYTES = 1_000_000
DEFAULT_MAX_PAYLOAD_BYTES = 256 * 1024 * 1024


def _recv_exact(connection: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = connection.recv(size - len(chunks))
        if not chunk:
            raise EOFError("rollout socket closed while receiving a packet")
        chunks.extend(chunk)
    return bytes(chunks)


def send_packet(
    connection: socket.socket,
    header: dict[str, Any],
    arrays: dict[str, np.ndarray] | None = None,
) -> None:
    metadata = dict(header)
    if int(metadata.get("protocol_version", PROTOCOL_VERSION)) != PROTOCOL_VERSION:
        raise ValueError("cannot send an unsupported rollout protocol version")
    metadata["protocol_version"] = PROTOCOL_VERSION
    descriptors = []
    payload_parts = []
    offset = 0
    for name, value in (arrays or {}).items():
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"object arrays are not supported: {name}")
        raw = memoryview(array).cast("B")
        descriptors.append(
            {
                "name": name,
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "offset": offset,
                "nbytes": raw.nbytes,
            }
        )
        payload_parts.append(raw)
        offset += raw.nbytes
    metadata["arrays"] = descriptors
    header_bytes = json.dumps(metadata, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    connection.sendall(PREFIX.pack(MAGIC, len(header_bytes), offset))
    connection.sendall(header_bytes)
    for part in payload_parts:
        connection.sendall(part)


def recv_packet(
    connection: socket.socket,
    *,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    max_payload_bytes: int = DEFAULT_MAX_PAYLOAD_BYTES,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    magic, header_size, payload_size = PREFIX.unpack(_recv_exact(connection, PREFIX.size))
    if magic != MAGIC:
        raise ValueError("invalid rollout protocol magic")
    if not 0 < header_size <= max_header_bytes:
        raise ValueError(f"invalid rollout header size: {header_size}")
    if payload_size > max_payload_bytes:
        raise ValueError(f"rollout payload exceeds limit: {payload_size}")
    header = json.loads(_recv_exact(connection, header_size).decode("ascii"))
    if not isinstance(header, dict):
        raise ValueError("rollout packet header must be a JSON object")
    if header.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(
            f"unsupported rollout protocol version {header.get('protocol_version')!r}"
        )
    payload = _recv_exact(connection, payload_size)
    arrays: dict[str, np.ndarray] = {}
    occupied: list[tuple[int, int]] = []
    for descriptor in header.pop("arrays", []):
        if not isinstance(descriptor, dict):
            raise ValueError("array descriptor must be an object")
        name = descriptor.get("name")
        if not isinstance(name, str) or not name or name in arrays:
            raise ValueError(f"invalid or duplicate array name: {name!r}")
        dtype = np.dtype(descriptor["dtype"])
        if dtype.hasobject:
            raise ValueError(f"object dtype is not supported: {name}")
        shape = tuple(int(value) for value in descriptor["shape"])
        if any(value < 0 for value in shape):
            raise ValueError(f"negative array shape for {name}: {shape}")
        offset = int(descriptor["offset"])
        nbytes = int(descriptor["nbytes"])
        expected = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
        if offset < 0 or nbytes != expected or offset + nbytes > payload_size:
            raise ValueError(f"invalid array bounds for {name}")
        for start, stop in occupied:
            if offset < stop and start < offset + nbytes:
                raise ValueError(f"overlapping array payload for {name}")
        occupied.append((offset, offset + nbytes))
        arrays[name] = np.frombuffer(payload, dtype=dtype, count=expected // dtype.itemsize, offset=offset).reshape(shape).copy()
    return header, arrays
