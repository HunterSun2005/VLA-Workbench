from __future__ import annotations

import queue
import socket
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vla_rollout_protocol import recv_packet, send_packet


class _RemoteTrainingProgress:
    def __init__(self, *, stream: Any = None) -> None:
        self.stream = sys.stdout if stream is None else stream
        self.last_width = 0
        self.rendered = False
        self.closed = False

    def update(self, message: dict[str, Any]) -> None:
        if self.closed:
            return
        if "stage" in message:
            line = f"Online processing: {message['stage']}"
            if "episode_id" in message:
                line += f" episode={int(message['episode_id'])}"
            self.stream.write("\r" + line.ljust(self.last_width))
            self.stream.flush()
            self.last_width = max(self.last_width, len(line))
            self.rendered = True
            return
        completed = int(message["completed_updates"])
        total = int(message["total_updates"])
        if total <= 0 or not 0 <= completed <= total:
            raise ValueError(
                f"invalid remote training progress: completed={completed} total={total}"
            )
        fraction = completed / total
        filled = min(24, int(24 * fraction))
        bar = "=" * filled + "-" * (24 - filled)
        critic = message.get("critic_loss")
        actor = message.get("actor_loss")
        critic_text = "--" if critic is None else f"{float(critic):.4f}"
        actor_text = "--" if actor is None else f"{float(actor):.4f}"
        line = (
            f"Online training [{bar}] {completed}/{total} "
            f"({100.0 * fraction:5.1f}%) critic_loss={critic_text} "
            f"actor_updates={int(message['actor_updates'])} actor_loss={actor_text}"
        )
        self.stream.write("\r" + line.ljust(self.last_width))
        self.stream.flush()
        self.last_width = max(self.last_width, len(line))
        self.rendered = True

    def close(self) -> None:
        if self.closed:
            return
        if self.rendered:
            self.stream.write("\n")
            self.stream.flush()
        self.closed = True


@dataclass(frozen=True)
class PolicyResult:
    request_id: int
    action: np.ndarray
    requested_phase: str
    reset_episode: bool
    phase: float
    source_type: float
    checkpoint_fingerprint: str
    latency_s: float
    received_at: float
    control_mode: str | None = None
    mode_version: int = -1
    probability: float | None = None


class AsyncPolicyClient:
    """Single-flight policy client; control loops never block on GPU inference."""

    def __init__(self, socket_path: str | Path, timeout_s: float) -> None:
        self.socket_path = str(Path(socket_path).expanduser().resolve())
        self.timeout_s = float(timeout_s)
        self._requests: queue.Queue[
            tuple[dict[str, Any], dict[str, np.ndarray], float] | None
        ] = queue.Queue(1)
        self._results: queue.Queue[PolicyResult] = queue.Queue()
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._pending = False
        self._closed = False
        self._next_request_id = 0
        self.control_mode = None
        self.mode_version = -1
        self._min_valid_request_id = 0
        self.contract = self._health_check()
        self._thread = threading.Thread(
            target=self._worker, name="franka-policy-client", daemon=True
        )
        self._thread.start()

    def set_control_mode(self, mode: str) -> None:
        if mode not in self.contract.get("control_modes", []):
            raise ValueError("policy does not support the requested runtime mode")
        with self._lock:
            self._min_valid_request_id = self._next_request_id
            self.control_mode = mode
            self.mode_version += 1

    def _connect(self) -> socket.socket:
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(self.timeout_s)
        connection.connect(self.socket_path)
        return connection

    def _health_check(self) -> dict[str, Any]:
        with self._connect() as connection:
            send_packet(connection, {"kind": "health", "request_id": -1})
            header, arrays = recv_packet(connection)
        if arrays or header.get("kind") != "health":
            raise RuntimeError(f"invalid policy server health response: {header}")
        return header

    def submit(
        self,
        *,
        state: np.ndarray,
        images: dict[str, np.ndarray],
        task: str,
        phase: str,
        reset_episode: bool = False,
    ) -> int | None:
        with self._lock:
            if self._closed:
                raise RuntimeError("policy client is closed")
            if self._error is not None:
                raise RuntimeError(
                    f"policy client failed: {self._error}"
                ) from self._error
            if self._pending:
                return None
            request_id = self._next_request_id
            self._next_request_id += 1
            self._pending = True
            managed = ({"control_mode": self.control_mode, "mode_version": self.mode_version}
                       if self.control_mode is not None else {})
            if self.control_mode in ("vla", "rlt"):
                phase = "rl" if self.control_mode == "rlt" else "vla"
        arrays = {"observation.state": np.asarray(state, dtype=np.float32), **images}
        self._requests.put(
            (
                {
                    "kind": "infer",
                    **managed,
                    "request_id": request_id,
                    "task": task,
                    "phase": phase,
                    "reset_episode": reset_episode,
                    "checkpoint_fingerprint": self.contract["checkpoint_fingerprint"],
                },
                arrays,
                time.monotonic(),
            )
        )
        return request_id

    def _worker(self) -> None:
        try:
            while True:
                request = self._requests.get()
                if request is None:
                    return
                header, arrays, submitted_at = request
                # One connection per request avoids reusing a server-side socket
                # that may have expired while the collector was between episodes.
                with self._connect() as connection:
                    send_packet(connection, header, arrays)
                    response, response_arrays = recv_packet(connection)
                if response.get("kind") == "error":
                    raise RuntimeError(
                        f"policy server {response.get('error_type')}: {response.get('error')}"
                    )
                if response.get("kind") != "action":
                    raise RuntimeError(f"invalid policy server response: {response}")
                if int(response.get("request_id", -1)) != int(header["request_id"]):
                    raise RuntimeError(
                        "policy server response request_id does not match the request"
                    )
                if response.get("checkpoint_fingerprint") != self.contract.get(
                    "checkpoint_fingerprint"
                ):
                    raise RuntimeError("policy server checkpoint fingerprint changed")
                if "control_mode" in header and (
                    response.get("control_mode") != header["control_mode"]
                    or response.get("mode_version") != header["mode_version"]
                ):
                    raise RuntimeError("policy control mode acknowledgement mismatch")
                action = np.asarray(response_arrays.get("action"), dtype=np.float32)
                received_at = time.monotonic()
                self._results.put(
                    PolicyResult(
                        request_id=int(response["request_id"]),
                        action=action,
                        requested_phase=str(header["phase"]),
                        reset_episode=bool(header["reset_episode"]),
                        phase=float(response["phase"]),
                        source_type=float(response["source_type"]),
                        checkpoint_fingerprint=str(response["checkpoint_fingerprint"]),
                        latency_s=received_at - submitted_at,
                        received_at=received_at,
                        control_mode=response.get("control_mode"),
                        mode_version=int(response.get("mode_version", -1)),
                        probability=response.get("probability"),
                    )
                )
                with self._lock:
                    self._pending = False
        except BaseException as error:
            with self._lock:
                self._error = error
                self._pending = False

    def poll_latest(self) -> PolicyResult | None:
        latest = None
        while True:
            try:
                candidate = self._results.get_nowait()
                with self._lock:
                    minimum = self._min_valid_request_id
                if candidate.request_id >= minimum:
                    latest = candidate
                else:
                    print(
                        "Discarded stale policy response "
                        f"id={candidate.request_id} "
                        f"requested_phase={candidate.requested_phase} "
                        f"minimum_valid_id={minimum}",
                        flush=True,
                    )
            except queue.Empty:
                return latest

    def invalidate(self) -> None:
        """Ignore queued replies and force the next request to use a fresh observation."""
        with self._lock:
            self._min_valid_request_id = self._next_request_id
        while self.poll_latest() is not None:
            pass

    def _wait_for_idle(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            self.raise_if_failed()
            with self._lock:
                pending = self._pending
            if not pending:
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "policy inference did not become idle before online training"
                )
            time.sleep(0.01)

    def process_saved_episode(
        self,
        *,
        dataset_root: str | Path,
        episode_id: int,
        outcome: str,
        task: str | None = None,
    ) -> dict[str, Any]:
        if not self.contract.get("online_enabled", False):
            raise RuntimeError("policy server is not configured for online RLT")
        timeout_s = float(self.contract.get("online_timeout_s", self.timeout_s))
        self._wait_for_idle(timeout_s)
        resolved_root = str(Path(dataset_root).expanduser().resolve())
        expected_root = str(self.contract.get("online_dataset_root", ""))
        if resolved_root != expected_root:
            raise ValueError(
                f"online dataset root mismatch: collector={resolved_root}, server={expected_root}"
            )
        with self._connect() as connection:
            connection.settimeout(timeout_s)
            send_packet(
                connection,
                {
                    "kind": "episode_saved",
                    "request_id": -2,
                    "task": self.contract["task"] if task is None else str(task),
                    "checkpoint_fingerprint": self.contract["checkpoint_fingerprint"],
                    "dataset_root": resolved_root,
                    "episode_id": int(episode_id),
                    "outcome": outcome,
                },
            )
            progress = _RemoteTrainingProgress()
            try:
                while True:
                    response, arrays = recv_packet(connection)
                    if response.get("kind") != "training_progress":
                        break
                    if arrays:
                        raise RuntimeError(
                            "online training progress must not contain array payloads"
                        )
                    if int(response.get("request_id", -1)) != -2:
                        raise RuntimeError(
                            "online training progress request_id does not match the request"
                        )
                    progress.update(response)
            finally:
                progress.close()
        if response.get("kind") == "error":
            raise RuntimeError(
                f"policy server {response.get('error_type')}: {response.get('error')}"
            )
        if arrays or response.get("kind") != "episode_processed":
            raise RuntimeError(f"invalid online episode response: {response}")
        return response

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError(f"policy client failed: {error}") from error

    def _online_request(self, kind: str) -> dict[str, Any]:
        with self._connect() as connection:
            connection.settimeout(float(self.contract.get("online_timeout_s", self.timeout_s)))
            send_packet(connection, {
                "kind": kind, "request_id": -3,
                "run_uuid": self.contract.get("run_uuid"),
                "dataset_uuid": self.contract.get("dataset_uuid"),
            })
            progress = _RemoteTrainingProgress()
            try:
                while True:
                    response, arrays = recv_packet(connection)
                    if response.get("kind") != "training_progress":
                        break
                    if arrays or response.get("request_id") != -3:
                        raise RuntimeError("invalid recovery progress response")
                    progress.update(response)
            finally:
                progress.close()
        if arrays or response.get("kind") != kind:
            raise RuntimeError(f"online {kind} failed: {response}")
        if response.get("online_status") != "READY":
            raise RuntimeError(f"online recovery remains blocked: {response.get('online_fault')}")
        self.contract.update({key: value for key, value in response.items() if key in self.contract or key.startswith("online_")})
        return response

    def begin_episode(self) -> dict[str, Any]:
        self._wait_for_idle(float(self.contract.get("online_timeout_s", self.timeout_s)))
        result = self._online_request("begin_episode")
        return {key: result[key] for key in ("run_uuid", "dataset_uuid", "episode_uuid", "online_warmup", "policy_revision")}

    def recover_online(self) -> None:
        self.invalidate()
        # An inference failure terminates the worker. Stop/join it before replacement.
        with self._lock:
            failed = self._error is not None
        if not failed:
            try:
                self._wait_for_idle(float(self.contract.get("online_timeout_s", self.timeout_s)))
            except RuntimeError:
                with self._lock:
                    failed = self._error is not None
                if not failed:
                    raise
        if failed:
            self._thread.join(timeout=self.timeout_s + 1.0)
            if self._thread.is_alive():
                raise RuntimeError("policy worker is still running; cannot recover")
        self._online_request("recover_online")
        if failed:
            with self._lock:
                self._error = None
                self._pending = False
                self._requests = queue.Queue(1)
            self._thread = threading.Thread(target=self._worker, name="franka-policy-client", daemon=True)
            self._thread.start()
        self.invalidate()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        if self._thread.is_alive():
            try:
                self._requests.put_nowait(None)
            except queue.Full:
                pass
        self._thread.join(timeout=self.timeout_s + 1.0)
