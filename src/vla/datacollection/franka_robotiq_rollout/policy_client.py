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
from vla.control.execution import PlanResponse, PlanRequest


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
        self._results: queue.Queue[PlanResponse] = queue.Queue(1)
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
        self._controls = queue.Queue(1)
        self._control_thread = threading.Thread(target=self._control_worker, name='policy-control', daemon=True)
        self._control_thread.start()

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

    def submit(self, *, plan, state, images, task, phase, reset_episode=False, anchor=None):
        with self._lock:
            if self._closed:
                raise RuntimeError('policy client is closed')
            if self._error is not None:
                raise RuntimeError(f'policy client failed: {self._error}') from self._error
            if self._pending:
                return None
            self._pending = True
            self._next_request_id = max(self._next_request_id, plan.request_id+1)
        header, arrays = plan.packet()
        header.update(task=task, phase=phase, reset_episode=reset_episode,
                      checkpoint_fingerprint=self.contract['checkpoint_fingerprint'])
        if self.control_mode is not None:
            header.update(control_mode=self.control_mode, mode_version=self.mode_version)
        arrays.update({'observation.state': np.asarray(state, dtype=np.float32), **images})
        self._requests.put_nowait((header, arrays, time.monotonic()))
        return plan.request_id

    def _worker(self):
        try:
            while True:
                item = self._requests.get()
                if item is None:
                    return
                header, arrays, _ = item
                with self._connect() as connection:
                    send_packet(connection, header, arrays)
                    response, values = recv_packet(connection)
                with self._lock:
                    stale = header['request_id'] < self._min_valid_request_id
                if stale:
                    with self._lock:
                        self._pending = False
                    continue
                if response.get('kind') != 'plan_result':
                    raise RuntimeError(f'invalid candidate response: {response}')
                plan = PlanResponse.from_packet(response, values)
                if plan.request.identity != PlanRequest.from_packet(header, arrays).identity:
                    raise ValueError('response identity mismatch')
                self._results.put_nowait(plan)
        except BaseException as error:
            with self._lock:
                self._error = error
                self._pending = False

    def poll_latest(self):
        try:
            candidate = self._results.get_nowait()
        except queue.Empty:
            return None
        with self._lock:
            self._pending = False
            return candidate if candidate.request.request_id >= self._min_valid_request_id else None

    def invalidate(self, session_id=None, generation=None):
        with self._lock:
            self._min_valid_request_id = self._next_request_id
        self.poll_latest()
        if session_id is not None:
            message = {'kind': 'invalidate', 'session_id': session_id, 'generation': generation}
            try:
                self._controls.get_nowait()
            except queue.Empty:
                pass
            self._controls.put_nowait(message)

    def _control_worker(self):
        try:
            while True:
                message = self._controls.get()
                if message is None:
                    return
                with self._connect() as connection:
                    send_packet(connection, message)
                    header, _ = recv_packet(connection)
                    if header.get('kind') != message['kind']:
                        raise RuntimeError(f'control message failed: {header}')
        except BaseException as error:
            with self._lock:
                self._error = error

    def _wait_for_idle(self, timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        while True:
            self.raise_if_failed()
            self.poll_latest()
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
        execution_context: dict,
        task: str | None = None,
    ) -> dict[str, Any]:
        if not self.contract.get("online_enabled", False):
            raise RuntimeError("policy server is not configured for online RLT")
        timeout_s = float(self.contract.get("online_timeout_s", self.timeout_s))
        self._wait_for_idle(timeout_s)
        if execution_context.get('state') != 'standby' or not execution_context.get('reset_acknowledged'):
            raise RuntimeError('online training requires acknowledged stopped motion')
        with self._connect() as connection:
            send_packet(connection, {'kind': 'progress', **execution_context})
            acknowledgement, _ = recv_packet(connection)
        if acknowledgement.get('kind') != 'progress':
            raise RuntimeError('executor stop progress was not acknowledged')
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
                    "execution_context": execution_context,
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

        try:
            self._controls.get_nowait()
        except queue.Empty:
            pass
        self._controls.put_nowait(None)
        self._control_thread.join(timeout=self.timeout_s+1)
