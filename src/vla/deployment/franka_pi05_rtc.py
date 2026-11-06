from __future__ import annotations

import json
import logging
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from vla.control.geometry import (
    ACTION_FIELDS,
    STATE_FIELDS,
    _rotation_angle,
    rot6d_to_matrix,
)


log = logging.getLogger(__name__)


PHYSICAL_CAMERAS = ("observation.images.base", "observation.images.left_wrist")
LEGACY_PADDING_CAMERAS = (
    "observation.images.base_0_rgb",
    "observation.images.left_wrist_0_rgb",
    "observation.images.right_wrist_0_rgb",
)


class JsonlDebugTrace:
    """Non-blocking JSONL trace writer for hardware deployment diagnostics."""

    def __init__(self, path: str | Path, max_pending: int = 4096) -> None:
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(maxsize=max_pending)
        self._dropped = 0
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._writer, name="franka-debug-trace", daemon=True)
        self._thread.start()

    def emit(self, event: str, **fields: Any) -> None:
        record = {"event": event, "wall_time": time.time(), "monotonic": time.perf_counter(), **fields}
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped += 1

    def _writer(self) -> None:
        try:
            with self.path.open("a", encoding="utf-8", buffering=1) as stream:
                while True:
                    record = self._queue.get()
                    if record is None:
                        break
                    stream.write(json.dumps(record, ensure_ascii=True, separators=(",", ":")) + "\n")
        except BaseException as error:
            self._error = error

    def close(self) -> None:
        if not self._thread.is_alive():
            return
        self.emit("trace_closed", dropped_records=self._dropped)
        try:
            self._queue.put(None, timeout=5.0)
        except queue.Full:
            log.warning("Debug trace queue did not drain within 5 seconds: %s", self.path)
            return
        self._thread.join(timeout=5.0)
        if self._thread.is_alive():
            log.warning("Debug trace writer did not stop within 5 seconds: %s", self.path)
        if self._error is not None:
            log.warning("Debug trace writer failed: %s", self._error)


@dataclass(frozen=True)
class CheckpointContract:
    path: Path
    config: dict[str, Any]
    tokenizer: str


def _feature_shape(config: dict[str, Any], group: str, key: str) -> tuple[int, ...]:
    try:
        return tuple(config[group][key]["shape"])
    except (KeyError, TypeError) as error:
        raise ValueError(f"checkpoint is missing {group}.{key}.shape") from error


def _tokenizer_name(checkpoint: Path) -> str:
    processor = json.loads((checkpoint / "policy_preprocessor.json").read_text(encoding="utf-8"))
    for step in processor.get("steps", []):
        if step.get("registry_name") == "tokenizer_processor":
            name = step.get("config", {}).get("tokenizer_name")
            if isinstance(name, str) and name:
                return name
    raise ValueError("checkpoint preprocessor has no tokenizer_processor tokenizer_name")


def _validate_action_unnormalizer(checkpoint: Path) -> None:
    from safetensors.torch import load_file

    path = checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint is missing action unnormalizer stats: {path}")
    tensors = load_file(path)
    try:
        minimum = tensors["action.min"].detach().cpu().numpy()
        maximum = tensors["action.max"].detach().cpu().numpy()
        q01 = tensors["action.q01"].detach().cpu().numpy()
        q99 = tensors["action.q99"].detach().cpu().numpy()
    except KeyError as error:
        raise ValueError(f"action unnormalizer is missing {error.args[0]!r}: {path}") from error
    if any(value.shape != (10,) for value in (minimum, maximum, q01, q99)):
        raise ValueError(f"action unnormalizer stats must be 10D: {path}")
    translation_span = float(np.linalg.norm(maximum[:3] - minimum[:3]))
    rot6d_span = float(np.linalg.norm(maximum[3:9] - minimum[3:9]))
    quantile_span = float(np.linalg.norm(q99[:9] - q01[:9]))
    if translation_span < 5e-3 and rot6d_span < 1e-2 and quantile_span < 1e-2:
        raise ValueError(
            "checkpoint action unnormalizer has degenerate constant TCP stats; "
            "the checkpoint was trained on invalid actions and cannot control the arm"
        )


def validate_checkpoint(checkpoint: str | Path, tokenizer_path: str | None = None) -> CheckpointContract:
    root = Path(checkpoint).expanduser().resolve()
    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json", "model.safetensors"):
        if not (root / name).is_file():
            raise FileNotFoundError(f"checkpoint is missing {name}: {root}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    from vla.actions import ActionSpec
    from vla.actions.metadata import checkpoint_contract
    contract = checkpoint_contract(root)
    spec = ActionSpec.from_dict(contract['action_spec'])
    sampling = contract['statistics']['contract']['sampling']
    expected_cameras = ('observation.images.base', 'observation.images.left_wrist') + (
        ('observation.images.right_wrist',) if len(spec.arms) == 2 else ())
    visual_features = tuple(key for key, feature in config.get('input_features', {}).items()
                            if feature.get('type') == 'VISUAL')
    checks = {
        'PI05 policy': config.get('type') in ('pi05', 'vla_pi05'),
        'state fields': _feature_shape(config, 'input_features', 'observation.state') == (len(contract['state_fields']),),
        'action dimensions': _feature_shape(config, 'output_features', 'action') == (spec.dim,),
        'project relative processor': config.get('use_relative_actions') is False,
        'chunk sampling': config.get('chunk_size') == sampling['chunk_length'],
        'physical cameras': visual_features == expected_cameras,
        'no empty cameras': config.get('empty_cameras', 0) == 0,
        'camera shapes': all(_feature_shape(config, 'input_features', key) == (3, 224, 224)
                             for key in expected_cameras),
    }
    if failed := [name for name, valid in checks.items() if not valid]:
        raise ValueError(f'checkpoint is incompatible with ROS deployment: {failed}')
    tokenizer = tokenizer_path or _tokenizer_name(root)
    tokenizer_root = Path(tokenizer).expanduser()
    if tokenizer_root.is_dir():
        tokenizer = str(tokenizer_root.resolve())
    else:
        from huggingface_hub import snapshot_download

        try:
            tokenizer = snapshot_download(tokenizer, local_files_only=True)
        except Exception as error:
            raise FileNotFoundError(
                f"tokenizer {tokenizer!r} is neither an explicit local snapshot "
                "nor available in offline cache"
            ) from error
    from transformers import AutoTokenizer

    try:
        AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
    except Exception as error:
        raise FileNotFoundError(f"could not load tokenizer from local snapshot {tokenizer!r}") from error
    return CheckpointContract(root, config, tokenizer)


def policy_observation(robot_observation: dict[str, Any], task: str) -> dict[str, Any]:
    state = np.asarray([robot_observation[name] for name in STATE_FIELDS], dtype=np.float32)
    if state.shape != (10,) or not np.isfinite(state).all():
        raise ValueError("robot observation did not produce a finite 10D state")
    result: dict[str, Any] = {"observation.state": state, "task": task}
    for role in ("base", "left_wrist"):
        image = np.asarray(robot_observation[role])
        if image.shape != (224, 224, 3) or image.dtype != np.uint8:
            raise ValueError(f"{role} must be 224x224 HWC uint8 RGB")
        result[f"observation.images.{role}"] = image
    return result


class Pi05RTC:
    """Thread-safe normalized/absolute RTC queue around a LeRobot pi0.5 policy."""

    def __init__(
        self,
        policy: Any,
        preprocessor: Callable[[dict[str, Any]], dict[str, Any]],
        postprocessor: Callable[[torch.Tensor], torch.Tensor],
        *,
        rtc_config: Any,
        fps: float = 30.0,
        queue_threshold: int = 30,
        clock: Callable[[], float] = time.perf_counter,
        trace: JsonlDebugTrace | None = None,
    ) -> None:
        from lerobot.policies.rtc.action_queue import ActionQueue
        from lerobot.policies.rtc.latency_tracker import LatencyTracker

        self.policy, self.preprocessor, self.postprocessor = policy, preprocessor, postprocessor
        self.fps, self.queue_threshold, self.clock = float(fps), int(queue_threshold), clock
        self.trace = trace
        from vla.adapters.lerobot.action_runtime import ActionRuntime
        contract = getattr(getattr(policy, 'config', None), 'action_contract', None)
        self.action_runtime = ActionRuntime(contract) if contract else None
        self.queue = ActionQueue(rtc_config)
        self.latency = LatencyTracker()
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._inference_lock = threading.Lock()
        self._worker: threading.Thread | None = None
        self._worker_active = False
        self._pending_request: tuple[dict[str, Any], torch.Tensor | None, int, int, float, int] | None = None
        self._worker_error: BaseException | None = None
        self._snapshot: dict[str, Any] | None = None
        self._generation = 0
        self._closed = False
        self.consecutive_failures = 0
        self._trace_observation_id = 0
        self._trace_chunk_id = 0
        self._action_provenance: list[dict[str, Any]] = []
        self._last_action_provenance: dict[str, Any] = {}

    def publish(self, observation: dict[str, Any]) -> None:
        with self._lock:
            self._trace_observation_id += 1
            self._snapshot = {k: v.detach().clone() if isinstance(v, torch.Tensor) else
                              v.copy() if isinstance(v, np.ndarray) else v for k, v in observation.items()}
            self._snapshot["_rtc_trace"] = {
                "observation_id": self._trace_observation_id,
                "published_monotonic_s": time.monotonic(),
                "input_state": np.asarray(observation["observation.state"]).tolist()
                if "observation.state" in observation else None,
            }

    def reset(self, *, wait_for_inference: bool = False) -> None:
        from lerobot.policies.rtc.action_queue import ActionQueue

        with self._condition:
            if self._closed:
                raise RuntimeError("RTC runtime is closed")
            self._generation += 1
            self.queue = ActionQueue(self.queue.cfg)
            self._action_provenance = []
            self._last_action_provenance = {}
            self.latency.reset()
            self._pending_request = None
            self._worker_error = None
            self.consecutive_failures = 0
            if wait_for_inference and threading.current_thread() is not self._worker:
                while self._worker_active:
                    self._condition.wait()

    def _request(self) -> tuple[dict[str, Any], torch.Tensor | None, int, int, float, int]:
        with self._lock:
            if self._closed:
                raise RuntimeError("RTC runtime is closed")
            if self._snapshot is None:
                raise RuntimeError("no observation snapshot has been published")
            snapshot = self._snapshot
            leftover = (self.queue.get_processed_left_over() if self.action_runtime else self.queue.get_left_over())
            leftover = None if leftover is None else leftover.clone()
            action_index = self.queue.get_action_index()
            inference_delay = math.ceil(float(self.latency.max() or 0.0) * self.fps)
            generation = self._generation
        return snapshot, leftover, inference_delay, action_index, self.clock(), generation

    def infer(self, request=None, *, account_for_latency: bool = True) -> None:
        snapshot, leftover, inference_delay, action_index, started, generation = request or self._request()
        with self._lock:
            self._trace_chunk_id += 1
            chunk_id = self._trace_chunk_id
        input_trace = snapshot.get("_rtc_trace", {})
        if self.trace is not None:
            self.trace.emit(
                "inference_started",
                chunk_id=chunk_id,
                input_observation=input_trace,
                generation=generation,
                queue_size=self.queue.qsize(),
                action_index=action_index,
                leftover_steps=0 if leftover is None else int(leftover.shape[0]),
                estimated_delay_steps=inference_delay,
            )
        try:
            # RTC prefix guidance locally re-enables autograd to differentiate
            # the denoised action with respect to its latent. no_grad() permits
            # that nested enable_grad(); inference_mode() does not.
            with self._inference_lock, torch.no_grad():
                with self._lock:
                    if generation != self._generation:
                        return
                batch = self.preprocessor({k: v for k, v in snapshot.items() if k != "_rtc_trace"})
                if self.action_runtime and leftover is not None:
                    leftover = self.action_runtime.encode(leftover, snapshot)
                normalized = self.policy.predict_action_chunk(
                    batch, inference_delay=inference_delay, prev_chunk_left_over=leftover
                )
                if normalized.ndim == 3 and normalized.shape[0] == 1:
                    normalized = normalized.squeeze(0)
                if normalized.ndim != 2 or normalized.shape[1] != (self.action_runtime.spec.dim if self.action_runtime else 10):
                    raise ValueError(f"pi0.5 returned invalid RTC chunk shape {tuple(normalized.shape)}")
                absolute = (self.action_runtime.decode(normalized, snapshot) if self.action_runtime
                            else self.postprocessor(normalized))
                if absolute.ndim == 3 and absolute.shape[0] == 1:
                    absolute = absolute.squeeze(0)
                if absolute.shape != normalized.shape:
                    raise ValueError(
                        "postprocessor changed RTC chunk shape "
                        f"{tuple(normalized.shape)} -> {tuple(absolute.shape)}"
                    )
            elapsed = self.clock() - started
            new_delay = math.ceil(elapsed * self.fps) if account_for_latency else 0
            with self._lock:
                if generation != self._generation:
                    return
                self.latency.add(elapsed)
                consumed_delay = max(0, self.queue.get_action_index() - action_index)
                if abs(consumed_delay - new_delay) > 1:
                    log.warning(
                        "RTC latency/consumption mismatch: consumed=%d, wall_clock=%d",
                        consumed_delay,
                        new_delay,
                    )
                self.queue.merge(
                    normalized.detach(),
                    absolute.detach(),
                    consumed_delay,
                    action_index,
                )
                provenance = [
                    {"generation": generation, "chunk_id": chunk_id,
                     "chunk_action_index": i,
                     "input_observation_id": input_trace.get("observation_id")}
                    for i in range(len(absolute))
                ]
                if self.queue.cfg.enabled:
                    self._action_provenance = provenance[min(consumed_delay, len(provenance)):]
                else:
                    self._action_provenance.extend(provenance)
                self.consecutive_failures = 0
                queue_size = self.queue.qsize()
            if self.trace is not None:
                debug_actions = absolute.detach().cpu()
                self.trace.emit(
                    "inference_completed",
                    chunk_id=chunk_id,
                    input_observation=input_trace,
                    generation=generation,
                    elapsed_ms=elapsed * 1000.0,
                    wall_clock_delay_steps=new_delay,
                    consumed_delay_steps=consumed_delay,
                    merged_queue_size=queue_size,
                    first_action=debug_actions[0].tolist(),
                    last_action=debug_actions[-1].tolist(),
                    predicted_xyz_min=debug_actions[:, :3].amin(dim=0).tolist(),
                    predicted_xyz_max=debug_actions[:, :3].amax(dim=0).tolist(),
                    predicted_gripper_min=float(debug_actions[:, 9].amin()),
                    predicted_gripper_max=float(debug_actions[:, 9].amax()),
                )
        except BaseException as error:
            with self._lock:
                if generation == self._generation:
                    self.consecutive_failures += 1
                    self._worker_error = error
            if self.trace is not None:
                self.trace.emit("inference_failed", generation=generation, error=repr(error))
            raise

    def _worker_main(self) -> None:
        while True:
            with self._condition:
                while self._pending_request is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                request = self._pending_request
                self._pending_request = None
                self._worker_active = True
            try:
                self.infer(request)
            except BaseException:
                pass
            finally:
                with self._condition:
                    self._worker_active = False
                    self._condition.notify_all()

    def maybe_start(self) -> None:
        with self._condition:
            if self._closed:
                raise RuntimeError("RTC runtime is closed")
            if self._worker_active or self._pending_request is not None:
                return
            if self.queue.qsize() > self.queue_threshold:
                return
            self._pending_request = self._request()
            if self._worker is None:
                self._worker = threading.Thread(
                    target=self._worker_main,
                    name="pi05-rtc-inference",
                    daemon=True,
                )
                self._worker.start()
            self._condition.notify()

    def get(self) -> torch.Tensor | None:
        with self._lock:
            action = self.queue.get()
            if action is not None:
                self._last_action_provenance = (
                    self._action_provenance.pop(0) if self._action_provenance else {}
                )
            return action

    def wait_for_refill(
        self,
        stop_event: threading.Event | None = None,
        *,
        poll_interval_s: float = 0.05,
    ) -> bool:
        """Wait for pending inference without turning a queue underrun into a failure."""
        if poll_interval_s <= 0:
            raise ValueError("refill wait poll interval must be positive")
        with self._condition:
            if self._closed:
                raise RuntimeError("RTC runtime is closed")
            if self.queue.qsize() > 0:
                return True
            if not self._worker_active and self._pending_request is None:
                return False
            while self._worker_active or self._pending_request is not None:
                if stop_event is not None and stop_event.is_set():
                    return False
                self._condition.wait(timeout=poll_interval_s if stop_event is not None else None)
            return stop_event is None or not stop_event.is_set()

    def debug_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "queue_size": self.queue.qsize(),
                "last_action_provenance": dict(self._last_action_provenance),
                "inference_active": self._worker_active or self._pending_request is not None,
                "consecutive_inference_failures": self.consecutive_failures,
                "max_inference_latency_ms": float(self.latency.max() or 0.0) * 1000.0,
            }

    def check_worker(self) -> None:
        with self._lock:
            error = self._worker_error
            self._worker_error = None
        if error is not None and self.consecutive_failures >= 3:
            raise RuntimeError(f"RTC inference failed: {error}") from error

    def stop(self) -> None:
        """Clear queued work and wait for in-flight inference, keeping the worker reusable."""
        self.reset(wait_for_inference=True)

    def reconfigure(self, *, rtc_config: Any, fps: float, queue_threshold: int) -> None:
        """Apply session-level RTC settings without rebuilding the resident policy."""
        from lerobot.policies.rtc.action_queue import ActionQueue

        fps = float(fps)
        queue_threshold = int(queue_threshold)
        if fps <= 0:
            raise ValueError("RTC fps must be positive")
        if queue_threshold < 0:
            raise ValueError("RTC queue threshold must be non-negative")
        self.stop()
        with self._condition:
            self.fps = fps
            self.queue_threshold = queue_threshold
            self.queue = ActionQueue(rtc_config)
            self.latency.reset()
            self._snapshot = None
            self._worker_error = None
            self.consecutive_failures = 0

    def close(self) -> None:
        """Invalidate all work and terminate the persistent inference worker."""
        with self._condition:
            if self._closed:
                return
            self._generation += 1
            from lerobot.policies.rtc.action_queue import ActionQueue

            self.queue = ActionQueue(self.queue.cfg)
            self.latency.reset()
            self._pending_request = None
            self._worker_error = None
            self.consecutive_failures = 0
            self._closed = True
            worker = self._worker
            self._condition.notify_all()
        if worker is not None and worker is not threading.current_thread():
            worker.join()
        with self._condition:
            self._worker = None
            self._worker_active = False


class DeploymentLoop:
    def __init__(
        self,
        robot: Any,
        rtc: Pi05RTC,
        *,
        task: str,
        fps: float = 30.0,
        execute: bool = False,
        trace: JsonlDebugTrace | None = None,
        console_log_every: int = 30,
        discontinuity_timeout_s: float | None = None,
        session_id: int = 0,
        clock: Callable[[], float] = time.perf_counter,
    ):
        self.robot, self.rtc, self.task = robot, rtc, task
        self.fps, self.execute = float(fps), bool(execute)
        if self.fps <= 0:
            raise ValueError("deployment fps must be positive")
        if discontinuity_timeout_s is not None and discontinuity_timeout_s <= 0:
            raise ValueError("discontinuity timeout must be positive")
        if discontinuity_timeout_s is not None and discontinuity_timeout_s <= 1.0 / self.fps:
            raise ValueError("discontinuity timeout must be greater than one control period")
        self.trace = trace
        self.console_log_every = max(1, int(console_log_every))
        self.discontinuity_timeout_s = discontinuity_timeout_s
        self.session_id = int(session_id)
        self.clock = clock
        self._stop = threading.Event()
        self._stop_reason: str | None = None

    def request_stop(self, reason: str = "requested") -> None:
        self._stop_reason = str(reason)
        self._stop.set()

    def _emit_robot_motion_events(self, *, step: int) -> None:
        drain = getattr(self.robot, "drain_motion_events", None)
        if not callable(drain):
            return
        for queued_event in drain():
            record = dict(queued_event)
            event = str(record.pop("event"))
            if self.trace is not None:
                self.trace.emit(event, session_id=self.session_id, step=step, **record)

    def preflight(self) -> np.ndarray:
        observation = policy_observation(self.robot.get_observation(), self.task)
        self.rtc.publish(observation)
        # Preflight is disarmed and no old action is being consumed while the
        # synchronous inference runs, so latency must not discard its output.
        self.rtc.infer(account_for_latency=False)
        action = self.rtc.get()
        if action is None:
            raise RuntimeError("preflight inference produced no executable action")
        return action.detach().cpu().numpy()

    def _resynchronize(self, *, step: int, control_gap_s: float, deadline_lateness_s: float) -> None:
        log.warning(
            "CONTROL discontinuity at step=%d gap=%.1fms lateness=%.1fms; "
            "stopping and rebuilding RTC from a fresh observation",
            step,
            control_gap_s * 1000.0,
            deadline_lateness_s * 1000.0,
        )
        if self.trace is not None:
            self.trace.emit(
                "control_resync_started",
                session_id=self.session_id,
                step=step,
                control_gap_ms=control_gap_s * 1000.0,
                deadline_lateness_ms=deadline_lateness_s * 1000.0,
            )
        self.robot.stop()
        self._emit_robot_motion_events(step=step)
        # Invalidate stale requests and wait for in-flight CUDA/native inference
        # before synchronously priming a new queue from the stopped robot state.
        self.rtc.stop()
        reset_policy = getattr(self.rtc.policy, "reset", None)
        if callable(reset_policy):
            reset_policy()
        observation = policy_observation(self.robot.get_observation(), self.task)
        self._emit_robot_motion_events(step=step)
        self.rtc.publish(observation)
        self.rtc.infer(account_for_latency=False)
        if self.trace is not None:
            self.trace.emit(
                "control_resynchronized",
                session_id=self.session_id,
                step=step,
                queue_size=self.rtc.debug_snapshot()["queue_size"],
            )

    def run(self) -> str | None:
        if not self.execute:
            raise PermissionError("execution is disarmed; pass --execute and confirm ARM")
        period = 1.0 / self.fps
        next_deadline = self.clock()
        step = 0
        previous_started: float | None = None
        previous_prediction: np.ndarray | None = None
        previous_prediction_rotation: np.ndarray | None = None
        previous_gripper: float | None = None
        stage = "starting"
        try:
            while not self._stop.is_set():
                stage = "deadline_wait"
                wait_s = next_deadline - self.clock()
                if wait_s > 0:
                    self._stop.wait(wait_s)
                    if self._stop.is_set():
                        break
                started = self.clock()
                loop_dt = None if previous_started is None else started - previous_started
                deadline_lateness = max(0.0, started - next_deadline)
                if (
                    loop_dt is not None
                    and self.discontinuity_timeout_s is not None
                    and loop_dt >= self.discontinuity_timeout_s
                ):
                    stage = "control_resync"
                    self._resynchronize(
                        step=step,
                        control_gap_s=loop_dt,
                        deadline_lateness_s=deadline_lateness,
                    )
                    next_deadline = self.clock()
                    previous_started = None
                    previous_prediction = None
                    previous_prediction_rotation = None
                    previous_gripper = None
                    continue
                previous_started = started
                missed_action_ticks = max(0, int(deadline_lateness // period))
                next_deadline += (missed_action_ticks + 1) * period
                if missed_action_ticks:
                    log.warning(
                        "CONTROL missed %d action tick(s), lateness=%.1fms; discarding stale actions",
                        missed_action_ticks,
                        deadline_lateness * 1000.0,
                    )
                stage = "rtc_worker_check"
                self.rtc.check_worker()
                stage = "observation"
                observation_started = self.clock()
                robot_observation = self.robot.get_observation()
                if self._stop.is_set():
                    break
                self._emit_robot_motion_events(step=step)
                observation_elapsed = self.clock() - observation_started
                observation_debug = (
                    self.robot.observation_debug_snapshot()
                    if hasattr(self.robot, "observation_debug_snapshot")
                    else {}
                )
                stage = "rtc_publish"
                self.rtc.publish(policy_observation(robot_observation, self.task))
                if self.trace is not None:
                    self.trace.emit(
                        "observed_state", session_id=self.session_id, step=step,
                        observation_id=getattr(self.rtc, "_trace_observation_id", None),
                        state=[float(robot_observation[name]) for name in STATE_FIELDS],
                        **observation_debug,
                    )
                rtc_before = self.rtc.debug_snapshot()
                stage = "rtc_refill"
                self.rtc.maybe_start()
                stage = "action_dequeue"
                action = None
                queue_wait_s = 0.0
                queue_wait_count = 0
                for _ in range(missed_action_ticks + 1):
                    queued_action = self.rtc.get()
                    if queued_action is None:
                        break
                    action = queued_action
                while action is None and not self._stop.is_set():
                    stage = "rtc_queue_wait"
                    queue_wait_count += 1
                    queue_wait_started = self.clock()
                    if self.trace is not None:
                        self.trace.emit(
                            "rtc_queue_wait_started",
                            session_id=self.session_id,
                            step=step,
                            attempt=queue_wait_count,
                            rtc=self.rtc.debug_snapshot(),
                        )
                    refill_completed = self.rtc.wait_for_refill(self._stop)
                    queue_wait_s += self.clock() - queue_wait_started
                    if self._stop.is_set():
                        break
                    self.rtc.check_worker()
                    action = self.rtc.get()
                    if action is not None:
                        resumed_at = self.clock()
                        previous_started = resumed_at
                        next_deadline = resumed_at + period
                        log.warning(
                            "CONTROL queue refilled after %.1fms at step=%d; rebasing deadline",
                            queue_wait_s * 1000.0,
                            step,
                        )
                        if self.trace is not None:
                            self.trace.emit(
                                "rtc_queue_wait_completed",
                                session_id=self.session_id,
                                step=step,
                                attempts=queue_wait_count,
                                wait_ms=queue_wait_s * 1000.0,
                                queue_size=self.rtc.debug_snapshot()["queue_size"],
                            )
                        break
                    rtc_after_wait = self.rtc.debug_snapshot()
                    if not refill_completed and rtc_after_wait["consecutive_inference_failures"] == 0:
                        raise RuntimeError("RTC action queue is empty and no refill is active")
                    self.rtc.maybe_start()
                if action is None:
                    if self._stop.is_set():
                        break
                    raise RuntimeError("RTC action queue is empty after refill")
                predicted = action.detach().cpu().numpy()
                prediction_delta = np.zeros(3) if previous_prediction is None else predicted[:3] - previous_prediction[:3]
                predicted_rotation = rot6d_to_matrix(predicted[3:9])
                prediction_delta_rotation = (
                    0.0 if previous_prediction_rotation is None else
                    _rotation_angle(predicted_rotation @ previous_prediction_rotation.T)
                )
                gripper_delta = 0.0 if previous_gripper is None else float(predicted[9] - previous_gripper)
                previous_prediction = predicted.copy()
                previous_prediction_rotation = predicted_rotation
                previous_gripper = float(predicted[9])
                stage = "robot_command"
                command_started = self.clock()
                if self._stop.is_set():
                    break
                self.robot.send_action(predicted)
                self._emit_robot_motion_events(step=step)
                command_elapsed = self.clock() - command_started
                command_debug = (
                    self.robot.command_debug_snapshot() if hasattr(self.robot, "command_debug_snapshot") else {}
                )
                if self.rtc.consecutive_failures >= 3:
                    raise RuntimeError("three consecutive RTC inference failures")
                elapsed = self.clock() - started
                record = {
                    "trace_schema_version": 2,
                    "action_provenance": self.rtc.debug_snapshot().get("last_action_provenance", {}),
                    "predicted_action": predicted.tolist(),
                    "session_id": self.session_id,
                    "step": step,
                    "loop_dt_ms": None if loop_dt is None else loop_dt * 1000.0,
                    "loop_work_ms": elapsed * 1000.0,
                    "deadline_overrun_ms": max(0.0, elapsed - period) * 1000.0,
                    "deadline_lateness_ms": deadline_lateness * 1000.0,
                    "missed_action_ticks": missed_action_ticks,
                    "queue_wait_ms": queue_wait_s * 1000.0,
                    "queue_wait_count": queue_wait_count,
                    "observation_ms": observation_elapsed * 1000.0,
                    "command_ms": command_elapsed * 1000.0,
                    "queue_before": rtc_before["queue_size"],
                    "queue_after": self.rtc.debug_snapshot()["queue_size"],
                    "inference_active": rtc_before["inference_active"],
                    "max_inference_latency_ms": rtc_before["max_inference_latency_ms"],
                    "predicted_xyz": predicted[:3].tolist(),
                    "prediction_delta_xyz": prediction_delta.tolist(),
                    "prediction_delta_rotation_rad": prediction_delta_rotation,
                    "predicted_gripper": float(predicted[9]),
                    "prediction_delta_gripper": gripper_delta,
                    "measured_gripper": float(robot_observation["gripper_position"]),
                    **observation_debug,
                    **command_debug,
                }
                if self.trace is not None:
                    self.trace.emit("control_step", **record)
                if step % self.console_log_every == 0:
                    log.info(
                        "CONTROL step=%d pred_xyz=%s dxyz_mm=%s measured_xyz=%s "
                        "gripper_pred/cmd/measured=%.3f/%.0f/%.3f queue=%d loop=%.1fms overrun=%.1fms",
                        step,
                        np.round(predicted[:3], 4).tolist(),
                        np.round(prediction_delta * 1000.0, 2).tolist(),
                        np.round(command_debug.get("measured_xyz", [math.nan] * 3), 4).tolist(),
                        float(predicted[9]),
                        float(command_debug.get("commanded_gripper", math.nan)),
                        float(robot_observation["gripper_position"]),
                        self.rtc.debug_snapshot()["queue_size"],
                        elapsed * 1000.0,
                        max(0.0, elapsed - period) * 1000.0,
                    )
                step += 1
            if self._stop_reason is not None:
                stage = "control_stop"
                if self.trace is not None:
                    self.trace.emit(
                        "control_stop_requested",
                        session_id=self.session_id,
                        step=step,
                        reason=self._stop_reason,
                    )
                self.rtc.reset()
                self.robot.stop()
                return self._stop_reason
        except BaseException as error:
            self._emit_robot_motion_events(step=step)
            if self.trace is not None:
                self.trace.emit(
                    "control_failed",
                    session_id=self.session_id,
                    step=step,
                    stage=stage,
                    error=repr(error),
                    rtc=self.rtc.debug_snapshot(),
                    last_prediction=None if previous_prediction is None else previous_prediction.tolist(),
                    last_command=(
                        self.robot.command_debug_snapshot()
                        if hasattr(self.robot, "command_debug_snapshot") else {}
                    ),
                )
            self.rtc.reset()
            try:
                self.robot.stop()
            except BaseException as stop_error:
                error.add_note(f"Robot stop also failed: {stop_error}")
            raise

    def stop(self) -> None:
        self._stop.set()
        try:
            self.robot.stop()
        finally:
            self.rtc.close()


class MockFrankaRobotiqRobot:
    """Deterministic hardware replacement for a full disarmed policy smoke test."""

    def __init__(self, workspace_min: tuple[float, ...], workspace_max: tuple[float, ...]) -> None:
        center = (np.asarray(workspace_min) + np.asarray(workspace_max)) / 2.0
        self._state = np.concatenate((center, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0425]))
        self.is_connected = False
        self.stop_calls = 0

    def connect(self) -> None:
        self.is_connected = True

    def check_health(self) -> None:
        if not self.is_connected:
            raise RuntimeError("mock robot is disconnected")

    def get_observation(self) -> dict[str, Any]:
        self.check_health()
        observation: dict[str, Any] = {
            name: float(value) for name, value in zip(STATE_FIELDS, self._state)
        }
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        image[..., 1] = 64
        observation["base"] = image
        observation["left_wrist"] = np.flip(image, axis=1).copy()
        return observation

    def send_action(self, action) -> None:
        raise AssertionError("mock deployment must never execute actions")

    def stop(self) -> None:
        self.stop_calls += 1

    def disconnect(self) -> None:
        self.stop()
        self.is_connected = False
