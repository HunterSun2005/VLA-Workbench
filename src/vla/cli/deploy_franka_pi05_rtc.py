from __future__ import annotations

from vla.terminal import (console_print as print, console_status, console_debug, terminal_session,
                          TerminalConfirmation, diagnostic_path, current_reporter)

import argparse
import dataclasses
import json
import logging
import math
from datetime import datetime
import os
import select
import sys
import tempfile
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Any, Callable

import yaml


log = logging.getLogger(__name__)


def _parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.split(","):
        bounds = item.strip().split("-", 1)
        if not bounds[0]:
            raise ValueError(f"invalid CPU list {value!r}")
        start = int(bounds[0])
        end = int(bounds[-1])
        if start < 0 or end < start:
            raise ValueError(f"invalid CPU range {item!r}")
        cpus.update(range(start, end + 1))
    return cpus


def _configure_housekeeping_affinity(value: str | None) -> set[int] | None:
    if value is None:
        return None
    cpus = _parse_cpu_list(value)
    online = set(range(os.cpu_count() or 1))
    if not cpus or not cpus <= online:
        raise ValueError(f"housekeeping_cpus must be a non-empty subset of online CPUs {sorted(online)}")
    os.sched_setaffinity(0, cpus)
    return cpus


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Deploy an absolute 10D pi0.5 SFT checkpoint on left FR3")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer-path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mock", action="store_true", help="run one full inference without hardware")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--debug-log", type=Path, default=None, help="write per-step deployment trace as JSONL")
    return parser.parse_args(argv)


def _load_yaml(path: Path) -> dict:
    value = yaml.safe_load(path.expanduser().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"deployment config must be a YAML mapping: {path}")
    return value


def _build_robot_config(values: dict):
    from vla.robots.franka_robotiq import franka_robotiq_config_from_mapping

    return franka_robotiq_config_from_mapping(values)


def _control_timing(deployment: dict, robot_config) -> tuple[float, float]:
    control_fps = float(deployment.get("fps", 30))
    if not math.isclose(control_fps, robot_config.control_frequency, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(
            "deployment.fps must equal robot.control_frequency: "
            f"{control_fps} != {robot_config.control_frequency}"
        )
    discontinuity_timeout_ms = float(
        deployment.get("control_discontinuity_timeout_ms", robot_config.velocity_motion_duration_ms)
    )
    if discontinuity_timeout_ms <= 1000.0 / control_fps:
        raise ValueError("control discontinuity timeout must be greater than one control period")
    if discontinuity_timeout_ms > robot_config.velocity_motion_duration_ms:
        raise ValueError("control discontinuity timeout must not exceed velocity waypoint hold")
    return control_fps, discontinuity_timeout_ms


def _confirm_execute(execute: bool) -> bool:
    return bool(execute)


class _ControlKeyListener:
    """Read one-key control commands without blocking the deployment loop."""

    def __init__(
        self,
        on_stop: Callable[[str], None],
        key: str = "s",
        *,
        home_enabled: bool = False,
        home_key: str = "h",
    ) -> None:
        self.on_stop = on_stop
        self.key = key.lower().encode("ascii")
        self.home_enabled = bool(home_enabled)
        self.home_key = home_key.lower().encode("ascii")
        if self.home_enabled and self.home_key == self.key:
            raise ValueError("home key must differ from the stop key")
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fd: int | None = None
        self._old_termios: list[Any] | None = None

    def start(self) -> None:
        if not sys.stdin.isatty():
            return
        self._fd = sys.stdin.fileno()
        self._old_termios = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        self._thread = threading.Thread(target=self._worker, name="franka-control-key", daemon=True)
        self._thread.start()
        message = f"press {self.key.decode('ascii')} to stop Franka control"
        if self.home_enabled:
            message += f"; press {self.home_key.decode('ascii')} to return to the demonstration home pose"
        log.info("Control keys active: %s", message)

    def _worker(self) -> None:
        assert self._fd is not None
        while not self._stop.is_set():
            try:
                ready, _, _ = select.select([self._fd], [], [], 0.05)
                if not ready:
                    continue
                value = os.read(self._fd, 1)
            except OSError:
                return
            if not value:
                return
            if value.lower() == self.key:
                self.on_stop("keyboard")
                return
            if self.home_enabled and value.lower() == self.home_key:
                self.on_stop("home")
                return
            if value.lower() == b"c" or value == b"\x1b":
                self.on_stop("exit")
                return

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        if self._fd is not None and self._old_termios is not None:
            try:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_termios)
            except termios.error:
                log.warning("Could not restore terminal settings after Franka control stopped")
        self._thread = None
        self._fd = None
        self._old_termios = None


def _confirm_restart(error: BaseException | None, session_id: int) -> bool:
    if not sys.stdin.isatty():
        log.error(
            "Control session %d failed and cannot be restarted without an interactive terminal",
            session_id,
        )
        return False
    if error is None:
        log.info("Franka control session %d was stopped by the operator", session_id)
    else:
        log.error("Franka control session %d stopped: %s", session_id, error)
    return _wait_control_action() == "restart"


def _wait_control_action() -> str:
    """Read the post-stop single-key menu without requiring Enter."""
    if not sys.stdin.isatty():
        return "exit"
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        log.info("Control menu: press r to restart, h to return home, c or Esc to exit")
        while True:
            value = os.read(fd, 1)
            if value.lower() == b"r":
                return "restart"
            if value.lower() == b"h":
                return "home"
            if value.lower() == b"c" or value == b"\x1b":
                return "exit"
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


@dataclasses.dataclass(frozen=True)
class _HomeConfig:
    target_action: tuple[float, ...]
    clearance_z: float
    gripper_hold_action: float
    phase_timeout_s: float = 20.0
    position_tolerance_m: float = 0.005
    rotation_tolerance_rad: float = math.radians(2.0)
    settled_steps: int = 5
    key: str = "h"


def _rot6d_to_matrix(value):
    import numpy as np

    value = np.asarray(value, dtype=np.float64)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("home target rot6d must be a finite 6-vector")
    first = value[:3]
    first_norm = float(np.linalg.norm(first))
    if first_norm < 1e-8:
        raise ValueError("home target rot6d first column has zero norm")
    first = first / first_norm
    second = value[3:] - np.dot(first, value[3:]) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm < 1e-8:
        raise ValueError("home target rot6d columns are collinear")
    second = second / second_norm
    return np.column_stack((first, second, np.cross(first, second)))


def _matrix_to_rot6d(value):
    import numpy as np

    matrix = np.asarray(value, dtype=np.float64)
    return np.concatenate((matrix[:, 0], matrix[:, 1]))


def _rotation_angle(value) -> float:
    import numpy as np

    return float(np.arccos(np.clip((np.trace(value) - 1.0) / 2.0, -1.0, 1.0)))


def _load_home_config(deployment: dict, robot_config) -> _HomeConfig | None:
    import numpy as np

    values = deployment.get("home")
    if values is None or values.get("enabled", True) is False:
        return None
    target = np.asarray(values.get("target_action"), dtype=np.float64)
    if target.shape != (10,) or not np.isfinite(target).all():
        raise ValueError("deployment.home.target_action must be a finite absolute 10D action")
    _rot6d_to_matrix(target[3:9])
    workspace_min = np.asarray(robot_config.workspace_min, dtype=np.float64)
    workspace_max = np.asarray(robot_config.workspace_max, dtype=np.float64)
    if np.any(target[:3] < workspace_min) or np.any(target[:3] > workspace_max):
        raise ValueError("deployment.home target position is outside the robot workspace")
    if not 0.0 <= target[9] <= 1.0:
        raise ValueError("deployment.home target gripper action must be in [0, 1]")
    clearance_z = float(values.get("clearance_z", target[2]))
    if not workspace_min[2] <= clearance_z <= workspace_max[2]:
        raise ValueError("deployment.home.clearance_z is outside the robot workspace")
    phase_timeout_s = float(values.get("phase_timeout_s", 20.0))
    position_tolerance_m = float(values.get("position_tolerance_m", 0.005))
    rotation_tolerance_rad = float(values.get("rotation_tolerance_rad", math.radians(2.0)))
    settled_steps = int(values.get("settled_steps", 5))
    gripper_hold_action = float(
        values.get(
            "gripper_hold_action",
            (robot_config.gripper_close_trigger + robot_config.gripper_open_trigger) / 2.0,
        )
    )
    if not robot_config.gripper_close_trigger < gripper_hold_action < robot_config.gripper_open_trigger:
        raise ValueError(
            "deployment.home.gripper_hold_action must lie strictly between the close and open triggers"
        )
    key = str(values.get("key", "h")).lower()
    if len(key) != 1 or not key.isascii():
        raise ValueError("deployment.home.key must be one ASCII character")
    if key == "s":
        raise ValueError("deployment.home.key must differ from the stop key 's'")
    if phase_timeout_s <= 0 or position_tolerance_m <= 0 or rotation_tolerance_rad <= 0:
        raise ValueError("deployment.home timeouts and tolerances must be positive")
    if settled_steps <= 0:
        raise ValueError("deployment.home.settled_steps must be positive")
    return _HomeConfig(
        target_action=tuple(float(item) for item in target),
        clearance_z=clearance_z,
        gripper_hold_action=gripper_hold_action,
        phase_timeout_s=phase_timeout_s,
        position_tolerance_m=position_tolerance_m,
        rotation_tolerance_rad=rotation_tolerance_rad,
        settled_steps=settled_steps,
        key=key,
    )


def _move_robot_home(
    robot: Any,
    config: _HomeConfig,
    *,
    fps: float,
    trace: Any | None,
    session_id: int,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Return through a lifted Cartesian route after policy control is stopped."""
    import numpy as np

    if fps <= 0:
        raise ValueError("home control frequency must be positive")
    target = np.asarray(config.target_action, dtype=np.float64)
    current = np.asarray(robot.measured_pose(), dtype=np.float64)
    transit_z = max(float(current[2, 3]), config.clearance_z, float(target[2]))
    lift = np.concatenate(
        (
            current[:3, 3],
            _matrix_to_rot6d(current[:3, :3]),
            [config.gripper_hold_action],
        )
    )
    lift[2] = transit_z
    transit = target.copy()
    transit[2] = transit_z
    phases = (("lift", lift), ("transit", transit), ("settle", target))
    if trace is not None:
        trace.emit(
            "home_return_started",
            session_id=session_id,
            target_action=target.tolist(),
            clearance_z=transit_z,
        )
    log.info(
        "HOME return started: target_xyz=%s clearance_z=%.3f",
        np.round(target[:3], 4).tolist(),
        transit_z,
    )
    period = 1.0 / fps
    try:
        for phase_name, action in phases:
            phase_rotation = _rot6d_to_matrix(action[3:9])
            deadline = clock() + config.phase_timeout_s
            settled = 0
            while True:
                measured = np.asarray(robot.measured_pose(), dtype=np.float64)
                position_error = float(np.linalg.norm(action[:3] - measured[:3, 3]))
                rotation_error = _rotation_angle(phase_rotation @ measured[:3, :3].T)
                if (
                    position_error <= config.position_tolerance_m
                    and rotation_error <= config.rotation_tolerance_rad
                ):
                    settled += 1
                    if settled >= config.settled_steps:
                        break
                else:
                    settled = 0
                if clock() >= deadline:
                    raise TimeoutError(
                        f"HOME {phase_name} timed out: position_error={position_error:.4f}m, "
                        f"rotation_error={math.degrees(rotation_error):.2f}deg"
                    )
                robot.send_action(action)
                sleep(period)
            robot.stop()
            if trace is not None:
                trace.emit(
                    "home_return_phase_completed",
                    session_id=session_id,
                    phase=phase_name,
                    position_error_m=position_error,
                    rotation_error_rad=rotation_error,
                )
            log.info(
                "HOME phase %s complete: position_error=%.1fmm rotation_error=%.2fdeg",
                phase_name,
                position_error * 1000.0,
                math.degrees(rotation_error),
            )
    except BaseException as error:
        try:
            robot.stop()
        finally:
            if trace is not None:
                trace.emit("home_return_failed", session_id=session_id, error=repr(error))
        raise
    if trace is not None:
        trace.emit("home_return_completed", session_id=session_id)
    log.info("HOME return complete; policy control remains stopped")


@dataclasses.dataclass(frozen=True)
class _ControlSessionConfig:
    robot_factory: Callable[[], Any]
    task: str
    fps: float
    console_log_every: int
    discontinuity_timeout_s: float
    rtc_config: Any | None = None
    queue_threshold: int | None = None
    home: _HomeConfig | None = None
    trace_fields: dict[str, Any] = dataclasses.field(default_factory=dict)


def _run_control_sessions(
    *,
    robot_factory: Callable[[], Any],
    loop_factory: Callable[..., Any],
    rtc: Any,
    policy: Any,
    task: str,
    fps: float,
    execute_requested: bool,
    trace: Any | None,
    console_log_every: int,
    discontinuity_timeout_s: float,
    observation_builder: Callable[[dict[str, Any], str], dict[str, Any]],
    session_config_loader: Callable[[], _ControlSessionConfig] | None = None,
    mock: bool = False,
    confirm_execute: Callable[[bool], bool] = _confirm_execute,
    confirm_restart: Callable[[BaseException | None, int], bool] = _confirm_restart,
    control_key_listener_factory: Callable[..., Any] = _ControlKeyListener,
    home_mover: Callable[..., None] = _move_robot_home,
) -> int:
    """Run disposable hardware sessions while keeping policy and RTC resident."""
    session_id = 0
    armed = False
    while True:
        robot = None
        session_error: BaseException | None = None
        operator_stopped = False
        requested_restart: bool | None = None
        try:
            # No inference from the previous robot session may survive into a
            # newly connected arm. This resets state, not model weights.
            rtc.stop()
            reset_policy = getattr(policy, "reset", None)
            if callable(reset_policy):
                reset_policy()

            session_config = (
                session_config_loader()
                if session_config_loader is not None
                else _ControlSessionConfig(
                    robot_factory=robot_factory,
                    task=task,
                    fps=fps,
                    console_log_every=console_log_every,
                    discontinuity_timeout_s=discontinuity_timeout_s,
                )
            )
            if session_config.rtc_config is not None:
                if session_config.queue_threshold is None:
                    raise ValueError("reloaded RTC config is missing queue_threshold")
                policy_config = getattr(policy, "config", None)
                if policy_config is not None:
                    policy_config.rtc_config = session_config.rtc_config
                init_rtc_processor = getattr(policy, "init_rtc_processor", None)
                if callable(init_rtc_processor):
                    # LeRobot's RTCProcessor retains the config object it was
                    # constructed with, so replace this lightweight helper too.
                    # The policy module and its loaded weights remain untouched.
                    init_rtc_processor()
                reconfigure_rtc = getattr(rtc, "reconfigure", None)
                if not callable(reconfigure_rtc):
                    raise RuntimeError("RTC runtime does not support session reconfiguration")
                reconfigure_rtc(
                    rtc_config=session_config.rtc_config,
                    fps=session_config.fps,
                    queue_threshold=session_config.queue_threshold,
                )
            if trace is not None:
                trace.emit(
                    "control_session_config_reloaded",
                    session_id=session_id,
                    task=session_config.task,
                    fps=session_config.fps,
                    console_log_every=session_config.console_log_every,
                    control_discontinuity_timeout_ms=(
                        session_config.discontinuity_timeout_s * 1000.0
                    ),
                    **session_config.trace_fields,
                )
            if session_config_loader is not None:
                log.info("Reloaded deployment YAML for control session %d", session_id)

            robot = session_config.robot_factory()
            loop = loop_factory(
                robot,
                rtc,
                task=session_config.task,
                fps=session_config.fps,
                execute=False,
                trace=trace,
                console_log_every=session_config.console_log_every,
                discontinuity_timeout_s=session_config.discontinuity_timeout_s,
                session_id=session_id,
            )
            if trace is not None:
                trace.emit("control_session_connecting", session_id=session_id)
            if mock:
                log.info("Starting mock FR3, cameras, and gripper")
            else:
                log.info(
                    "Connecting FR3, two RealSense cameras, and Robotiq serial device "
                    "for control session %d",
                    session_id,
                )
            robot.connect()
            first_action = loop.preflight()
            log.info(
                "Control session %d preflight passed; first absolute action shape=%s",
                session_id,
                first_action.shape,
            )
            if not execute_requested:
                log.info("Deployment remains disarmed; no robot command was sent")
                return 0
            if not armed:
                # The normal deployment path starts immediately after preflight.
                # Keep the injectable callback for tests and downstream wrappers.
                armed = True if confirm_execute is _confirm_execute else confirm_execute(True)
                if not armed:
                    log.info("Deployment remains disarmed; no robot command was sent")
                    return 0

            # Rebuild the queue from a fresh observation without reconstructing
            # the policy after the startup checks.
            log.info("Armed session %d; opening Robotiq before Franka control", session_id)
            prepare_for_control = getattr(robot, "prepare_for_control", None)
            if callable(prepare_for_control):
                if trace is not None:
                    trace.emit("control_session_gripper_opening", session_id=session_id)
                measured_open_fraction = float(prepare_for_control())
                if trace is not None:
                    trace.emit(
                        "control_session_gripper_opened",
                        session_id=session_id,
                        measured_open_fraction=measured_open_fraction,
                    )
                log.info(
                    "Robotiq ready for session %d; measured open fraction=%.3f",
                    session_id,
                    measured_open_fraction,
                )
            log.info("Refreshing observation and priming RTC action queue for session %d", session_id)
            rtc.stop()
            if callable(reset_policy):
                reset_policy()
            rtc.publish(observation_builder(robot.get_observation(), session_config.task))
            rtc.infer(account_for_latency=False)
            loop.execute = True
            if trace is not None:
                trace.emit("control_session_started", session_id=session_id)
            log.info("RTC queue primed; starting session %d at %g Hz", session_id, loop.fps)
            key_listener = control_key_listener_factory(
                loop.request_stop,
                home_enabled=session_config.home is not None,
                home_key=(session_config.home.key if session_config.home is not None else "h"),
            )
            try:
                key_listener.start()
                stop_reason = loop.run()
            finally:
                key_listener.close()
            if stop_reason == "exit":
                return 0
            if stop_reason == "home":
                if session_config.home is None:
                    raise RuntimeError("HOME was requested but deployment.home is disabled")
                rtc.stop()
                home_mover(
                    robot,
                    session_config.home,
                    fps=session_config.fps,
                    trace=trace,
                    session_id=session_id,
                )
                operator_stopped = True
                if trace is not None:
                    trace.emit(
                        "control_session_operator_stopped",
                        session_id=session_id,
                        reason="home_completed",
                    )
            elif stop_reason == "keyboard":
                operator_stopped = True
                if trace is not None:
                    trace.emit(
                        "control_session_operator_stopped",
                        session_id=session_id,
                        reason="keyboard",
                    )
            else:
                if trace is not None:
                    trace.emit("control_session_completed", session_id=session_id)
                return 0
            if operator_stopped and confirm_restart is _confirm_restart:
                action = _wait_control_action()
                while action == "home":
                    if session_config.home is None:
                        log.error("HOME is disabled in deployment configuration")
                    else:
                        home_mover(
                            robot,
                            session_config.home,
                            fps=session_config.fps,
                            trace=trace,
                            session_id=session_id,
                        )
                    action = _wait_control_action()
                requested_restart = action == "restart"
        except KeyboardInterrupt:
            log.info("Stopping all deployment resources on Ctrl+C")
            return 0
        except Exception as error:
            session_error = error
            log.exception("Control session %d failed; the loaded model remains resident", session_id)
            if trace is not None:
                trace.emit("control_session_failed", session_id=session_id, error=repr(error))
        finally:
            # Pi05RTC.stop() waits for in-flight inference but deliberately does
            # not terminate its worker or unload policy weights from the GPU.
            try:
                rtc.stop()
            except BaseException as error:
                log.exception("Failed to quiesce RTC after control session %d", session_id)
                if session_error is None:
                    session_error = error
            if robot is not None:
                try:
                    robot.disconnect()
                except BaseException:
                    log.exception("Failed to disconnect hardware after control session %d", session_id)
            if trace is not None:
                trace.emit("control_session_stopped", session_id=session_id)

        if not operator_stopped and (session_error is None or not execute_requested):
            return 1
        try:
            if requested_restart is not None:
                restart = requested_restart
            else:
                restart = confirm_restart(None if operator_stopped else session_error, session_id)
        except KeyboardInterrupt:
            log.info("Stopping all deployment resources on Ctrl+C")
            return 0
        if not restart:
            log.info("Control restart declined; closing the resident model")
            return 0 if operator_stopped else 1
        session_id += 1
        armed = True  # The single-key restart menu has already selected r.
        if trace is not None:
            trace.emit("control_session_restart_requested", session_id=session_id)
        log.info("Restarting Franka control with reloaded YAML; checkpoint weights remain loaded")


def _load_pi05_config(config_class, contract):
    try:
        return config_class.from_pretrained(contract.path, local_files_only=True)
    except Exception:
        # Older 0.5.x patch releases reject metadata fields introduced by the
        # SFT writer. Contract validation already checked them; retain only the
        # fields understood by the installed concrete config class.
        import draccus

        known = {field.name for field in dataclasses.fields(config_class)}
        raw = {key: value for key, value in contract.config.items() if key in known}
        with tempfile.NamedTemporaryFile("w", suffix=".json") as stream:
            json.dump(raw, stream)
            stream.flush()
            with draccus.config_type("json"):
                return draccus.parse(config_class, stream.name, args=[])


@terminal_session("rtc-deploy")
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mock and args.execute:
        raise ValueError("--mock and --execute cannot be used together")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    values = _load_yaml(args.config)
    deployment = values.get("deployment", {})
    housekeeping_cpus = _configure_housekeeping_affinity(deployment.get("housekeeping_cpus"))
    if housekeeping_cpus is not None:
        log.info("Pinned deployment/PyTorch/camera threads to housekeeping CPUs %s", sorted(housekeeping_cpus))
    torch_num_threads = int(deployment.get("torch_num_threads", 1))
    torch_num_interop_threads = int(deployment.get("torch_num_interop_threads", 1))
    if torch_num_threads <= 0 or torch_num_interop_threads <= 0:
        raise ValueError("deployment torch thread counts must be positive")
    # Configure CPU pools before model construction and the first torch op. The
    # PI05 forward pass runs on CUDA, so allowing PyTorch/OMP to occupy every CPU
    # only starves libfranka's 1 kHz thread during background RTC refills.
    os.environ["OMP_NUM_THREADS"] = str(torch_num_threads)
    os.environ["MKL_NUM_THREADS"] = str(torch_num_threads)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    import torch

    torch.set_num_threads(torch_num_threads)
    try:
        torch.set_num_interop_threads(torch_num_interop_threads)
    except RuntimeError as error:
        if torch.get_num_interop_threads() != torch_num_interop_threads:
            raise RuntimeError(
                "PyTorch inter-op threads were initialized before deployment; "
                "start inference from a fresh process"
            ) from error
    from vla.adapters.lerobot import register
    from vla.deployment.franka_pi05_rtc import (
        DeploymentLoop,
        JsonlDebugTrace,
        Pi05RTC,
        policy_observation,
        validate_checkpoint,
    )

    register()
    contract = validate_checkpoint(args.checkpoint, args.tokenizer_path)
    device_name = args.device or deployment.get("device", "cuda")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA deployment requested but torch.cuda.is_available() is false")

    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    from lerobot.processor import PolicyProcessorPipeline
    from lerobot.processor.converters import policy_action_to_transition, transition_to_policy_action
    from lerobot.utils.constants import POLICY_POSTPROCESSOR_DEFAULT_NAME, POLICY_PREPROCESSOR_DEFAULT_NAME
    from lerobot.policies.utils import prepare_observation_for_inference

    # Read through LeRobot after strict JSON validation so incompatible local
    # installations fail before any hardware connection is attempted.
    config = _load_pi05_config(PI05Config, contract)
    config.device = device_name

    def make_rtc_config(rtc_values: dict) -> RTCConfig:
        return RTCConfig(
            enabled=True,
            execution_horizon=int(rtc_values.get("execution_horizon", 10)),
            max_guidance_weight=float(rtc_values.get("max_guidance_weight", 10.0)),
            prefix_attention_schedule=RTCAttentionSchedule(
                rtc_values.get("prefix_attention_schedule", "EXP")
            ),
        )

    rtc_values = values.get("rtc", {})
    rtc_config = make_rtc_config(rtc_values)
    config.rtc_config = rtc_config
    policy = PI05Policy.from_pretrained(
        contract.path, config=config, local_files_only=True
    ).to(device).eval()
    policy.reset()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(contract.tokenizer, local_files_only=True)
    overrides = {
        "tokenizer_processor": {
            "tokenizer_name": contract.tokenizer,
            "tokenizer": tokenizer,
        },
        "device_processor": {"device": device_name},
    }
    try:
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(contract.path),
            preprocessor_overrides=overrides,
        )
    except TypeError:
        # LeRobot 0.5.1 exposes local-only loading through the pipeline. Keeping
        # this path explicit also guarantees tokenizer resolution stays offline.
        preprocessor = PolicyProcessorPipeline.from_pretrained(
            contract.path, config_filename=POLICY_PREPROCESSOR_DEFAULT_NAME + ".json",
            local_files_only=True, overrides=overrides,
        )
        postprocessor = PolicyProcessorPipeline.from_pretrained(
            contract.path, config_filename=POLICY_POSTPROCESSOR_DEFAULT_NAME + ".json",
            local_files_only=True, to_transition=policy_action_to_transition,
            to_output=transition_to_policy_action,
        )

    task = str(deployment.get("task", "Place the yellow block on the white paper."))

    def preprocess(observation):
        observation_task = str(observation.get("task", task))
        raw = {key: value for key, value in observation.items() if key != "task"}
        prepared = prepare_observation_for_inference(
            raw, device, observation_task, "franka_robotiq"
        )
        return preprocessor(prepared)

    if args.mock:
        from vla.deployment.franka_pi05_rtc import MockFrankaRobotiqRobot
    else:
        from vla.robots.franka_robotiq import FrankaRobotiqRobot

    def session_trace_fields(robot_config, queue_threshold: int) -> dict[str, Any]:
        return {
            "queue_threshold": queue_threshold,
            "robot_ip": robot_config.robot_ip,
            "dynamics_factor": robot_config.dynamics_factor,
            "max_linear_velocity": robot_config.max_linear_velocity,
            "max_angular_velocity": robot_config.max_angular_velocity,
            "target_smoothing_alpha": robot_config.target_smoothing_alpha,
            "gripper_action_threshold": robot_config.gripper_action_threshold,
            "gripper_close_trigger": robot_config.gripper_close_trigger,
            "gripper_open_trigger": robot_config.gripper_open_trigger,
            "gripper_post_close_lock_s": robot_config.gripper_post_close_lock_s,
            "gripper_startup_open_fraction": robot_config.gripper_startup_open_fraction,
            "gripper_startup_open_timeout_s": robot_config.gripper_startup_open_timeout_s,
            "velocity_smoothing": robot_config.velocity_smoothing,
            "smooth_angular_velocity": robot_config.smooth_angular_velocity,
            "velocity_tracking_tau_s": robot_config.velocity_tracking_tau_s,
            "velocity_motion_duration_ms": robot_config.velocity_motion_duration_ms,
            "min_command_position_delta": robot_config.min_command_position_delta,
            "min_command_rotation_delta": robot_config.min_command_rotation_delta,
            "max_consecutive_motion_failures": robot_config.max_consecutive_motion_failures,
            "camera_max_pair_skew_ms": robot_config.camera_max_pair_skew_s * 1000.0,
            "observation_max_image_age_ms": robot_config.observation_max_image_age_s * 1000.0,
            "robot_realtime_cpu": robot_config.realtime_cpu,
        }

    def load_session_config() -> _ControlSessionConfig:
        reloaded_values = _load_yaml(args.config)
        reloaded_deployment = reloaded_values.get("deployment", {})
        reloaded_device = args.device or reloaded_deployment.get("device", "cuda")
        if reloaded_device != device_name:
            raise ValueError(
                "deployment.device changed after model load; restart the deployment process"
            )
        reloaded_housekeeping_value = reloaded_deployment.get("housekeeping_cpus")
        reloaded_housekeeping = (
            None
            if reloaded_housekeeping_value is None
            else _parse_cpu_list(str(reloaded_housekeeping_value))
        )
        if reloaded_housekeeping != housekeeping_cpus:
            raise ValueError(
                "deployment.housekeeping_cpus changed after process affinity was set; "
                "restart the deployment process"
            )
        reloaded_torch_threads = int(reloaded_deployment.get("torch_num_threads", 1))
        reloaded_torch_interop = int(
            reloaded_deployment.get("torch_num_interop_threads", 1)
        )
        if (
            reloaded_torch_threads != torch_num_threads
            or reloaded_torch_interop != torch_num_interop_threads
        ):
            raise ValueError(
                "deployment torch thread counts changed after PyTorch initialization; "
                "restart the deployment process"
            )

        reloaded_robot_config = _build_robot_config(reloaded_values)
        reloaded_fps, reloaded_timeout_ms = _control_timing(
            reloaded_deployment, reloaded_robot_config
        )
        if (
            housekeeping_cpus is not None
            and reloaded_robot_config.realtime_cpu in housekeeping_cpus
        ):
            raise ValueError(
                f"robot realtime_cpu {reloaded_robot_config.realtime_cpu} must not be in "
                "deployment housekeeping_cpus"
            )
        reloaded_rtc_values = reloaded_values.get("rtc", {})
        reloaded_queue_threshold = int(reloaded_rtc_values.get("queue_threshold", 30))
        reloaded_rtc_config = make_rtc_config(reloaded_rtc_values)
        reloaded_home = _load_home_config(reloaded_deployment, reloaded_robot_config)

        if args.mock:

            def reloaded_robot_factory(config=reloaded_robot_config):
                return MockFrankaRobotiqRobot(config.workspace_min, config.workspace_max)

        else:

            def reloaded_robot_factory(config=reloaded_robot_config):
                return FrankaRobotiqRobot(config)

        return _ControlSessionConfig(
            robot_factory=reloaded_robot_factory,
            task=str(
                reloaded_deployment.get(
                    "task", "Place the yellow block on the white paper."
                )
            ),
            fps=reloaded_fps,
            console_log_every=int(reloaded_deployment.get("debug_console_every", 30)),
            discontinuity_timeout_s=reloaded_timeout_ms / 1000.0,
            rtc_config=reloaded_rtc_config,
            queue_threshold=reloaded_queue_threshold,
            home=reloaded_home,
            trace_fields=session_trace_fields(
                reloaded_robot_config, reloaded_queue_threshold
            ),
        )

    robot_config = _build_robot_config(values)
    control_fps, discontinuity_timeout_ms = _control_timing(deployment, robot_config)
    if args.mock:

        def robot_factory():
            return MockFrankaRobotiqRobot(
                robot_config.workspace_min, robot_config.workspace_max
            )

    else:

        def robot_factory():
            return FrankaRobotiqRobot(robot_config)

    trace = None
    if args.execute or args.debug_log is not None:
        debug_path = args.debug_log
        if debug_path is None:
            debug_dir = Path(deployment.get("debug_log_dir", "logs"))
            debug_path = diagnostic_path("rtc.jsonl", debug_dir / f"franka_pi05_rtc_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
        trace = JsonlDebugTrace(debug_path)
        log.info("Writing deployment debug trace to %s", trace.path)
        trace.emit(
            "deployment_config",
            checkpoint=str(contract.path),
            device=device_name,
            fps=control_fps,
            queue_threshold=int(rtc_values.get("queue_threshold", 30)),
            robot_ip=robot_config.robot_ip,
            dynamics_factor=robot_config.dynamics_factor,
            max_linear_velocity=robot_config.max_linear_velocity,
            max_angular_velocity=robot_config.max_angular_velocity,
            target_smoothing_alpha=robot_config.target_smoothing_alpha,
            gripper_action_threshold=robot_config.gripper_action_threshold,
            gripper_close_trigger=robot_config.gripper_close_trigger,
            gripper_open_trigger=robot_config.gripper_open_trigger,
            gripper_post_close_lock_s=robot_config.gripper_post_close_lock_s,
            gripper_startup_open_fraction=robot_config.gripper_startup_open_fraction,
            gripper_startup_open_timeout_s=robot_config.gripper_startup_open_timeout_s,
            velocity_smoothing=robot_config.velocity_smoothing,
            smooth_angular_velocity=robot_config.smooth_angular_velocity,
            velocity_tracking_tau_s=robot_config.velocity_tracking_tau_s,
            velocity_motion_duration_ms=robot_config.velocity_motion_duration_ms,
            min_command_position_delta=robot_config.min_command_position_delta,
            min_command_rotation_delta=robot_config.min_command_rotation_delta,
            max_consecutive_motion_failures=robot_config.max_consecutive_motion_failures,
            camera_max_pair_skew_ms=robot_config.camera_max_pair_skew_s * 1000.0,
            observation_max_image_age_ms=robot_config.observation_max_image_age_s * 1000.0,
            control_discontinuity_timeout_ms=discontinuity_timeout_ms,
            torch_num_threads=torch.get_num_threads(),
            torch_num_interop_threads=torch.get_num_interop_threads(),
            housekeeping_cpus=(sorted(housekeeping_cpus) if housekeeping_cpus is not None else None),
            robot_realtime_cpu=robot_config.realtime_cpu,
            cuda_device=(torch.cuda.get_device_name(device) if device.type == "cuda" else None),
        )
    rtc = Pi05RTC(
        policy, preprocess, postprocessor, rtc_config=rtc_config,
        fps=control_fps,
        queue_threshold=int(rtc_values.get("queue_threshold", 30)),
        trace=trace,
    )
    try:
        return _run_control_sessions(
            robot_factory=robot_factory,
            loop_factory=DeploymentLoop,
            rtc=rtc,
            policy=policy,
            task=task,
            fps=control_fps,
            execute_requested=args.execute,
            trace=trace,
            console_log_every=int(deployment.get("debug_console_every", 30)),
            discontinuity_timeout_s=discontinuity_timeout_ms / 1000.0,
            mock=args.mock,
            observation_builder=policy_observation,
            session_config_loader=load_session_config,
        )
    finally:
        try:
            rtc.close()
        finally:
            if trace is not None:
                trace.close()


if __name__ == "__main__":
    raise SystemExit(main())
