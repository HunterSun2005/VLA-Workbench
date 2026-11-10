"""Franky Cartesian impedance worker.

The module is importable without Franky installed.  The binding is imported
only when a real arm is opened, which lets ROS and protocol tests run in the
system Python environment.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import json
import math
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any, Mapping

import numpy as np
from scipy.spatial.transform import Rotation

from vla.core.hardware_lease import HardwareLease
from .ipc import decode_packet, encode_packet


def check_binding() -> dict[str, str]:
    """Validate the selected interpreter before opening an arm connection."""
    try:
        import franky
    except Exception as error:
        raise RuntimeError(f"Franky binding unavailable: {error}") from error
    required = ("Robot", "RealtimeConfig", "CartesianImpedanceTracker", "Affine")
    missing = [name for name in required if not hasattr(franky, name)]
    if missing:
        raise RuntimeError("Franky binding lacks required API: " + ", ".join(missing))
    version = getattr(franky, "__version__", "unknown")
    libfranka = (getattr(franky, "libfranka_version", None)
                 or getattr(franky, "__libfranka_version__", None))
    if callable(libfranka):
        try:
            libfranka = libfranka()
        except Exception:
            libfranka = None
    try:
        import importlib.metadata
        for distribution in ("franky-control", "franky"):
            try:
                version = importlib.metadata.version(distribution)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    except Exception:
        pass
    result = {"franky": str(version)}
    if libfranka:
        result["libfranka"] = str(libfranka)
    binding_file = getattr(franky, "__file__", None)
    if binding_file:
        try:
            result["binding"] = str(Path(binding_file).resolve())
        except OSError:
            result["binding"] = str(binding_file)
    return result


@dataclass
class Compliance:
    translational_stiffness: float = 1000.0
    rotational_stiffness: float = 50.0
    nullspace_stiffness: float = 20.0
    translational_clip: float = .008
    rotational_clip: float = .04
    max_step: float = .03
    max_step_rad: float = .10
    max_delta_tau: float = .2
    gains_time_constant: float = .1

    def validate(self) -> "Compliance":
        values = asdict(self)
        if any(not math.isfinite(float(value)) for value in values.values()):
            raise ValueError("Franky compliance parameters must be finite")
        if not (0 < self.translational_stiffness <= 1200):
            raise ValueError("translational_stiffness must be in (0, 1200]")
        if not (0 < self.rotational_stiffness <= 80):
            raise ValueError("rotational_stiffness must be in (0, 80]")
        if self.nullspace_stiffness < 0 or self.translational_clip < .005:
            raise ValueError("nullspace stiffness or translational clip is unsafe")
        if self.rotational_clip < .02 or self.max_step <= 0 or self.max_step_rad <= 0:
            raise ValueError("rotational clip and target steps must be positive")
        if self.max_delta_tau <= 0 or self.gains_time_constant <= 0:
            raise ValueError("tracker dynamics must be positive")
        return self

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None,
                     current: "Compliance | None" = None) -> "Compliance":
        result = cls(**(asdict(current) if current else {}))
        for key, value in (values or {}).items():
            if key not in asdict(result):
                raise ValueError(f"unknown Franky compliance parameter: {key}")
            setattr(result, key, float(value))
        return result.validate()


def _pose_to_affine(franky: Any, pose: np.ndarray) -> Any:
    pose = np.asarray(pose, dtype=np.float64)
    quat = pose[3:] / np.linalg.norm(pose[3:])
    matrix = np.eye(4)
    matrix[:3, :3] = Rotation.from_quat(quat).as_matrix()
    matrix[:3, 3] = pose[:3]
    return franky.Affine(matrix)


def _affine_pose(value: Any) -> np.ndarray:
    """Normalize Franky Affine or libfranka 4x4 column-major arrays."""
    if hasattr(value, "translation") and hasattr(value, "quaternion"):
        return np.r_[np.asarray(value.translation, dtype=float),
                     np.asarray(value.quaternion, dtype=float)]
    matrix = np.asarray(value, dtype=float)
    if matrix.size == 16:
        matrix = matrix.reshape(4, 4, order="F")
        return np.r_[matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_quat()]
    raise ValueError("Franky state pose is not an Affine or 4x4 matrix")


class FrankyArm:
    """One exclusive CartesianImpedanceTracker session."""

    def __init__(self, name: str, robot_ip: str, *, gripper_port: str | None = None,
                 compliance: Compliance | None = None, realtime_config: str = "ignore"):
        self.name = name
        self.robot_ip = robot_ip
        self.compliance = (compliance or Compliance()).validate()
        self.realtime_config = realtime_config
        # The Franky worker owns the FCI arm exclusively. Gripper ownership
        # remains with the ROS peripheral stack, so gripper_port is optional.
        self._lease = HardwareLease(robot_ip, gripper_port)
        self.franky = None
        self.robot = None
        self.tracker = None
        self.target_pose: np.ndarray | None = None
        self.target_generation = -1
        self.target_sequence = -1
        self.fault: str | None = None
        self.binding_fingerprint: dict[str, str] = {}

    def open(self) -> None:
        try:
            self.binding_fingerprint = check_binding()
            import franky
        except Exception as error:
            if self._lease:
                self._lease.close()
            raise RuntimeError(f"Franky binding unavailable: {error}") from error
        self.franky = franky
        self._lease = self._lease or None
        mode = getattr(franky.RealtimeConfig, "Enforce" if self.realtime_config == "enforce" else "Ignore")
        self.robot = franky.Robot(self.robot_ip, realtime_config=mode)
        self.robot.recover_from_errors()
        self.robot.relative_dynamics_factor = .2
        self.robot.set_collision_behavior(
            [80.0] * 4 + [11.0] * 3,
            [100.0] * 3 + [25.0] * 3,
        )

    def _state(self) -> dict[str, Any]:
        if self.robot is None:
            raise RuntimeError("Franky arm is not open")
        # Surface a tracker thread that stopped between target packets. The
        # collector polls health independently of command traffic, so an
        # asynchronous Franky exception cannot remain hidden until the next
        # user action.
        self._check_tracker()
        raw = self.robot.state
        affine = raw.O_T_EE
        pose = _affine_pose(affine)
        desired = getattr(raw, "O_T_EE_d", None)
        desired_pose = _affine_pose(desired) if desired is not None else self.target_pose
        flange = getattr(raw, "F_T_EE", None)
        flange_pose = (_affine_pose(flange) if flange is not None
                       else np.array([0., 0., 0., 0., 0., 0., 1.]))
        q = np.asarray(raw.q, dtype=float)
        dq = np.asarray(raw.dq, dtype=float)
        qd = np.asarray(getattr(raw, "q_d", q), dtype=float)
        tau = np.asarray(getattr(raw, "tau_J", np.zeros(7)), dtype=float)
        tau_ext = np.asarray(getattr(raw, "tau_ext_hat_filtered", np.zeros(7)), dtype=float)
        force = np.asarray(getattr(raw, "K_F_ext_hat_K", np.zeros(6)), dtype=float)
        return {"arm": self.name, "observed_at_ns": time.monotonic_ns(),
                "tcp_pose": pose.tolist(), "joints": q.tolist(),
                "joint_velocity": dq.tolist(), "joint_torque": tau.tolist(),
                "external_joint_torque": tau_ext.tolist(), "target_joints": qd.tolist(),
                "flange_to_tcp_pose": flange_pose.tolist(),
                "desired_tcp_pose": desired_pose.tolist() if desired_pose is not None else [],
                "force": force[:3].tolist(), "torque": force[3:].tolist(),
                "generation": self.target_generation, "sequence": self.target_sequence,
                "tracker_running": bool(self.tracker is not None and self._tracker_running()),
                "state_valid": True,
                "fault": self.fault}

    def _tracker_running(self) -> bool:
        if self.tracker is None:
            return False
        value = getattr(self.tracker, "is_running", False)
        try:
            return bool(value() if callable(value) else value)
        except Exception as error:
            self.fault = f"Cartesian tracker state unavailable: {error}"
            return False

    def _check_tracker(self) -> None:
        if self.fault:
            raise RuntimeError(self.fault)
        if self.tracker is not None and not self._tracker_running():
            try:
                self.robot.poll_motion()
            except Exception as error:
                self.fault = f"Cartesian tracker stopped: {error}"
            else:
                self.fault = "Cartesian tracker stopped unexpectedly"
            raise RuntimeError(self.fault)

    def _ensure_tracker(self) -> None:
        self._check_tracker()
        if self.tracker is not None:
            return
        # Franky cannot splice a new impedance motion into a previous motion
        # command. Join the old motion and clear any reflex before constructing
        # the tracker, matching the RLinf lifecycle.
        try:
            self.robot.join_motion()
        except Exception:
            pass
        self.robot.recover_from_errors()
        nullspace = np.asarray(self.robot.state.q, dtype=np.float64).copy()
        c = self.compliance
        self.tracker = self.franky.CartesianImpedanceTracker(
            self.robot,
            translational_stiffness=c.translational_stiffness,
            rotational_stiffness=c.rotational_stiffness,
            nullspace_target=nullspace,
            nullspace_stiffness=c.nullspace_stiffness,
            translational_error_clip=np.full(3, c.translational_clip),
            rotational_error_clip=np.full(3, c.rotational_clip),
            max_delta_tau=c.max_delta_tau,
            gains_time_constant=c.gains_time_constant,
        )
        # Seed the target accumulator from the measured TCP. The first client
        # packet is therefore limited by max_step instead of causing a pose
        # discontinuity.
        self.target_pose = _affine_pose(self.robot.state.O_T_EE)

    def set_target(self, pose: list[float], generation: int, sequence: int) -> None:
        self._ensure_tracker()
        pose_array = np.asarray(pose, dtype=np.float64)
        pose_array[3:] /= np.linalg.norm(pose_array[3:])
        if self.target_pose is not None:
            delta = pose_array[:3] - self.target_pose[:3]
            distance = float(np.linalg.norm(delta))
            if distance > self.compliance.max_step:
                pose_array[:3] = self.target_pose[:3] + delta * self.compliance.max_step / distance
            previous = Rotation.from_quat(self.target_pose[3:])
            current = Rotation.from_quat(pose_array[3:])
            delta_r = current * previous.inv()
            vector = delta_r.as_rotvec()
            angle = float(np.linalg.norm(vector))
            if angle > self.compliance.max_step_rad:
                pose_array[3:] = (Rotation.from_rotvec(vector * self.compliance.max_step_rad / angle) * previous).as_quat()
        self.tracker.set_target(_pose_to_affine(self.franky, pose_array))
        self.target_pose = pose_array
        self.target_generation = generation
        self.target_sequence = sequence
        self._check_tracker()

    def reset(self, generation: int) -> None:
        self.stop(recover=False)
        if self.fault:
            raise RuntimeError(self.fault)
        self.target_generation = generation
        self.target_sequence = -1

    def stop(self, *, recover: bool = False) -> None:
        if self.tracker is not None:
            try:
                self.tracker.stop()
            finally:
                self.tracker = None
        try:
            if self.robot is not None:
                self.robot.join_motion()
        except Exception:
            pass
        self.target_pose = None
        if recover and self.robot is not None:
            self.robot.recover_from_errors()

    def recover(self) -> None:
        if self.robot is None:
            raise RuntimeError("Franky arm is not open")
        if self.tracker is not None:
            try:
                self.tracker.stop()
            finally:
                self.tracker = None
        self.robot.recover_from_errors()
        self.fault = None

    def update_compliance(self, values: Mapping[str, Any]) -> Compliance:
        next_value = Compliance.from_mapping(values, self.compliance)
        if self.tracker is not None and any(key in values for key in (
                "translational_clip", "rotational_clip", "max_delta_tau", "gains_time_constant")):
            pose = self.target_pose.copy() if self.target_pose is not None else None
            self.stop()
            self.compliance = next_value
            if pose is not None:
                self.set_target(pose.tolist(), self.target_generation, self.target_sequence)
        else:
            self.compliance = next_value
            if self.tracker is not None:
                self.tracker.set_gains(
                    translational_stiffness=next_value.translational_stiffness,
                    rotational_stiffness=next_value.rotational_stiffness,
                    nullspace_stiffness=next_value.nullspace_stiffness,
                )
        return self.compliance

    def close(self) -> None:
        try:
            self.stop(recover=False)
        finally:
            if self._lease:
                self._lease.close()
                self._lease = None


class FrankyWorker:
    def __init__(self, arms: Mapping[str, Mapping[str, Any]], *, session_id: str,
                 compliance: Mapping[str, Any] | None = None):
        self.session_id = session_id
        base = Compliance.from_mapping(compliance)
        self.arms = {name: FrankyArm(name, str(value["robot_ip"]),
                                     gripper_port=value.get("gripper_port"),
                                     compliance=Compliance.from_mapping({}, base))
                     for name, value in arms.items()}
        self._lock = threading.RLock()
        self.generation = -1
        self.sequence = -1
        self.timestamp_ns = -1
        self.fault: str | None = None
        self.binding_fingerprint: dict[str, str] = {}

    def open(self) -> None:
        opened = []
        try:
            for arm in self.arms.values():
                arm.open()
                self.binding_fingerprint.update(arm.binding_fingerprint)
                opened.append(arm)
        except Exception:
            for arm in reversed(opened):
                arm.close()
            raise

    def request(self, message: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if message.get("session_id") != self.session_id:
                raise ValueError("wrong Franky worker session")
            kind = message.get("kind")
            if kind == "health":
                response = self.snapshot()
                response["request_id"] = str(message.get("request_id", ""))
                return response
            generation = int(message.get("generation", -1))
            if kind == "reset":
                if generation < self.generation:
                    return self.snapshot()
                for arm in self.arms.values():
                    arm.reset(generation)
                self.generation = generation
                self.sequence = -1
                self.timestamp_ns = -1
            elif kind == "stop":
                if generation < self.generation:
                    return self.snapshot()
                for arm in self.arms.values():
                    arm.stop()
                self.generation = generation
            elif kind == "recover":
                if generation < self.generation:
                    return self.snapshot()
                for arm in self.arms.values():
                    arm.recover()
                self.generation = generation
                self.fault = None
            elif kind == "target":
                timestamp_ns = int(message.get("timestamp_ns", -1))
                if (generation < self.generation
                        or int(message.get("sequence", -1)) <= self.sequence
                        or timestamp_ns <= self.timestamp_ns):
                    return self.snapshot()
                if set(message.get("arms", {})) != set(self.arms):
                    raise ValueError("target must contain every configured Franky arm")
                targets = message.get("arms", {})
                # Validate the complete packet before touching either tracker.
                # A failed second arm must never leave a dual-arm command
                # half-installed while the first arm continues moving.
                for name, value in targets.items():
                    if name not in self.arms:
                        raise ValueError(f"unknown Franka arm: {name}")
                    if not isinstance(value, Mapping) or "pose" not in value:
                        raise ValueError(f"missing TCP pose for Franka arm {name}")
                    from .ipc import _validate_pose
                    _validate_pose(value["pose"])
                try:
                    for name, value in targets.items():
                        self.arms[name].set_target(value["pose"], generation, int(message["sequence"]))
                except Exception as error:
                    self.fault = f"Franky target installation failed: {error}"
                    for arm in self.arms.values():
                        try:
                            arm.stop()
                        except Exception:
                            pass
                    raise RuntimeError(self.fault) from error
                self.generation = generation
                self.sequence = int(message["sequence"])
                self.timestamp_ns = timestamp_ns
            elif kind == "compliance":
                if generation < self.generation:
                    return self.snapshot()
                for arm in self.arms.values():
                    arm.update_compliance(message.get("values", {}))
            else:
                raise ValueError(f"unknown Franky worker request: {kind}")
            response = self.snapshot()
            response["request_id"] = str(message.get("request_id", ""))
            if kind == "compliance":
                response["request_id"] = str(message.get("request_id", ""))
            return response

    def snapshot(self) -> dict[str, Any]:
        states = []
        for arm in self.arms.values():
            try:
                states.append(arm._state())
            except Exception as error:
                arm.fault = str(error)
                states.append({"arm": arm.name, "observed_at_ns": time.monotonic_ns(),
                               "tcp_pose": [0, 0, 0, 0, 0, 0, 1], "joints": [0] * 7,
                               "joint_velocity": [0] * 7, "state_valid": False,
                               "fault": arm.fault})
        return {"kind": "state", "session_id": self.session_id,
                "generation": self.generation, "sequence": self.sequence,
                "states": states, "fault": self.fault or next((x.get("fault") for x in states if x.get("fault")), None),
                "binding_fingerprint": dict(self.binding_fingerprint),
                "compliance": asdict(next(iter(self.arms.values())).compliance) if self.arms else asdict(Compliance())}

    def close(self) -> None:
        for arm in reversed(tuple(self.arms.values())):
            arm.close()


class FrankyWorkerServer:
    def __init__(self, worker: FrankyWorker, path: str | os.PathLike[str]):
        self.worker = worker
        self.path = Path(path).expanduser().resolve()
        self._closed = threading.Event()
        self._listener: socket.socket | None = None

    def serve(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Acquire the hardware lease before touching an existing socket. If a
        # second worker starts while the first is live, removing its socket
        # before the lease check would strand the active client.
        self.worker.open()
        try:
            if self.path.exists():
                if not self.path.is_socket():
                    raise FileExistsError(f"refusing to replace non-socket path: {self.path}")
                self.path.unlink()
        except BaseException:
            self.worker.close()
            raise
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(str(self.path))
        os.chmod(self.path, 0o600)
        listener.listen(16)
        listener.settimeout(.2)
        self._listener = listener
        try:
            while not self._closed.is_set():
                try:
                    connection, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    if self._closed.is_set():
                        break
                    raise
                with connection:
                    try:
                        response = self.worker.request(decode_packet(connection))
                    except Exception as error:
                        response = {"kind": "error", "error": str(error), **self.worker.snapshot()}
                    connection.sendall(encode_packet(response))
        finally:
            listener.close()
            self.worker.close()
            if self.path.is_socket():
                self.path.unlink()

    def close(self) -> None:
        self._closed.set()
        if self._listener is not None:
            try:
                self._listener.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            self._listener.close()
