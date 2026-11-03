"""Collect single-arm VLA/RLT rollouts through the proven Franky backend."""

from __future__ import annotations

from vla.terminal import (console_print as print, console_status, console_debug, terminal_session,
                          TerminalConfirmation, diagnostic_path, current_reporter)

import argparse
from collections import Counter
from dataclasses import dataclass, replace
from datetime import datetime
import math
import os
from pathlib import Path
import sys
import tempfile
import time
import traceback
from typing import Any

import numpy as np
import yaml

from vla.datacollection.franka_robotiq_single_left import schema
from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter
from vla.deployment.franka_pi05_rtc import JsonlDebugTrace
from vla.robots.franka_robotiq import (
    FrankaRobotiqRobot,
    load_franka_robotiq_config,
)
from vla.robots.franka_robotiq.robot import (
    ACTION_FIELDS,
    FrankaRolloutSnapshot,
    action_vector,
    rot6d_to_matrix,
)

from .common import AsyncVideoWriter, KeyboardEvents
from .config import RolloutConfig, load_rollout_config
from .gamepad import decode_cartesian_gamepad_values
from .joy_transport import JoyReceiver
from .policy_client import AsyncPolicyClient, PolicyResult
from .safety import clamp_gripper_action_roundoff, validate_rollout_episode
from vla.core.online_contract import (
    MAX_STEP_PERIODS, MIN_STEP_PERIODS, RECORDING_SCHEMA_VERSION, TIMING_FEATURES, load_run,
)

COLLECTOR_HUMAN = 0
COLLECTOR_PI05 = 1
COLLECTOR_RLT_ACTOR = 2
COLLECTOR_NAMES = {
    COLLECTOR_HUMAN: "HUMAN",
    COLLECTOR_PI05: "VLA",
    COLLECTOR_RLT_ACTOR: "RLT_ACTOR",
}
MIN_EPISODE_FRAMES = 12

EXTRA_FEATURES = {
    "complementary_info.is_intervention": {
        "dtype": "float32",
        "shape": [1],
        "names": ["is_intervention"],
    },
    "complementary_info.phase": {
        "dtype": "float32",
        "shape": [1],
        "names": ["phase"],
    },
    "complementary_info.collector_policy_id": {
        "dtype": "int64",
        "shape": [1],
        "names": ["collector_policy_id"],
        "info": {"codebook": {"0": "human", "1": "pi0.5", "2": "pi_rlt_actor"}},
    },
}


def episode_source_frame_counts(
    phase_values: list[float], collector_values: list[int]
) -> dict[str, int]:
    if len(phase_values) != len(collector_values):
        raise ValueError("phase and collector annotations must have the same length")
    critical = [float(value) > 0.5 for value in phase_values]
    return {
        "critical_actor": sum(
            active and int(source) == COLLECTOR_RLT_ACTOR
            for active, source in zip(critical, collector_values, strict=True)
        ),
        "critical_human": sum(
            active and int(source) == COLLECTOR_HUMAN
            for active, source in zip(critical, collector_values, strict=True)
        ),
        "critical_vla": sum(
            active and int(source) == COLLECTOR_PI05
            for active, source in zip(critical, collector_values, strict=True)
        ),
    }


@dataclass(frozen=True)
class CommandSnapshot:
    action: np.ndarray
    source: int
    phase: float
    applied_at: float
    command_id: int = 0


def rollout_state(
    snapshot: FrankaRolloutSnapshot,
    previous: tuple[np.ndarray, np.ndarray, int] | None,
) -> np.ndarray:
    position = snapshot.pose[:3, 3]
    rotation = snapshot.pose[:3, :3]
    velocity = schema.base_tcp_velocity(
        previous[0] if previous else None,
        previous[1] if previous else None,
        previous[2] if previous else None,
        position,
        rotation,
        snapshot.observed_at_ns,
    )
    value = np.concatenate(
        (
            position,
            schema.matrix_to_rot6d(rotation),
            [snapshot.gripper_width],
            snapshot.base_wrench,
            velocity,
            snapshot.joint_positions,
            snapshot.joint_torques,
            snapshot.external_joint_torques,
            snapshot.joint_velocities,
        )
    ).astype(np.float32)
    if value.shape != (len(schema.STATE_FIELDS),) or not np.isfinite(value).all():
        raise ValueError(f"invalid Franky rollout state: {value.shape}")
    return value


class FrankyRolloutCollector:
    _MAX_CONSECUTIVE_OBSERVATION_TIMEOUTS = 3

    def __init__(
        self,
        config: RolloutConfig,
        robot: FrankaRobotiqRobot,
        *,
        policy_client: AsyncPolicyClient | None = None,
        joy_receiver: JoyReceiver | None = None,
        writer: V30DatasetWriter | None = None,
        trace: JsonlDebugTrace | None = None,
        log_actions_every: int = 0,
        require_online: bool = False,
    ) -> None:
        if config.backend != "franky" or len(config.arms) != 1:
            raise ValueError("Franky rollout currently supports exactly one arm")
        self.config = config
        self.arm = config.arms[0]
        self.robot = robot
        self.policy_client = policy_client or AsyncPolicyClient(
            config.socket_path, config.policy_timeout_s
        )
        self.joy_receiver = joy_receiver
        self.trace = trace
        if log_actions_every < 0:
            raise ValueError("log_actions_every must be non-negative")
        self.log_actions_every = int(log_actions_every)
        self.require_online = bool(require_online)
        self._sft_interface = config.gamepad_profile == "sft_record"
        self._setup_teleop = None
        self.preview = None
        self._preview_episode = "-"
        self._processing = False
        self._full_episode = False
        self._frame_features = {**EXTRA_FEATURES, **(TIMING_FEATURES if require_online else {})}
        self._segment_id = 0
        self._command_sequence = 0
        self._pending_frame: int | None = None
        self._pending_event: tuple[str, str | None] | None = None
        self._at_boundary = False
        self._critical_entered = False
        self._episode_identity: dict[str, Any] = {}
        self._owns_policy_client = policy_client is None
        self._owns_joy_receiver = joy_receiver is None
        self._episode_active = False
        self._priming = False
        self._intervention = False
        self._phase = "vla"
        self._reset_policy_on_request = True
        self._latest_policy_result: PolicyResult | None = None
        self._latest_policy_action: np.ndarray | None = None
        self._command: CommandSnapshot | None = None
        self._states: list[np.ndarray] = []
        self._actions: list[np.ndarray] = []
        self._annotations = {key: [] for key in self._frame_features}
        self._video_writers: dict[str, AsyncVideoWriter] = {}
        self._last_pose: tuple[np.ndarray, np.ndarray, int] | None = None
        self._last_gamepad_tick = time.monotonic()
        self._last_action_log = 0
        self._consecutive_observation_timeouts = 0
        self._stale_joy_reported = False
        self._faulted = False
        self._fault_reason: str | None = None
        self._closed = False
        self._gamepad_controls = None
        self._skip_reasons: Counter[str] = Counter()
        try:
            self._validate_contract()
            if self._full_episode:
                self._phase = "rl"
            if config.gamepad_profile == "sft_record":
                from vla.datacollection.start_teleop.rollout_gamepad import RolloutGamepadControls
                self._gamepad_controls = RolloutGamepadControls(robot, self.joy_receiver)
                human_robot = self._gamepad_controls.config.robot
                if (tuple(human_robot.workspace_min) != tuple(self.arm.workspace_min)
                        or tuple(human_robot.workspace_max) != tuple(self.arm.workspace_max)):
                    raise ValueError("SFT gamepad workspace must match the online rollout workspace")
            run = load_run(config.dataset_root) if require_online else None
            if run is not None and any(self.policy_client.contract.get(key) != run[key] for key in ("run_uuid", "dataset_uuid")):
                raise ValueError("collector/server online dataset identity mismatch")
            self.writer = writer or V30DatasetWriter(
                config.dataset_root,
                config.fps,
                config.task,
                tuple(camera.feature_key for camera in config.cameras),
                *config.image_size,
                state_fields=tuple(schema.STATE_FIELDS),
                action_fields=tuple(schema.ACTION_FIELDS),
                extra_features=self._frame_features,
                robot_type="franka_robotiq",
                info_metadata={
                    "recording_schema_version": RECORDING_SCHEMA_VERSION if require_online else 3,
                    **({key: run[key] for key in ("run_uuid", "dataset_uuid")} if run else {}),
                    "rlt_episode_metadata_fields": {
                        "rl_intervals": "RL frame intervals with episode outcome.",
                        "human_intervention_intervals": "Human intervention frame intervals.",
                    },
                    "policy_checkpoint_fingerprint": self.policy_client.contract[
                        "checkpoint_fingerprint"
                    ],
                    "rollout_control_backend": "franky",
                    **(
                        {
                            "online_rlt": True,
                            "online_replay_root": self.policy_client.contract[
                                "online_replay_root"
                            ],
                            "online_warmup_steps": self.policy_client.contract[
                                "online_warmup_steps"
                            ],
                        }
                        if self.require_online
                        else {}
                    ),
                },
                strict_metadata=True,
            )
            if self.joy_receiver is None:
                self.joy_receiver = JoyReceiver(config.joy_socket_path)
            if self._sft_interface:
                from vla.datacollection.start_teleop.gamepad_teleop import GamepadTeleop
                self._setup_teleop = GamepadTeleop(
                    self._gamepad_controls.config, robot, self.joy_receiver,
                )
        except BaseException:
            if self._owns_joy_receiver and self.joy_receiver is not None:
                self.joy_receiver.close()
            if self._owns_policy_client:
                self.policy_client.close()
            raise
        self._emit(
            "collector_ready",
            dataset=str(config.dataset_root),
            task=config.task,
            policy_socket=str(config.socket_path),
            joy_socket=str(config.joy_socket_path),
        )

    @property
    def episode_active(self) -> bool:
        return self._episode_active

    def update_preview(self, *, mode: str | None = None) -> None:
        viewer = getattr(self, "preview", None)
        if viewer is None:
            return
        setup = getattr(self, "_setup_teleop", None)
        controls = getattr(self, "_gamepad_controls", None)
        if mode is None:
            if self._faulted:
                mode = "FAULT"
            elif getattr(self, "_processing", False):
                mode = "SAVING/TRAINING"
            elif self._episode_active:
                mode = "HUMAN REC" if self._intervention else "PRIMING" if self._priming else "RLT REC" if self._phase == "rl" else "VLA REC"
            else:
                mode = "TELEOP" if setup is not None and setup.mode == "active" else "STANDBY"
        speed_control = controls if self._episode_active else setup
        viewer.update_status({
            "mode": mode,
            "episode": getattr(self, "_preview_episode", "-"),
            "frames": len(self._states),
            "speed": "HIGH" if speed_control is not None and speed_control.high_speed else "LOW",
            "critical": "critical" if self._episode_active and self._phase == "rl" else "before",
            "full_episode": getattr(self, "_full_episode", False),
        })

    @property
    def faulted(self) -> bool:
        return self._faulted

    def _emit(self, event: str, **fields: Any) -> None:
        if self.trace is not None:
            self.trace.emit(event, **fields)

    def _validate_contract(self) -> None:
        contract = self.policy_client.contract
        expected_cameras = [camera.feature_key for camera in self.config.cameras]
        expected_shapes = {
            key: [*self.config.image_size, 3] for key in expected_cameras
        }
        expected_state_names = list(schema.TCP_STATE_FIELDS)
        expected_action_names = list(schema.ACTION_FIELDS)
        mismatches = []
        online_enabled = bool(contract.get("online_enabled", False))
        self._full_episode = bool(contract.get("online_full_episode", False))
        self._sft_only = contract.get("inference_mode") == "sft_action_queue"
        if self._full_episode and not online_enabled:
            mismatches.append("full-episode mode requires online learning")
        if online_enabled != self.require_online:
            mismatches.append(
                "collector --online-warmup and policy server online mode must match"
            )
        if self.require_online and str(self.config.dataset_root.resolve()) != str(
            contract.get("online_dataset_root", "")
        ):
            mismatches.append("online dataset root does not match the policy server")
        checks = (
            (int(contract["state_dim"]) == 10, "state_dim must be 10"),
            (int(contract["action_dim"]) == 10, "action_dim must be 10"),
            (contract.get("robot_type") == "franka_robotiq", "robot_type mismatch"),
            (contract.get("task") == self.config.task, "task mismatch"),
            (
                list(contract.get("camera_keys", [])) == expected_cameras,
                "camera order mismatch",
            ),
            (
                contract.get("camera_shapes_hwc") == expected_shapes,
                "camera shape mismatch",
            ),
            (
                list(contract.get("state_names", [])) == expected_state_names,
                "state fields mismatch",
            ),
            (
                list(contract.get("action_names", [])) == expected_action_names,
                "action fields mismatch",
            ),
            (
                contract.get("actions_stored_as") == "absolute",
                "action storage mismatch",
            ),
            (
                not bool(contract.get("use_relative_actions", True)),
                "relative actions are unsupported",
            ),
            (
                math.isclose(float(contract.get("fps", 0.0)), self.config.fps),
                "FPS mismatch",
            ),
        )
        mismatches.extend(message for valid, message in checks if not valid)
        if self.config.rtc_deployment_config is not None and not self._sft_only:
            profile = yaml.safe_load(
                self.config.rtc_deployment_config.read_text(encoding="utf-8")
            )
            rtc_values = profile.get("rtc", {}) if isinstance(profile, dict) else {}
            expected_horizon = int(rtc_values.get("execution_horizon", 10))
            expected_threshold = int(rtc_values.get("queue_threshold", 30))
            rtc_checks = (
                (
                    bool(contract.get("rtc_enabled", False)),
                    "RTC runtime is not enabled",
                ),
                (
                    contract.get("rtc_execution_horizon") == expected_horizon,
                    "RTC execution_horizon mismatch",
                ),
                (
                    contract.get("rtc_queue_threshold") == expected_threshold,
                    "RTC queue_threshold mismatch",
                ),
            )
            mismatches.extend(message for valid, message in rtc_checks if not valid)
        if mismatches:
            raise ValueError(
                "rollout policy contract mismatch: " + "; ".join(mismatches)
            )

    def _images(self, snapshot: FrankaRolloutSnapshot) -> dict[str, np.ndarray]:
        images = {
            camera.feature_key: np.asarray(snapshot.images[camera.name])
            for camera in self.config.cameras
        }
        expected = (*self.config.image_size, 3)
        for key, image in images.items():
            if image.shape != expected or image.dtype != np.uint8:
                raise ValueError(
                    f"{key} must be uint8 RGB with shape {expected}, "
                    f"got {image.shape}/{image.dtype}"
                )
        return images

    def _record_frame(
        self,
        snapshot: FrankaRolloutSnapshot,
        state: np.ndarray,
        images: dict[str, np.ndarray],
    ) -> None:
        command = self._command
        if command is None or self._priming:
            return
        for key, video_writer in self._video_writers.items():
            video_writer.write(images[key])
        self._states.append(state)
        self._actions.append(command.action.copy())
        self._annotations["complementary_info.is_intervention"].append(
            [float(command.source == COLLECTOR_HUMAN)]
        )
        self._annotations["complementary_info.phase"].append([command.phase])
        self._annotations["complementary_info.collector_policy_id"].append(
            [command.source]
        )
        if self.require_online:
            values = {
                "observed_at_ns": snapshot.observed_at_ns,
                "command_at_ns": int(command.applied_at * 1e9),
                "command_id": command.command_id,
                "segment_id": self._segment_id,
                "action_valid": 0,
                "segment_terminal": 0,
            }
            for name, value in values.items():
                self._annotations[f"complementary_info.{name}"].append([value])
            self._pending_frame = len(self._states) - 1
        self._emit(
            "frame_recorded",
            frame_index=len(self._states) - 1,
            source=COLLECTOR_NAMES[command.source],
            phase="rl" if command.phase > 0.5 else "vla",
            observed_at_ns=snapshot.observed_at_ns,
        )

    def set_control_mode(self, mode: str) -> None:
        """Called on the coordinator thread, between command ticks."""
        self.policy_client.set_control_mode(mode)
        self._web_mode = mode
        self._web_switch_pending = True
        self._latest_policy_result = None
        self._reset_policy_on_request = True
        self._latest_policy_action = None
        self._command = None
        if not self._intervention and self._episode_active:
            self.robot.stop()
            self._priming = True

    def _consume_policy_result(self) -> bool:
        result = self.policy_client.poll_latest()
        if result is None:
            return False
        managed = getattr(result, "control_mode", None) is not None
        if managed and result.mode_version != self.policy_client.mode_version:
            return False
        if self._intervention or (not managed and result.requested_phase != self._phase):
            print(
                "Discarded policy response after source change "
                f"id={result.request_id} requested={result.requested_phase}",
                flush=True,
            )
            return False
        returned_phase = "rl" if result.phase > 0.5 else "vla"
        if getattr(result, "control_mode", None) != "auto" and returned_phase != result.requested_phase:
            raise RuntimeError(
                f"policy phase mismatch: requested={result.requested_phase}, "
                f"returned={returned_phase}"
            )
        if getattr(result, "control_mode", None) != "auto" and result.requested_phase == "vla" and result.source_type > 0.5:
            raise RuntimeError(
                "policy returned RLT actor output for requested VLA phase"
            )
        if getattr(self, "_full_episode", False) and not result.source_type > 0.5:
            raise RuntimeError("full-episode RLT requires Actor output; refusing VLA response")
        action = clamp_gripper_action_roundoff(result.action, arm_count=1)
        if action.shape != (10,) or not np.isfinite(action).all():
            raise ValueError(
                f"policy action must be a finite 10-vector, got {action.shape}"
            )
        rot6d_to_matrix(action[3:9])
        self._latest_policy_result = result
        if managed:
            self._web_switch_pending = False
        self._latest_policy_action = action
        return True

    def _submit_policy(
        self,
        snapshot: FrankaRolloutSnapshot,
        images: dict[str, np.ndarray],
    ) -> None:
        submitted = self.policy_client.submit(
            state=snapshot.policy_state,
            images=images,
            task=self.config.task,
            phase=self._phase,
            reset_episode=self._reset_policy_on_request,
        )
        if submitted is not None:
            self._emit(
                "policy_submitted",
                request_id=submitted,
                phase=self._phase,
                reset_policy=self._reset_policy_on_request,
            )
            self._reset_policy_on_request = False

    def _policy_stream_discontinuous(self, fresh_result: bool) -> float | None:
        if fresh_result or self._latest_policy_result is None:
            return None
        age = time.monotonic() - self._latest_policy_result.received_at
        if age <= self.config.control_discontinuity_timeout_s:
            return None
        return age

    def _reprime_policy_stream(self, age: float) -> None:
        self._break_timed_segment()
        self.robot.stop()
        self.policy_client.invalidate()
        self._latest_policy_result = None
        self._latest_policy_action = None
        self._command = None
        self._priming = True
        self._reset_policy_on_request = True
        self._last_pose = None
        reason = (
            "policy action stream exceeded control discontinuity timeout: "
            f"age={age * 1000.0:.1f}ms"
        )
        print(f"{reason}; stopped Franky and requested policy reprime", flush=True)
        self._emit(
            "policy_stream_reprime_started",
            phase=self._phase,
            age_ms=age * 1000.0,
        )

    def _apply_policy(self) -> None:
        if self._latest_policy_action is None or self._latest_policy_result is None:
            return
        sent = action_vector(self.robot.send_action(self._latest_policy_action))
        self._command_sequence += 1
        source = (
            COLLECTOR_RLT_ACTOR
            if self._latest_policy_result.source_type > 0.5
            else COLLECTOR_PI05
        )
        self._command = CommandSnapshot(
            action=sent,
            source=source,
            phase=float(self._latest_policy_result.phase),
            applied_at=time.monotonic(),
            command_id=self._command_sequence,
        )
        was_priming = self._priming
        self._priming = False
        self._last_action_log += 1
        if self.log_actions_every > 0 and (
            was_priming or self._last_action_log % self.log_actions_every == 0
        ):
            debug = self.robot.command_debug_snapshot()
            console_debug(
                "Franky action applied "
                f"source={COLLECTOR_NAMES[source]} phase={self._phase} "
                f"policy_xyz={np.round(self._latest_policy_action[:3], 4).tolist()} "
                f"effective_xyz={np.round(sent[:3], 4).tolist()} "
                f"sent={debug.get('motion_command_sent')} "
                f"v={np.round(debug.get('commanded_linear_velocity', []), 4).tolist()}",
                flush=True,
            )
        self._emit(
            "action_applied",
            source=COLLECTOR_NAMES[source],
            phase=self._phase,
            policy_latency_ms=self._latest_policy_result.latency_s * 1000.0,
            policy_xyz=self._latest_policy_action[:3].tolist(),
            effective_xyz=sent[:3].tolist(),
            robot_debug=self.robot.command_debug_snapshot(),
        )

    def tick(self) -> None:
        if self._closed or self._faulted:
            return
        try:
            self.policy_client.raise_if_failed()
            try:
                snapshot = self.robot.get_rollout_snapshot()
            except TimeoutError as error:
                self._handle_observation_timeout(error)
                return
            self._consecutive_observation_timeouts = 0
            images = self._images(snapshot)
            if not self._episode_active:
                return
            state = rollout_state(snapshot, self._last_pose)
            self._last_pose = (
                snapshot.pose[:3, 3].copy(),
                snapshot.pose[:3, :3].copy(),
                snapshot.observed_at_ns,
            )
            if self.require_online:
                self._tick_timed(snapshot, state, images)
                return
            if self._intervention:
                self._record_frame(snapshot, state, images)
                return
            fresh_result = self._consume_policy_result()
            discontinuity_age = self._policy_stream_discontinuous(fresh_result)
            if discontinuity_age is not None:
                self._reprime_policy_stream(discontinuity_age)
                return
            # The snapshot belongs to the command that was active before this
            # cycle's fresh policy result is sent to Franky.
            self._record_frame(snapshot, state, images)
            self._apply_policy()
            self._submit_policy(snapshot, images)
        except (
            KeyError,
            OSError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            self._skip_reasons[f"control error: {error}"] += 1
            self._latch_fault(error)

    def _handle_observation_timeout(self, error: TimeoutError) -> None:
        if getattr(self, "_sft_interface", False):
            self.pause_control()
        self._break_timed_segment()
        self._consecutive_observation_timeouts += 1
        consecutive = self._consecutive_observation_timeouts
        self._skip_reasons[f"observation timeout: {error}"] += 1
        self._emit(
            "observation_timeout",
            reason=str(error),
            consecutive=consecutive,
        )
        if self._episode_active:
            try:
                self.robot.stop()
            except (OSError, RuntimeError, ValueError) as stop_error:
                self._latch_fault(stop_error)
                return
            self.policy_client.invalidate()
            self._latest_policy_result = None
            self._latest_policy_action = None
            self._command = None
            self._priming = True
            self._reset_policy_on_request = True
            self._last_pose = None
        if consecutive >= self._MAX_CONSECUTIVE_OBSERVATION_TIMEOUTS:
            self._latch_fault(error)
            return
        print(
            "Transient rollout observation timeout; stopped Franky and will retry "
            f"({consecutive}/{self._MAX_CONSECUTIVE_OBSERVATION_TIMEOUTS}): {error}",
            file=sys.stderr,
            flush=True,
        )

    def _break_timed_segment(self) -> None:
        if not self.require_online:
            return
        self._pending_frame = None
        self._segment_id += 1
        self._last_pose = None

    def _tick_timed(self, snapshot, state, images) -> None:
        previous = self._pending_frame
        if previous is not None:
            observed = self._annotations["complementary_info.observed_at_ns"][previous][0]
            commanded = self._annotations["complementary_info.command_at_ns"][previous][0]
            dt = (snapshot.observed_at_ns - observed) / 1e9
            if 0 <= dt < MIN_STEP_PERIODS / self.config.fps:
                # An early scheduler tick is not a missing action interval. Keep
                # the pending command and wait for a valid endpoint observation.
                return
            if not (0 < dt <= MAX_STEP_PERIODS / self.config.fps and observed <= commanded < snapshot.observed_at_ns):
                reason = f"recording interval invalid: dt={dt * 1000:.1f}ms, observed={observed}, command={commanded}, next={snapshot.observed_at_ns}"
                self._emit("recording_interval_invalid", reason=reason)
                if getattr(self, "_sft_interface", False):
                    print(f"{reason}; discarding episode and returning to standby", flush=True)
                    self.pause_control()
                else:
                    self._reprime_policy_stream(dt)
                return
            self._annotations["complementary_info.action_valid"][previous] = [1]
            self._pending_frame = None
        if self._pending_event is not None:
            event, outcome = self._pending_event
            self._pending_event = None
            if previous is not None:
                self._record_frame(snapshot, state, images)
                terminal = event in {"phase", "finish"} and self._phase == "rl"
                self._annotations["complementary_info.segment_terminal"][-1] = [int(terminal)]
                self._pending_frame = None
            self._at_boundary = True
            try:
                if event == "finish":
                    self.finish_episode(outcome)
                elif event == "phase":
                    self.toggle_phase()
                elif event == "home":
                    self.return_home()
                else:
                    self.toggle_intervention()
            finally:
                self._at_boundary = False
            return
        if self._intervention:
            self.gamepad_tick(decide=True)
        else:
            fresh = self._consume_policy_result()
            age = self._policy_stream_discontinuous(fresh)
            if age is not None:
                if previous is not None:
                    self._record_frame(snapshot, state, images)
                    self._pending_frame = None
                self._reprime_policy_stream(age)
                return
            self._apply_policy()
        if self._faulted:
            return
        self._record_frame(snapshot, state, images)
        if not self._intervention:
            self._submit_policy(snapshot, images)

    def gamepad_tick(self, *, decide: bool = False) -> None:
        setup = getattr(self, "_setup_teleop", None)
        if setup is not None and not self._episode_active:
            if not self._faulted:
                setup.tick()
                if setup.mode == "fault":
                    self._latch_fault(RuntimeError(setup.reason))
            return
        now = time.monotonic()
        minimum_dt = 1.0 / (2.0 * self.config.gamepad_control_rate)
        dt = min(max(now - self._last_gamepad_tick, minimum_dt), 0.05)
        self._last_gamepad_tick = now
        # Keep the datagram queue current while VLA/RL owns the robot. If it is
        # allowed to fill, the ROS sender blocks and the first HUMAN sample is
        # stale even though the gamepad itself is publishing at 100 Hz.
        self.joy_receiver.poll_latest()
        controls = getattr(self, "_gamepad_controls", None)
        if controls is not None:
            try:
                packet = self.joy_receiver.require_fresh(self.config.gamepad_joy_timeout_s)
                phase_pressed = controls.observe(
                    packet, active=self._episode_active and not self._faulted,
                    human=self._intervention,
                )
                if phase_pressed:
                    self.toggle_phase()
            except (TimeoutError, RuntimeError):
                controls.reset_episode()
            except (ValueError, IndexError) as error:
                self._latch_fault(error)
                return
        if self.require_online and not decide:
            return
        if self.require_online:
            dt = 1.0 / self.config.fps
        if not self._episode_active or not self._intervention or self._faulted:
            return
        try:
            limits = dict(
                damping=self.config.damping, max_joint_speed=self.config.max_joint_speed,
                joint_limit_margin=self.config.joint_limit_margin,
                position_gain=self.config.gamepad_position_gain,
                orientation_gain=self.config.gamepad_orientation_gain,
                max_linear_speed=2.0 * self.config.gamepad_xy_speed,
                max_angular_speed=self.config.gamepad_max_angular_speed,
                gripper_speed_scale=self.config.gamepad_gripper_speed,
            )
            try:
                packet = self.joy_receiver.require_fresh(
                    self.config.gamepad_joy_timeout_s
                )
            except TimeoutError as error:
                if getattr(self, "_sft_interface", False):
                    print(f"Joy input unavailable; pausing and discarding episode: {error}", flush=True)
                    self.pause_control()
                    return
                packet = None
                if not self._stale_joy_reported:
                    print(
                        f"Joy input unavailable; holding Franky target: {error}",
                        flush=True,
                    )
                    self._stale_joy_reported = True
            if packet is None:
                linear = np.zeros(3, dtype=np.float64)
                angular_tool = np.zeros(3, dtype=np.float64)
                gripper_motion = 0.0
            else:
                if self._stale_joy_reported:
                    print("Joy input restored", flush=True)
                    self._stale_joy_reported = False
                if controls is not None:
                    linear, angular_tool, gripper_motion, limits = controls.make_command(
                        packet, self.robot.get_control_state(), dt,
                    )
                else:
                    command = decode_cartesian_gamepad_values(
                        packet.axes, packet.buttons, self.arm.gamepad, self.config,
                    )
                    linear = command.linear_velocity_base
                    angular_tool = command.angular_velocity_tool
                    gripper_motion = command.gripper_motion
            if self.require_online and controls is not None:
                limits.update(anchor_to_measured=True, immediate_gripper_hold=True)
            sent = action_vector(
                (self.robot.send_human_target if self.require_online else self.robot.send_human_command)(
                    linear,
                    0.0,
                    gripper_motion,
                    dt,
                    angular_velocity_tool=angular_tool,
                    **limits,
                )
            )
            self._command = CommandSnapshot(
                action=sent,
                source=COLLECTOR_HUMAN,
                phase=float(self._phase == "rl"),
                applied_at=time.monotonic(),
                command_id=self._command_sequence + 1,
            )
            self._command_sequence += 1
            if self.require_online:
                self._priming = False
            self._emit(
                "human_action_applied",
                phase=self._phase,
                linear=linear.tolist(),
                angular_velocity_tool=angular_tool.tolist(),
                gripper_motion=gripper_motion,
                effective_xyz=sent[:3].tolist(),
                robot_debug=self.robot.command_debug_snapshot(),
            )
        except (
            KeyError,
            OSError,
            RuntimeError,
            ValueError,
            np.linalg.LinAlgError,
        ) as error:
            self._skip_reasons[f"gamepad error: {error}"] += 1
            self._latch_fault(error)

    def _preflight(self) -> FrankaRolloutSnapshot:
        self.policy_client.raise_if_failed()
        self.joy_receiver.require_fresh(self.config.gamepad_joy_timeout_s)
        snapshot = self.robot.get_rollout_snapshot()
        self._images(snapshot)
        return snapshot

    def enable_teleop(self) -> None:
        if self._faulted or self._episode_active:
            print("无法启用：请先结束回合或恢复故障；回合中可使用人工接管。", flush=True)
            return
        if self._setup_teleop.mode == "active":
            return
        try:
            self._preflight()
            self._setup_teleop.enable()
            if self._setup_teleop.mode == "active":
                controls = self._gamepad_controls
                if controls._initial_position is None:
                    controls.capture_initial_pose(self.robot.get_control_state())
                # One session origin for both unrecorded positioning and intervention.
                self._setup_teleop._initial_position = controls._initial_position.copy()
                self._setup_teleop._initial_upright_rotation = controls._initial_upright_rotation.copy()
                print("遥操作已启用，尚未录制；可开始推理回合或暂停。", flush=True)
        except (OSError, RuntimeError, ValueError) as error:
            self._latch_fault(error)

    def pause_control(self) -> None:
        if self._episode_active:
            self.finish_episode(None, discard=True)
        setup = getattr(self, "_setup_teleop", None)
        if setup is not None:
            setup.pause("operator pause; current episode discarded")
            if setup.mode == "fault":
                self._latch_fault(RuntimeError(setup.reason))

    def start_episode(self) -> None:
        if self._episode_active:
            return
        if self._faulted:
            print(
                f"Cannot start rollout: fault latched ({self._fault_reason})",
                flush=True,
            )
            return
        episode_index: int | None = None
        try:
            if getattr(self, "_sft_interface", False):
                setup = self._setup_teleop
                if setup.mode != "active":
                    print("请先启用遥操作，再开始推理与录制。", flush=True)
                    return
                if not setup._neutral(setup._packet()):
                    print("开始推理与录制前，请松开摇杆、扳机及按钮。", flush=True)
                    return
                # No motion while begin_episode reconciles data or trains pending work.
                setup.pause("starting RLT episode")
                if setup.mode == "fault":
                    raise RuntimeError(setup.reason)
            self._preflight()
            if self.require_online:
                self._episode_identity = self.policy_client.begin_episode()
            episode_index = self.writer.next_episode_index()
            self._preview_episode = episode_index
            self.writer.discard_uncommitted_files(episode_index)
            self._video_writers = {}
            for camera in self.config.cameras:
                self._video_writers[camera.feature_key] = AsyncVideoWriter(
                    self.writer.video_path(episode_index, camera.feature_key),
                    width=self.config.image_size[1],
                    height=self.config.image_size[0],
                    fps=self.config.fps,
                )
            self.robot.stop()
            self.policy_client.invalidate()
            self._phase = "rl" if getattr(self, "_full_episode", False) else "vla"
            self._priming = True
            self._intervention = False
            self._reset_policy_on_request = True
            self._latest_policy_result = None
            self._latest_policy_action = None
            self._command = None
            self._states = []
            self._actions = []
            self._annotations = {key: [] for key in self._frame_features}
            self._segment_id = 0
            self._pending_frame = None
            self._pending_event = None
            self._critical_entered = False
            if getattr(self, "_gamepad_controls", None) is not None:
                self._gamepad_controls.reset_episode()
            self._last_pose = None
            self._skip_reasons.clear()
            self._episode_active = True
            self._emit("episode_started", episode_index=episode_index, phase=self._phase)
            print(
                f"Priming rollout episode {episode_index} in {self._phase.upper()} phase; "
                "recording starts after the first Franky action",
                flush=True,
            )
        except (OSError, RuntimeError, ValueError) as error:
            self._abort_videos()
            self._video_writers = {}
            if episode_index is not None:
                try:
                    self.writer.discard_uncommitted_files(episode_index)
                except (OSError, RuntimeError, ValueError):
                    pass
            print(f"Rollout start blocked: {error}", flush=True)
            if getattr(self, "_sft_interface", False):
                self._latch_fault(error)

    @staticmethod
    def _intervals(values: list[int | float], predicate) -> list[dict[str, int]]:
        intervals = []
        start = None
        for index, value in enumerate(values + [None]):
            active = index < len(values) and predicate(value)
            if active and start is None:
                start = index
            elif not active and start is not None:
                intervals.append({"start_frame": start, "end_frame": index})
                start = None
        return intervals

    def _abort_videos(self) -> None:
        for video_writer in self._video_writers.values():
            try:
                video_writer.abort()
            except BaseException:
                pass

    def _clear_episode(self) -> None:
        self._video_writers = {}
        self._states = []
        self._actions = []
        self._annotations = {key: [] for key in self._frame_features}
        self._pending_frame = None
        self._pending_event = None
        self._last_pose = None
        self._command = None
        self._latest_policy_result = None
        self._latest_policy_action = None

    def finish_episode(self, outcome: str | None, *, discard: bool = False) -> None:
        if not self._episode_active:
            setup = getattr(self, "_setup_teleop", None)
            if setup is not None:
                setup.pause("no episode to save; returning to standby")
                if setup.mode == "fault":
                    self._latch_fault(RuntimeError(setup.reason))
            return
        if self.require_online and not discard and not self._at_boundary:
            self._pending_event = ("finish", outcome)
            return
        episode_index = self.writer.next_episode_index()
        self._processing = True
        self.update_preview()
        try:
            if getattr(self, "_sft_interface", False):
                self.robot.end_human_control()
                self._setup_teleop.pause("episode ended; saving/cleanup")
                if self._setup_teleop.mode == "fault":
                    raise RuntimeError(self._setup_teleop.reason)
            else:
                self.robot.stop()
            self.policy_client.invalidate()
            should_save = (
                not discard
                and outcome in {"success", "failure"}
                and len(self._states) >= MIN_EPISODE_FRAMES
            )
            if should_save:
                validate_rollout_episode(
                    np.asarray(self._states),
                    np.asarray(self._actions),
                    arm_count=1,
                    workspace_bounds=[
                        (
                            np.asarray(self.arm.workspace_min),
                            np.asarray(self.arm.workspace_max),
                        )
                    ],
                    maximum_gripper_width=schema.ROBOTIQ_MAX_OPENING_WIDTH_M,
                )
                for video_writer in self._video_writers.values():
                    video_writer.close()
                collector_values = [
                    int(value[0])
                    for value in self._annotations[
                        "complementary_info.collector_policy_id"
                    ]
                ]
                phase_values = [
                    float(value[0])
                    for value in self._annotations["complementary_info.phase"]
                ]
                source_counts = episode_source_frame_counts(
                    [phase if not self.require_online or self._annotations["complementary_info.action_valid"][index][0] else 0.0 for index, phase in enumerate(phase_values)], collector_values
                )
                rl_intervals = self._intervals(
                    phase_values, lambda value: float(value) > 0.5
                )
                for interval in rl_intervals:
                    interval["outcome"] = outcome
                saved = self.writer.save_episode(
                    np.asarray(self._states),
                    np.asarray(self._actions),
                    {
                        key: video_writer.image_stats.as_dict()
                        for key, video_writer in self._video_writers.items()
                    },
                    frame_features={
                        key: np.asarray(values)
                        for key, values in self._annotations.items()
                    },
                    extra_episode_metadata={
                        **self._episode_identity,
                        **({"human_control_profile": self._gamepad_controls.profile_metadata}
                           if getattr(self, "_gamepad_controls", None) is not None else {}),
                        "episode_success": outcome,
                        "rl_intervals": rl_intervals,
                        "human_intervention_intervals": self._intervals(
                            collector_values,
                            lambda value: int(value) == COLLECTOR_HUMAN,
                        ),
                    },
                )
                print(
                    f"Saved rollout episode {saved}: outcome={outcome}, "
                    f"frames={len(self._states)}, "
                    f"critical+Actor={source_counts['critical_actor']}, "
                    f"critical+Human={source_counts['critical_human']}, "
                    f"critical+VLA={source_counts['critical_vla']}",
                    flush=True,
                )
                if self.require_online:
                    metrics = self.policy_client.process_saved_episode(
                        dataset_root=self.config.dataset_root,
                        episode_id=saved,
                        outcome=str(outcome),
                    )
                    print(
                        "Online RLT episode processed: "
                        f"warmup={metrics['warmup_env_steps']}/"
                        f"{metrics['warmup_steps']} "
                        f"remaining={metrics['warmup_remaining']} "
                        f"transitions={metrics['transitions']}",
                        flush=True,
                    )
                self._emit(
                    "episode_saved",
                    episode_index=saved,
                    outcome=outcome,
                    frames=len(self._states),
                    **source_counts,
                )
            else:
                self._abort_videos()
                self.writer.discard_uncommitted_files(episode_index)
                print(f"Discarded rollout episode {episode_index}", flush=True)
                self._emit("episode_discarded", episode_index=episode_index)
        except (OSError, RuntimeError, ValueError) as error:
            self._latch_fault(error)
            return
        finally:
            self._processing = False
            self._episode_active = False
            self._priming = False
            self._intervention = False
            self._reset_policy_on_request = True
            self._clear_episode()
            self.update_preview()
            if getattr(self, "_sft_interface", False) and not self._faulted:
                print("已返回待命；重新启用遥操作后可开始推理与录制。", flush=True)

    def toggle_intervention(self) -> None:
        if not self._episode_active:
            print("Ignoring gamepad intervention outside an active episode", flush=True)
            return
        if self._faulted:
            return
        if self.require_online and not self._at_boundary:
            if self._pending_event is not None:
                print("Ignoring intervention: a boundary event is already pending", flush=True)
                return
            self._pending_event = ("intervention", None)
            return
        self._break_timed_segment()
        try:
            if not self._intervention:
                self.joy_receiver.require_fresh(self.config.gamepad_joy_timeout_s)
                if getattr(self, "_sft_interface", False):
                    if not self._gamepad_controls._neutral(self.joy_receiver.require_fresh(self.config.gamepad_joy_timeout_s)):
                        print("请松开手柄输入，然后重新请求人工接管。", flush=True)
                        return
                self.policy_client.invalidate()
                self._latest_policy_result = None
                self._latest_policy_action = None
                self._command = None
                self.robot.begin_human_control()
                if getattr(self, "_gamepad_controls", None) is not None:
                    self._gamepad_controls.reset_human()
                self._intervention = True
                self._priming = False
                self._last_gamepad_tick = time.monotonic()
                self._stale_joy_reported = False
                print("Applied control source=HUMAN", flush=True)
                self._emit("source_changed", source="HUMAN", phase=self._phase)
            else:
                if getattr(self, "_sft_interface", False):
                    if not self._gamepad_controls._neutral(self.joy_receiver.require_fresh(self.config.gamepad_joy_timeout_s)):
                        print("请松开手柄输入，然后重新请求归还策略控制。", flush=True)
                        return
                self.robot.end_human_control()
                if getattr(self, "_gamepad_controls", None) is not None:
                    self._gamepad_controls.reset_human()
                self._intervention = False
                self._priming = True
                self._command = None
                self._latest_policy_result = None
                self._latest_policy_action = None
                self._reset_policy_on_request = True
                self.policy_client.invalidate()
                print(
                    f"Requested return to policy phase={self._phase}; reprime pending",
                    flush=True,
                )
                self._emit("source_reprime_started", phase=self._phase)
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Cannot switch intervention: {error}", flush=True)

    def return_home(self) -> None:
        """Return to the configured mean demonstration start pose."""
        home = getattr(self, "home_config", None)
        if home is None:
            print("Dataset home return is not configured", flush=True)
            return
        if self._faulted:
            return
        controls = getattr(self, "_gamepad_controls", None)
        if controls is None:
            print("Dataset home return requires the single-left gamepad profile", flush=True)
            return
        if not self._episode_active:
            setup = getattr(self, "_setup_teleop", None)
            if setup is None:
                return
            if setup.mode != "active":
                self.enable_teleop()
            if setup.mode != "active":
                print("Cannot return home until teleop is safely enabled", flush=True)
                return
            self._request_dataset_home(setup, home)
            return
        if self.require_online and not self._at_boundary:
            if self._pending_event is not None:
                print("Ignoring home return: a boundary event is already pending", flush=True)
                return
            self._pending_event = ("home", None)
            print("Home return queued until the current online action interval ends", flush=True)
            return
        if not self._intervention:
            self.toggle_intervention()
        if self._intervention:
            self._request_dataset_home(controls, home)

    def _request_dataset_home(self, controls, home) -> None:
        from vla.datacollection.start_teleop.gamepad_teleop import axis_aligned_upright_rotation

        try:
            target = np.asarray(home.target_action, dtype=np.float64)
            rotation = axis_aligned_upright_rotation(rot6d_to_matrix(target[3:9]))
            controls.request_dataset_home_return(
                target[:3], rotation, home.clearance_z, home.position_tolerance_m,
            )
            self._last_gamepad_tick = time.monotonic()
            self._emit("home_return_requested", phase=self._phase, target_xyz=target[:3].tolist())
        except (OSError, RuntimeError, ValueError) as error:
            print(f"Cannot return home: {error}", flush=True)

    def toggle_phase(self) -> None:
        if getattr(self, "_sft_only", False):
            print("SFT 部署可切换人工／策略控制。", flush=True)
            return
        if getattr(self, "_full_episode", False):
            print("完整回合 RLT 不支持阶段切换，可使用人工接管。", flush=True)
            return
        if not self._episode_active:
            print("Ignoring phase toggle outside an active episode", flush=True)
            return
        if self._intervention and getattr(self, "_gamepad_controls", None) is None:
            print(
                "Ignoring phase toggle while gamepad intervention is active", flush=True
            )
            return
        if self.require_online and self._phase == "vla" and self._critical_entered:
            print("Critical phase already ended; start a new episode to enter again", flush=True)
            return
        if self.require_online and not self._at_boundary:
            if self._pending_event is not None:
                print("Ignoring phase toggle: a boundary event is already pending", flush=True)
                return
            self._pending_event = ("phase", None)
            return
        try:
            self._break_timed_segment()
            if self._intervention:
                self._phase = "vla" if self._phase == "rl" else "rl"
                self._critical_entered |= self._phase == "rl"
                print(f"Applied HUMAN critical={self._phase == 'rl'}; control remains HUMAN", flush=True)
                self._emit("human_phase_changed", phase=self._phase)
                return
            self.robot.stop()
            self._phase = "vla" if self._phase == "rl" else "rl"
            self._critical_entered |= self._phase == "rl"
            self._priming = True
            self._command = None
            self._latest_policy_result = None
            self._latest_policy_action = None
            self.policy_client.invalidate()
            print(f"Requested policy phase={self._phase}; reprime pending", flush=True)
            self._emit("phase_reprime_started", phase=self._phase)
        except (OSError, RuntimeError, ValueError) as error:
            self._latch_fault(error)

    def discard_latest(self) -> None:
        if self._faulted:
            print(
                "故障锁定时不能丢弃回合；请先执行故障恢复。", flush=True
            )
            return
        if self._episode_active:
            self.finish_episode(None, discard=True)
            return
        if self.require_online:
            if getattr(self, "_sft_interface", False):
                self.pause_control()
            print(
                "Online RLT keeps saved episodes immutable; only the active episode can be discarded",
                flush=True,
            )
            return
        discarded = self.writer.discard_latest_episode()
        if discarded is not None:
            print(f"Deleted latest saved episode {discarded}", flush=True)

    def recover_fault(self) -> bool:
        if not self._faulted:
            if not self.require_online:
                print("No rollout fault is latched", flush=True)
                return True
            if self._episode_active:
                print("Cannot recover online training during an active episode", flush=True)
                return False
            print("Verifying and recovering online training state", flush=True)
        previous_reason = self._fault_reason
        self.update_preview(mode="RECOVERING")
        try:
            if getattr(self, "_sft_interface", False):
                self._setup_teleop.pause("manual recovery")
            if getattr(self, "_sft_interface", False):
                try:
                    self.robot.check_health()
                except (OSError, RuntimeError):
                    self.robot.recover_motion_fault()
            self.robot.end_human_control()
            if self.require_online:
                self.policy_client.recover_online()
            self.policy_client.raise_if_failed()
            self.robot.get_rollout_snapshot()
        except (OSError, RuntimeError, ValueError) as error:
            if getattr(self, "_sft_interface", False):
                self._faulted = True
            self._fault_reason = f"{type(error).__name__}: {error}"
            self._emit(
                "collector_fault_recovery_failed",
                previous_reason=previous_reason,
                reason=self._fault_reason,
            )
            print(
                "Fault recovery failed; Franky commands remain disabled: "
                f"{self._fault_reason}",
                file=sys.stderr,
                flush=True,
            )
            self.update_preview(mode="FAULT")
            return False
        self.policy_client.invalidate()
        self._faulted = False
        self._fault_reason = None
        self._episode_active = False
        self._priming = False
        self._intervention = False
        self._reset_policy_on_request = True
        self._consecutive_observation_timeouts = 0
        self._stale_joy_reported = False
        self._last_gamepad_tick = time.monotonic()
        self._clear_episode()
        if getattr(self, "_sft_interface", False):
            setup = self._setup_teleop
            setup.mode, setup.reason = "standby", ""
            setup._state_time = None
            setup._state_advanced = False
            setup._paused_at = time.monotonic()
            setup._reset_speed()
        self._emit("collector_fault_recovered", previous_reason=previous_reason)
        self.update_preview()
        print(
            ("故障已清除，当前待命；启用遥操作后可开始推理与录制。"
             if getattr(self, "_sft_interface", False) else
             "故障已清除，当前无活动回合；确认现场状态后可开始新回合。"),
            flush=True,
        )
        return True

    def _latch_fault(self, error: BaseException) -> None:
        if self._faulted:
            return
        self._faulted = True
        self._fault_reason = f"{type(error).__name__}: {error}"
        try:
            self.robot.stop()
        except BaseException:
            pass
        self.policy_client.invalidate()
        self._abort_videos()
        if self._episode_active:
            try:
                self.writer.discard_uncommitted_files(self.writer.next_episode_index())
            except BaseException:
                pass
        self._episode_active = False
        self._priming = False
        self._intervention = False
        setup = getattr(self, "_setup_teleop", None)
        if setup is not None:
            setup.pause(self._fault_reason, fault=True)
        self._emit(
            "collector_fault_latched",
            reason=self._fault_reason,
            traceback=traceback.format_exc(),
        )
        self._clear_episode()
        self.update_preview()
        print(
            "\nROLLOUT FAULT LATCHED; Franky commands disabled. "
            "请排查健康状态并执行故障恢复： "
            f"{self._fault_reason}\n{traceback.format_exc()}",
            file=sys.stderr,
            flush=True,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._episode_active:
                self.finish_episode(None, discard=True)
            else:
                self.robot.stop()
        finally:
            try:
                if self._owns_policy_client:
                    self.policy_client.close()
            finally:
                try:
                    if self._owns_joy_receiver:
                        self.joy_receiver.close()
                finally:
                    self.robot.disconnect()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--no-preview", action="store_true", help="Disable the optional live camera window")
    parser.add_argument("--online", "--online-warmup", dest="online_warmup", action="store_true",
                        help="Require online server mode; warmup/full-episode behavior comes from the server contract")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--debug-log", type=Path)
    parser.add_argument(
        "--log-actions-every",
        type=int,
        default=0,
        help="Record every Nth applied policy action in diagnostic logs; 0 disables action-value logs.",
    )
    return parser.parse_args()


def _load_deployment(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"deployment config must be a mapping: {path}")
    return value


def _parse_cpu_list(value: str) -> set[int]:
    cpus: set[int] = set()
    for item in value.split(","):
        bounds = item.strip().split("-", 1)
        start, end = int(bounds[0]), int(bounds[-1])
        if start < 0 or end < start:
            raise ValueError(f"invalid CPU range: {item}")
        cpus.update(range(start, end + 1))
    return cpus


def _configure_affinity(deployment: dict[str, Any], realtime_cpu: int | None) -> None:
    value = deployment.get("housekeeping_cpus")
    if value is None:
        return
    cpus = _parse_cpu_list(str(value))
    online = set(range(os.cpu_count() or 1))
    if not cpus or not cpus <= online:
        raise ValueError(f"housekeeping CPUs must be within {sorted(online)}")
    if realtime_cpu is not None and realtime_cpu in cpus:
        raise ValueError("Franky realtime CPU must not be a housekeeping CPU")
    os.sched_setaffinity(0, cpus)


def _confirm_execute(execute: bool, *, sft: bool = False) -> None:
    if not execute:
        raise PermissionError("Franky rollout requires --execute")
    if not sys.stdin.isatty():
        raise PermissionError("--execute requires an interactive terminal")
    if sft:
        print("Connecting in STANDBY; no home/open target. Clear workspace and keep emergency stop accessible. s=enable teleop", flush=True)
        return
    print("已接受 --execute；连接后按 Right 开始回合。", flush=True)


def next_control_deadline(started: float, finished: float, period: float) -> float:
    """Do not squeeze a missed cycle into a short catch-up interval."""
    deadline = started + period
    return deadline if finished < deadline else finished + period


def dispatch_event(collector, event: str) -> bool:
    if event == "quit":
        return True
    handlers = {
        "enable": collector.enable_teleop, "right": collector.start_episode,
        "left": collector.discard_latest, "pause": collector.pause_control,
        "intervention": collector.toggle_intervention, "phase": collector.toggle_phase,
        "home": collector.return_home,
        "recover": collector.recover_fault,
        "success": lambda: collector.finish_episode("success"),
        "failure": lambda: collector.finish_episode("failure"),
    }
    handlers[event]()
    return False


@terminal_session("franky-rollout")
def main() -> None:
    args = parse_args()
    config = load_rollout_config(args.config)
    if args.dataset_root is not None:
        config = replace(
            config,
            dataset_root=args.dataset_root.expanduser().resolve(),
        )
    if args.online_warmup and args.dataset_root is None:
        raise ValueError("--online-warmup requires an explicit new --dataset-root")
    if config.backend != "franky":
        raise ValueError("collect_franky requires a config with backend: franky")
    if config.rtc_deployment_config is None:
        raise ValueError("Franky rollout requires control.rtc_deployment_config")
    deployment_values = _load_deployment(config.rtc_deployment_config)
    deployment = deployment_values.get("deployment", {})
    if deployment.get("task") != config.task:
        raise ValueError(
            "rollout task must exactly match RTC deployment task: "
            f"{config.task!r} != {deployment.get('task')!r}"
        )
    robot_config = load_franka_robotiq_config(config.rtc_deployment_config)
    from vla.cli.deploy_franka_pi05_rtc import _load_home_config
    home_config = _load_home_config(deployment, robot_config)
    if tuple(config.arms[0].workspace_min) != tuple(
        robot_config.workspace_min
    ) or tuple(config.arms[0].workspace_max) != tuple(robot_config.workspace_max):
        raise ValueError(
            "Franky rollout workspace must exactly match RTC deployment workspace"
        )
    sft_interface = config.gamepad_profile == "sft_record"
    if sft_interface:
        # These switches affect HUMAN control only; Actor deployment limits stay unchanged.
        robot_config = replace(robot_config, human_linear_velocity_control=True,
                               human_angular_velocity_control=True, human_gripper_immediate_hold=True)
    _confirm_execute(args.execute, sft=sft_interface)
    _configure_affinity(deployment, robot_config.realtime_cpu)
    debug_path = args.debug_log or diagnostic_path("rollout.jsonl", f"logs/franka_rollout_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    trace = JsonlDebugTrace(debug_path)
    robot = FrankaRobotiqRobot(robot_config)
    collector: FrankyRolloutCollector | None = None
    keyboard: KeyboardEvents | None = None
    preview = preview_runtime = None
    try:
        # Validate the policy and dataset contracts before touching hardware.
        collector = FrankyRolloutCollector(
            config,
            robot,
            trace=trace,
            log_actions_every=args.log_actions_every,
            require_online=args.online_warmup,
        )
        collector.home_config = home_config
        robot.connect()
        opened = None if sft_interface else robot.prepare_for_control()
        if collector._gamepad_controls is not None:
            if home_config is not None:
                target = np.asarray(home_config.target_action, dtype=np.float64)
                collector._gamepad_controls.configure_dataset_home_position(target[:3], rot6d_to_matrix(target[3:9]))
            collector._gamepad_controls.capture_initial_pose(robot.get_control_state())
            rb_label = "disabled" if collector._full_episode else "critical"
            print(f"SFT gamepad profile: LB=XYZ 1x/5x, RB={rb_label}; HUMAN only: Menu=dataset-home XYZ, View=initial pose", flush=True)
        if collector._full_episode:
            print("FULL-EPISODE RLT: Actor from first action, all valid intervals trainable; phase switching disabled", flush=True)
        print("Franky connected in STANDBY; gripper unchanged" if sft_interface else
              f"Franky connected; Robotiq open_fraction={opened:.3f}", flush=True)
        if not args.no_preview:
            from vla.datacollection.start_teleop.gamepad_preview import CameraPreview
            try:
                preview_runtime = tempfile.TemporaryDirectory(prefix="vla-online-preview-")
                preview = CameraPreview(preview_runtime.name, robot.get_camera_preview_snapshot)
                collector.preview = preview
                collector.update_preview()
                preview.start()
            except (OSError, RuntimeError, ValueError) as error:
                print(f"Camera preview unavailable: {error}; collector remains available", flush=True)
        confirmation = TerminalConfirmation()
        keyboard = KeyboardEvents(sft=sft_interface, confirmation=confirmation)
        print(
            ("STANDBY: s=teleop, n=RLT inference+record, Space=human/RLT, "
             "y=success/save, f=failure/save, x=discard (confirm), d=pause+discard immediately, h=demo mean home, r=recover, c/q/Esc=exit"
             if sft_interface else
             "Franky rollout ready; Right=start, r=phase, Space=human, s/f=outcome, Left=discard-active, c=recover, q=quit"),
            flush=True,
        )
        control_period = 1.0 / config.fps
        gamepad_period = 1.0 / config.gamepad_control_rate
        next_control = next_gamepad = time.monotonic()
        next_communication_sample = next_control
        stop = False
        while not stop:
            confirmation.expire()
            for event in confirmation.prioritize(keyboard.consume()):
                if event == "force_quit":
                    stop = True
                    break
                active = collector.episode_active is True
                index = collector.writer.next_episode_index()
                destructive = event == "quit" and active or event == "left" and (active or not collector.require_online and index > 0)
                event = confirmation.route(event, destructive=destructive,
                    valid=lambda recording=active, episode=index: collector.episode_active is recording and collector.writer.next_episode_index() == episode and not collector.faulted,
                    label="退出并丢弃当前未保存回合" if event == "quit" else "丢弃当前或最近已保存回合")
                if event is None:
                    continue
                stop = dispatch_event(collector, event)
                if stop or sft_interface:
                    break
            if stop:
                break
            collector.update_preview()
            now = time.monotonic()
            if now >= next_gamepad:
                collector.gamepad_tick()
                next_gamepad = next_control_deadline(now, time.monotonic(), gamepad_period)
            now = time.monotonic()
            if now >= next_control:
                was_active = collector.episode_active
                collector.tick()
                next_control = next_control_deadline(now, time.monotonic(), control_period)
                if sft_interface and was_active and not collector.episode_active:
                    # Discard keystrokes entered while save/train was blocking.
                    keyboard.consume()
            collector.update_preview()
            if time.monotonic() >= next_communication_sample:
                console_status(f"{ 'FAULT' if collector.faulted else 'RECORDING' if collector.episode_active else 'STANDBY' } episode={collector.writer.next_episode_index()} frames={len(collector._states)} source={'HUMAN' if collector._intervention else collector._phase.upper()}")
                collector._emit(
                    "communication_sample", episode_active=collector.episode_active,
                    **robot.communication_debug_snapshot(),
                )
                next_communication_sample = time.monotonic() + 1.0
            time.sleep(max(0.0, min(next_gamepad, next_control) - time.monotonic()))
    finally:
        try:
            if keyboard is not None:
                keyboard.close()
        finally:
            try:
                try:
                    if preview is not None:
                        preview.close()
                finally:
                    if collector is not None:
                        collector.close()
                    else:
                        robot.disconnect()
            finally:
                try:
                    if preview_runtime is not None:
                        preview_runtime.cleanup()
                finally:
                    trace.close()


if __name__ == "__main__":
    main()
