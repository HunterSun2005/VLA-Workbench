"""Serve Franka RLT actions to the ROS2 rollout collector over a Unix socket."""

from __future__ import annotations

import argparse
import hashlib
import os
import pathlib
import signal
import socket
import time
from types import SimpleNamespace
from typing import Any, Callable

import numpy as np
import yaml

from vla_rollout_protocol import recv_packet, send_packet


class _TrainingProgressSender:
    def __init__(self, connection: socket.socket, request_id: int) -> None:
        self.connection = connection
        self.request_id = int(request_id)
        self.available = True

    def __call__(self, progress: dict[str, float | int | str | None]) -> None:
        if not self.available:
            return
        try:
            send_packet(
                self.connection,
                {
                    "kind": "training_progress",
                    "request_id": self.request_id,
                    **progress,
                },
            )
        except (BrokenPipeError, ConnectionError, OSError):
            # Collector display failure must not interrupt training/checkpointing.
            self.available = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-path", required=True)
    parser.add_argument("--socket-path", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--task", required=True, help="default task prompt")
    parser.add_argument("--tasks", nargs="+", help="all task prompts accepted by this server")
    parser.add_argument("--robot-type", default="franka_robotiq")
    parser.add_argument(
        "--tokenizer-path",
        default=None,
        help="PaliGemma tokenizer repo id or local snapshot path; use a local path offline.",
    )
    parser.add_argument(
        "--rtc-deployment-config",
        type=pathlib.Path,
        default=None,
        help=(
            "RTC deployment YAML for the shared ROS collector. Used by the "
            "ROS rollout path."
        ),
    )
    parser.add_argument("--socket-timeout-s", type=float, default=30.0)
    parser.add_argument("--online-replay-root", type=pathlib.Path)
    parser.add_argument("--online-dataset-root", type=pathlib.Path)
    parser.add_argument("--online-dataset-repo-id", default=None)
    parser.add_argument("--online-warmup-steps", type=int, default=1000)
    parser.add_argument("--online-batch-size", type=int, default=32)
    parser.add_argument(
        "--online-encoding-batch-size", type=int, default=16,
        help="Batch size for per-episode PI0.5/RL Token encoding before replay commit.",
    )
    parser.add_argument(
        "--online-video-backend", choices=("torchcodec", "pyav", "video_reader"),
        default="torchcodec", help="Video decoder used while encoding online episodes.",
    )
    parser.add_argument(
        "--online-encoding-num-workers", type=int, default=0,
        help="DataLoader workers used while encoding online episodes.",
    )
    parser.add_argument("--online-replay-capacity", type=int, default=100000)
    parser.add_argument("--online-timeout-s", type=float, default=3600.0)
    parser.add_argument("--online-resume", action="store_true")
    parser.add_argument(
        "--online-inherit-actor-critic", action="store_true",
        help="Initialize a new online run from --policy-path A/C weights; keep this flag on resume. Warmup applies unless --online-full-episode is set.",
    )
    parser.add_argument(
        "--online-full-episode", action="store_true",
        help="Use inherited Actor from the first action and train the whole episode; disables VLA warmup and manual phase switching.",
    )
    parser.add_argument(
        "--log-inference-every",
        type=int,
        default=0,
        help="Print every Nth inference result; 0 disables per-inference logs.",
    )
    args = parser.parse_args()
    prompts = tuple(dict.fromkeys(item.strip() for item in (args.tasks or [args.task]) if item.strip()))
    if not prompts or args.task.strip() not in prompts:
        parser.error("--tasks must contain the default --task")
    args.tasks = prompts
    if args.online_full_episode and (not args.online_inherit_actor_critic or args.online_replay_root is None):
        parser.error("--online-full-episode requires --online-inherit-actor-critic and online mode")
    return args


def _checkpoint_fingerprint(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    files = [
        path / name
        for name in (
            "config.json",
            "model.safetensors",
            "policy_preprocessor.json",
            "policy_postprocessor.json",
        )
        if (path / name).is_file()
    ]
    if not files:
        raise FileNotFoundError(f"no checkpoint files found in {path}")
    for file_path in files:
        digest.update(file_path.name.encode("utf-8"))
        with file_path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _load_rollout_rtc_profile(
    path: pathlib.Path,
) -> tuple[pathlib.Path, Any, float, int, str | None]:
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    profile_path = path.expanduser().resolve()
    values = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError(
            f"RTC deployment config must be a YAML mapping: {profile_path}"
        )
    rtc_values = values.get("rtc")
    deployment = values.get("dataset") if values.get("backend") == "ros2" else values.get("deployment")
    if not isinstance(rtc_values, dict) or not isinstance(deployment, dict):
        raise ValueError(
            "RTC deployment config must contain rtc and deployment mappings"
        )

    execution_horizon = int(rtc_values.get("execution_horizon", 10))
    queue_threshold = int(rtc_values.get("queue_threshold", 30))
    fps = float(deployment.get("fps", 30.0))
    if execution_horizon <= 0:
        raise ValueError("RTC execution_horizon must be positive")
    if queue_threshold < 0:
        raise ValueError("RTC queue_threshold must be non-negative")
    if fps <= 0:
        raise ValueError("RTC deployment fps must be positive")
    rtc_config = RTCConfig(
        enabled=True,
        execution_horizon=execution_horizon,
        max_guidance_weight=float(rtc_values.get("max_guidance_weight", 10.0)),
        prefix_attention_schedule=RTCAttentionSchedule(
            rtc_values.get("prefix_attention_schedule", "EXP")
        ),
    )
    task = deployment.get("task")
    return (
        profile_path,
        rtc_config,
        fps,
        queue_threshold,
        None if task is None else str(task),
    )


class FrankaRolloutPolicyServer:
    def __init__(self, args: argparse.Namespace) -> None:
        from vla.adapters.lerobot import register

        register()
        from vla.adapters.lerobot.policies.configuration_rlt_ac import (
            ChunkACConfig,
        )
        from vla.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
        from vla.adapters.lerobot.policies.processor_rlt_ac import (
            make_rlt_ac_pre_post_processors,
        )
        from vla.adapters.lerobot.inference import predict_action
        from lerobot.utils.device_utils import get_safe_torch_device

        self._predict_action = predict_action
        if args.log_inference_every < 0:
            raise ValueError("log-inference-every must be non-negative")
        self.log_inference_every = int(args.log_inference_every)
        self.policy_path = pathlib.Path(args.policy_path).expanduser().resolve()
        self.fingerprint = _checkpoint_fingerprint(self.policy_path)
        config = ChunkACConfig.from_pretrained(self.policy_path)
        config.device = args.device
        if getattr(args, "handover_path", None):
            config.handover_pretrained_path = args.handover_path
            config.phase_mode = "learned"
        if args.tokenizer_path is not None:
            config.tokenizer_path = args.tokenizer_path
        self.policy = (
            ChunkACPolicy.from_pretrained(self.policy_path, config=config)
            .to(args.device)
            .eval()
        )
        self.policy.prepare_for_inference()
        from vla.actions.metadata import checkpoint_contract
        from vla.adapters.lerobot.action_runtime import ActionRuntime
        self.action_contract = checkpoint_contract(config.vla_pretrained_path)
        self.action_stride = self.action_contract["statistics"]["contract"]["sampling"]["frame_stride"]
        self.policy.action_runtime = ActionRuntime(self.action_contract)
        self.preprocessor, self.postprocessor = make_rlt_ac_pre_post_processors(config)
        self.device = get_safe_torch_device(args.device)
        self.task = str(args.task)
        if not self.task.strip():
            raise ValueError("--task must not be empty")
        self.tasks = tuple(dict.fromkeys(str(item).strip() for item in getattr(args, "tasks", (self.task,)) if str(item).strip()))
        if self.task not in self.tasks:
            raise ValueError("--tasks must contain --task")
        self.robot_type = args.robot_type
        self.action_dim = int(config.action_dim)
        self.state_dim = int(config.proprio_dim)
        self.camera_keys = [f"observation.images.{key}" for key in config.camera_keys]
        contract = config.cache_contract
        if not isinstance(contract, dict):
            raise ValueError(
                "rollout checkpoint must contain an embedded cache_contract"
            )
        dataset_contract = contract.get("dataset", {})
        policy_contract = contract.get("policy", {})
        self.state_names = list(dataset_contract.get("state_names", []))
        self.action_names = list(dataset_contract.get("action_names", []))
        if (
            len(self.state_names) != self.state_dim
            or len(self.action_names) != self.action_dim
        ):
            raise ValueError(
                "checkpoint cache contract has invalid state/action feature names"
            )
        self.fps = float(dataset_contract.get("fps", 0.0))
        if self.fps <= 0:
            raise ValueError("checkpoint cache contract has invalid fps")
        self.rtc_enabled = False
        self.rtc_profile_path: pathlib.Path | None = None
        self.rtc_execution_horizon: int | None = None
        self.rtc_queue_threshold: int | None = None
        rtc_profile = getattr(args, "rtc_deployment_config", None)
        if rtc_profile is not None:
            (
                self.rtc_profile_path,
                rtc_config,
                rtc_fps,
                self.rtc_queue_threshold,
                rtc_task,
            ) = _load_rollout_rtc_profile(pathlib.Path(rtc_profile))
            if not np.isclose(rtc_fps, self.fps):
                raise ValueError(
                    f"RTC deployment fps does not match policy cache contract: {rtc_fps} != {self.fps}"
                )
            if rtc_task is not None and rtc_task != self.task:
                raise ValueError(
                    f"RTC deployment task does not match --task: {rtc_task!r} != {self.task!r}"
                )
            # Only PI05 guidance configuration; execution queues belong to ROS.
            self.policy._rtc_config = rtc_config
            self.rtc_enabled = True
            self.rtc_execution_horizon = int(rtc_config.execution_horizon)
        self.policy.reset()
        from vla.control.plan_service import PlanGate
        self.plan_gate = PlanGate()
        self.actions_stored_as = str(dataset_contract.get("actions_stored_as", ""))
        self.use_relative_actions = bool(
            policy_contract.get("use_relative_actions", True)
        )
        self.camera_shapes_hwc = {}
        for key in self.camera_keys:
            feature = config.input_features.get(key)
            shape = tuple(int(value) for value in getattr(feature, "shape", ()))
            if len(shape) != 3 or shape[0] != 3:
                raise ValueError(
                    f"checkpoint camera feature {key!r} is not CHW RGB: {shape}"
                )
            self.camera_shapes_hwc[key] = [shape[1], shape[2], shape[0]]
        self.use_amp = bool(config.use_amp)
        self.phase = "rl" if config.phase_mode == "always_rl" else "vla"
        self.phase_mode = str(config.phase_mode)
        self.control_mode = None
        self.mode_version = -1
        self.control_modes = ["vla", "rlt"] + (["auto"] if config.phase_mode == "learned" else [])
        self._infer_count = 0
        self.online_session = None
        self.online_timeout_s = float(getattr(args, "online_timeout_s", 3600.0))
        replay_root = getattr(args, "online_replay_root", None)
        dataset_root = getattr(args, "online_dataset_root", None)
        dataset_repo_id = getattr(args, "online_dataset_repo_id", None)
        online_values = (replay_root, dataset_root, dataset_repo_id)
        if getattr(args, "online_inherit_actor_critic", False) and replay_root is None:
            raise ValueError("--online-inherit-actor-critic requires online mode")
        if any(value is not None for value in online_values) and not all(
            value is not None for value in online_values
        ):
            raise ValueError(
                "online RLT requires --online-replay-root, --online-dataset-root, "
                "and --online-dataset-repo-id together"
            )
        if replay_root is not None:
            from vla.core.online_training import (
                OnlineRLTSession,
                OnlineRLTTrainingConfig,
            )

            if self.online_timeout_s <= 0:
                raise ValueError("online-timeout-s must be positive")
            self.online_dataset_root = pathlib.Path(dataset_root).expanduser().resolve()
            self.online_replay_root = pathlib.Path(replay_root).expanduser().resolve()
            resume = bool(getattr(args, "online_resume", False))
            if (
                not resume
                and self.online_dataset_root.exists()
                and any(self.online_dataset_root.iterdir())
            ):
                raise FileExistsError(
                    "new online run requires an empty dataset root: "
                    f"{self.online_dataset_root}"
                )
            online_config = OnlineRLTTrainingConfig(
                enable=True,
                replay_root=str(self.online_replay_root),
                replay_capacity=int(args.online_replay_capacity),
                warmup_steps=0 if getattr(args, "online_full_episode", False) else int(args.online_warmup_steps),
                batch_size=int(args.online_batch_size),
                encoding_batch_size=int(args.online_encoding_batch_size),
                encoding_num_workers=int(args.online_encoding_num_workers),
                video_backend=str(args.online_video_backend),
                inherit_actor_critic=bool(getattr(args, "online_inherit_actor_critic", False)),
                full_episode=bool(getattr(args, "online_full_episode", False)),
            )
            dataset = SimpleNamespace(
                repo_id=str(dataset_repo_id),
                root=self.online_dataset_root,
            )
            self.online_session = OnlineRLTSession(
                policy=self.policy,
                dataset=dataset,
                fps=self.fps,
                task_instruction=self.task,
                config=online_config,
                resume=resume,
                state_feature_names=tuple(self.state_names),
                runtime_contract={
                    "checkpoint_fingerprint": self.fingerprint,
                    "state_names": self.state_names,
                    "action_names": self.action_names,
                    "camera_shapes_hwc": self.camera_shapes_hwc,
                    "robot_type": self.robot_type,
                    "actions_stored_as": self.actions_stored_as,
                    "use_relative_actions": self.use_relative_actions,
                    "action_spec": self.action_contract["action_spec"],
                    "wire_action_representation": "absolute",
                    "rtc": None if self.rtc_profile_path is None else yaml.safe_load(self.rtc_profile_path.read_text()),
                    "tokenizer_sha256": self._tokenizer_digest(config.tokenizer_path),
                },
            )
            if online_config.full_episode:
                self._set_phase("rl", force=True)

    @staticmethod
    def _tokenizer_digest(path):
        from vla.core.online_contract import model_digest
        return model_digest(path) if path else None

    @property
    def contract(self) -> dict[str, Any]:
        contract = {
            "checkpoint_fingerprint": self.fingerprint,
            "plan_horizons": {"vla": self.policy.config.chunk_exec_steps, "rlt": self.policy.config.chunk_length},
            "frame_stride": self.action_stride,
            "action_contract": self.action_contract,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "camera_keys": self.camera_keys,
            "camera_shapes_hwc": self.camera_shapes_hwc,
            "robot_type": self.robot_type,
            "task": self.task,
            "tasks": list(self.tasks),
            "fps": self.fps,
            "state_names": self.state_names,
            "action_names": self.action_names,
            "actions_stored_as": self.actions_stored_as,
            "use_relative_actions": self.use_relative_actions,
                    "action_spec": self.action_contract["action_spec"],
                    "wire_action_representation": "absolute",
            "phase_mode": self.phase_mode,
            "control_modes": getattr(self, "control_modes", []),
            "control_mode_protocol": 1,
            "rtc_enabled": self.rtc_enabled,
            "rtc_profile_path": (
                None if self.rtc_profile_path is None else str(self.rtc_profile_path)
            ),
            "rtc_execution_horizon": self.rtc_execution_horizon,
            "rtc_queue_threshold": self.rtc_queue_threshold,
        }
        if getattr(self, "online_session", None) is None:
            contract["online_enabled"] = False
        else:
            contract.update(
                online_enabled=True,
                online_full_episode=self.online_session.config.full_episode,
                online_dataset_root=str(self.online_dataset_root),
                online_replay_root=str(self.online_replay_root),
                online_warmup_steps=self.online_session.config.warmup_steps,
                online_warmup_env_steps=self.online_session.replay.warmup_env_steps,
                online_timeout_s=self.online_timeout_s,
            )
            if hasattr(self.online_session.replay, "trainable_env_steps"):
                contract.update(
                    online_trainable_env_steps=self.online_session.replay.trainable_env_steps,
                    online_actor_enabled=not self.online_session.is_warmup,
                )
            if hasattr(self.online_session, "run"):
                contract.update(
                    run_uuid=self.online_session.run["run_uuid"],
                    dataset_uuid=self.online_session.run["dataset_uuid"],
                    recording_schema_version=self.online_session.run["recording_schema_version"],
                    online_status=self.online_session.status,
                    online_fault=self.online_session.fault_reason,
                )
        return contract

    def _set_phase(self, phase: str, *, force: bool = False) -> None:
        if phase not in {"vla", "rl"}:
            raise ValueError(f"phase must be 'vla' or 'rl', got {phase!r}")
        if phase == self.phase and not force:
            return
        if phase == "rl":
            self.policy.set_rl_mode()
        else:
            self.policy.set_vla_mode()
        self.phase = phase

    def handle(
        self,
        header: dict[str, Any],
        arrays: dict[str, np.ndarray],
        progress_callback: (
            Callable[[dict[str, float | int | str | None]], None] | None
        ) = None,
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        kind = header.get("kind")
        request_id = int(header.get("request_id", -1))
        if kind == "health":
            return {
                "kind": "health",
                "request_id": request_id,
                "active_phase": self.phase,
                **self.contract,
            }, {}
        if kind in {"begin_episode", "recover_online"}:
            session = getattr(self, "online_session", None)
            if session is None:
                raise RuntimeError("online session is not enabled")
            for key in ("run_uuid", "dataset_uuid"):
                if header.get(key) != session.run[key]:
                    raise ValueError(f"online {key} mismatch")
            result = session.recover(progress_callback) if kind == "recover_online" else session.begin_episode()
            return {"kind": kind, "request_id": request_id, **result, **self.contract}, {}
        if kind == "episode_saved":
            if getattr(self, "online_session", None) is None:
                raise RuntimeError("policy server is not configured for online RLT")
            expected_fingerprint = header.get("checkpoint_fingerprint")
            if expected_fingerprint != self.fingerprint:
                raise ValueError(
                    "collector and server checkpoint fingerprints do not match"
                )
            if str(header.get("task", "")) not in getattr(self, "tasks", (self.task,)):
                raise ValueError("online episode task does not match the policy server")
            dataset_root = pathlib.Path(str(header.get("dataset_root", ""))).resolve()
            if dataset_root != self.online_dataset_root:
                raise ValueError(
                    "online episode dataset root does not match the policy server"
                )
            metrics = self.online_session.process_saved_episode(
                episode_id=int(header.get("episode_id", -1)),
                outcome=str(header.get("outcome", "")),
                progress_callback=progress_callback,
            )
            warmup_env_steps = self.online_session.replay.warmup_env_steps
            warmup_steps = self.online_session.config.warmup_steps
            return {
                "kind": "episode_processed",
                "request_id": request_id,
                **metrics,
                "warmup_env_steps": warmup_env_steps,
                "warmup_steps": warmup_steps,
                "warmup_remaining": max(0, warmup_steps - warmup_env_steps),
            }, {}
        from vla.control.plan_service import control_message
        from vla.adapters.lerobot.plan_inference import infer_plan
        result = control_message(self, header)
        if result is not None:
            return result
        if kind != 'plan':
            raise ValueError('only protocol-v2 candidate plans are supported')
        return infer_plan(self, header, arrays)



def _send_error(
    connection: socket.socket, request_id: int, error: BaseException
) -> None:
    send_packet(
        connection,
        {
            "kind": "error",
            "request_id": request_id,
            "error_type": type(error).__name__,
            "error": str(error),
        },
    )


def serve(args: argparse.Namespace) -> None:
    if args.socket_timeout_s <= 0:
        raise ValueError("socket-timeout-s must be positive")
    policy_server = FrankaRolloutPolicyServer(args)
    socket_path = pathlib.Path(args.socket_path).expanduser().resolve()
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    if socket_path.exists():
        if not socket_path.is_socket():
            raise FileExistsError(f"refusing to replace non-socket path: {socket_path}")
        socket_path.unlink()
    from vla.web.policy_server import serve as serve_concurrent
    try:
        serve_concurrent(policy_server, socket_path)
    finally:
        if policy_server.online_session is not None:
            policy_server.online_session.close()


def main() -> None:
    serve(parse_args())


if __name__ == "__main__":
    main()
