"""Dual ROS2 gamepad demonstrations without a model server or GELLO."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config


DEFAULT_CONFIG = Path(__file__).parents[1] / "franka_robotiq_rollout/configs/dual.yaml"
ACTION_ENCODING = "controller_joint_command_fk_absolute_tcp_rot6d_v1"


def validate_dual_config(config):
    if config.backend != "ros2" or tuple(arm.name for arm in config.arms) != ("left", "right"):
        raise ValueError("dual gamepad recording requires ROS2 arms ordered left, right")
    if tuple(camera.name for camera in config.cameras) != ("base", "left_wrist", "right_wrist"):
        raise ValueError("dual cameras must be ordered base, left_wrist, right_wrist")
    if len({camera.serial for camera in config.cameras}) != 3:
        raise ValueError("dual recording requires three distinct camera serials")
    for field in ("robot_state_topic", "measured_joint_topic", "gripper_state_topic",
                  "command_joint_topic", "command_gripper_topic", "gripper_action_topic",
                  "gripper_speed_topic"):
        if len({getattr(arm, field) for arm in config.arms}) != 2:
            raise ValueError(f"left/right {field} must be distinct")
    if len({arm.gamepad.joy_topic for arm in config.arms}) != 2:
        raise ValueError("left and right gamepads must use distinct Joy topics")
    if config.fps != 30 or config.image_size != (224, 224) or config.gamepad_control_rate != 100:
        raise ValueError("dual recording requires 30 FPS, 224x224 RGB and 100 Hz gamepad control")
    if not config.task.strip():
        raise ValueError("task must be nonempty")
    if not config.tasks or config.task not in config.tasks:
        raise ValueError("dataset.tasks must contain dataset.task")


def recording_contract(config):
    settings = asdict(config)
    # Model transport and output paths do not affect recorded actions.
    for key in ("dataset_root", "socket_path", "joy_socket_path", "policy_timeout_s"):
        settings.pop(key)
    return json.loads(json.dumps({
        "version": 1, "collection_kind": "human_demonstration", "source": "demonstration",
        "action_encoding": ACTION_ENCODING, "settings": settings,
        "state_gripper_unit": "metres", "action_gripper_unit": "open_fraction",
        "recording_clock": "base_camera_host_receive",
    }, default=str))


def validate_dataset_root(root, *, resume):
    root = Path(root).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise ValueError("dataset root must be a directory")
    if resume and not (root / "meta/info.json").is_file():
        raise ValueError("--resume requires an existing dataset")
    if not resume and root.exists() and any(root.iterdir()):
        raise ValueError("dataset root is not empty; use --resume with the original recording contract")


def validate_resume_contract(config):
    info = json.loads((config.dataset_root / "meta/info.json").read_text())
    if (info.get("robot_type") != "franka_robotiq_dual"
            or info.get("recording_contract") != recording_contract(config)):
        raise ValueError("existing dataset recording_contract differs; use the original settings or a new root")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--check-only", action="store_true")
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--tasks", nargs="+", help="task prompts available for episode selection")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.resume and args.dataset_root is None:
        parser.error("--resume requires --dataset-root")
    if args.execute and (args.dataset_root is None or (not args.task and not args.tasks)):
        parser.error("--execute requires --dataset-root and --task or --tasks")
    config = load_rollout_config(args.config)
    if args.dataset_root is not None:
        config = replace(config, dataset_root=args.dataset_root.expanduser().resolve())
    if args.task is not None:
        config = replace(config, task=args.task,
                         tasks=(args.task, *tuple(t for t in config.tasks if t != args.task)))
    if args.tasks is not None:
        prompts = tuple(dict.fromkeys(item.strip() for item in args.tasks if item.strip()))
        if not prompts:
            parser.error("--tasks requires at least one nonempty prompt")
        config = replace(config, task=prompts[0], tasks=prompts)
    validate_dual_config(config)
    if args.dataset_root is not None:
        validate_dataset_root(config.dataset_root, resume=args.resume)
        if args.resume:
            validate_resume_contract(config)
    if not args.execute:
        print("Dual configuration passed: left/right, 58D raw state, 20D action, 3 RGB cameras; "
              "no hardware connection, policy server or dataset initialization")
        return
    if not sys.stdin.isatty():
        parser.error("--execute requires an interactive terminal")
    # The configuration check above also runs in the model environment without ROS imports.
    from vla.datacollection.franka_robotiq_rollout.collect_ros import run
    run(config, demonstration=True, resume=args.resume)


if __name__ == "__main__":
    main()
