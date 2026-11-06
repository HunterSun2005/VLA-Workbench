"""Dual ROS2 gamepad demonstrations without a model server or GELLO."""
from __future__ import annotations

from vla.terminal import (console_print as print, console_status, console_debug, terminal_session,
                          TerminalConfirmation, diagnostic_path, current_reporter)

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import sys

from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config


DEFAULT_CONFIG = Path(__file__).parents[1] / "franka_robotiq_rollout/configs/dual.yaml"
ACTION_ENCODING = "ros_accepted_absolute_target_v1"


def validate_dual_config(config):
    if config.backend != "ros2" or tuple(arm.name for arm in config.arms) not in (("left",), ("left", "right")):
        raise ValueError("dual gamepad recording requires ROS2 arms ordered left, right")
    if len(config.arms) == 2 and tuple(camera.name for camera in config.cameras) not in (("base", "left_wrist", "right_wrist"), ("base", "left_wrist", "right_wrist", "side")):
        raise ValueError("dual cameras must be ordered base, left_wrist, right_wrist, with optional side last")
    if len({camera.serial for camera in config.cameras}) != len(config.cameras):
        raise ValueError("dual recording requires distinct camera serials")
    for field in ("robot_state_topic", "measured_joint_topic", "gripper_state_topic",
                  "command_joint_topic", "command_gripper_topic", "gripper_action_topic",
                  "gripper_speed_topic"):
        if len({getattr(arm, field) for arm in config.arms}) != len(config.arms):
            raise ValueError(f"left/right {field} must be distinct")
    if len({arm.gamepad.joy_topic for arm in config.arms}) != len(config.arms):
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
        "version": 2, "raw_state_format": "full50_per_arm", "collection_kind": "human_demonstration", "source": "demonstration",
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


def main():
    from vla.cli.ros_session import main as shared_main
    shared_main(['--arms', 'dual', '--source', 'gamepad', *sys.argv[1:]])


if __name__ == "__main__":
    main()
