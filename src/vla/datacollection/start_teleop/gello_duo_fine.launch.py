#!/usr/bin/env python3
"""Launch configured GELLO publishers while routing left commands through the arbiter."""

from __future__ import annotations

import argparse
import os

import yaml
from launch import LaunchDescription, LaunchService
from launch.actions import DeclareLaunchArgument, OpaqueFunction, Shutdown
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_yaml(path: str):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"GELLO configuration not found: {path}")
    with open(path, encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _nodes_from_config(config_path: str):
    configs = _load_yaml(config_path)
    nodes = []
    for item_name, config in configs.items():
        namespace = config["namespace"]
        remappings = [
            (
                "gripper/gripper_client/target_gripper_width_percent",
                "gello/gripper_width_percent",
            )
        ]
        nodes.append(
            Node(
                package="franka_gello_state_publisher",
                executable="gello_publisher",
                name="gello_publisher",
                namespace=namespace,
                output="screen",
                on_exit=Shutdown(),
                remappings=remappings,
                parameters=[
                    {"com_port": "/dev/serial/by-id/" + config["com_port"]},
                    {"gello_name": item_name},
                    {"num_arm_joints": config["num_arm_joints"]},
                    {"joint_signs": config["joint_signs"]},
                    {"gripper": config["gripper"]},
                    {"gripper_range_rad": config["gripper_range_rad"]},
                    {"assembly_offsets": config["assembly_offsets"]},
                    {"dynamixel_kp_p": config["dynamixel_kp_p"]},
                    {"dynamixel_kp_i": config["dynamixel_kp_i"]},
                    {"dynamixel_kp_d": config["dynamixel_kp_d"]},
                    {"dynamixel_goal_position": config["dynamixel_goal_position"]},
                    {"dynamixel_torque_enable": config["dynamixel_torque_enable"]},
                ],
            )
        )
    return nodes


def _generate_nodes(context):
    config_path = LaunchConfiguration("config_path").perform(context)
    return _nodes_from_config(config_path)


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument("config_path", description="Absolute GELLO YAML path"),
            OpaqueFunction(function=_generate_nodes),
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--expected-namespaces", default="left,right")
    args = parser.parse_args()
    config_path = os.path.abspath(args.config_path)
    if args.check:
        namespaces = {config["namespace"] for config in _load_yaml(config_path).values()}
        expected = {item.strip() for item in args.expected_namespaces.split(",") if item.strip()}
        if not expected or namespaces != expected:
            raise ValueError(
                f"Expected GELLO namespaces={sorted(expected)}, got {sorted(namespaces)}"
            )
        print(f"GELLO fine launch check passed: {args.config_path}")
        return

    nodes = _nodes_from_config(config_path)
    service = LaunchService()
    service.include_launch_description(LaunchDescription(nodes))
    raise SystemExit(service.run())


if __name__ == "__main__":
    main()
