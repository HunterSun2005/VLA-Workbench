#!/usr/bin/env python3
"""Launch configured Robotiq grippers with explicit source and teleop topics."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    control_owner = LaunchConfiguration("control_owner")
    rollout_condition = PythonExpression(["'", control_owner, "' == 'rollout'"])
    upstream_launch = PathJoinSubstitution(
        [
            FindPackageShare("franka_gripper_manager"),
            "launch",
            "robotiq_gripper_controller_client.launch.py",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                description="Gripper configuration filename in the package config directory",
            ),
            DeclareLaunchArgument(
                "control_owner",
                default_value="teleop",
                description="teleop keeps right GELLO input; rollout routes both grippers",
            ),
            SetRemap(
                src=(
                    "/left/gripper/gripper_client/"
                    "target_gripper_width_percent"
                ),
                dst="/left/teleop/gripper_drive_width_percent",
            ),
            SetRemap(
                src=(
                    "/right/gripper/gripper_client/"
                    "target_gripper_width_percent"
                ),
                dst="/right/gello/gripper_width_percent",
                condition=UnlessCondition(rollout_condition),
            ),
            SetRemap(
                src=(
                    "/right/gripper/gripper_client/"
                    "target_gripper_width_percent"
                ),
                dst="/right/teleop/gripper_drive_width_percent",
                condition=IfCondition(rollout_condition),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(upstream_launch),
                launch_arguments={"config_file": config_file}.items(),
            ),
        ]
    )
