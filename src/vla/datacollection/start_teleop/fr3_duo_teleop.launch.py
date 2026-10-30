#!/usr/bin/env python3
"""Launch FR3 controllers with teleop or rollout command routing."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, PythonExpression
from launch_ros.actions import SetRemap
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_config_file = LaunchConfiguration("robot_config_file")
    control_owner = LaunchConfiguration("control_owner")
    upstream_launch = PathJoinSubstitution(
        [
            FindPackageShare("franka_fr3_arm_controllers"),
            "launch",
            "franka_fr3_arm_controllers.launch.py",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "robot_config_file",
                description="Arm configuration filename in the package config directory",
            ),
            DeclareLaunchArgument(
                "control_owner",
                default_value="teleop",
                description="teleop keeps right GELLO input; rollout routes both arms",
            ),
            SetRemap(
                src="/left/gello/joint_states",
                dst="/left/teleop/joint_states",
            ),
            SetRemap(
                src="/right/gello/joint_states",
                dst="/right/teleop/joint_states",
                condition=IfCondition(
                    PythonExpression(["'", control_owner, "' == 'rollout'"])
                ),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(upstream_launch),
                launch_arguments={"robot_config_file": robot_config_file}.items(),
            ),
        ]
    )
