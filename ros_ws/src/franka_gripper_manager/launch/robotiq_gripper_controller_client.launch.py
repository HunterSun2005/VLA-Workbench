import os
import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def load_yaml(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    with open(file_path, "r") as file:
        return yaml.safe_load(file)


def generate_robot_nodes(context):
    config_file_name = LaunchConfiguration("config_file").perform(context)
    package_config_dir = FindPackageShare("franka_gripper_manager").perform(context)
    config_file = os.path.join(package_config_dir, "config", config_file_name)
    configs = load_yaml(config_file)
    nodes = []
    for item_name, config in configs.items():
        namespace = config["namespace"]
        nodes.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [FindPackageShare("franka_gripper_manager"), "launch", "robotiq.launch.py"]
                    )
                ),
                launch_arguments={
                    "namespace": namespace,
                    "com_port": str(config["com_port"]),
                    "open_position": str(config.get("open_position", 0.0)),
                    "closed_position": str(config.get("closed_position", 0.8)),
                    "max_effort": str(config.get("max_effort", 50.0)),
                    "command_deadband": str(config.get("command_deadband", 0.02)),
                    "endpoint_snap_threshold": str(
                        config.get("endpoint_snap_threshold", 0.25)
                    ),
                }.items(),
            )
        )
    return nodes


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value="example_fr3_config_robotiq.yaml",
                description="Path to the robot configuration file to load",
            ),
            OpaqueFunction(function=generate_robot_nodes),
        ]
    )
