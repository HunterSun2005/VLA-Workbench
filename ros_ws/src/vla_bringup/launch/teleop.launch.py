"""Explicit controller/Robotiq bringup; configs are external runtime files."""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from vla.paths import resource

def generate_launch_description():
    root = resource("datacollection/start_teleop")
    return LaunchDescription([
        DeclareLaunchArgument("robot_config_file"),
        DeclareLaunchArgument("config_file"),
        DeclareLaunchArgument("control_owner", default_value="teleop"),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(root / "fr3_duo_teleop.launch.py")), launch_arguments={"robot_config_file": LaunchConfiguration("robot_config_file"), "control_owner": LaunchConfiguration("control_owner")}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(root / "robotiq_duo_teleop.launch.py")), launch_arguments={"config_file": LaunchConfiguration("config_file"), "control_owner": LaunchConfiguration("control_owner")}.items()),
    ])
