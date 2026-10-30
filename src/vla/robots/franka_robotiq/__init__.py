from vla.robots.franka_robotiq.config import (
    FrankaRobotiqConfig,
    franka_robotiq_config_from_mapping,
    load_franka_robotiq_config,
)
from vla.robots.franka_robotiq.robot import FrankaRobotiqRobot

# LeRobot's generic device factory derives this name by removing "Config".
FrankaRobotiq = FrankaRobotiqRobot

__all__ = [
    "FrankaRobotiq",
    "FrankaRobotiqConfig",
    "FrankaRobotiqRobot",
    "franka_robotiq_config_from_mapping",
    "load_franka_robotiq_config",
]
