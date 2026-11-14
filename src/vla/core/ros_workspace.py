"""One managed ROS workspace selector shared by shell, CLI, and Web."""
import os
from pathlib import Path


def ros_workspace(root, configured=None):
    root = Path(root).resolve()
    path = Path(configured or os.environ.get('VLA_ROS_WORKSPACE') or root/'runtime/ros/current').expanduser()
    if path.absolute() == root/'runtime/ros/current' and not path.exists():
        return path.absolute()
    resolved = path.resolve()
    if not resolved.is_relative_to(root/'runtime/ros/workspaces'):
        raise ValueError('ROS workspace must be a managed runtime/ros/workspaces build; rebuild for libfranka 0.21.2')
    return resolved
