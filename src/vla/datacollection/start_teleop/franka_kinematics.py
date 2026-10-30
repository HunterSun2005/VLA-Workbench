"""Shared Franka kinematic helpers for teleoperation and rollout control."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import PyKDL
from urdf_parser_py.urdf import Joint, Pose, Robot


@dataclass(frozen=True)
class KinematicModel:
    chain: PyKDL.Chain
    joint_names: tuple[str, ...]
    lower_limits: np.ndarray
    upper_limits: np.ndarray


def joint7_step(
    current: float,
    x_button: float,
    b_button: float,
    speed: float,
    direction_sign: float,
    dt: float,
    lower: float,
    upper: float,
) -> float:
    target = current + direction_sign * (x_button - b_button) * speed * dt
    return float(np.clip(target, lower, upper))


def _pose_to_frame(origin: Pose | None) -> PyKDL.Frame:
    if origin is None:
        return PyKDL.Frame.Identity()
    xyz = origin.xyz if origin.xyz is not None else [0.0, 0.0, 0.0]
    rpy = origin.rpy if origin.rpy is not None else [0.0, 0.0, 0.0]
    return PyKDL.Frame(
        PyKDL.Rotation.RPY(float(rpy[0]), float(rpy[1]), float(rpy[2])),
        PyKDL.Vector(float(xyz[0]), float(xyz[1]), float(xyz[2])),
    )


def _joint_to_kdl(joint: Joint, origin: PyKDL.Frame) -> PyKDL.Joint:
    if joint.type == "fixed":
        return PyKDL.Joint(joint.name, PyKDL.Joint.Fixed)

    axis_values = joint.axis if joint.axis is not None else [1.0, 0.0, 0.0]
    axis = origin.M * PyKDL.Vector(*(float(value) for value in axis_values))
    if joint.type in {"revolute", "continuous"}:
        return PyKDL.Joint(joint.name, origin.p, axis, PyKDL.Joint.RotAxis)
    if joint.type == "prismatic":
        return PyKDL.Joint(joint.name, origin.p, axis, PyKDL.Joint.TransAxis)
    raise ValueError(f"Unsupported URDF joint type {joint.type!r} for {joint.name!r}")


def build_kinematic_model(
    robot_description: str,
    base_link: str,
    tip_link: str,
    joint_limit_margin: float,
) -> KinematicModel:
    robot = Robot.from_xml_string(robot_description)
    path: list[Joint] = []
    current_link = tip_link
    visited: set[str] = set()

    while current_link != base_link:
        if current_link in visited:
            raise ValueError(f"Cycle found while resolving {base_link} -> {tip_link}")
        visited.add(current_link)
        try:
            joint_name, parent_link = robot.parent_map[current_link]
            joint = robot.joint_map[joint_name]
        except KeyError as exc:
            raise ValueError(
                f"No URDF chain from {base_link!r} to {tip_link!r}; "
                f"stopped at {current_link!r}"
            ) from exc
        path.append(joint)
        current_link = parent_link

    chain = PyKDL.Chain()
    joint_names: list[str] = []
    lower_limits: list[float] = []
    upper_limits: list[float] = []

    for joint in reversed(path):
        origin = _pose_to_frame(joint.origin)
        chain.addSegment(PyKDL.Segment(joint.child, _joint_to_kdl(joint, origin), origin))
        if joint.type == "fixed":
            continue
        if joint.type == "continuous":
            lower, upper = -math.pi, math.pi
        elif joint.limit is None or joint.limit.lower is None or joint.limit.upper is None:
            raise ValueError(f"Joint {joint.name!r} has no finite position limits")
        else:
            lower = float(joint.limit.lower)
            upper = float(joint.limit.upper)
        if upper - lower <= 2.0 * joint_limit_margin:
            raise ValueError(f"Joint limit margin is too large for {joint.name!r}")
        joint_names.append(joint.name)
        lower_limits.append(lower + joint_limit_margin)
        upper_limits.append(upper - joint_limit_margin)

    if chain.getNrOfJoints() != 7:
        raise ValueError(
            f"Expected a 7-DoF chain from {base_link} to {tip_link}, "
            f"got {chain.getNrOfJoints()}"
        )

    return KinematicModel(
        chain=chain,
        joint_names=tuple(joint_names),
        lower_limits=np.asarray(lower_limits, dtype=float),
        upper_limits=np.asarray(upper_limits, dtype=float),
    )
