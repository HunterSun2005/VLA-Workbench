#!/usr/bin/env python3
"""Build mode-specific Franka, Robotiq, and optional GELLO configurations."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


MODES = {
    "single_left": ("LEFT",),
    "duo": ("LEFT", "RIGHT"),
}


def _load_selected(path: str | Path, sides: tuple[str, ...]) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        data = yaml.safe_load(stream)
    if not isinstance(data, dict) or not set(sides).issubset(data):
        raise ValueError(f"Configuration must contain {sides}: {path}")
    return {side: data[side] for side in sides}


def _save(path: str | Path, data: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(data, stream, sort_keys=False)


def _serial_path(value: object, *, expand_by_id: bool = False) -> str:
    path = str(value)
    if expand_by_id and not os.path.isabs(path):
        path = os.path.join("/dev/serial/by-id", path)
    return os.path.realpath(path)


def prepare_runtime_configs(
    *,
    mode: str,
    gello_source: str | Path | None,
    arm_source: str | Path,
    gripper_source: str | Path,
    gello_output: str | Path | None,
    arm_output: str | Path,
    gripper_output: str | Path,
    gripper_speed_output: str | Path,
    left_gello: str = "",
    right_gello: str = "",
    left_ip: str = "",
    right_ip: str = "",
    left_gripper: str = "",
    right_gripper: str = "",
    control_owner: str = "teleop",
) -> None:
    try:
        sides = MODES[mode]
    except KeyError as error:
        raise ValueError(f"unsupported teleop mode: {mode}") from error

    if control_owner not in {"teleop", "rollout"}:
        raise ValueError(f"unsupported control owner: {control_owner}")
    if control_owner == "teleop" and (gello_source is None or gello_output is None):
        raise ValueError("teleop control requires GELLO source and output paths")

    gello = _load_selected(gello_source, sides) if control_owner == "teleop" else None
    arms = _load_selected(arm_source, sides)
    grippers = _load_selected(gripper_source, sides)

    overrides = {
        "LEFT": (left_gello, left_ip, left_gripper),
        "RIGHT": (right_gello, right_ip, right_gripper),
    }
    for side in sides:
        gello_port, robot_ip, gripper_port = overrides[side]
        if gello is not None and gello_port:
            gello[side]["com_port"] = gello_port
        if robot_ip:
            arms[side]["robot_ip"] = robot_ip
        if gripper_port:
            grippers[side]["com_port"] = gripper_port

    gripper_ports = {side: _serial_path(grippers[side]["com_port"]) for side in sides}
    if len(set(gripper_ports.values())) != len(gripper_ports):
        raise ValueError("LEFT and RIGHT Robotiq devices resolve to the same serial port")
    if gello is not None:
        gello_ports = {
            side: _serial_path(gello[side]["com_port"], expand_by_id=True)
            for side in sides
        }
        if len(set(gello_ports.values())) != len(gello_ports):
            raise ValueError("LEFT and RIGHT GELLO devices resolve to the same serial port")
        conflicts = set(gello_ports.values()) & set(gripper_ports.values())
        if conflicts:
            raise ValueError(
                "GELLO and Robotiq serial ports conflict: "
                + ", ".join(sorted(conflicts))
            )

    if gello is not None:
        _save(gello_output, gello)
    _save(arm_output, arms)
    _save(gripper_output, grippers)
    _save(
        gripper_speed_output,
        {
            "/**": {
                "ros__parameters": {
                    "joints": ["robotiq_85_left_knuckle_joint"],
                    "interface_name": "set_gripper_max_velocity",
                }
            }
        },
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=tuple(MODES), required=True)
    parser.add_argument(
        "--control-owner", choices=("teleop", "rollout"), default="teleop"
    )
    parser.add_argument("--gello-source")
    parser.add_argument("--arm-source", required=True)
    parser.add_argument("--gripper-source", required=True)
    parser.add_argument("--gello-output")
    parser.add_argument("--arm-output", required=True)
    parser.add_argument("--gripper-output", required=True)
    parser.add_argument("--gripper-speed-output", required=True)
    parser.add_argument("--left-gello", default="")
    parser.add_argument("--right-gello", default="")
    parser.add_argument("--left-ip", default="")
    parser.add_argument("--right-ip", default="")
    parser.add_argument("--left-gripper", default="")
    parser.add_argument("--right-gripper", default="")
    return parser.parse_args()


def main() -> None:
    prepare_runtime_configs(**vars(parse_args()))


if __name__ == "__main__":
    main()
