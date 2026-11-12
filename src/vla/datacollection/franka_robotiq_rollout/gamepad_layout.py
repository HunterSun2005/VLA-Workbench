"""Device-independent SDL GameController layout; no device discovery or probing."""
from __future__ import annotations


def axis_buttons(layout):
    if layout not in ("standard_sdl", "raw_joy"):
        raise ValueError("gamepad input_layout must be standard_sdl or raw_joy")
    standard = layout == "standard_sdl"
    return dict(left_stick_x_axis=0, left_stick_y_axis=1,
                right_stick_x_axis=2 if standard else 3,
                right_stick_y_axis=3 if standard else 4,
                left_trigger_axis=4 if standard else 2, right_trigger_axis=5,
                a_button=0, b_button=1, x_button=2, y_button=3,
                speed_button=9 if standard else 4, phase_button=10 if standard else 5,
                upright_button=6 if standard else 7, reset_pose_button=4 if standard else 6,
                yaw_reset_button=8 if standard else 10)


def validate_layout(values):
    layout = values.get("input_layout", "raw_joy")
    expected = axis_buttons(layout)
    for key, default in expected.items():
        value = values.get(key, default)
        if type(value) is not int or value < 0:
            raise ValueError(f"gamepad {key} must be a non-negative integer")
        if layout == "standard_sdl" and value != default:
            raise ValueError(f"gamepad {key}: custom indices require input_layout: raw_joy")


def node_for_layout(layout):
    axis_buttons(layout)
    return "game_controller_node" if layout == "standard_sdl" else "joy_node"


if __name__ == "__main__":
    import sys
    from .config import load_rollout_config
    config = load_rollout_config(sys.argv[1])
    arm = next(a for a in config.arms if a.name == sys.argv[2])
    print(node_for_layout(arm.gamepad.input_layout))
