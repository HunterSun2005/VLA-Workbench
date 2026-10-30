import pytest

from franka_gripper_manager.robotiq_command import (
    snap_width_percent_to_endpoints,
    width_percent_to_joint_position,
)


@pytest.mark.parametrize(
    ("width_percent", "expected_position"),
    [
        (0.0, 0.8),
        (0.25, 0.6),
        (0.5, 0.4),
        (1.0, 0.0),
        (-1.0, 0.8),
        (2.0, 0.0),
    ],
)
def test_width_percent_to_joint_position(width_percent, expected_position):
    assert width_percent_to_joint_position(width_percent, 0.0, 0.8) == pytest.approx(
        expected_position
    )


@pytest.mark.parametrize(
    ("width_percent", "expected_width"),
    [
        (-0.1, 0.0),
        (0.0, 0.0),
        (0.019, 0.0),
        (0.02, 0.0),
        (0.021, 0.021),
        (0.5, 0.5),
        (0.979, 0.979),
        (0.98, 1.0),
        (1.0, 1.0),
        (1.1, 1.0),
    ],
)
def test_snap_width_percent_to_endpoints(width_percent, expected_width):
    assert snap_width_percent_to_endpoints(width_percent, 0.02) == pytest.approx(
        expected_width
    )


@pytest.mark.parametrize(
    ("width_percent", "expected_width"),
    [
        (0.249, 0.0),
        (0.25, 0.0),
        (0.251, 0.251),
        (0.749, 0.749),
        (0.75, 1.0),
        (0.751, 1.0),
    ],
)
def test_snap_width_percent_with_calibration_tolerance(width_percent, expected_width):
    assert snap_width_percent_to_endpoints(width_percent, 0.25) == pytest.approx(
        expected_width
    )
