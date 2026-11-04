"""No hardware: verify the GELLO reference, startup ordering and failed setup."""
from pathlib import Path
import re

import pytest
import yaml

from vla.datacollection.start_teleop.gamepad_config import load_config
from vla.robots.franka_robotiq.collision import GELLO_COLLISION_THRESHOLDS
from vla.robots.franka_robotiq.config import FrankaRobotiqConfig
from vla.robots.franka_robotiq.robot import FrankaRobotiqRobot
from tests.rlt.test_franka_pi05_robot import FakeArm, FakeCamera, FakeGripper

ROOT = Path(__file__).parents[2]
TELEOP = ROOT / 'src/vla/datacollection/start_teleop'


def test_collision_thresholds_match_gello_full_service_request():
    shell = (TELEOP / 'lib/controllers.sh').read_text()
    request = yaml.safe_load(re.search(r"local request='(\{.*?\})'", shell, re.S).group(1))
    names = [f'{bound}_{kind}_thresholds_{phase}'
             for kind in ('torque', 'force') for phase in ('acceleration', 'nominal')
             for bound in ('lower', 'upper')]
    assert set(names) == set(request)
    assert GELLO_COLLISION_THRESHOLDS == tuple(tuple(request[name]) for name in names)


@pytest.mark.parametrize('name', ['single_left_gamepad', 'single_left_gamepad_record', 'single_left_gamepad_31d'])
def test_gamepad_profiles_explicitly_select_gello(name):
    config = load_config(TELEOP / 'configs' / f'{name}.yaml', recording=name != 'single_left_gamepad')
    assert config.robot.collision_behavior_profile == 'gello'


@pytest.mark.parametrize('profile', ['default', 'gello'])
def test_profile_applies_before_peripherals_or_motion(tmp_path, profile):
    camera, gripper = FakeCamera(), FakeGripper()

    class Arm(FakeArm):
        def set_collision_behavior(self, *thresholds):
            assert not camera.connected and not gripper.connected and not self.moves
            super().set_collision_behavior(*thresholds)

    arm = Arm('unused')
    config = FrankaRobotiqConfig(id='test', calibration_dir=tmp_path, collision_behavior_profile=profile)
    robot = FrankaRobotiqRobot(config, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    try:
        robot.connect()
        assert arm.collision_calls == ([GELLO_COLLISION_THRESHOLDS] if profile == 'gello' else [])
        assert not arm.moves
    finally:
        robot.disconnect()


def test_collision_setup_failure_aborts_connection_and_cleans_up(tmp_path):
    class Arm(FakeArm):
        def set_collision_behavior(self, *thresholds):
            raise RuntimeError('configuration rejected')

    arm, camera, gripper = Arm('unused'), FakeCamera(), FakeGripper()
    config = FrankaRobotiqConfig(id='test', calibration_dir=tmp_path, collision_behavior_profile='gello')
    robot = FrankaRobotiqRobot(config, arm_factory=lambda _: arm, camera=camera, gripper=gripper)
    with pytest.raises(RuntimeError, match='Failed to apply GELLO collision profile before motion'):
        robot.connect()
    assert not robot.is_connected and not arm.moves
    assert not camera.connected and not gripper.connected
    assert camera.closed and gripper.closed


@pytest.mark.parametrize('profile', ['unknown', None, 1, True])
def test_invalid_collision_profile_is_rejected(profile):
    with pytest.raises(ValueError, match='collision_behavior_profile'):
        FrankaRobotiqConfig(collision_behavior_profile=profile)
