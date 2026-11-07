from tests.rlt.test_dual_gamepad_record import ros_module
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vla.cli.serve_franka_sft import Server, STATE_NAMES, ACTION_NAMES, TASK
from vla.datacollection.franka_robotiq_single_left.schema import TCP_STATE_FIELDS


def test_takeover_collector_accepts_pure_sft_without_online_or_rtc(ros_module):
    from pathlib import Path
    cls = ros_module.RolloutCollector
    from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config

    collector = object.__new__(cls)
    config = Path(__file__).resolve().parents[2] / "src/vla/datacollection/franka_robotiq_rollout/configs/single_left_sft_takeover.yaml"
    collector.config = load_rollout_config(config)
    collector.require_online = False
    cameras = ["observation.images.base", "observation.images.left_wrist"]
    collector.policy_client = SimpleNamespace(contract={
        "online_enabled": False, "rtc_enabled": False,
        "state_dim": 10, "action_dim": 10, "robot_type": "franka_robotiq", "task": TASK,
        "camera_keys": cameras, "camera_shapes_hwc": {key: [224, 224, 3] for key in cameras},
        "state_names": STATE_NAMES, "action_names": ACTION_NAMES, "fps": 30,
        "actions_stored_as": "absolute", "use_relative_actions": False,
        "wire_action_representation":"absolute", "action_spec":collector.config.action_spec.to_dict(),
    })
    collector._validate_policy_contract()


@pytest.mark.parametrize("arm_count", [1, 2])
def test_sft_server_rejects_single_action_reset_requests(arm_count):
    server = object.__new__(Server)
    with pytest.raises(ValueError, match='protocol-v2'):
        server.handle({'kind': 'infer', 'reset_episode': True}, {})
