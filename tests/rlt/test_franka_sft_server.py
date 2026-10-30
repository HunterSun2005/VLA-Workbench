from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

from vla.cli.serve_franka_sft import Server, STATE_NAMES, ACTION_NAMES, TASK
from vla.datacollection.franka_robotiq_single_left.schema import TCP_STATE_FIELDS


def test_takeover_collector_accepts_pure_sft_without_online_or_rtc():
    from pathlib import Path
    from vla.datacollection.franka_robotiq_rollout.collect_franky import FrankyRolloutCollector
    from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config

    collector = object.__new__(FrankyRolloutCollector)
    config = Path(__file__).resolve().parents[2] / "src/vla/datacollection/franka_robotiq_rollout/configs/single_left_sft_takeover.yaml"
    collector.config = load_rollout_config(config)
    collector.require_online = False
    cameras = ["observation.images.base", "observation.images.left_wrist"]
    collector.policy_client = SimpleNamespace(contract={
        "inference_mode": "sft_action_queue", "online_enabled": False, "rtc_enabled": False,
        "state_dim": 10, "action_dim": 10, "robot_type": "franka_robotiq", "task": TASK,
        "camera_keys": cameras, "camera_shapes_hwc": {key: [224, 224, 3] for key in cameras},
        "state_names": STATE_NAMES, "action_names": ACTION_NAMES, "fps": 30,
        "actions_stored_as": "absolute", "use_relative_actions": False,
    })
    collector._validate_contract()
    assert collector._sft_only
    collector._phase = "vla"
    collector.toggle_phase()
    assert collector._phase == "vla"


def test_sft_server_resets_queue_on_episode_or_takeover_return():
    server = object.__new__(Server)
    server.task, server.fingerprint, server.device = "task", "weights", "cpu"
    server.torch = torch
    server.prepare = lambda obs, *_: obs
    server.pre = server.post = lambda value: value
    server.policy = SimpleNamespace(reset=Mock(), select_action=Mock(return_value=torch.zeros(1, 10)))
    arrays = {"observation.state": np.zeros(10),
              "observation.images.base": np.zeros((224, 224, 3), dtype=np.uint8),
              "observation.images.left_wrist": np.zeros((224, 224, 3), dtype=np.uint8)}
    header = {"kind": "infer", "phase": "vla", "reset_episode": True}
    _, action = server.handle(header, arrays)
    server.policy.reset.assert_called_once()
    assert action["action"].shape == (10,)
    server.handle({**header, "reset_episode": False}, arrays)
    assert server.policy.reset.call_count == 1
    server.handle(header, arrays)
    assert server.policy.reset.call_count == 2
    with pytest.raises(ValueError, match="only supports vla"):
        server.handle({**header, "phase": "rl"}, arrays)
    assert STATE_NAMES == TCP_STATE_FIELDS
