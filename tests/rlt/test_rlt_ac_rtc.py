from types import SimpleNamespace
import pytest
import torch
from vla.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig


def test_configure_rtc_accepts_vla_rtc_config_and_switches_by_phase() -> None:
    policy = object.__new__(ChunkACPolicy)
    policy.config = SimpleNamespace(chunk_length=10, action_dim=3, proprio_dim=3)
    policy.vla_seen_configs = []

    class FakePi05:
        def __init__(self) -> None:
            self.config = SimpleNamespace(rtc_config=None)
            self.init_calls: list[RTCConfig] = []

        def init_rtc_processor(self) -> None:
            self.init_calls.append(self.config.rtc_config)

        def predict_action_chunk(self, batch, **kwargs):
            policy.vla_seen_configs.append(self.config.rtc_config)
            return torch.zeros(1, 4, 3)

    fake_pi05 = FakePi05()
    policy._rl_token_policy = SimpleNamespace(_pi05=fake_pi05)
    policy._prefix_capture = SimpleNamespace(consume=lambda: torch.zeros(1, 1, 1))
    policy.modifier = SimpleNamespace(
        is_rl_phase=False,
        compute_chunk=lambda vla_chunk, proprio, prefix_tokens: vla_chunk,
    )
    object.__setattr__(policy, "eval", lambda: policy)
    object.__setattr__(policy, "_ensure_modifier", lambda: policy.modifier)

    rlt_rtc_config = RTCConfig(enabled=True, execution_horizon=10)
    vla_rtc_config = RTCConfig(enabled=True, execution_horizon=25)
    policy.configure_rtc(rlt_rtc_config, fps=30, vla_rtc_config=vla_rtc_config)

    policy.predict_action_chunk({"observation.state": torch.ones(1, 3)})
    policy.modifier.is_rl_phase = True
    policy.predict_action_chunk({"observation.state": torch.ones(1, 3)})

    assert fake_pi05.init_calls == [rlt_rtc_config, vla_rtc_config, rlt_rtc_config]
    assert policy.vla_seen_configs == [vla_rtc_config, rlt_rtc_config]




def test_model_cannot_pop_actions_or_own_an_execution_cursor():
    policy = object.__new__(ChunkACPolicy)
    with pytest.raises(RuntimeError, match='ROS executor'):
        policy.select_action({})
    assert not hasattr(policy, '_prepare_rtc_request')
    assert not hasattr(policy, '_select_action_rtc')
