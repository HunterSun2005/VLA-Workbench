"""Tests for RLT LeRobot ecosystem integration.

Validates config serialization, state_dict exclusion, model construction,
and action selection using DummyVLAAdapter (no real pi0.5 required).
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch

from vla.adapters.lerobot.policies.configuration_rlt import RLTPretrainedConfig
from vla.adapters.lerobot.policies.modeling_rlt import RLTPretrainedPolicy, _VLA_PREFIX
from vla.core.vla_adapter import DummyVLAAdapter


def _make_policy_with_dummy_vla(
    action_dim: int = 12,
    proprio_dim: int = 12,
    chunk_length: int = 10,
    device: str = "cpu",
) -> RLTPretrainedPolicy:
    """Build an RLTPretrainedPolicy with a DummyVLAAdapter for testing."""
    cfg = RLTPretrainedConfig(
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        chunk_length=chunk_length,
        device=device,
        # Exercise real serialization and inference with a tiny synthetic head.
        rl_token_dim=32,
        rl_token_nhead=2,
        rl_token_enc_layers=1,
        rl_token_dec_layers=1,
        rl_token_ff_dim=64,
        rl_token_num_rl_tokens=2,
        actor_hidden_dim=16,
        actor_num_layers=1,
        token_pool_size=8,
        camera_keys=["cam0"],
        proprio_keys=["proprio"],
        action_keys=[f"a{i}" for i in range(action_dim)],
    )
    policy = RLTPretrainedPolicy(cfg)
    dummy_vla = DummyVLAAdapter(
        token_dim=cfg.rl_token_dim,
        num_tokens=8,
        action_dim=action_dim,
        horizon=50,
    )
    policy.set_vla(dummy_vla)
    policy.eval()
    return policy


def _make_batch(
    action_dim: int = 12,
    proprio_dim: int = 12,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Create a minimal batch dict with the keys the policy expects."""
    return {
        "cam0": torch.randn(1, 3, 224, 224, device=device),
        "proprio": torch.randn(1, proprio_dim, device=device),
    }


class TestConfigSerialization:
    def test_round_trip_json(self):
        """Config should survive save/load via PreTrainedConfig.from_pretrained."""
        cfg = RLTPretrainedConfig(
            vla_pretrained_path="lerobot/pi05_base",
            task_instruction="insert screw",
            rl_token_num_rl_tokens=4,
            actor_residual=True,
            phase_mode="always_rl",
            action_dim=12,
            device="cpu",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg.save_pretrained(tmpdir)

            # Verify config.json was written
            config_path = Path(tmpdir) / "config.json"
            assert config_path.exists()
            with open(config_path) as f:
                data = json.load(f)
            assert data["type"] == "rlt"

            # Load via the standard from_pretrained path
            from lerobot.configs.policies import PreTrainedConfig
            loaded = PreTrainedConfig.from_pretrained(tmpdir)

        assert loaded.vla_pretrained_path == "lerobot/pi05_base"
        assert loaded.task_instruction == "insert screw"
        assert loaded.rl_token_num_rl_tokens == 4
        assert loaded.actor_residual is True
        assert loaded.phase_mode == "always_rl"
        assert loaded.action_dim == 12

    def test_invalid_phase_mode_raises(self):
        with pytest.raises(ValueError, match="phase_mode"):
            RLTPretrainedConfig(phase_mode="invalid", device="cpu")


class TestSaveLoad:
    def test_save_and_load_roundtrip(self):
        """Save a policy, reload it, verify weights match."""
        policy = _make_policy_with_dummy_vla()
        assert isinstance(policy, RLTPretrainedPolicy)
        assert isinstance(policy.rl_token, torch.nn.Module)
        assert isinstance(policy.actor, torch.nn.Module)
        assert not hasattr(policy.rl_token, "decoder") or policy.rl_token.inference_only
        sd_before = {k: v.clone() for k, v in policy.state_dict().items()}

        assert not any(key.startswith(_VLA_PREFIX) for key in sd_before)
        assert any(key.startswith("modifier.rl_token.") for key in sd_before)
        assert any(key.startswith("modifier.actor.") for key in sd_before)

        with tempfile.TemporaryDirectory() as tmpdir:
            policy.save_pretrained(tmpdir)

            # Verify files exist
            assert (Path(tmpdir) / "model.safetensors").exists()
            assert (Path(tmpdir) / "config.json").exists()

            # Check config.json has correct type
            with open(Path(tmpdir) / "config.json") as f:
                config_data = json.load(f)
            assert config_data["type"] == "rlt"

            # Reload
            loaded = RLTPretrainedPolicy.from_pretrained(tmpdir, device="cpu")
            sd_after = loaded.state_dict()

        # Same keys
        assert set(sd_before.keys()) == set(sd_after.keys())

        # Same values
        for key in sd_before:
            assert torch.allclose(sd_before[key], sd_after[key]), f"Mismatch on {key}"


class TestValidateFeatures:
    def test_validate_features_populates(self):
        cfg = RLTPretrainedConfig(device="cpu")
        cfg.input_features = {}
        cfg.output_features = {}
        cfg.validate_features()
        assert "observation.state" in cfg.input_features
        assert "action" in cfg.output_features


def test_candidate_chunk_and_reset_do_not_own_execution():
    policy = _make_policy_with_dummy_vla(chunk_length=5)
    batch = _make_batch()
    chunk = policy.predict_action_chunk(batch)
    assert chunk.shape == (1, 5, 12)
    assert len(policy.modifier._step_metadata) == 5
    with pytest.raises(RuntimeError, match='ROS executor'):
        policy.select_action(batch)
    policy.reset()
    assert not policy.modifier._step_metadata
    assert not hasattr(policy.modifier, '_action_queue')
    policy.set_rl_mode()
    assert policy.predict_action_chunk(batch).shape == (1, 5, 12)
