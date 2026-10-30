from __future__ import annotations

from pathlib import Path

import pytest
from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature

from vla.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
from vla.adapters.lerobot.policies.configuration_rlt_token import RLTokenPolicyConfig
from vla.cli.common import load_training_config


def _write_schema(path: Path) -> None:
    path.write_text(
        """
action_dim: 16
proprio_dim: 30
vla_horizon: 40
chunk_length: 8
cameras:
  - camera_base
  - camera_left_wrist
  - camera_right_wrist
""".lstrip()
    )


def test_common_loader_preserves_yaml_schema(tmp_path: Path) -> None:
    config_path = tmp_path / "franka.yaml"
    _write_schema(config_path)

    config = load_training_config(str(config_path))

    assert config.action_dim == 16
    assert config.proprio_dim == 30
    assert config.vla_horizon == 40
    assert config.chunk_length == 8
    assert config.cameras == [
        "camera_base",
        "camera_left_wrist",
        "camera_right_wrist",
    ]


@pytest.mark.parametrize("config_cls", [RLTokenPolicyConfig, ChunkACPolicyConfig])
def test_policy_schema_can_be_loaded_from_rlt_yaml(config_cls, tmp_path: Path) -> None:
    config_path = tmp_path / "franka.yaml"
    _write_schema(config_path)

    config = config_cls(rlt_config_path=str(config_path), device="cpu")
    config.validate_features()

    assert config.action_dim == 16
    assert config.proprio_dim == 30
    assert config.camera_keys == [
        "camera_base",
        "camera_left_wrist",
        "camera_right_wrist",
    ]
    assert config.input_features["observation.state"].shape == (30,)
    assert config.output_features["action"].shape == (16,)


def test_resolved_schema_checkpoint_does_not_depend_on_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "franka.yaml"
    _write_schema(config_path)
    config = RLTokenPolicyConfig(rlt_config_path=str(config_path), device="cpu")
    config.validate_features()
    checkpoint_dir = tmp_path / "policy"
    config.save_pretrained(checkpoint_dir)
    config_path.unlink()

    loaded = PreTrainedConfig.from_pretrained(checkpoint_dir)

    assert loaded.action_dim == 16
    assert loaded.proprio_dim == 30
    assert loaded.camera_keys == [
        "camera_base",
        "camera_left_wrist",
        "camera_right_wrist",
    ]


@pytest.mark.parametrize("config_cls", [RLTokenPolicyConfig, ChunkACPolicyConfig])
def test_concrete_config_loader_accepts_saved_type_discriminator(config_cls, tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / config_cls.__name__
    config = config_cls(
        device="cpu",
        action_dim=10,
        proprio_dim=10,
        camera_keys=["base", "left_wrist"],
    )
    config.validate_features()
    config.save_pretrained(checkpoint_dir)

    loaded = config_cls.from_pretrained(checkpoint_dir, local_files_only=True)

    assert isinstance(loaded, config_cls)
    assert loaded.type == config.type
    assert loaded.action_dim == 10
    assert loaded.proprio_dim == 10
    assert loaded.camera_keys == ["base", "left_wrist"]


def test_rl_token_schema_is_inferred_from_dataset_features() -> None:
    cameras = ["camera_base", "camera_left_wrist", "camera_right_wrist"]
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(30,)),
        **{
            f"observation.images.{camera}": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 480, 640)
            )
            for camera in cameras
        },
    }
    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(16,)),
    }
    config = RLTokenPolicyConfig(
        input_features=input_features,
        output_features=output_features,
        device="cpu",
    )

    config.validate_features()

    assert config.action_dim == 16
    assert config.proprio_dim == 30
    assert config.camera_keys == cameras


def test_explicit_schema_mismatch_is_rejected() -> None:
    config = RLTokenPolicyConfig(
        action_dim=12,
        proprio_dim=30,
        camera_keys=["camera_base"],
        input_features={
            "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(30,)),
            "observation.images.camera_base": PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, 480, 640)
            ),
        },
        output_features={
            "action": PolicyFeature(type=FeatureType.ACTION, shape=(16,)),
        },
        device="cpu",
    )

    with pytest.raises(ValueError, match="does not match"):
        config.validate_features()
