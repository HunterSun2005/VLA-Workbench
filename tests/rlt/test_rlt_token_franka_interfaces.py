from __future__ import annotations

from types import SimpleNamespace

import pytest

from vla.adapters.lerobot.policies.configuration_rlt_token import RLTokenConfig
from vla.adapters.lerobot.policies.modeling_rlt_token import RLTokenPolicy


class _PolicyStub:
    def __init__(self, config: RLTokenConfig, *, patch_size: int = 14) -> None:
        self.config = config
        vision_config = SimpleNamespace(patch_size=patch_size)
        paligemma = SimpleNamespace(config=SimpleNamespace(vision_config=vision_config))
        self.model = SimpleNamespace(
            paligemma_with_expert=SimpleNamespace(paligemma=paligemma)
        )


@pytest.mark.parametrize(
    ("state_dim", "action_dim", "cameras"),
    [
        (10, 10, ["base", "left_wrist"]),
        (20, 20, ["base", "left_wrist", "right_wrist"]),
    ],
)
def test_rl_token_config_accepts_franka_single_and_dual_schema(
    state_dim: int, action_dim: int, cameras: list[str]
) -> None:
    config = RLTokenConfig(
        action_dim=action_dim,
        proprio_dim=state_dim,
        camera_keys=cameras,
        device="cpu",
    )

    config.validate_features()

    assert config.input_features["observation.state"].shape == (state_dim,)
    assert config.output_features["action"].shape == (action_dim,)
    assert config.tokenizer_path is None


def test_image_token_geometry_is_inferred_for_camera_subset() -> None:
    config = RLTokenConfig(
        action_dim=10,
        proprio_dim=10,
        camera_keys=["base", "left_wrist"],
        active_camera_indices=[1],
        image_resolution=(224, 224),
        device="cpu",
    )
    stub = _PolicyStub(config)

    total = RLTokenPolicy._compute_num_image_tokens(stub, stub)

    assert config.num_per_camera == 256
    assert total == 512


def test_invalid_camera_subset_index_is_rejected() -> None:
    config = RLTokenConfig(
        action_dim=10,
        proprio_dim=10,
        camera_keys=["base", "left_wrist"],
        active_camera_indices=[2],
        device="cpu",
    )
    stub = _PolicyStub(config)

    with pytest.raises(ValueError, match="invalid indices"):
        RLTokenPolicy._compute_num_image_tokens(stub, stub)
