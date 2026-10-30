from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.utils.constants import ACTION, OBS_IMAGES, OBS_STATE


PolicyConfigT = TypeVar("PolicyConfigT", bound=PreTrainedConfig)


def load_concrete_policy_config(
    config_class: type[PolicyConfigT],
    pretrained_name_or_path: str | Path,
    **kwargs: Any,
) -> PolicyConfigT:
    """Load a saved choice config through LeRobot's registered base class.

    LeRobot 0.5.1 writes a ``type`` discriminator into config.json. Calling its
    inherited loader on a concrete dataclass tries to decode that discriminator
    as a dataclass field and fails before the loader can remove it.
    """
    ensure_registered = getattr(config_class, "ensure_registered", None)
    if callable(ensure_registered):
        ensure_registered()
    config = PreTrainedConfig.from_pretrained(pretrained_name_or_path, **kwargs)
    if not isinstance(config, config_class):
        raise TypeError(
            f"checkpoint contains {type(config).__name__}, expected {config_class.__name__}"
        )
    return config


def apply_rlt_schema_file(config) -> None:
    """Fill unresolved policy schema fields from an VLA YAML config."""
    config_path = getattr(config, "rlt_config_path", None)
    unresolved = (
        config.action_dim is None
        or config.proprio_dim is None
        or config.camera_keys is None
    )
    if not config_path or not unresolved:
        return

    from vla.core.config import RLTConfig

    schema = RLTConfig.from_yaml(Path(config_path))
    if config.action_dim is None:
        config.action_dim = schema.action_dim
    if config.proprio_dim is None:
        config.proprio_dim = schema.proprio_dim
    if config.camera_keys is None:
        config.camera_keys = list(schema.cameras)


def resolve_policy_schema(config) -> None:
    """Resolve dimensions/cameras from explicit config or LeRobot features.

    Explicit values take precedence. Missing values are inferred from dataset
    metadata populated by LeRobot's policy factory. Transition-cache training
    has no raw dataset features, so its schema must be supplied explicitly or
    through ``rlt_config_path``.
    """
    input_features = config.input_features or {}
    output_features = config.output_features or {}
    state_feature = input_features.get(OBS_STATE)
    action_feature = output_features.get(ACTION)

    config.proprio_dim = _resolve_dim(
        name="proprio_dim",
        configured=config.proprio_dim,
        feature=state_feature,
        feature_key=OBS_STATE,
    )
    config.action_dim = _resolve_dim(
        name="action_dim",
        configured=config.action_dim,
        feature=action_feature,
        feature_key=ACTION,
    )

    if config.proprio_dim > config.max_state_dim:
        raise ValueError(
            f"proprio_dim ({config.proprio_dim}) exceeds max_state_dim ({config.max_state_dim})"
        )
    if config.action_dim > config.max_action_dim:
        raise ValueError(
            f"action_dim ({config.action_dim}) exceeds max_action_dim ({config.max_action_dim})"
        )

    if config.camera_keys is None:
        prefix = f"{OBS_IMAGES}."
        config.camera_keys = [
            key.removeprefix(prefix)
            for key, feature in input_features.items()
            if key.startswith(prefix) and feature.type is FeatureType.VISUAL
        ]

    if not config.camera_keys:
        raise ValueError(
            "camera_keys is empty and no visual features were available to infer it; "
            "set camera_keys or rlt_config_path"
        )

    if input_features:
        missing = [
            key for key in config.camera_keys if f"{OBS_IMAGES}.{key}" not in input_features
        ]
        if missing:
            raise ValueError(f"configured camera_keys are missing from dataset features: {missing}")


def populate_policy_features(config) -> None:
    """Add schema features when no raw LeRobot dataset metadata is available."""
    if not config.input_features:
        config.input_features = {}
    if OBS_STATE not in config.input_features:
        config.input_features[OBS_STATE] = PolicyFeature(
            type=FeatureType.STATE, shape=(config.proprio_dim,)
        )
    for camera_key in config.camera_keys:
        image_key = f"{OBS_IMAGES}.{camera_key}"
        if image_key not in config.input_features:
            config.input_features[image_key] = PolicyFeature(
                type=FeatureType.VISUAL, shape=(3, *config.image_resolution)
            )

    if not config.output_features:
        config.output_features = {}
    if ACTION not in config.output_features:
        config.output_features[ACTION] = PolicyFeature(
            type=FeatureType.ACTION, shape=(config.action_dim,)
        )


def _resolve_dim(
    *,
    name: str,
    configured: int | None,
    feature: PolicyFeature | None,
    feature_key: str,
) -> int:
    inferred = int(feature.shape[-1]) if feature is not None else None
    if configured is None:
        if inferred is None:
            raise ValueError(
                f"{name} is not configured and cannot be inferred from {feature_key!r}; "
                f"set {name} or rlt_config_path"
            )
        return inferred
    if configured <= 0:
        raise ValueError(f"{name} must be positive, got {configured}")
    if inferred is not None and configured != inferred:
        raise ValueError(
            f"configured {name} ({configured}) does not match {feature_key!r} feature size ({inferred})"
        )
    return configured
