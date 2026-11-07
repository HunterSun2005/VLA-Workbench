from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.optim.optimizers import AdamWConfig, OptimizerConfig
from lerobot.optim.schedulers import LRSchedulerConfig
from vla.adapters.lerobot.policies.configuration_schema import (
    apply_rlt_schema_file,
    load_concrete_policy_config,
    populate_policy_features,
    resolve_policy_schema,
)


@dataclass
class ChunkACConfig(PreTrainedConfig):
    """Config for the chunk-level TD3+BC actor-critic policy on top of RL Token.

    Train via vla-train-actor-critic with a ChunkTransitionDataset; deploy via
    make_policy + lerobot-record. forward(batch) returns a single scalar TD3+BC
    loss combining critic MSE + (every actor_update_interval steps) the actor
    Q-maximization + BC reg. Target critic is soft-updated with tau after every
    critic step. UTD ratio is achieved by setting outer --steps=outer*utd
    (successful optimizer updates advance the counter, not forward calls).
    """

    # --- VLA + RL Token backbones (frozen, not serialized) ---
    action_contract: dict | None = None
    vla_pretrained_path: str = ""
    vla_revision: str | None = None
    vla_dtype: str = "bfloat16"
    rl_token_pretrained_path: str = ""
    cache_manifest_path: str | None = None
    cache_contract: dict[str, Any] | None = None

    # --- RL Token arch (must match the loaded RLTokenPolicy ckpt) ---
    rl_token_dim: int = 2048
    rl_token_num_rl_tokens: int = 1
    token_pool_size: int = 0
    image_only: bool = False
    active_camera_indices: list[int] | None = None
    num_per_camera: int = 0

    # --- Actor ---
    actor_hidden_dim: int = 256
    actor_num_layers: int = 2
    actor_fixed_std: float = 0.05
    actor_ref_dropout_p: float = 0.5
    actor_activation: str = "relu"
    actor_layer_norm: bool = False
    actor_residual: bool = False

    # --- Critic + target ---
    critic_hidden_dim: int = 256
    critic_num_layers: int = 2
    critic_activation: str = "relu"
    critic_layer_norm: bool = False
    critic_residual: bool = False

    # --- TD3+BC hyperparams ---
    gamma: float = 0.99
    beta: float = 0.3
    prefix_bc_weight: float | None = None
    tau: float = 0.005
    utd_ratio: int = 5
    actor_update_interval: int = 2
    target_q_clip: float = 100.0

    # --- Shapes ---
    chunk_length: int = 10
    action_dim: int | None = None
    proprio_dim: int | None = None

    # --- Deploy ---
    chunk_exec_steps: int = 25
    phase_mode: str = "always_rl"
    deterministic: bool = True
    handover_pretrained_path: str = ""
    handover_enter_threshold: float = 0.8
    handover_exit_threshold: float = 0.2
    handover_debounce_chunks: int = 2
    handover_manual_cooldown_chunks: int = 3

    # --- Observation mapping (for deploy preprocessor) ---
    camera_keys: list[str] | None = None
    rlt_config_path: str | None = None

    # --- Normalization MATCHES PI05 (deploy parity) ---
    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.QUANTILES,
            "ACTION": NormalizationMode.QUANTILES,
        }
    )

    # --- pi05 proxy fields ---
    max_state_dim: int = 32
    max_action_dim: int = 32
    image_resolution: tuple[int, int] = (224, 224)
    tokenizer_max_length: int = 200
    tokenizer_path: str | None = None

    @classmethod
    def ensure_registered(cls) -> None:
        import lerobot.policies  # noqa: F401

        if "rlt_ac" not in PreTrainedConfig.get_known_choices():
            PreTrainedConfig.register_subclass("rlt_ac")(cls)

    @classmethod
    def from_pretrained(cls, pretrained_name_or_path, **kwargs):
        return load_concrete_policy_config(cls, pretrained_name_or_path, **kwargs)

    def __post_init__(self) -> None:
        self.prefix_bc_weight = self.beta if self.prefix_bc_weight is None else self.prefix_bc_weight
        if self.prefix_bc_weight < 0:
            raise ValueError('prefix_bc_weight must be nonnegative')
        from pathlib import Path
        import json
        contract_path = Path(self.vla_pretrained_path) / 'config.json'
        if contract_path.is_file():
            source_contract = json.loads(contract_path.read_text()).get('action_contract')
            if self.action_contract is not None and source_contract != self.action_contract:
                raise ValueError('RLT action contract differs from SFT checkpoint')
            self.action_contract = source_contract
            if source_contract is None or source_contract.get('version') != 2:
                raise ValueError('A/C requires SFT model action contract v2')
        self.ensure_registered()
        self._apply_cache_contract()
        if self.action_contract is not None:
            if self.action_contract.get('version') != 2:
                raise ValueError('A/C requires model action contract v2')
            reference_horizon = self.action_contract['statistics']['contract']['sampling']['chunk_length']
            if not 1 <= self.chunk_length <= reference_horizon:
                raise ValueError('Actor horizon must fit the reference model sampling horizon')
        apply_rlt_schema_file(self)
        super().__post_init__()
        if self.phase_mode not in ("always_rl", "always_vla", "manual", "learned"):
            raise ValueError(
                "phase_mode must be 'always_rl', 'always_vla', 'manual', or "
                f"'learned', got {self.phase_mode!r}"
            )
        if self.phase_mode == "learned" and not self.handover_pretrained_path:
            raise ValueError("phase_mode='learned' requires handover_pretrained_path")
        if not 0 <= self.handover_exit_threshold < self.handover_enter_threshold <= 1:
            raise ValueError("handover thresholds must satisfy 0 <= exit < enter <= 1")
        if self.handover_debounce_chunks <= 0:
            raise ValueError("handover_debounce_chunks must be positive")
        if self.handover_manual_cooldown_chunks < 0:
            raise ValueError("handover_manual_cooldown_chunks cannot be negative")

    def _apply_cache_contract(self) -> None:
        contract = self.cache_contract
        if self.cache_manifest_path:
            from vla.adapters.lerobot.policies.cache_manifest import load_cache_manifest

            manifest_path = Path(self.cache_manifest_path).expanduser()
            if manifest_path.exists():
                manifest = load_cache_manifest(manifest_path)
                if manifest.get("status") != "complete":
                    raise ValueError("transition cache manifest is not complete")
                loaded = manifest["contract"]
                if contract is not None and contract != loaded:
                    raise ValueError("embedded AC cache contract does not match cache_manifest_path")
                contract = loaded
                self.cache_contract = loaded
            elif contract is None:
                raise FileNotFoundError(f"cache manifest does not exist: {manifest_path}")
        if contract is None:
            return

        policy = contract["policy"]
        if policy.get("action_contract") is not None:
            self._contract_value("action_contract", policy["action_contract"])
        elif self.action_contract is not None:
            raise ValueError("legacy cache has no action contract; rebuild it")
        cache = contract["cache"]
        self._contract_value("action_dim", policy["action_dim"])
        self._contract_value("proprio_dim", policy["proprio_dim"])
        self._contract_value("camera_keys", list(policy["camera_keys"]))
        self._contract_value("chunk_length", cache["chunk_length"], default=10)
        self._contract_value("rl_token_dim", policy["rl_token_dim"], default=2048)
        self._contract_value(
            "rl_token_num_rl_tokens", policy["rl_token_num_rl_tokens"], default=1
        )
        self._contract_value("token_pool_size", policy["token_pool_size"], default=0)
        self._contract_value("image_only", policy["image_only"], default=False)
        self._contract_value("active_camera_indices", policy["active_camera_indices"])
        self._contract_value("num_per_camera", policy["num_per_camera"], default=0)
        self._contract_path("rl_token_pretrained_path", policy["rl_token_path"])
        self._contract_path("vla_pretrained_path", policy["sft_path"])

    def _contract_value(self, name: str, expected: Any, default: Any = None) -> None:
        current = getattr(self, name)
        if current is None or current == default:
            setattr(self, name, expected)
            return
        if current != expected:
            raise ValueError(
                f"configured {name}={current!r} does not match transition cache {expected!r}"
            )

    def _contract_path(self, name: str, recorded: str) -> None:
        """Allow checkpoint relocation; model construction verifies content hashes."""
        current = getattr(self, name)
        if not current:
            setattr(self, name, recorded)

    @property
    def type(self) -> str:
        return "rlt_ac"

    def validate_features(self) -> None:
        resolve_policy_schema(self)
        populate_policy_features(self)

    def get_optimizer_preset(self) -> OptimizerConfig:
        return AdamWConfig(lr=3e-4, weight_decay=0.0, grad_clip_norm=1.0)

    def get_scheduler_preset(self) -> LRSchedulerConfig | None:
        return None

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> list[int]:
        return list(range(self.chunk_length))

    @property
    def reward_delta_indices(self) -> None:
        return None


ChunkACConfig.ensure_registered()
