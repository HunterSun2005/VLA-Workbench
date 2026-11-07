from __future__ import annotations

import copy
import logging
from typing import Any

import torch
from torch import Tensor
from typing_extensions import Unpack

from lerobot.policies.pretrained import ActionSelectKwargs, PreTrainedPolicy
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from vla.adapters.lerobot.policies.action_modifier import PrefixOutputCapture, RLTActionModifier
from vla.adapters.lerobot.policies.configuration_rlt_ac import ChunkACConfig
from vla.adapters.lerobot.policies.configuration_rlt_token import RLTokenConfig
from vla.adapters.lerobot.policies.modeling_rlt_token import RLTokenPolicy
from vla.adapters.lerobot.checkpoint_contract import (
    assert_same_fingerprint,
    fingerprint_rl_token_checkpoint,
    fingerprint_sft_checkpoint,
)
from vla.core.actor import ChunkActor
from vla.core.critic import TwinCritic
from vla.core.losses import actor_loss, critic_loss
from vla.core.handover_checkpoint import load_handover_classifier
from vla.core.phase_controller import PhaseController
from vla.core.utils import soft_update

log = logging.getLogger(__name__)

class ChunkACPolicy(PreTrainedPolicy):
    """Chunk-level TD3+BC offline RL policy with VLA reference.

    Deployment lazily loads the frozen RL Token + pi0.5 backbone into
    ``_rl_token_policy``. Offline AC training consumes cached state/reference
    tensors, so loading the backbone there would only waste startup time and
    accelerator memory. The backbone remains outside ``nn.Module`` tracking and
    therefore does not land in the AC safetensors checkpoint.

    Trainable modules (registered as nn submodules → in safetensors + optimizer):
      * `actor`: ChunkActor — refines or replaces the VLA reference chunk.
      * `critic`: TwinCritic — twin Q networks.
      * `target_critic`: TwinCritic — Polyak-averaged copy of critic.

    forward(batch) returns one scalar loss without advancing training state.
    The native trainer calls on_optimizer_step after successful updates.
    Actor is updated every `actor_update_interval` successful steps. UTD=k is achieved by
    setting lerobot-train `--steps` to outer*k.
    """

    config_class = ChunkACConfig
    name = "rlt_ac"

    def __init__(
        self,
        config: ChunkACConfig,
        dataset_stats: dict[str, dict[str, Tensor]] | None = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(config, *args, **kwargs)
        from vla.adapters.lerobot.action_runtime import ActionRuntime
        self.action_runtime = ActionRuntime(config.action_contract) if config.action_contract else None
        config.validate_features()
        self.config: ChunkACConfig = config

        rl_token_config = self._load_rl_token_config()
        object.__setattr__(self, "_rl_token_config", rl_token_config)
        object.__setattr__(self, "_rl_token_policy", None)
        self._validate_rl_token_arch_config(rl_token_config)

        state_dim = config.rl_token_dim + config.proprio_dim
        chunk_dim = config.chunk_length * config.action_dim

        self.actor = ChunkActor(
            state_dim=state_dim,
            chunk_dim=chunk_dim,
            hidden_dim=config.actor_hidden_dim,
            num_layers=config.actor_num_layers,
            fixed_std=config.actor_fixed_std,
            ref_dropout_p=config.actor_ref_dropout_p,
            activation=config.actor_activation,
            layer_norm=config.actor_layer_norm,
            residual=config.actor_residual,
        )
        self.critic = TwinCritic(
            state_dim=state_dim,
            chunk_dim=chunk_dim,
            hidden_dim=config.critic_hidden_dim,
            num_layers=config.critic_num_layers,
            activation=config.critic_activation,
            layer_norm=config.critic_layer_norm,
            residual=config.critic_residual,
        )
        self.target_critic = copy.deepcopy(self.critic)
        for p in self.target_critic.parameters():
            p.requires_grad = False
        self.target_critic.eval()

        # Persistent step counter — survives ckpt save/load.
        self.register_buffer("_critic_step", torch.zeros((), dtype=torch.long), persistent=True)

        # Deploy-only action modifier, built on the first control request.
        self.modifier: RLTActionModifier | None = None
        # Deploy toggle (set by lerobot_rlt_record). False => the actor sees a
        # zeroed VLA reference chunk in RL phase (mirrors training ref-dropout).
        self.vla_ref: bool = True
        # Online warmup keeps critical-phase labels while executing the frozen
        # VLA. OnlineRLTSession changes this only between episodes.
        self.online_warmup: bool = False
        self._rtc_config: RTCConfig | None = None
        self._vla_rtc_config: RTCConfig | None = None
        self._active_pi05_rtc_config: RTCConfig | None = None
        self._last_phase_changed: bool = False

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def _load_rl_token_config(self) -> RLTokenConfig:
        if not self.config.rl_token_pretrained_path:
            raise ValueError(
                "ChunkACPolicy requires config.rl_token_pretrained_path "
                "pointing at a saved RLTokenPolicy checkpoint dir."
            )
        rl_config = RLTokenConfig.from_pretrained(self.config.rl_token_pretrained_path)
        rl_config.device = self.config.device
        rl_config.vla_dtype = self.config.vla_dtype
        if self.config.vla_pretrained_path:
            rl_config.vla_pretrained_path = self.config.vla_pretrained_path
        if self.config.tokenizer_path is not None:
            rl_config.tokenizer_path = self.config.tokenizer_path
        return rl_config

    def _load_rl_token_policy(self) -> RLTokenPolicy:
        rl_config = self._rl_token_config
        policy = RLTokenPolicy.from_pretrained(
            self.config.rl_token_pretrained_path,
            config=rl_config,
        )
        for p in policy.parameters():
            p.requires_grad = False
        policy.to(self.config.device).eval()
        return policy

    def _validate_rl_token_arch(self, rl_token_policy: RLTokenPolicy) -> None:
        self._validate_rl_token_arch_config(rl_token_policy.config)

    def _validate_rl_token_arch_config(self, rtp_cfg: RLTokenConfig) -> None:
        fields = (
            "action_dim",
            "proprio_dim",
            "camera_keys",
            "rl_token_dim",
            "rl_token_num_rl_tokens",
            "token_pool_size",
            "image_only",
            "active_camera_indices",
            "num_per_camera",
        )
        mismatches = [
            f"{name}: AC={getattr(self.config, name)!r}, RL Token={getattr(rtp_cfg, name)!r}"
            for name in fields
            if getattr(self.config, name) != getattr(rtp_cfg, name)
        ]
        if self.config.chunk_length > rtp_cfg.chunk_size:
            mismatches.append(
                f"chunk_length: AC={self.config.chunk_length} exceeds PI0.5={rtp_cfg.chunk_size}"
            )
        if mismatches:
            raise ValueError("AC/RL Token contract mismatch: " + "; ".join(mismatches))

        contract = self.config.cache_contract
        if contract is not None:
            policy_contract = contract["policy"]
            actual_rl_token = fingerprint_rl_token_checkpoint(
                self.config.rl_token_pretrained_path
            )
            assert_same_fingerprint(
                policy_contract.get("rl_token_contract"),
                actual_rl_token,
                label="RL Token",
            )
            actual_sft = fingerprint_sft_checkpoint(rtp_cfg.vla_pretrained_path)
            assert_same_fingerprint(
                policy_contract.get("sft_contract"),
                actual_sft,
                label="SFT cache",
            )
            assert_same_fingerprint(
                rtp_cfg.sft_checkpoint_contract,
                actual_sft,
                label="SFT RL Token",
            )

    def _ensure_rl_token_policy(self) -> RLTokenPolicy:
        policy = self._rl_token_policy
        if policy is None:
            policy = self._load_rl_token_policy()
            object.__setattr__(self, "_rl_token_policy", policy)
        return policy

    def prepare_for_inference(self) -> None:
        """Load frozen deployment backbones before accepting control requests."""
        self._ensure_rl_token_policy()
        if self.config.phase_mode == "learned":
            self._ensure_modifier()

    # ------------------------------------------------------------------
    # Training: forward
    # ------------------------------------------------------------------

    def _coerce_batch(self, batch: dict[str, Any]) -> dict[str, Tensor]:
        """Convert ChunkTransition-style batch into the dict expected by losses.

        Required keys: state_vec, exec_chunk, ref_chunk, reward_seq,
        next_state_vec, next_ref_chunk, done, actual_steps.
        Adds flattened views: exec_chunk_flat, ref_chunk_flat, next_ref_flat.
        """
        out: dict[str, Tensor] = {}
        for k in (
            "state_vec",
            "exec_chunk",
            "ref_chunk",
            "reward_seq",
            "next_state_vec",
            "next_ref_chunk",
            "done",
            "actual_steps",
        ):
            if k not in batch:
                raise KeyError(f"ChunkACPolicy.forward missing batch key: {k!r}")
            v = batch[k]
            if not isinstance(v, Tensor):
                v = torch.as_tensor(v)
            out[k] = v
        context_keys = ('start_idx', 'valid_mask', 'prefix_mask', 'actor_mask', 'next_start_idx',
                        'next_valid_mask', 'next_prefix_mask', 'next_actor_mask', 'next_exec_chunk',
                        'action_anchor', 'next_action_anchor', 'reward_offsets', 'elapsed_steps', 'reward_valid_mask')
        for key in context_keys:
            if key not in batch:
                raise ValueError(f'training semantics v3 requires {key}')
            out[key] = torch.as_tensor(batch[key])
        out['next_exec_chunk_flat'] = out['next_exec_chunk'].flatten(start_dim=-2)
        out["exec_chunk_flat"] = out["exec_chunk"].flatten(start_dim=-2)
        out["ref_chunk_flat"] = out["ref_chunk"].flatten(start_dim=-2)
        out["next_ref_flat"] = out["next_ref_chunk"].flatten(start_dim=-2)
        return out

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict | None]:
        tx = self._coerce_batch(batch)

        c_loss = critic_loss(
            self.critic,
            self.target_critic,
            self.actor,
            tx,
            gamma=self.config.gamma,
            C=self.config.chunk_length,
            target_q_clip=self.config.target_q_clip,
            projection=self.execution_projection,
        )
        next_step = int(self._critic_step.item()) + 1
        do_actor = (next_step % self.config.actor_update_interval) == 0
        if do_actor:
            a_loss = self._actor_loss_without_critic_grads(tx)
            total = c_loss + a_loss
            info = {
                "loss_critic": c_loss.detach().item(),
                "loss_actor": a_loss.detach().item(),
                **self.loss_metrics,
                "critic_step": int(self._critic_step.item()),
            }
            return total, info

        info = {
            "loss_critic": c_loss.detach().item(),
            "critic_step": int(self._critic_step.item()),
        }
        return c_loss, info

    @torch.no_grad()
    def on_optimizer_step(self) -> None:
        soft_update(self.target_critic, self.critic, self.config.tau)
        self._critic_step += 1

    def execution_projection(self, mu, anchor):
        from vla.actions.projection import ActionProjection
        if self.config.action_contract is None:
            raise ValueError('A/C training requires a versioned action projection contract')
        return ActionProjection(self.config.action_contract).torch(mu, anchor)

    def _actor_loss_without_critic_grads(self, tx: dict[str, Tensor]) -> Tensor:
        critic_params = [p for p in self.critic.parameters() if p.requires_grad]
        for p in critic_params:
            p.requires_grad_(False)
        try:
            self.loss_metrics = {}
            return actor_loss(self.actor, self.critic, tx, beta=self.config.beta,
                              prefix_bc_weight=self.config.prefix_bc_weight,
                              projection=self.execution_projection, metrics=self.loss_metrics)
        finally:
            for p in critic_params:
                p.requires_grad_(True)

    # ------------------------------------------------------------------
    # Inference: predict_action_chunk + select_action
    # ------------------------------------------------------------------

    def _ensure_modifier(self) -> RLTActionModifier:
        if self.modifier is None:
            rl_token_policy = self._ensure_rl_token_policy()
            phase_ctrl = self._build_phase_controller()
            rl_token_module = rl_token_policy.rl_token
            self.modifier = RLTActionModifier(
                rl_token=rl_token_module,
                actor=self.actor,
                phase_ctrl=phase_ctrl,
                chunk_length=self.config.chunk_length,
                action_dim=self.config.action_dim,
                proprio_dim=self.config.proprio_dim,
                chunk_exec_steps=self.config.chunk_exec_steps,
                vla_ref=self.vla_ref,
                online_warmup=self.online_warmup,
            )
            self._prefix_capture = PrefixOutputCapture(
                token_pool_size=self.config.token_pool_size,
                image_only=self.config.image_only,
                num_image_tokens=self._compute_num_image_tokens(),
                num_per_camera=self.config.num_per_camera,
                active_camera_indices=self.config.active_camera_indices,
            )
            self._prefix_capture.attach(rl_token_policy._pi05)
        return self.modifier

    def _apply_phase_mode(self, ctrl: PhaseController) -> None:
        """Pin the phase controller to the phase encoded by phase_mode.

        always_rl -> critical, always_vla -> vla, manual -> left as-is. Must be
        re-applied after every reset(): modifier.reset() drops the controller
        back to VLA, which would otherwise silently disable always_rl from the
        second episode onward (record_loop resets the policy each episode).
        """
        if self.config.phase_mode == "always_rl":
            ctrl.trigger_critical()
        elif self.config.phase_mode == "always_vla":
            ctrl.trigger_vla()

    def _build_phase_controller(self) -> PhaseController:
        # ChunkACConfig.phase_mode (always_rl / always_vla / manual) is
        # validated in the config __post_init__. PhaseController only does
        # manual/learned transitions, so we pin the deploy phase explicitly.
        if self.config.phase_mode == "learned":
            actual_rl_token = fingerprint_rl_token_checkpoint(
                self.config.rl_token_pretrained_path
            )
            classifier, _ = load_handover_classifier(
                self.config.handover_pretrained_path,
                expected_input_dim=self.config.rl_token_dim,
                expected_rl_token_contract=actual_rl_token,
                device=self.config.device,
            )
            return PhaseController(
                mode="learned",
                classifier=classifier,
                enter_threshold=self.config.handover_enter_threshold,
                exit_threshold=self.config.handover_exit_threshold,
                debounce_chunks=self.config.handover_debounce_chunks,
                manual_cooldown_chunks=self.config.handover_manual_cooldown_chunks,
            )
        ctrl = PhaseController(mode="manual")
        self._apply_phase_mode(ctrl)
        return ctrl

    def _compute_num_image_tokens(self) -> int:
        pi05 = self._ensure_rl_token_policy()._pi05
        pwx = pi05.model.paligemma_with_expert
        vision_cfg = pwx.paligemma.config.vision_config
        tokens_per_camera = (self.config.image_resolution[0] // vision_cfg.patch_size) ** 2
        return tokens_per_camera * len(self.config.camera_keys)

    def configure_rtc(self, rtc_config: RTCConfig, fps: float = 30,
                      vla_rtc_config: RTCConfig | None = None) -> None:
        """Configure denoising guidance; the ROS executor supplies every prefix."""
        if fps != 30:
            raise ValueError('the execution contract requires 30 Hz')
        self._rtc_config = rtc_config
        self._vla_rtc_config = vla_rtc_config or rtc_config
        self._active_pi05_rtc_config = None
        self._set_pi05_rtc_config(rtc_config)

    def _rtc_config_for_phase(self, mod: RLTActionModifier) -> RTCConfig:
        if self._rtc_config is None:
            raise RuntimeError("RTC runtime is not configured")
        if not mod.is_rl_phase and self._vla_rtc_config is not None:
            return self._vla_rtc_config
        return self._rtc_config

    def _set_pi05_rtc_config(self, rtc_config: RTCConfig) -> None:
        if self._active_pi05_rtc_config is rtc_config:
            return
        pi05 = self._ensure_rl_token_policy()._pi05
        pi05.config.rtc_config = rtc_config
        pi05.init_rtc_processor()
        self._active_pi05_rtc_config = rtc_config

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor], **kwargs: Unpack[ActionSelectKwargs]) -> Tensor:
        self.eval()
        mod = self._ensure_modifier()
        pi05 = self._ensure_rl_token_policy()._pi05
        if self._rtc_config is not None:
            self._set_pi05_rtc_config(self._rtc_config_for_phase(mod))
        vla_chunk = pi05.predict_action_chunk(batch, **kwargs)
        vla_chunk = vla_chunk[:, :, : self.config.action_dim]
        prefix_tokens = self._prefix_capture.consume()
        proprio = batch["observation.state"][:, : self.config.proprio_dim]
        z_rl = None
        self._last_phase_changed = False
        phase_ctrl = getattr(mod, "phase_ctrl", None)
        if phase_ctrl is not None and phase_ctrl.mode == "learned":
            z_rl = mod.rl_token.encode(prefix_tokens)
            previous_phase = phase_ctrl.phase
            phase_ctrl.update(z_rl)
            self._last_phase_changed = phase_ctrl.phase != previous_phase
            if self._last_phase_changed:
                log.info(
                    "learned handover phase=%s probability=%.4f",
                    phase_ctrl.phase.value,
                    phase_ctrl.last_probability,
                )
        if z_rl is None:
            return mod.compute_chunk(vla_chunk, proprio, prefix_tokens)
        return mod.compute_chunk(vla_chunk, proprio, prefix_tokens, z_rl=z_rl)

    def select_action(self, batch, **kwargs):
        raise RuntimeError('single-action inference is unsupported; use protocol-v2 plans with the ROS executor')

    def reset(self) -> None:
        if self.modifier is not None:
            self.modifier.reset()
            # modifier.reset() resets the phase controller to VLA; re-pin the
            # configured deploy phase so always_rl survives episode resets.
            self._apply_phase_mode(self.modifier.phase_ctrl)

    def set_control_mode(self, mode: str) -> None:
        """Managed inference override; all weights/classifiers must already be loaded."""
        if mode not in {"vla", "rlt", "auto"}:
            raise ValueError("unknown control mode")
        ctrl = self._ensure_modifier().phase_ctrl
        if mode == "auto" and ctrl.classifier is None:
            raise ValueError("automatic control requires a preloaded handover classifier")
        self.config.phase_mode = {"vla": "always_vla", "rlt": "always_rl", "auto": "learned"}[mode]
        ctrl.mode = "learned" if mode == "auto" else "manual"
        self.reset()

    def set_rl_mode(self) -> None:
        self._ensure_modifier().set_rl_mode()

    def set_vla_mode(self) -> None:
        self._ensure_modifier().set_vla_mode()

    def set_online_warmup(self, enabled: bool) -> None:
        enabled = bool(enabled)
        self.online_warmup = enabled
        if self.modifier is not None:
            self.modifier.online_warmup = enabled
            self.modifier.interrupt_chunk()

    def trigger_critical_phase(self) -> None:
        self._ensure_modifier().trigger_critical_phase()

    def interrupt_chunk(self) -> None:
        if self.modifier is not None:
            self.modifier.interrupt_chunk()

    def quiesce_for_online_training(self) -> None:
        """The service PlanGate already serializes inference and training."""
        if self.modifier is not None:
            self.modifier.interrupt_chunk()

    def get_optim_params(self) -> list:
        return [
            {"params": list(self.actor.parameters())},
            {"params": list(self.critic.parameters())},
        ]

    # ------------------------------------------------------------------
    # Device + train-mode plumbing for the stashed backbone
    # ------------------------------------------------------------------

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        if self._rl_token_policy is not None:
            self._rl_token_policy.to(*args, **kwargs)
        return self

    def cuda(self, device=None):
        super().cuda(device)
        if self._rl_token_policy is not None:
            self._rl_token_policy.cuda(device)
        return self

    def cpu(self):
        super().cpu()
        if self._rl_token_policy is not None:
            self._rl_token_policy.cpu()
        return self

    def train(self, mode: bool = True):
        super().train(mode)
        # Frozen backbones always in eval.
        if self._rl_token_policy is not None:
            self._rl_token_policy.eval()
        self.target_critic.eval()
        return self
