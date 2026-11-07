from __future__ import annotations

import torch
import torch.nn.functional as F

from vla.core.actor import ChunkActor
from vla.core.critic import TwinCritic
from vla.core.utils import compute_discount_vector


def _valid_steps(actual_steps: torch.Tensor, batch_size: int, C: int) -> torch.Tensor:
    if actual_steps.shape != (batch_size,) or actual_steps.dtype == torch.bool or actual_steps.is_complex():
        raise ValueError("actual_steps must have shape (batch_size,)")
    if not torch.isfinite(actual_steps).all() or not (
        (actual_steps >= 1) & (actual_steps <= C) & (actual_steps == actual_steps.floor())
    ).all():
        raise ValueError("actual_steps must contain integers in [1, chunk_length]")
    return torch.arange(C, device=actual_steps.device)[None, :] < actual_steps[:, None]


def _action_mask(batch: dict[str, torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    if ref.ndim != 2:
        raise ValueError("flattened action chunks must have shape (batch_size, chunk_dims)")
    steps = batch.get("actual_steps")
    if steps is None:
        return torch.ones_like(ref, dtype=torch.bool)
    if "reward_seq" not in batch or batch["reward_seq"].ndim != 2:
        raise ValueError("actual_steps requires a (batch_size, chunk_length) reward_seq")
    B, C = batch["reward_seq"].shape
    if B != ref.shape[0] or C <= 0 or ref.shape[1] % C:
        raise ValueError("action and reward chunk dimensions do not match")
    return _valid_steps(steps.to(ref.device), B, C).repeat_interleave(ref.shape[1] // C, dim=1)


def execution_masks(batch, ref, *, next_state=False):
    prefix = 'next_' if next_state else ''
    keys = {prefix+name for name in ('valid_mask', 'prefix_mask', 'actor_mask')}
    if not keys & batch.keys():
        valid = _action_mask(batch, ref) if not next_state else torch.ones_like(ref, dtype=torch.bool)
        return valid, torch.zeros_like(valid), valid
    if not keys <= batch.keys():
        raise ValueError('incomplete execution masks')
    masks = []
    for name in ('valid_mask', 'prefix_mask', 'actor_mask'):
        mask = batch[prefix+name].to(ref.device)
        if mask.ndim != 2 or mask.shape[0] != ref.shape[0] or ref.shape[1] % mask.shape[1]:
            raise ValueError('invalid execution mask shape')
        if not ((mask == 0) | (mask == 1)).all():
            raise ValueError('execution masks must be binary')
        masks.append(mask.bool().repeat_interleave(ref.shape[1]//mask.shape[1], dim=1))
    valid, committed, actor = masks
    if (committed & actor).any() or not torch.equal(committed | actor, valid):
        raise ValueError('prefix and Actor masks must partition valid execution')
    return valid, committed, actor


def assembled_action(mu, batch, *, next_state=False, projection=None):
    valid, prefix, actor = execution_masks(batch, mu, next_state=next_state)
    name = 'next_exec_chunk_flat' if next_state else 'exec_chunk_flat'
    executed = batch.get(name)
    if prefix.any() and executed is None:
        raise ValueError('committed prefix requires actual execution context')
    if executed is None:
        executed = torch.zeros_like(mu)
    if executed.shape != mu.shape:
        raise ValueError('execution context shape mismatch')
    projected = (projection(mu, batch['next_action_anchor' if next_state else 'action_anchor'])
                 if projection is not None else mu.clamp(-1, 1))
    return torch.where(prefix, executed.detach(), torch.where(actor, projected, torch.zeros_like(mu)))


def discounted_chunk_return(
    reward_seq: torch.Tensor, gamma: float, actual_steps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute discounted return over a chunk of rewards.

    Args:
        reward_seq: (B, C) rewards for each timestep (padded with 0 beyond actual_steps)
        gamma: discount factor
        actual_steps: (B,) number of valid steps per chunk (if None, assume all C are valid)

    Returns:
        (B, 1) discounted return
    """
    C = reward_seq.shape[1]
    if actual_steps is not None:
        mask = _valid_steps(actual_steps.to(reward_seq.device), reward_seq.shape[0], C)
        reward_seq = reward_seq.masked_fill(~mask, 0)
    discounts = compute_discount_vector(gamma, C, device=reward_seq.device)
    return (reward_seq * discounts.unsqueeze(0)).sum(dim=1, keepdim=True)


def critic_loss(
    critic: TwinCritic,
    target_critic: TwinCritic,
    actor: ChunkActor,
    batch: dict[str, torch.Tensor],
    gamma: float,
    C: int,
    target_q_clip: float | None = 100.0,
    projection=None,
) -> torch.Tensor:
    """TD3-style chunk-level TD loss with correct truncated-chunk handling.

    Uses actual_steps to compute the correct bootstrap exponent gamma^k
    instead of always using gamma^C.
    """
    x = batch["state_vec"]
    a = batch["exec_chunk_flat"]
    x_next = batch["next_state_vec"]
    ref_next = batch["next_ref_flat"]
    reward_seq = batch["reward_seq"]
    done = batch["done"]
    actual_steps = batch.get("actual_steps")

    physical_timing = 'reward_offsets' in batch or 'elapsed_steps' in batch
    if physical_timing and (actual_steps is None or not {'reward_offsets', 'elapsed_steps'} <= batch.keys()):
        raise ValueError('physical timing requires reward_offsets, elapsed_steps and actual_steps')
    if reward_seq.ndim != 2 or (not physical_timing and reward_seq.shape[1] != C):
        raise ValueError('reward chunk dimensions do not match chunk length')

    with torch.no_grad():
        # Use deterministic mean for target action (TD3-style), clamped to [-1,1]
        mu_next, _ = actor.forward(x_next, ref_next)
        mu_next = assembled_action(mu_next, batch, next_state=True, projection=projection)
        q_next = target_critic.min_q(x_next, mu_next)
        if target_q_clip is not None and target_q_clip > 0:
            q_next = q_next.clamp(-target_q_clip, target_q_clip)
        if physical_timing:
            offsets = batch['reward_offsets'].to(reward_seq.device)
            elapsed = batch['elapsed_steps'].to(reward_seq.device)
            actual_steps = actual_steps.to(reward_seq.device)
            mask = (batch['reward_valid_mask'].to(reward_seq.device).bool() if 'reward_valid_mask' in batch
                    else _valid_steps(actual_steps.to(reward_seq.device), reward_seq.shape[0], C))
            if (offsets.shape != reward_seq.shape or elapsed.shape != actual_steps.shape
                    or offsets.dtype == torch.bool or elapsed.dtype == torch.bool
                    or offsets.is_complex() or elapsed.is_complex()):
                raise ValueError('invalid physical reward timing dimensions')
            if not torch.isfinite(offsets).all() or not torch.isfinite(elapsed).all() or (elapsed <= 0).any():
                raise ValueError('invalid elapsed control steps')
            if ((offsets < 0) | ((offsets >= elapsed[:, None]) & mask)).any():
                raise ValueError('invalid reward control-frame offset')
            r = (reward_seq.masked_fill(~mask, 0) * gamma ** offsets).sum(1, keepdim=True)
        else:
            r = discounted_chunk_return(reward_seq, gamma, actual_steps)

        # Bootstrap with gamma^k where k = actual steps executed
        if actual_steps is not None:
            bootstrap_exp = batch.get("elapsed_steps", actual_steps).to(q_next.device).unsqueeze(-1).float()
        else:
            bootstrap_exp = torch.full_like(done.unsqueeze(-1), C, dtype=torch.float32)
        bootstrap = (gamma ** bootstrap_exp) * (1.0 - done.unsqueeze(-1)) * q_next
        target = (r + bootstrap).to(a.dtype)

    valid, _, _ = execution_masks(batch, a)
    q1, q2 = critic(x, a.masked_fill(~valid, 0))
    return F.mse_loss(q1, target) + F.mse_loss(q2, target)


def actor_loss(actor, critic, batch, beta, prefix_bc_weight=None, *, projection=None, metrics=None):
    """Q gradients only reach the effective suffix; prefix BC trains raw output.

    Both BC terms sum action dimensions then average the batch. No mask-count
    rescaling: a one-action suffix has exactly one action's regularization.
    """
    prefix_bc_weight = beta if prefix_bc_weight is None else prefix_bc_weight
    if beta < 0 or prefix_bc_weight < 0:
        raise ValueError('BC weights must be nonnegative')
    x, ref = batch['state_vec'], batch['ref_chunk_flat']
    valid, prefix, suffix = execution_masks(batch, ref)
    mu, _ = actor.forward(x, ref, training=True)
    if mu.shape != ref.shape:
        raise ValueError('Actor output and reference chunk dimensions do not match')
    q = critic.min_q(x, assembled_action(mu, batch, projection=projection))
    suffix_bc = ((mu-ref.clamp(-1, 1)).masked_fill(~suffix, 0)).square().sum(-1).mean()
    target = batch.get('exec_chunk_flat', ref).detach()
    prefix_bc = ((mu-target).masked_fill(~prefix, 0)).square().sum(-1).mean()
    if metrics is not None:
        metrics.update(bc_suffix=float(suffix_bc.detach()), bc_prefix=float(prefix_bc.detach()),
                       prefix_bc_weight=float(prefix_bc_weight), beta=float(beta),
                       prefix_dims=int(prefix.sum()), actor_dims=int(suffix.sum()), valid_dims=int(valid.sum()))
    return -q.mean()+beta*suffix_bc+prefix_bc_weight*prefix_bc
