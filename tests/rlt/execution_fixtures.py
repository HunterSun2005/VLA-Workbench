"""Explicit d=0 execution context for small network and cache unit fixtures."""
import torch
from vla.core.training_contract import TRAINING_SEMANTICS_VERSION, REFERENCE_POLICY


def semantics():
    return {'training_semantics_version': TRAINING_SEMANTICS_VERSION,
            'annotation_policy': {'ref_chunk': REFERENCE_POLICY}}


def execution_context(tx, *, start_slot=0):
    action = tx['exec_chunk']
    H, D = action.shape[-2:]
    shape = action.shape[:-2]
    mask = torch.ones((*shape, H), dtype=torch.bool)
    zero = torch.zeros_like(mask)
    scalar = torch.zeros(shape, dtype=torch.int64)
    done = torch.as_tensor(tx['done']).bool()
    next_valid = mask & ~done[..., None]
    tx.update(start_idx=scalar.clone(), valid_mask=mask, prefix_mask=zero.clone(), actor_mask=mask.clone(),
              next_start_idx=scalar.clone(), next_exec_chunk=torch.zeros_like(action),
              next_valid_mask=next_valid, next_prefix_mask=zero.clone(), next_actor_mask=next_valid.clone(),
              action_anchor=torch.zeros((*shape, D)), next_action_anchor=torch.zeros((*shape,D)),
              absolute_exec_chunk=action.clone(), reward_offsets=torch.arange(H).expand(*shape,H).clone(),
              reward_valid_mask=mask.clone(), elapsed_steps=torch.full(shape, float(H)),
              endpoint_ns=torch.full(shape, 1_000_000_000+round(H*1e9/30), dtype=torch.int64),
              interval_slot_ids=torch.arange(start_slot,start_slot+H).expand(*shape,H).clone(),
              plan_version=scalar.clone(), generation=scalar.clone(), termination_reason=done.long())
    tx['transition_start_ns'] = torch.full(shape, 1_000_000_000, dtype=torch.int64)
    tx['reward_interval_start_ns'] = (1_000_000_000+(torch.arange(H).double()*1e9/30).round().long()).expand(*shape, H).clone()
    tx['absolute_vla_reference'] = tx['ref_chunk'].clone()
    tx['action_sources'] = torch.zeros((*shape,H), dtype=torch.int64)
    for key in ('model_version_digest', 'session_id_digest', 'episode_id_digest'):
        tx[key] = torch.zeros((*shape,32), dtype=torch.uint8)
    return tx


def human_rows(length):
    return {i: dict(slot_id=i, segment_id=0, action_valid=int(i<length-1), observation_valid=1,
                    supervision_valid=int(i<length-1), collector_policy_id=0,
                    task_terminal=int(i==length-1), command_at_ns=1_000_000_000+round(i*1e9/30),
                    observed_at_ns=1_000_000_000+round(i*1e9/30)) for i in range(length)}


def encoded_anchors(encoded):
    for value in encoded.values():
        value['action_anchor'] = torch.zeros(value['exec_chunk'].shape[-1])
        value['absolute_exec_chunk'] = value['exec_chunk'].clone()
    return encoded
