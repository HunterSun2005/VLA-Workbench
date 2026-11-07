"""Build schema-4 Replay from installed plans and actual execution, never proposals alone."""
from pathlib import Path
import json
import numpy as np
import torch
from vla.actions import ActionSpec, QuantileNormalizer
from vla.actions.transforms import DeltaActions
from vla.core.online_contract import file_digest
from vla.core.online_replay import ReplayTransitionRecord


def load_execution_events(root, episode_id):
    path = Path(root)/'execution'/f'episode-{episode_id:06d}'
    manifest = json.loads((path/'manifest.json').read_text())
    if manifest.get('schema_version') != 5 or manifest['episode_index'] != episode_id:
        raise ValueError('missing schema-5 episode execution audit')
    events = []
    for name, digest in sorted(manifest['files'].items()):
        if Path(name).name != name or file_digest(path/name) != digest:
            raise ValueError('execution audit checksum mismatch')
        events.extend(json.loads((path/name).read_text())['events'])
    return events


def build_execution_replay(rows, events, contract, *, episode_id, success, encode_endpoint):
    """rows are frame-ordered scalars plus raw actions; encode_endpoint is model-only.

    Rejected/revoked candidates are absent from installs. An interrupted plan is
    usable only when its endpoint is an explicitly marked real task terminal.
    Bootstrap needs a captured commitment at the endpoint, not an empty prefix.
    """
    spec = ActionSpec.from_dict(contract['action_spec'])
    normalizer = QuantileNormalizer(contract['statistics'])
    transform = DeltaActions(spec)
    by_slot = {int(row['slot_id']): (i, row) for i, row in enumerate(rows)}
    if len(by_slot) != len(rows):
        raise ValueError('duplicate execution slots in episode')
    contexts = {int(e[1]['slot_id']): e[1] for e in events if e[0] == 'execution_context'}
    records = []
    for event in events:
        if event[0] != 'install':
            continue
        plan = event[1]
        req = plan['request']
        H, r, d = len(plan['actions']), int(req['frame_stride']), len(req['prefix'])
        if 'actor' not in plan['sources'][d:] and not any(plan['phases'][d:]):
            continue
        start = int(req['start_slot'])
        if start not in by_slot or 'state_vec' not in plan['context']:
            continue
        first_index, first = by_slot[start]
        segment = first['segment_id']
        executed = []
        endpoint = None
        terminal = False
        for slot in range(start, start+H*r+1):
            entry = by_slot.get(slot)
            if entry is None:
                break
            _, row = entry
            if not row['observation_valid'] or row['segment_id'] != segment:
                break
            if slot > start and row.get('task_terminal', 0):
                endpoint, terminal = row, True
                break
            if slot == start+H*r:
                endpoint = row
                break
            if not row['action_valid']:
                break
            action_index = (slot-start)//r
            if action_index >= d and (row['plan_version'] != req['plan_version']
                                      or row['action_index'] != action_index):
                break
            executed.append(row)
        if endpoint is None or not executed or len(executed) <= d*r:
            continue
        if len(executed) != H*r and not terminal:
            continue
        actual_steps = (len(executed)+r-1)//r
        absolute = np.zeros((H, spec.dim), dtype=np.float64)
        for i in range(actual_steps):
            absolute[i] = executed[i*r]['action']
        # Check the fixed prefix against the targets that really reached the controller.
        if not np.allclose(absolute[:d], np.asarray(req['prefix']), atol=1e-6, rtol=0):
            continue
        anchor = np.asarray(req['anchor'])
        normalized = np.zeros_like(absolute)
        normalized[:actual_steps] = normalizer.normalize(transform(absolute[:actual_steps], anchor))
        reference = normalizer.normalize(transform(np.asarray(plan['vla_reference']), anchor))
        reference[actual_steps:] = 0
        valid = np.arange(H) < actual_steps
        prefix = valid & (np.arange(H) < d)
        actor = valid & ~prefix
        next_context = contexts.get(int(endpoint['slot_id']))
        if not terminal and next_context is None:
            continue
        if terminal:
            next_state = np.zeros_like(np.asarray(plan['context']['state_vec']).reshape(-1))
            next_ref, next_exec = np.zeros_like(normalized), np.zeros_like(normalized)
            next_anchor = np.asarray(endpoint['anchor'])
            next_valid = next_prefix = next_actor = np.zeros(H, bool)
            next_d = 0
        else:
            if (next_context['generation'] != req['generation'] or next_context['horizon'] != H
                    or next_context['model_version'] != req['model_version']):
                continue
            next_anchor = np.asarray(next_context['anchor'])
            next_d = next_context['start_idx']
            if not 0 <= next_d < H:
                continue
            next_state, next_ref = encode_endpoint(endpoint, next_context)
            next_exec = np.zeros_like(normalized)
            next_exec[:next_d] = normalizer.normalize(transform(np.asarray(next_context['prefix']), next_anchor))
            next_valid = np.ones(H, bool)
            next_prefix = np.arange(H) < next_d
            next_actor = ~next_prefix
        reward = np.zeros(H*r, dtype=np.float32)
        reward_valid = np.arange(H*r) < len(executed)
        reward[len(executed)-1] = float(terminal and success)
        start_ns = int(executed[0]['command_at_ns'])
        end_ns = int(endpoint['observed_at_ns'])
        if end_ns <= int(executed[-1]['command_at_ns']):
            continue
        offsets = np.zeros(H*r, dtype=np.float64)
        offsets[:len(executed)] = [(int(row['command_at_ns'])-start_ns)*30/1e9 for row in executed]
        ids = np.full(H*r, -1, dtype=np.int64)
        ids[:len(executed)] = [row['slot_id'] for row in executed]
        values = dict(state_vec=np.asarray(plan['context']['state_vec']).reshape(-1),
                      exec_chunk=normalized, absolute_exec_chunk=absolute, action_anchor=anchor,
                      ref_chunk=reference, next_state_vec=next_state, next_ref_chunk=next_ref,
                      next_exec_chunk=next_exec, next_action_anchor=next_anchor,
                      start_idx=d, valid_mask=valid, prefix_mask=prefix, actor_mask=actor,
                      next_start_idx=next_d, next_valid_mask=next_valid,
                      next_prefix_mask=next_prefix, next_actor_mask=next_actor,
                      reward_seq=reward, reward_valid_mask=reward_valid, reward_offsets=offsets,
                      elapsed_steps=(end_ns-start_ns)*30/1e9, endpoint_ns=end_ns, transition_start_ns=start_ns,
                      reward_interval_start_ns=np.array([int(row['command_at_ns']) for row in executed]+[-1]*(H*r-len(executed))),
                      interval_slot_ids=ids, done=float(terminal), actual_steps=actual_steps,
                      source=2 if 'actor' in plan['sources'][d:] else 1, intervention=0., episode_id=episode_id, is_critical=1.,
                      plan_version=req['plan_version'], generation=req['generation'],
                      termination_reason=int(terminal))
        tensors = {k: torch.as_tensor(np.array(v, copy=True)) for k, v in values.items()}
        import hashlib
        for name in ('model_version', 'session_id', 'episode_id'):
            tensors[name+'_digest'] = torch.tensor(list(hashlib.sha256(str(req[name]).encode()).digest()), dtype=torch.uint8)
        tensors['absolute_vla_reference'] = torch.as_tensor(np.asarray(plan['vla_reference']).copy())
        tensors['action_sources'] = torch.tensor([int(executed[i*r].get('collector_policy_id', 2))
                                                  if i < actual_steps else -1 for i in range(H)])
        for key in ('state_vec', 'exec_chunk', 'ref_chunk', 'next_state_vec', 'next_ref_chunk',
                    'next_exec_chunk', 'reward_seq', 'done', 'intervention', 'is_critical'):
            tensors[key] = tensors[key].float()
        records.append(ReplayTransitionRecord(first_index, tensors))
    return records


def validate_execution_transition(value):
    required = {'start_idx', 'valid_mask', 'prefix_mask', 'actor_mask', 'next_start_idx',
                'next_valid_mask', 'next_prefix_mask', 'next_actor_mask', 'next_exec_chunk',
                'action_anchor', 'next_action_anchor', 'absolute_exec_chunk',
                'reward_offsets', 'reward_valid_mask', 'elapsed_steps', 'endpoint_ns',
                'interval_slot_ids', 'plan_version', 'generation', 'termination_reason',
                'transition_start_ns', 'reward_interval_start_ns', 'absolute_vla_reference', 'action_sources',
                'model_version_digest', 'session_id_digest', 'episode_id_digest'}
    if not required <= value.keys():
        raise ValueError('Replay v4 requires complete execution/next-state context')
    H = value['exec_chunk'].shape[0]
    for stem in ('', 'next_'):
        valid, prefix, actor = [value[stem+k].bool() for k in ('valid_mask', 'prefix_mask', 'actor_mask')]
        d = int(value[stem+'start_idx'])
        if (valid.shape != (H,) or prefix.shape != (H,) or actor.shape != (H,) or not 0 <= d <= H
                or (prefix & actor).any() or not torch.equal(prefix | actor, valid)
                or not torch.equal(prefix, valid & (torch.arange(H) < d))):
            raise ValueError('invalid Replay execution masks')
    if not bool(value['done']) and not value['next_valid_mask'].any():
        raise ValueError('nonterminal Replay is missing next execution context')
    if not value['actor_mask'].any():
        raise ValueError('unexecuted candidate cannot produce an RL transition')
    action = value['exec_chunk']
    if action.ndim != 2 or any(value[k].shape != action.shape for k in
            ('ref_chunk', 'absolute_exec_chunk', 'next_exec_chunk', 'next_ref_chunk', 'absolute_vla_reference')):
        raise ValueError('Replay action shapes disagree')
    for item in value.values():
        if not isinstance(item, torch.Tensor) or not torch.isfinite(item).all():
            raise ValueError('Replay requires finite tensor fields')
    valid = value['valid_mask'].bool()
    if int(value['actual_steps']) != int(valid.sum()) or not torch.equal(valid, torch.arange(H) < valid.sum()):
        raise ValueError('Replay execution window must be a contiguous prefix')
    if not bool(value['done']) and not valid.all():
        raise ValueError('interrupted nonterminal execution cannot bootstrap')
    for key in ('model_version_digest', 'session_id_digest', 'episode_id_digest'):
        if value[key].shape != (32,) or value[key].dtype != torch.uint8:
            raise ValueError('Replay requires SHA256 model/session/episode identities')
    if value['action_sources'].shape != (H,):
        raise ValueError('Replay requires per-action sources')
    reward = value['reward_seq']
    mask = value['reward_valid_mask'].bool()
    offsets, ids = value['reward_offsets'], value['interval_slot_ids']
    elapsed = float(value['elapsed_steps'])
    if (reward.ndim != 1 or mask.shape != reward.shape or offsets.shape != reward.shape
            or ids.shape != reward.shape or not mask.any() or elapsed <= 0
            or int(value['endpoint_ns']) <= 0 or (offsets[mask] < 0).any()
            or (offsets[mask] >= elapsed).any() or (offsets[mask].diff() <= 0).any()
            or (ids[mask] < 0).any() or (ids[mask].diff() != 1).any()):
        raise ValueError('invalid causal Replay reward intervals')
    starts = value['reward_interval_start_ns']
    start = int(value['transition_start_ns'])
    if (starts.shape != reward.shape or start < 0 or int(starts[mask][0]) != start
            or not torch.allclose((starts[mask].double()-start)*30/1e9, offsets[mask].double(), atol=1e-6, rtol=0)
            or abs((int(value['endpoint_ns'])-start)*30/1e9-elapsed) > 1e-6):
        raise ValueError('Replay physical timestamp/discount mismatch')


def encode_execution_dataset(*, dataset, episode_id, episode_start, episode_stop, contract,
                             success, pi05, rl_token, preprocessor, capture, chunk_length,
                             proprio_dim, state_indices, task, device, online_mode=None):
    """Use saved request state/reference; only evaluate the constrained bootstrap policy."""
    from vla.core.online_contract import TIMING_FIELDS
    from vla.actions.metadata import require_recording_source
    info = getattr(dataset.meta, 'info', None)
    if info is not None:
        require_recording_source(contract, info.get('recording_contract'))
    rows = []
    for index in range(episode_start, episode_stop):
        raw = dataset.hf_dataset[index]
        fields = (*TIMING_FIELDS, 'collector_policy_id', 'phase', 'is_intervention')
        missing = {f'complementary_info.{k}' for k in fields} - raw.keys()
        if missing:
            raise ValueError(f'schema-5 execution fields missing: {sorted(missing)}')
        row = {k: int(torch.as_tensor(raw['complementary_info.'+k]).reshape(-1)[0]) for k in fields}
        row.update(action=np.asarray(raw['action']), anchor=np.asarray(raw['observation.anchor']),
                   dataset_index=index)
        rows.append(row)
    if online_mode == 'full' and any(r['action_valid'] and r['collector_policy_id'] >= 0 and
            (r['phase'] != 1 or r['is_intervention']) for r in rows):
        raise ValueError('full-episode online RLT requires autonomous critical-phase execution')
    if online_mode == 'single':
        phases = np.asarray([r['phase'] == 1 for r in rows])
        if np.count_nonzero(phases & ~np.r_[False, phases[:-1]]) > 1:
            raise ValueError('online RLT supports only one contiguous critical phase')
    root = getattr(dataset, 'root', getattr(dataset.meta, 'root', None))
    events = load_execution_events(root, episode_id)
    normalizer = QuantileNormalizer(contract['statistics'])
    transform = DeltaActions(ActionSpec.from_dict(contract['action_spec']))
    endpoint_cache = {}

    def encode(row, context):
        index = row['dataset_index']
        if index in endpoint_cache:
            return endpoint_cache[index]
        item = dict(dataset[index])
        state = torch.as_tensor(item['observation.state'])
        if state_indices is not None:
            state = state[list(state_indices)]
        item['observation.state'] = state
        item['observation.anchor'] = torch.as_tensor(context['anchor'])
        item.pop('action_is_pad', None)
        item.pop('action', None)
        batch = {key: value.unsqueeze(0).to(device) if isinstance(value, torch.Tensor) else value
                 for key, value in item.items() if key.startswith('observation.')}
        batch['task'] = [task or item.get('task')]
        processed = preprocessor(batch)
        prefix = normalizer.normalize(transform(np.asarray(context['prefix']), np.asarray(context['anchor'])))
        with torch.no_grad():
            reference = pi05.predict_action_chunk(processed, inference_delay=context['start_idx'],
                         prev_chunk_left_over=torch.as_tensor(prefix, device=device, dtype=torch.float32))
            latent = rl_token.encode(capture.consume().float())
            state_vec = torch.cat((latent, processed['observation.state'][:, :proprio_dim]), -1)
        result = (state_vec[0].cpu().numpy(), reference[0, :chunk_length, :normalizer.low.size].cpu().numpy())
        endpoint_cache[index] = result
        return result

    records = build_execution_replay(rows, events, contract, episode_id=episode_id,
                                     success=success, encode_endpoint=encode)
    # Count unique recorded intervals, independent of overlapping Replay windows.
    critical = sum(bool(r['action_valid']) and r['phase'] == 1 and r['plan_version'] >= 0 for r in rows)
    return records, critical
