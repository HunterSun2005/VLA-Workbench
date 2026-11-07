"""Candidate-only model inference using the executor's fixed absolute prefix."""
from copy import deepcopy
from contextlib import nullcontext
import numpy as np
import torch
from vla.control.execution import PlanRequest, PlanResponse
from vla.adapters.lerobot.action_runtime import ActionRuntime


def infer_plan(server, header, arrays, *, sft=False):
    from lerobot.policies.utils import prepare_observation_for_inference
    request = PlanRequest.from_packet(header, arrays)
    fingerprint = server.fingerprint
    if request.model_version != fingerprint:
        raise ValueError('wrong plan model version')
    contract = server.contract
    mode = header.get('control_mode') or ('rlt' if header.get('phase') == 'rl' else 'vla')
    if mode not in contract['control_modes']:
        raise ValueError('unsupported control mode')
    horizons = contract['plan_horizons']
    allowed = {max(horizons.values())} if mode == 'auto' else {horizons[mode]}
    if request.horizon not in allowed or request.frame_stride != contract['frame_stride']:
        raise ValueError('plan horizon/stride does not match model sampling')
    task = header.get('task', server.task)
    if task not in server.tasks:
        raise ValueError('task mismatch')
    observation = {'observation.state': np.asarray(arrays['observation.state'], dtype=np.float32),
                   'observation.anchor': request.anchor.copy()}
    if observation['observation.state'].shape != (server.state_dim,):
        raise ValueError('invalid observation state shape')
    if not np.isfinite(observation['observation.state']).all():
        raise ValueError('non-finite observation state')
    for key in server.camera_keys:
        image = arrays[key]
        if image.dtype != np.uint8 or list(image.shape) != contract['camera_shapes_hwc'][key]:
            raise ValueError('invalid camera image')
        observation[key] = image
    gate = server.plan_gate
    with gate.planning(request):
        policy = server.policy
        phase_snapshot = None
        ctrl = None
        if not sft:
            ctrl = policy._ensure_modifier().phase_ctrl
            phase_snapshot = deepcopy(ctrl.__dict__)
            if (getattr(server, '_plan_mode', None) != mode or header.get('reset_episode')):
                if getattr(server, 'online_session', None) is not None:
                    server.online_session.require_ready()
                    server._set_phase('rl' if mode == 'rlt' else 'vla', force=True)
                else:
                    policy.set_control_mode(mode)
                server._plan_mode = mode
        runtime = ActionRuntime(server.action_contract)
        device = torch.device(server.device)
        pre = server.pre if sft else server.preprocessor
        prepared = prepare_observation_for_inference(observation, device, task, server.robot_type)
        prefix = torch.as_tensor(request.prefix.copy(), device=device, dtype=torch.float32)
        encoded = runtime.encode(prefix, prepared)
        precision = torch.autocast('cuda') if getattr(server, 'use_amp', False) and device.type == 'cuda' else nullcontext()
        try:
            with torch.no_grad(), precision:
                batch = pre(prepared)
                chunk = policy.predict_action_chunk(batch, inference_delay=request.start_idx,
                                                    prev_chunk_left_over=encoded)
                if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[1] not in (set(horizons.values()) if mode == 'auto' else allowed):
                    raise ValueError('model returned a horizon different from the requested contract')
                absolute = runtime.decode(chunk, prepared)[0].cpu().numpy()
                if sft:
                    sources, phases, reference, context = ('vla',)*len(absolute), (0,)*len(absolute), absolute, {}
                else:
                    mod = policy._ensure_modifier()
                    metadata = [mod.pop_step_metadata() for _ in range(len(absolute))]
                    if any(m is None for m in metadata):
                        raise ValueError('missing per-action provenance')
                    sources = tuple('actor' if m.source_type > .5 else 'vla' for m in metadata)
                    phases = tuple(int(m.phase > .5) for m in metadata)
                    reference = runtime.decode(mod.plan_reference, prepared)[0].cpu().numpy()
                    context = {k: v.detach().cpu().numpy() for k, v in mod.plan_context.items()}
                result = PlanResponse(request, absolute, sources, phases, reference, context)
            with gate.lock:
                if not gate.current(request):
                    raise ValueError('plan revoked during inference')
                return result.packet()
        except BaseException:
            if ctrl is not None:
                ctrl.__dict__.clear()
                ctrl.__dict__.update(phase_snapshot)
                server._plan_mode = None
                policy._ensure_modifier().interrupt_chunk()
            raise
