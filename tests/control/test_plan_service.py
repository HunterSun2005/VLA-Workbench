from threading import Event, Thread
from types import SimpleNamespace
import numpy as np
import pytest
from vla.control.plan_service import PlanGate, control_message
from tests.control.test_execution import executor, request


def test_invalidation_remains_responsive_during_inference():
    gate = PlanGate()
    req = request(executor())
    entered, finish = Event(), Event()
    seen = []
    def infer():
        with gate.planning(req):
            entered.set()
            assert finish.wait(2)
            seen.append(gate.current(req))
    thread = Thread(target=infer)
    thread.start()
    assert entered.wait(2)
    result = control_message(SimpleNamespace(plan_gate=gate),
                             {'kind': 'invalidate', 'session_id': req.session_id, 'generation': req.generation+1})
    assert result[0]['kind'] == 'invalidate'
    finish.set()
    thread.join(2)
    assert seen == [False]
    with pytest.raises(ValueError, match='revoked'):
        with gate.planning(req):
            pass


@pytest.mark.parametrize('arms', [('left',), ('left', 'right')])
def test_sft_plan_is_full_chunk_with_reanchored_prefix(arms):
    import torch
    from vla.actions import ActionSpec
    from vla.actions.statistics import SamplingSpec, compute_statistics
    from vla.control.execution import ExecutionContract, Executor, PlanResponse
    from vla.cli.serve_franka_sft import Server
    spec = ActionSpec('joint', arms)
    hold = np.tile(np.r_[np.zeros(7), .5], len(arms))
    stats = compute_statistics([(np.tile(hold, (12, 1)), np.tile(hold, (12, 1)), np.ones(12, bool))],
                               spec, SamplingSpec(12), 'data')
    server = object.__new__(Server)
    server.task, server.tasks, server.fingerprint, server.device = 'task', ('task',), 'model', 'cpu'
    server.state_dim = spec.dim
    server.robot_type, server.camera_keys = 'franka', []
    server.action_contract = {'action_spec': spec.to_dict(), 'statistics': stats}
    server.contract = {'control_modes': ['vla'], 'plan_horizons': {'vla': 12}, 'frame_stride': 1,
                       'camera_shapes_hwc': {}}
    server.plan_gate = PlanGate()
    server.pre = lambda x: x
    captures = []
    def predict(batch, **kw):
        captures.append(kw)
        return torch.zeros(1, 12, spec.dim)
    server.policy = SimpleNamespace(predict_action_chunk=predict)
    contract = ExecutionContract(spec, {'vla': 12}, 'c', 'k', 't')
    e = Executor(contract, session_id='s', model_version='model', epoch_ns=0)
    e.enable(hold, episode_id='e')
    req = e.request(slot=0, anchor=hold, observation_id=0, observed_at_ns=0, policy='vla')
    h, a = req.packet()
    h.update(task='task', phase='vla')
    a['observation.state'] = hold.copy()
    response = PlanResponse.from_packet(*server.handle(h, a))
    assert response.actions.shape == (12, spec.dim)
    assert captures[0]['inference_delay'] == 5
    assert captures[0]['prev_chunk_left_over'].shape == (5, spec.dim)
    assert e.install(response, 1)
    with pytest.raises(ValueError, match='protocol-v2'):
        server.handle({'kind': 'infer'}, {})
