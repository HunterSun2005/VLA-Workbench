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
@pytest.mark.parametrize('representation', ['absolute', 'chunk_relative'])
def test_sft_plan_is_full_chunk_with_reanchored_prefix(arms, representation):
    import torch
    from vla.actions import ActionSpec
    from vla.actions.statistics import SamplingSpec, compute_statistics
    from vla.control.execution import ExecutionContract, Executor, PlanResponse
    from vla.cli.serve_franka_sft import Server
    spec = ActionSpec('joint', arms, representation)
    hold = np.tile(np.r_[np.zeros(7), .5], len(arms))
    stats = compute_statistics([(np.tile(hold, (12, 1)), np.tile(hold, (12, 1)), np.ones(12, bool))],
                               spec, SamplingSpec(12), 'data')
    server = object.__new__(Server)
    server.task, server.tasks, server.fingerprint, server.device = 'task', ('task',), 'model', 'cpu'
    server.state_dim = spec.dim
    server.robot_type, server.camera_keys = 'franka', ['observation.images.base']
    server.action_contract = {'action_spec': spec.to_dict(), 'statistics': stats}
    server.contract = {'control_modes': ['vla'], 'plan_horizons': {'vla': 12}, 'frame_stride': 1,
                       'camera_shapes_hwc': {'observation.images.base': [4, 6, 3]}}
    server.plan_gate = PlanGate()
    def preprocess(batch):
        assert batch['observation.images.base'].shape == (1, 3, 4, 6)
        assert torch.all(batch['observation.images.base'] == 1)
        assert batch['task'] == 'task'
        assert not torch.is_grad_enabled()
        return batch
    server.pre = preprocess
    captures = []
    def predict(batch, **kw):
        captures.append(kw)
        return torch.zeros(1, 12, spec.dim)
    server.policy = SimpleNamespace(predict_action_chunk=predict)
    contract = ExecutionContract(ActionSpec('joint', arms), {'vla': 12}, 'c', 'k', 't')
    e = Executor(contract, session_id='s', model_version='model', epoch_ns=0)
    e.enable(hold, episode_id='e')
    anchor = hold.copy()
    anchor[0] = .07
    req = e.request(slot=0, anchor=anchor, observation_id=0, observed_at_ns=0, policy='vla')
    h, a = req.packet()
    h.update(task='task', phase='vla')
    a['observation.state'] = hold.copy()
    image = np.full((4, 6, 3), 255, dtype=np.uint8)
    a['observation.images.base'] = image
    with pytest.raises(ValueError, match='task mismatch'):
        server.handle({**h, 'task': 'different task'}, a)
    with pytest.raises(ValueError, match='camera image'):
        server.handle(h, {**a, 'observation.images.base': image[:2]})
    response = PlanResponse.from_packet(*server.handle(h, a))
    assert a['observation.images.base'] is image
    assert np.all(image == 255)
    assert response.actions.shape == (12, spec.dim)
    assert captures[0]['inference_delay'] == 5
    assert captures[0]['prev_chunk_left_over'].shape == (5, spec.dim)
    from vla.actions import DeltaActions, QuantileNormalizer
    expected = QuantileNormalizer(stats).normalize(DeltaActions(spec)(req.prefix, anchor))
    np.testing.assert_allclose(captures[0]['prev_chunk_left_over'], expected, atol=1e-6)
    assert e.install(response, 1)
    with pytest.raises(ValueError, match='protocol-v2'):
        server.handle({'kind': 'infer'}, {})


@pytest.mark.parametrize('failure', ['preprocess', 'revoked'])
def test_candidate_errors_cannot_commit_classifier_state(failure):
    import torch
    from vla.adapters.lerobot.plan_inference import infer_plan
    from vla.adapters.lerobot.policies.action_modifier import RLTStepMetadata
    req = request(executor())
    ctrl = SimpleNamespace(count=0)
    interruptions = []
    mod = SimpleNamespace(phase_ctrl=ctrl, plan_reference=torch.zeros(1,12,8), plan_context={},
                          pop_step_metadata=lambda: RLTStepMetadata(phase=1,source_type=1),
                          interrupt_chunk=lambda: interruptions.append(True))
    gate = PlanGate()
    def predict(batch, **kwargs):
        ctrl.count += 1
        gate.invalidate(req.session_id, req.generation+1)
        return torch.zeros(1,12,8)
    def preprocess(batch):
        if failure == 'preprocess':
            ctrl.count += 1
            raise ValueError('injected preprocess failure')
        return batch
    server = SimpleNamespace(contract={'plan_horizons': {'rlt':12}, 'control_modes':['rlt'], 'frame_stride':1},
        fingerprint='model', task='task', tasks=['task'], camera_keys=[], state_dim=8, robot_type='franka',
        plan_gate=gate, device='cpu', preprocessor=preprocess,
        action_contract={'action_spec': executor().contract.action_spec.to_dict(),
                         'statistics':{'q01':[-1]*8,'q99':[1]*8}},
        policy=SimpleNamespace(_ensure_modifier=lambda:mod, set_control_mode=lambda mode:None,
                               predict_action_chunk=predict))
    h, a = req.packet()
    h.update(control_mode='rlt', task='task')
    a['observation.state'] = req.anchor.copy()
    with pytest.raises(ValueError, match='preprocess failure|revoked'):
        infer_plan(server,h,a)
    assert ctrl.count == 0 and interruptions == [True]
    assert server._plan_mode is None
