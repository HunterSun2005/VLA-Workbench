"""Two independent protocol implementations over real loopback TCP; no model/GPU."""
from contextlib import contextmanager
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import sys
import threading
import time
import types

import numpy as np
import pytest

from vla.actions import ActionSpec
from vla.control.execution import PlanResponse
from vla.control.serial_execution import SerialExecutionContract, SerialExecutor
from vla.datacollection.franka_robotiq_rollout.policy_client import ExternalPolicyClient
from vla.deployment.external import inspect_service
from vla_rollout_protocol import send_packet, recv_packet
from vla_rollout_protocol.external import connect, validate_health, require_tool

ROOT = Path(__file__).resolve().parents[2]
FIXTURE = json.loads((ROOT/'tests/fixtures/lingbot_wire_v2.json').read_text())


@pytest.fixture
def server_module():
    root = Path(os.environ.get('VLA_LINGBOT_ROOT', ROOT.parent/'lingbot-vla-v2'))
    if not (root/'deploy/workbench_server.py').is_file():
        pytest.skip('independent LingBot service checkout unavailable; set VLA_LINGBOT_ROOT')
    assert json.loads((root/'tests/fixtures/workbench_wire_v2.json').read_text()) == FIXTURE
    package = types.ModuleType('_lingbot_wire_test')
    package.__path__ = [str(root/'deploy')]
    sys.modules[package.__name__] = package
    return importlib.import_module(package.__name__+'.workbench_server')


class Model:
    deployment = FIXTURE['deployment']

    def __init__(self):
        self.started, self.release = threading.Event(), threading.Event()
        self.release.set()

    def infer(self, arrays, task):
        self.started.set()
        assert self.release.wait(3), 'test model was not released'
        assert arrays['observation.state'][-1] == np.float32(.04)
        assert arrays['observation.anchor'][-1] == .5
        return np.tile(np.asarray(FIXTURE['anchor'], dtype=np.float32), (50, 1))


@contextmanager
def running(module, model=None):
    model = Model() if model is None else model
    service = module.Service(model, FIXTURE['tasks'])
    stop, ready = threading.Event(), threading.Event()
    addresses = []
    def bound(address):
        addresses.append(address)
        ready.set()
    worker = threading.Thread(target=module.serve, args=(service,),
                              kwargs=dict(port=0, stop=stop, ready=bound), daemon=True)
    worker.start()
    assert ready.wait(3)
    endpoint = 'tcp://%s:%s' % addresses[0]
    try:
        yield service, model, endpoint
    finally:
        model.release.set()
        stop.set()
        worker.join(4)
        assert not worker.is_alive()


def plan(model_version):
    ex = SerialExecutor(SerialExecutionContract(ActionSpec(), 'c', 'k', 't'),
                        session_id='test', model_version=model_version, epoch_ns=time.monotonic_ns())
    ex.enable(np.asarray(FIXTURE['anchor']), episode_id='episode')
    return ex, ex.request(slot=0, anchor=np.asarray(FIXTURE['anchor']), observation_id=0,
                          observed_at_ns=time.monotonic_ns(), policy='vla')


def observation():
    return {'state': np.asarray(FIXTURE['state'], dtype=np.float32),
            'images': {k: np.zeros(shape, dtype=np.uint8) for k, shape in FIXTURE['deployment']['camera_shapes_hwc'].items()},
            'task': FIXTURE['tasks'][0], 'phase': 'vla'}


def wait_for(callback):
    deadline = time.monotonic()+3
    while time.monotonic() < deadline:
        value = callback()
        if value:
            return value
        time.sleep(.005)
    pytest.fail('timed out')


def test_actual_cross_repository_client_and_service(server_module):
    with running(server_module) as (service, model, endpoint):
        health = inspect_service(endpoint, tasks=FIXTURE['tasks'], image_size=(224, 224))
        client = ExternalPolicyClient(endpoint, 2)
        try:
            ex, req = plan(health['model_version'])
            client.set_control_mode('vla')
            client.submit(plan=req, **observation())
            candidate = wait_for(client.poll_latest)
            assert req.observed_at_ns <= candidate.received_at_ns <= time.monotonic_ns()
            assert ex.install(candidate, time.monotonic_ns(), start_slot=1)
            np.testing.assert_array_equal(candidate.actions, model.infer(
                {'observation.state': observation()['state'], 'observation.anchor': req.anchor}, 'Pick and place.'))
            assert candidate.request.packet()[0] == req.packet()[0]
            # Duplicate requests fail explicitly rather than running the model again.
            h, a = req.packet(); h.update(task=FIXTURE['tasks'][0], control_mode='vla')
            a.update({'observation.state': observation()['state'], **observation()['images']})
            with connect(endpoint, 2) as sock:
                send_packet(sock, h, a)
                reply, _ = recv_packet(sock)
            assert reply['kind'] == 'error' and 'duplicate' in reply['error']
        finally:
            client.close()


def test_cli_check_only_queries_external_service_without_model_or_hardware(server_module, monkeypatch, capsys):
    from vla.cli.ros_session import main
    import vla.core.franky_runtime as runtime
    import vla.web.backends as backends
    monkeypatch.setattr(runtime, 'check_worker', lambda *a: ('/mock/python', {'libfranka': 'mock'}))
    monkeypatch.setattr(backends, 'make_backend', lambda *a: pytest.fail('check-only must not start hardware'))
    with running(server_module) as (_, model, endpoint):
        main(['--workflow', 'deploy', '--model-backend', 'lingbot_v2',
              '--lingbot-execution-mode', 'serial', '--policy-endpoint', endpoint, '--task', FIXTURE['tasks'][0],
              '--tasks', *FIXTURE['tasks'], '--check-only'])
        assert not model.started.is_set()
    assert 'hardware not started' in capsys.readouterr().out


def test_control_remains_responsive_during_inference_and_revokes_result(server_module):
    with running(server_module) as (service, model, endpoint):
        model.release.clear()
        client = ExternalPolicyClient(endpoint, 2)
        try:
            ex, req = plan(client.contract['model_version'])
            client.submit(plan=req, **observation())
            assert model.started.wait(1)
            assert inspect_service(endpoint, .5)['model_version'] == req.model_version
            client.invalidate(req.session_id, req.generation+1)
            wait_for(lambda: service.generations.get(req.session_id) == req.generation+1)
            model.release.set()
            wait_for(lambda: not client._pending)
            assert client.poll_latest() is None
            client.raise_if_failed()
        finally:
            model.release.set()
            client.close()


def test_delayed_control_does_not_fault_or_revoke_newer_plan(server_module):
    service = server_module.Service(Model(), FIXTURE['tasks'])
    _, req = plan(service.contract['model_version'])
    req = replace(req, generation=3)
    header, arrays = req.packet(); obs = observation()
    header.update(task=obs['task'])
    arrays.update({'observation.state': obs['state'], **obs['images']})
    assert service.handle(header, arrays)[0]['kind'] == 'plan_result'
    # An earlier generation's control connection arrives after the new plan.
    reply, _ = service.handle({'kind': 'invalidate', 'session_id': req.session_id, 'generation': 2}, {})
    assert reply['kind'] == 'invalidate' and reply['current_generation'] == 3
    stale = {**header, 'generation': 2, 'request_id': req.request_id+1}
    with pytest.raises(ValueError, match='revoked'):
        service.handle(stale, arrays)
    assert service.handle({**header, 'request_id': req.request_id+2}, arrays)[0]['kind'] == 'plan_result'


def test_external_request_owns_observation_buffers(server_module, monkeypatch):
    with running(server_module) as (_, model, endpoint):
        client = ExternalPolicyClient(endpoint, 2)
        gate, connecting = threading.Event(), threading.Event()
        connect_original = client._connect
        def delayed_connect():
            connecting.set()
            assert gate.wait(2)
            return connect_original()
        monkeypatch.setattr(client, '_connect', delayed_connect)
        try:
            _, req = plan(client.contract['model_version'])
            obs = observation()
            def infer(arrays, task):
                assert arrays['observation.state'][-1] == np.float32(.04)
                assert not arrays['observation.images.base'].any()
                return np.tile(np.asarray(FIXTURE['anchor']), (50, 1))
            model.infer = infer
            client.submit(plan=req, **obs)
            assert connecting.wait(1)
            obs['state'][:] = 99
            for image in obs['images'].values():
                image[:] = 255
            gate.set()
            assert wait_for(client.poll_latest).actions.shape == (50, 10)
            client.raise_if_failed()
        finally:
            gate.set()
            client.close()
            client.close()


@pytest.mark.parametrize('field,value', [('execution_mode', 'rtc'), ('state_dim', 55),
                                       ('rtc_enabled', True), ('online_enabled', True)])
def test_health_cannot_override_contract(server_module, field, value):
    header = server_module.health_contract(FIXTURE['deployment'], FIXTURE['tasks'])
    header[field] = value
    with pytest.raises(ValueError):
        validate_health(header)


def test_tcp_mismatch_and_native_statistics_identity_are_explicit():
    deployment = FIXTURE['deployment']
    require_tool(deployment, deployment['tool_definition'])
    other = json.loads(json.dumps(deployment['tool_definition']))
    other['left']['position'][2] = .1
    with pytest.raises(ValueError, match='TCP'):
        require_tool(deployment, other)


def frame_adapted_deployment():
    deployment = json.loads(json.dumps(FIXTURE['deployment']))
    deployment['tool_definition']['left']['position'] = [0., 0., .174]
    model_tool = json.loads(json.dumps(deployment['tool_definition']))
    c = 2**-.5
    model_tool['left']['rotation'] = [[c, c, 0.], [-c, c, 0.], [0., 0., 1.]]
    deployment['model_frame_adapter'] = {
        'kind': 'same_origin_tcp_rotation_v1', 'model_tool_definition': model_tool,
        'weights_sha256': deployment['artifacts']['weights']}
    return deployment


def test_checkpoint_bound_frame_contract_conformance_and_wire_tool(server_module):
    from vla_rollout_protocol.external import health_contract
    d = frame_adapted_deployment()
    expected = health_contract(d, FIXTURE['tasks'])
    assert expected == server_module.health_contract(d, FIXTURE['tasks'])
    validate_health(expected)
    require_tool(d, d['tool_definition'])
    with pytest.raises(ValueError, match='TCP transform mismatch'):
        require_tool(d, d['model_frame_adapter']['model_tool_definition'])
    without = {k: v for k, v in d.items() if k != 'model_frame_adapter'}
    assert expected['model_version'] != health_contract(without, FIXTURE['tasks'])['model_version']
    other = json.loads(json.dumps(d))
    other['model_frame_adapter']['model_tool_definition']['left']['rotation'] = np.eye(3).tolist()
    assert expected['model_version'] != health_contract(other, FIXTURE['tasks'])['model_version']


@pytest.mark.parametrize('change', ['kind', 'missing_tool', 'null', 'extra', 'weights',
                                   'unbound', 'bad_artifacts', 'position', 'rotation', 'base', 'tip'])
def test_frame_adapter_malformed_or_unbound_is_rejected_by_both_repos(server_module, change):
    from vla_rollout_protocol.external import validate_deployment
    d = frame_adapted_deployment()
    adapter = d['model_frame_adapter']
    tool = adapter['model_tool_definition']['left']
    if change == 'kind':
        adapter['kind'] = 'implicit_rotation'
    elif change == 'missing_tool':
        del adapter['model_tool_definition']
    elif change == 'null':
        d['model_frame_adapter'] = None
    elif change == 'extra':
        adapter['unknown'] = True
    elif change == 'weights':
        adapter['weights_sha256'] = 'f'*64
    elif change == 'unbound':
        del d['artifacts']
    elif change == 'bad_artifacts':
        d['artifacts'] = None
    elif change == 'position':
        tool['position'][2] = .175
    elif change == 'rotation':
        tool['rotation'][0][0] = 2.
    else:
        tool[change] = 'wrong_frame'
    from importlib import import_module
    server_validator = import_module(server_module.__package__+'.workbench_contract').validate_deployment
    for validator in (validate_deployment, server_validator):
        with pytest.raises(ValueError):
            validator(d, require_artifacts=False)


def test_server_rejects_bad_observation_and_nonempty_prefix(server_module):
    contract = server_module.health_contract(FIXTURE['deployment'], FIXTURE['tasks'])
    _, req = plan(contract['model_version'])
    h, a = req.packet(); obs = observation()
    h.update(task=obs['task'])
    a.update({'observation.state': obs['state'], **obs['images']})
    server_module.validate_request(h, a, contract)
    a['committed_prefix'] = np.zeros((1, 10), dtype=np.float32)
    with pytest.raises(ValueError):
        server_module.validate_request(h, a, contract)
    a['committed_prefix'] = np.empty((0, 10), dtype=np.float32)
    a['observation.state'][0] = np.nan
    with pytest.raises(ValueError):
        server_module.validate_request(h, a, contract)


def rtc_model():
    model = Model()
    model.deployment = json.loads(json.dumps(FIXTURE['deployment']))
    model.deployment.update(execution_mode='rtc', rtc={
        'prefix_conditioning': 'clean_tokens_time_zero_training_mask', 'max_prefix_steps': 15})
    def infer(arrays, task, committed_prefix):
        np.testing.assert_array_equal(committed_prefix, arrays['committed_prefix'])
        # Deliberately return low precision prefix: service must restore wire values.
        return np.tile(np.asarray(FIXTURE['anchor'], np.float32), (50, 1))
    model.infer = infer
    return model


def rtc_request(server_module):
    from vla.control.lingbot_execution import LingBotRTCContract, LingBotRTCExecutor
    service = server_module.Service(rtc_model(), FIXTURE['tasks'])
    ex = LingBotRTCExecutor(LingBotRTCContract(ActionSpec(), {'vla': 50}, 'c', 'k', 't'),
        session_id='rtc-test', model_version=service.contract['model_version'], epoch_ns=0)
    anchor = np.asarray(FIXTURE['anchor']); anchor[-1] = .512345678901
    ex.enable(anchor, episode_id='episode')
    req = ex.request(slot=0, anchor=anchor, observation_id=0, observed_at_ns=0, policy='vla')
    h, a = req.packet(); h.update(task=FIXTURE['tasks'][0])
    a.update({'observation.state': observation()['state'], **observation()['images']})
    return service, ex, h, a


def test_rtc_contract_conformance_and_exact_wire_prefix(server_module):
    from vla_rollout_protocol.external import health_contract
    service, ex, h, a = rtc_request(server_module)
    assert service.contract == health_contract(service.model.deployment, FIXTURE['tasks'])
    # Client monotonic epoch need not match server time (deadline is far in the past here).
    rh, ra = service.handle(h, a)
    result = PlanResponse.from_packet(rh, ra)
    np.testing.assert_array_equal(result.actions[:15], a['committed_prefix'])
    assert ex.install(result, 1)
    validate_health(service.contract, execution_mode='rtc')
    with pytest.raises(ValueError, match='mode mismatch'):
        validate_health(service.contract, execution_mode='serial')
    with pytest.raises(ValueError, match='rtc_max_delay'):
        validate_health(service.contract, execution_mode='rtc', inference_slots=15)


@pytest.mark.parametrize('change', ['too_long', 'boundary', 'negative', 'identity', 'horizon', 'stride', 'prefix_nan'])
def test_rtc_service_rejects_bad_request(server_module, change):
    service, ex, h, a = rtc_request(server_module)
    if change == 'too_long':
        h['replace_slot'] = 16; a['committed_prefix'] = np.tile(ex.hold, (16, 1))
    elif change == 'boundary':
        h['replace_slot'] = 14
    elif change == 'negative':
        h['start_slot'] = 16
    elif change == 'identity':
        h['model_version'] = 'other'
    elif change == 'horizon':
        h['horizon'] = 49
    elif change == 'stride':
        h['frame_stride'] = 2
    else:
        a['committed_prefix'] = a['committed_prefix'].copy(); a['committed_prefix'][0, 0] = np.nan
    with pytest.raises(ValueError):
        service.handle(h, a)


def test_rtc_actual_tcp_client_to_independent_service(server_module):
    from vla.control.lingbot_execution import LingBotRTCContract, LingBotRTCExecutor
    with running(server_module, rtc_model()) as (service, _, endpoint):
        health = inspect_service(endpoint, execution_mode='rtc', tasks=FIXTURE['tasks'])
        client = ExternalPolicyClient(endpoint, 2)
        try:
            epoch = time.monotonic_ns()
            ex = LingBotRTCExecutor(LingBotRTCContract(ActionSpec(), {'vla': 50}, 'c', 'k', 't'),
                session_id='rtc-wire', model_version=health['model_version'], epoch_ns=epoch)
            ex.enable(np.asarray(FIXTURE['anchor']), episode_id='e')
            req = ex.request(slot=0, anchor=ex.hold, observation_id=0, observed_at_ns=epoch, policy='vla')
            client.set_control_mode('vla')
            assert client.submit(plan=req, **observation()) is not None
            assert client.submit(plan=req, **observation()) is None
            result = wait_for(client.poll_latest)
            assert ex.install(result, time.monotonic_ns())
            np.testing.assert_array_equal(result.actions[:15], req.prefix)
            assert ex.queue[14][1] == 'system_hold' and ex.queue[15][1] == 'vla'
            client.raise_if_failed()
        finally:
            client.close()
