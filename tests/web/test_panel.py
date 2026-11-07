from tests.rlt.test_dual_gamepad_record import ros_module
"""Focused lifecycle, transport and mode regressions; never opens robot devices."""
from dataclasses import replace
import json
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid

import numpy as np
import pytest

from vla.web.config import SessionConfig, preflight
from vla.web.demo import DemoBackend
from vla.web.runtime import Runtime
from vla.web.manager import Manager


@pytest.mark.parametrize('arms,source', [('single_left','gello'),('single_left','gamepad'),('dual','gello'),('dual','gamepad')])
def test_collection_lifecycle_and_deduplicated_commands(tmp_path,arms,source):
    cfg=SessionConfig(task='test',arms=arms,source=source)
    backend=DemoBackend(cfg,tmp_path)
    runtime=Runtime('test',cfg,tmp_path,backend)
    def command(op,**args):
        message={'kind':'command','session_id':'test','command_id':str(uuid.uuid4()),
                 'revision':runtime.revision,'operation':op,'args':args}
        first=runtime.rpc(message)
        assert runtime.rpc(message)==first
        runtime.poll_command()
        result=runtime.rpc(message)
        return result
    assert command('record')['status']=='failed'
    assert command('enable')['status']=='completed'
    assert command('record')['status']=='completed'
    assert command('stop')['status']=='failed'
    assert backend.recording  # Stop must not silently discard an episode.
    assert command('save')['status']=='completed'
    assert backend.episode==1
    assert command('stop')['status']=='completed'
    with pytest.raises(RuntimeError,match='ended'):
        command('enable')


@pytest.mark.parametrize('arms', ['single_left','dual'])
@pytest.mark.parametrize('mode', ['vla','rlt','auto'])
def test_deployment_modes_and_human_precedence(tmp_path,arms,mode):
    cfg=SessionConfig(task='test',workflow='deploy',arms=arms,mode=mode)
    backend=DemoBackend(cfg,tmp_path)
    backend.command('enable',{});backend.command('record',{})
    backend.tick();backend.command('intervention',{})
    backend.command('mode',{'mode':mode});backend.tick()
    assert backend.sample()['actual_source']=='human'
    backend.command('intervention',{});backend.tick()
    assert backend.sample()['actual_source'] in ('vla','rlt')
    sft=replace(cfg,policy_kind='sft',mode='vla')
    with pytest.raises(ValueError,match='unavailable'):
        DemoBackend(sft,tmp_path).command('mode',{'mode':'auto'})


def test_save_completion_failure_is_not_reported_as_success(tmp_path):
    cfg=SessionConfig(task='test')
    calls=[]
    def completed():
        raise OSError('disk full')
    backend=SimpleNamespace(command=lambda *args: calls.append(args) or completed)
    runtime=Runtime('test',cfg,tmp_path,backend)
    msg={'kind':'command','session_id':'test','command_id':str(uuid.uuid4()),'revision':0,'operation':'save','args':{}}
    runtime.rpc(msg);runtime.poll_command()
    assert runtime.results[msg['command_id']]['status']=='accepted'
    runtime.poll_command()
    assert runtime.results[msg['command_id']]['status']=='failed'
    assert 'disk full' in runtime.results[msg['command_id']]['error']
    runtime.rpc(msg);assert len(calls)==1


def test_bounded_history_and_camera_without_subscribers(tmp_path):
    backend=DemoBackend(SessionConfig(task='test'),tmp_path)
    runtime=Runtime('test',backend.config,tmp_path,backend)
    for _ in range(6010):
        runtime.last_sample=0
        runtime.publish()
    assert len(runtime.samples)==6000
    response=runtime.rpc({'kind':'state','session_id':'test','since':-1})
    assert len(response['samples'])==100
    assert not runtime.frames and not runtime.frame_interest
    with pytest.raises(RuntimeError,match='state changed'):
        runtime.rpc({'kind':'command','session_id':'test','command_id':str(uuid.uuid4()),
                     'operation':'enable','revision':-1,'args':{}})


def test_http_security_and_packaged_static_files(tmp_path):
    from fastapi.testclient import TestClient
    from vla.web.app import create_app
    with TestClient(create_app(tmp_path,demo=True),base_url="http://127.0.0.1") as client:
        assert client.get('/').status_code==200
        assert 'uPlot' in client.get('/static/vendor/uPlot.iife.min.js').text
        token=client.get('/api/bootstrap').json()['token']
        assert client.post('/api/preflight',json={'task':'test'}).status_code==403
        assert client.post('/api/preflight',json={'task':'test'},headers={
            'X-VLA-Token':token,'Origin':'http://evil.example'}).status_code==403
        assert client.post('/api/preflight',json={'task':'test'},headers={'X-VLA-Token':token}).json()['ok']
        assert client.get('/',headers={'host':'evil.example'}).status_code==400


def test_detached_session_reattach_camera_and_cleanup():
    with tempfile.TemporaryDirectory(prefix='vw-') as name:
        manager=Manager(name,demo=True)
        active=manager.create({'task':'test','arms':'dual','model_python':sys.executable,
                               'dataset_root':'datasets/raw/test'})
        try:
            assert active['config']['dataset_root']=='datasets/raw/test'
            saved=json.loads(Path(active['directory'],'config.json').read_text())
            assert saved['dataset_root']==str(Path(name)/'datasets/raw/test')
            deadline=time.monotonic()+10
            while True:
                try:
                    state=manager.rpc('state')
                    if state['status']['state']=='standby':break
                except (OSError,RuntimeError):pass
                if time.monotonic()>deadline:
                    pytest.fail(Path(active['directory'],'runner.log').read_text())
                time.sleep(.05)
            second=Manager(name,demo=True)
            assert second.active()['pid']==active['pid']
            with pytest.raises(RuntimeError,match='already running'):
                second.create({'task':'second'})
            for role in ('base','left_wrist','right_wrist'):
                second.rpc('camera',role=role)
            deadline=time.monotonic()+5
            while True:
                previews=[second.rpc('camera',role=role)['jpeg'] for role in ('base','left_wrist','right_wrist')]
                if all(previews):break
                assert time.monotonic()<deadline, second.rpc('state')
                time.sleep(.05)
            msg={'command_id':str(uuid.uuid4()),'revision':0,'operation':'stop','args':{}}
            assert second.rpc('command',**msg)['status']=='accepted'
            for _ in range(100):
                if not manager.active()['alive']:break
                time.sleep(.05)
            assert not manager.active()['alive']
            assert second.rpc('state')['status']['state']=='stopped'
        finally:
            if manager.active()['alive']:
                import os,signal
                os.kill(active['pid'],signal.SIGTERM)
                time.sleep(.6)


def test_managed_server_accepts_actual_auto_phase_and_rejects_old_version():
    from tests.control.test_execution import executor, request, response
    from dataclasses import replace
    e = executor()
    req = request(e)
    old = replace(req, generation=req.generation-1)
    assert not e.install(response(old), 1)
    assert e.install(response(req), 1)
    assert e.advance(req.replace_slot, e.control_clock.deadline(req.replace_slot)).phase == 1


def test_ros_collector_drops_old_mode_and_preserves_human_control(ros_module):
    cls = ros_module.RolloutCollector
    from vla.datacollection.franka_robotiq_rollout.policy_client import PolicyResult
    from unittest.mock import Mock
    c=object.__new__(cls)
    c.policy_client=Mock(mode_version=2)
    import threading
    c._lock=threading.RLock()
    c._intervention=True;c._recording=True;c._processing=False;c._saving=False;c.robot=Mock()
    from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
    c.config=load_rollout_config(SessionConfig().default_config())
    c._control_boundary=Mock();c._reset_servo_tracking=Mock()
    c.set_control_mode('auto')
    c.robot.stop.assert_not_called()
    assert c._intervention and c._web_switch_pending
    action=np.array([.4,0,.3,1,0,0,0,1,0,.5],dtype=np.float32)
    result=PolicyResult(1,action,'vla',False,1.,1.,'x',.01,time.monotonic(),'auto',1,.9)
    c.policy_client.poll_latest.return_value=result
    c.executor = Mock()
    c.executor.install.return_value = False
    c._consume_policy_result()
    assert c._latest_policy_result is None
    c._intervention=False;c._phase='vla'
    c.policy_client.poll_latest.return_value=replace(result,mode_version=2)
    c.executor = Mock()
    c.executor.install.return_value = False
    c._consume_policy_result()
    assert c._latest_policy_result is None
    assert c._web_switch_pending  # Only an accepted plan can finish a switch.


def test_policy_mode_reset_retains_classifier_and_clears_queues():
    from vla.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
    from vla.core.phase_controller import PhaseController,Phase
    from collections import deque
    policy=object.__new__(ChunkACPolicy)
    # Module's own __setattr__ requires initialization.
    import torch
    torch.nn.Module.__init__(policy)
    ctrl=PhaseController(classifier=object())
    policy.config=SimpleNamespace(phase_mode='manual')
    policy._rtc_config=None
    class Modifier:
        phase_ctrl=ctrl
        queue=deque([1,2])
        def reset(self):
            self.queue.clear();self.phase_ctrl.reset()
    policy.modifier=Modifier()
    policy._ensure_modifier=lambda:policy.modifier
    policy.set_control_mode('rlt')
    assert ctrl.phase==Phase.CRITICAL_PHASE and not policy.modifier.queue
    policy.reset();assert ctrl.phase==Phase.CRITICAL_PHASE
    policy.set_control_mode('auto');assert ctrl.mode=='learned' and ctrl.classifier is not None
    policy.set_control_mode('vla');assert ctrl.mode=='manual' and ctrl.phase==Phase.VLA_PHASE


def test_protocol_import_does_not_load_torch_under_system_python():
    result=subprocess.run(['/usr/bin/python3','-c',
        'import vla.web.protocol, vla.web.config, vla.web.processes; import sys; assert "torch" not in sys.modules'],
        env={**__import__('os').environ,'PYTHONPATH':str(Path(__file__).resolve().parents[2]/'src')},capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_startup_cancel_and_slow_monitor_do_not_block_commands(tmp_path):
    import threading
    cfg=SessionConfig(task='test')
    runtime=Runtime('test',cfg,tmp_path)
    msg={'kind':'command','session_id':'test','command_id':str(uuid.uuid4()),
         'revision':0,'operation':'stop','args':{}}
    runtime.rpc(msg)
    assert runtime.cancel_start.is_set()
    runtime=Runtime('test',cfg,tmp_path)
    entered,released=threading.Event(),threading.Event()
    backend=DemoBackend(cfg,tmp_path)
    original=backend.sample
    def slow_sample():
        entered.set()
        released.wait(3)
        return original()
    backend.sample=slow_sample
    runtime.backend=backend
    runtime.start()
    try:
        assert entered.wait(2)
        msg={**msg,'operation':'enable'}
        runtime.rpc(msg)
        runtime.poll_command()
        assert backend.enabled
        assert runtime.results[msg['command_id']]['status']=='completed'
    finally:
        released.set()
        runtime.close()


@pytest.mark.parametrize('path_kind', ['relative', 'absolute', 'default'])
def test_dataset_path_preflight_and_resume_use_workspace(tmp_path, monkeypatch, path_kind, ros_ready):
    workspace=tmp_path/'workspace'
    workspace.mkdir()
    elsewhere=tmp_path/'elsewhere'
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    value={'relative':'datasets/raw/custom', 'absolute':str(tmp_path/'external'), 'default':''}[path_kind]
    cfg=ros_ready(workspace,SessionConfig(task='test',dataset_root=value))
    expected=(tmp_path/'external' if path_kind=='absolute' else
              workspace/('datasets/raw/custom' if path_kind=='relative' else 'datasets/raw/franka_single_left_gamepad_full'))
    result=preflight(cfg,root=workspace)
    assert result['ok'],result
    assert result['dataset_root']==str(expected)
    assert not expected.exists()  # A check never creates a dataset.
    expected.mkdir(parents=True)
    (expected/'meta').mkdir()
    from vla.datacollection.franka_robotiq_single_left.schema import STATE_FIELDS
    (expected/'meta/info.json').write_text(json.dumps({'gripper_recording_version': 2, 'features': {
        'observation.state': {'shape': [50], 'names': STATE_FIELDS},
        **{f'observation.images.{r}': {'dtype':'video'} for r in cfg.cameras}}}))
    result=preflight(cfg,root=workspace)
    assert not result['ok'] and any('非空' in error for error in result['errors'])
    assert preflight(replace(cfg,resume=True),root=workspace)['ok']
    (expected/'meta/info.json').unlink()
    assert not preflight(replace(cfg,resume=True),root=workspace)['ok']


@pytest.mark.parametrize('arms,width', [('single_left',29),('single_left',31),('dual',58)])
def test_preflight_refuses_legacy_raw_format(tmp_path, arms, width):
    cfg = SessionConfig(task='test', arms=arms, resume=True)
    target = cfg.dataset_path(tmp_path)
    (target/'meta').mkdir(parents=True)
    (target/'meta/info.json').write_text(json.dumps({'gripper_recording_version': 2, 'features': {
        'observation.state': {'shape': [width], 'names': ['old'] * width}}}))
    result = preflight(cfg, root=tmp_path)
    assert not result['ok']
    assert any('旧 29D/31D/58D' in error for error in result['errors'])
