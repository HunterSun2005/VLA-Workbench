from dataclasses import asdict, replace
from types import SimpleNamespace as NS
import uuid

import pytest

from vla.web.config import SessionConfig, preflight
from vla.web.backends import BaseBackend
from vla.web.runtime import Runtime
from vla.control.franky.worker import VelocityControl
from tests.web.test_session_ui import session_browser, short_session_root, phase


@pytest.mark.parametrize('workflow', ['collect', 'deploy'])
@pytest.mark.parametrize('arms', ['single_left', 'dual'])
def test_initial_velocity_control_matches_preflight_and_backend(tmp_path, workflow, arms):
    config = SessionConfig.parse({'task': 'test', 'workflow': workflow, 'arms': arms,
                                 'velocity_control': {'max_linear_speed': 0.05, 'max_angular_speed': 0.2}})
    backend = BaseBackend()
    backend.config, backend.directory = config, tmp_path
    rollout = backend.rollout_config()
    assert rollout.franky_velocity_control == preflight(config, demo=True)['velocity_control']
    assert rollout.franky_velocity_control['max_linear_speed'] == 0.05
    assert rollout.franky_velocity_control['tracking_tau_s'] == .15
    with pytest.raises(ValueError):
        SessionConfig.parse({'task':'test', 'velocity_control':{'max_linear_speed':0.17}})
    with pytest.raises(ValueError):
        SessionConfig.parse({'task':'test', 'velocity_control':{'target_smoothing_alpha':True}})


def test_runtime_tuning_ack_does_not_block_pause(tmp_path):
    done, calls = [], []
    def command(op,args):
        calls.append((op,args))
        if op == 'velocity_control': return lambda: bool(done)
    rt = Runtime('s', SessionConfig(task='test'), tmp_path, NS(command=command))
    def submit(op,args):
        cid = uuid.uuid4().hex
        rt.rpc({'kind':'command','session_id':'s','command_id':cid,'revision':rt.revision,
                'operation':op,'args':args})
        rt.poll_command()
        return cid
    cid = submit('velocity_control', {'values':{'max_angular_speed':0.15}})
    assert calls[0][1]['_request_id'] == cid and rt.results[cid]['status'] == 'accepted'
    pause = submit('pause', {})
    assert rt.results[pause]['status'] == 'completed'
    done.append(True); rt.poll_command()
    assert rt.results[cid]['status'] == 'completed'


@pytest.mark.parametrize('session_browser', [False, True], indirect=True)
def test_browser_draft_and_correlated_ack(session_browser):
    b, scenario, root = session_browser
    phase(b,'configure')
    b.wait('document.querySelector("[data-velocity_control=max_linear_speed]").value==="0.05"')
    b.fill('[data-velocity_control=max_linear_speed]', '0.04', 'input')
    b.fill('[data-velocity_control=max_angular_speed]', '0.15', 'input')
    b.click('#preflight')
    b.wait('!document.querySelector("#preflight").disabled')
    assert scenario['checks'][-1]['velocity_control']['max_linear_speed'] == 0.04
    b.click('#start'); phase(b,'running')
    assert scenario['posts'][-1]['velocity_control']['max_angular_speed'] == 0.15
    values = {**asdict(VelocityControl()), 'max_linear_speed':0.04, 'max_angular_speed':0.15}
    scenario.update(state='standby', velocity_control={'pending':False,'values':values})
    b.wait('document.querySelector("#state").textContent==="待命"')
    b.fill('[data-velocity_control=max_linear_speed]', '0.03', 'input')
    b.fill('[data-velocity_control=max_angular_speed]', '0.25', 'input')
    b.click('#viewConfig')
    assert b.js('return Array.from(document.querySelectorAll("[data-velocity_control]"),el=>el.disabled).every(Boolean)')
    b.key('\ue00c')
    b.wait('!document.querySelector("#configDrawer").open')
    assert not b.js('return document.querySelector("[data-velocity_control=target_smoothing_enabled]").disabled')
    b.wait('document.querySelector("#velocity_controlStatus").textContent.includes("未应用")')
    assert b.js('return document.querySelector("[data-velocity_control=max_linear_speed]").value') == '0.03'
    b.click('#applyVelocityControl')
    b.wait('!!document.querySelector("#velocity_controlPanel").dataset.pendingRequest')
    cid = b.js('return document.querySelector("#velocity_controlPanel").dataset.pendingRequest')
    b.wait('document.querySelector("#velocity_controlStatus").textContent.includes("应用中")')
    b.fill('[data-velocity_control=max_linear_speed]', '0.02', 'input')
    scenario['velocity_control']={'values':values,'pending':True,'request':{'request_id':'other','status':'applied'}}
    # A telemetry refresh and drawer blur must not revert either field.
    assert b.js('return document.querySelector("#velocity_controlPanel").dataset.pendingRequest') == cid
    scenario['velocity_control']={'values':{**values,'max_linear_speed':0.03,'max_angular_speed':0.25},
                            'pending':False,'request':{'request_id':cid,'status':'applied'}}
    scenario['command_results']=[{'command_id':cid,'operation':'velocity_control','status':'completed'}]
    b.wait('!document.querySelector("#velocity_controlPanel").dataset.pendingRequest')
    assert b.js('return document.querySelector("[data-velocity_control=max_linear_speed]").value') == '0.02'
    assert b.js('return document.querySelector("#velocity_controlPanel").dataset.dirtyFields') == 'max_linear_speed'
    b.click('#applyVelocityControl'); b.wait('!!document.querySelector("#velocity_controlPanel").dataset.pendingRequest')
    cid = b.js('return document.querySelector("#velocity_controlPanel").dataset.pendingRequest')
    scenario['velocity_control']={**scenario['velocity_control'], 'request':{'request_id':cid,'status':'failed','error':'mock failure'}}
    scenario['command_results'].append({'command_id':cid,'operation':'velocity_control','status':'failed','error':'mock failure'})
    b.wait('document.querySelector("#velocity_controlStatus").textContent.includes("mock failure")')
    assert b.js('return document.querySelector("[data-velocity_control=max_linear_speed]").value') == '0.02'
    scenario['disconnect']()
    b.wait('document.querySelector("#connection").textContent==="已连接"')
    assert b.js('return document.querySelector("[data-velocity_control=max_linear_speed]").value') == '0.02'
    scenario.update(state='stopped', final=True)
    scenario['record']={**scenario['record'],'alive':False}
    phase(b,'finished'); b.click('#nextSession'); phase(b,'configure')
    assert b.js('return document.querySelector("[data-velocity_control=max_linear_speed]").value') == '0.02'
    b.click('#resetVelocityControl')
    assert b.js('return document.querySelector("[data-velocity_control=max_linear_speed]").value') == '0.03'
    b.wait('!document.getAnimations().some(a=>a.playState==="running")')
    b.shot(root/'velocity_control-draft.png')
    b.call('WebDriver:SetWindowRect', width=390, height=900)
    b.js('document.documentElement.dataset.theme="light"')
    b.wait('document.documentElement.scrollWidth<=innerWidth')
    b.shot(root/'velocity_control-narrow-light.png')


@pytest.mark.parametrize('width', [1440, 390])
def test_restart_retains_confirmed_velocity_control_and_reload_keeps_draft(session_browser, width):
    b, scenario, root = session_browser
    def editor_values():
        return b.js('return Object.fromEntries(Array.from(document.querySelectorAll("[data-velocity_control]"),'
                    'input=>[input.dataset.velocity_control,input.dataset.velocity_control.endsWith("_enabled")?input.value==="true":Number(input.value)]))')
    b.call('WebDriver:SetWindowRect', width=width, height=1000)
    phase(b, 'configure')
    b.click('#start'); phase(b, 'running')
    # Runtime settings differ from the immutable initial session config.
    values = dict(asdict(VelocityControl()), max_linear_speed=.04, max_angular_speed=.1,
                  tracking_tau_s=.1, target_smoothing_alpha=.3)
    scenario.update(state='standby', velocity_control={'pending': False, 'values': values})
    b.wait('document.querySelector("[data-velocity_control=target_smoothing_alpha]").value==="0.3"')
    scenario.update(state='stopped', final=True)
    scenario['record'] = {**scenario['record'], 'alive': False}
    phase(b, 'finished')
    # Recover a session tuned by the previous UI, which never stored preferences.
    b.js('localStorage.removeItem("vla.velocity_control.v1")')
    url = b.js('return location.href')
    b.call('WebDriver:Navigate', url=url); phase(b, 'finished')
    b.click('#nextSession'); phase(b, 'configure')
    assert editor_values() == values
    # Untouched acknowledged fields must survive source changes and reconnect.
    b.fill('[name=source]', 'gello')
    scenario['disconnect']()
    b.wait('document.querySelector("#connection").textContent.includes("连接中断")')
    b.wait('document.querySelector("#connection").textContent==="已连接"')
    assert editor_values() == values
    # A draft is preserved across a full document reload, but is not a worker ACK.
    b.fill('[data-velocity_control=max_angular_speed]', '0.15', 'input')
    b.call('WebDriver:Navigate', url=url); phase(b, 'finished')
    b.wait('document.querySelector("[data-velocity_control=max_angular_speed]").value==="0.15"')
    b.wait('document.querySelector("[data-effective=max_angular_speed]").textContent.endsWith("0.1")')
    b.click('#nextSession'); phase(b, 'configure')
    b.fill('[name=arms]', 'dual')
    b.wait('document.querySelector("[data-velocity_control=max_angular_speed]").value==="0.2"')
    b.fill('[name=arms]', 'single_left')
    assert editor_values() == {**values, 'max_angular_speed': 0.15}
    b.fill('[data-velocity_control=max_angular_speed]', '', 'input')
    b.call('WebDriver:Navigate', url=url); phase(b, 'finished')
    b.click('#nextSession'); phase(b, 'configure')
    assert b.js('return document.querySelector("[data-velocity_control=max_angular_speed]").value') == ''
    b.fill('[data-velocity_control=max_angular_speed]', '0.15', 'input')
    b.click('#preflight')
    b.wait('!document.querySelector("#preflight").disabled')
    assert scenario['checks'][-1]['velocity_control'] == {**values, 'max_angular_speed': 0.15}
    b.click('#start'); phase(b, 'running')
    assert scenario['posts'][-1]['velocity_control'] == scenario['checks'][-1]['velocity_control']
    b.shot(root/'velocity_control-restart.png')


def test_reloading_dual_session_does_not_save_uninitialized_single_values(session_browser):
    b, scenario, _ = session_browser
    phase(b, 'configure')
    b.fill('[name=arms]', 'dual')
    b.click('#start'); phase(b, 'running')
    b.js('localStorage.removeItem("vla.velocity_control.v1")')
    b.call('WebDriver:Refresh'); phase(b, 'running')
    scenario.update(state='stopped', final=True)
    scenario['record'] = {**scenario['record'], 'alive': False}
    phase(b, 'finished')
    b.click('#nextSession'); phase(b, 'configure')
    b.fill('[name=arms]', 'single_left')
    b.wait('document.querySelector("[data-velocity_control=max_angular_speed]").value==="0.2"')
    assert b.js('return Array.from(document.querySelectorAll("[data-velocity_control]"),'
                'input=>input.value!==""&&input.checkValidity()).every(Boolean)')
