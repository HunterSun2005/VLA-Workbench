from dataclasses import asdict, replace
from types import SimpleNamespace as NS
import uuid

import pytest

from vla.web.config import SessionConfig, preflight
from vla.web.backends import BaseBackend
from vla.web.runtime import Runtime
from vla.control.franky.worker import Compliance
from tests.web.test_session_ui import session_browser, short_session_root, phase


@pytest.mark.parametrize('workflow', ['collect', 'deploy'])
@pytest.mark.parametrize('arms', ['single_left', 'dual'])
def test_initial_compliance_matches_preflight_and_backend(tmp_path, workflow, arms):
    config = SessionConfig.parse({'task': 'test', 'workflow': workflow, 'arms': arms,
                                 'compliance': {'translational_stiffness': 1100, 'rotational_stiffness': 65}})
    backend = BaseBackend()
    backend.config, backend.directory = config, tmp_path
    rollout = backend.rollout_config()
    assert rollout.franky_compliance == preflight(config, demo=True)['compliance']
    assert rollout.franky_compliance['translational_stiffness'] == 1100
    assert rollout.franky_compliance['translational_clip'] == .008
    with pytest.raises(ValueError):
        SessionConfig.parse({'task':'test', 'compliance':{'translational_stiffness':1201}})
    with pytest.raises(ValueError):
        SessionConfig.parse({'task':'test', 'compliance':{'rotational_clip':True}})


def test_runtime_tuning_ack_does_not_block_pause(tmp_path):
    done, calls = [], []
    def command(op,args):
        calls.append((op,args))
        if op == 'compliance': return lambda: bool(done)
    rt = Runtime('s', SessionConfig(task='test'), tmp_path, NS(command=command))
    def submit(op,args):
        cid = uuid.uuid4().hex
        rt.rpc({'kind':'command','session_id':'s','command_id':cid,'revision':rt.revision,
                'operation':op,'args':args})
        rt.poll_command()
        return cid
    cid = submit('compliance', {'values':{'rotational_stiffness':60}})
    assert calls[0][1]['_request_id'] == cid and rt.results[cid]['status'] == 'accepted'
    pause = submit('pause', {})
    assert rt.results[pause]['status'] == 'completed'
    done.append(True); rt.poll_command()
    assert rt.results[cid]['status'] == 'completed'


def test_browser_static_correction_is_explicit_and_requires_ack(session_browser):
    b, scenario, _ = session_browser
    phase(b, 'configure')
    b.wait('document.querySelector("[data-compliance=tracking_integral_gain]").value==="0"')
    b.fill('[data-compliance=tracking_integral_gain]', '0.3', 'input')
    b.fill('[data-compliance=tracking_max_offset_m]', '0.008', 'input')
    b.click('#preflight')
    b.wait('!document.querySelector("#preflight").disabled')
    assert scenario['checks'][-1]['compliance']['tracking_integral_gain'] == .3
    assert scenario['checks'][-1]['compliance']['tracking_max_offset_m'] == .008
    b.click('#start'); phase(b, 'running')
    assert scenario['posts'][-1]['compliance']['tracking_integral_gain'] == .3
    values = {**asdict(Compliance()), 'tracking_integral_gain': .3, 'tracking_max_offset_m': .008}
    scenario.update(state='standby', compliance={'pending': False, 'values': values})
    b.wait('document.querySelector("[data-effective=tracking_integral_gain]").textContent.endsWith("0.3")')
    b.fill('[data-compliance=tracking_integral_gain]', '0', 'input')
    b.wait('!document.querySelector("#applyCompliance").disabled')
    b.click('#applyCompliance')
    b.wait('!!document.querySelector("#compliancePanel").dataset.pendingRequest')
    cid = b.js('return document.querySelector("#compliancePanel").dataset.pendingRequest')
    assert scenario['commands'][-1]['args']['values']['tracking_integral_gain'] == 0
    assert b.js('return document.querySelector("[data-effective=tracking_integral_gain]").textContent').endswith('0.3')
    scenario['compliance'] = {'values': {**values, 'tracking_integral_gain': 0}, 'pending': False,
                             'request': {'request_id': cid, 'status': 'applied'}}
    scenario['command_results'] = [{'command_id': cid, 'operation': 'compliance', 'status': 'completed'}]
    b.wait('!document.querySelector("#compliancePanel").dataset.pendingRequest')
    b.wait('document.querySelector("[data-effective=tracking_integral_gain]").textContent.endsWith("0")')


@pytest.mark.parametrize('session_browser', [False, True], indirect=True)
def test_browser_draft_and_correlated_ack(session_browser):
    b, scenario, root = session_browser
    phase(b,'configure')
    b.wait('document.querySelector("[data-compliance=translational_stiffness]").value==="1100"')
    b.fill('[data-compliance=translational_stiffness]', '1120', 'input')
    b.fill('[data-compliance=rotational_stiffness]', '60', 'input')
    b.click('#preflight')
    b.wait('!document.querySelector("#preflight").disabled')
    assert scenario['checks'][-1]['compliance']['translational_stiffness'] == 1120
    b.click('#start'); phase(b,'running')
    assert scenario['posts'][-1]['compliance']['rotational_stiffness'] == 60
    values = {**asdict(Compliance()), 'translational_stiffness':1120, 'rotational_stiffness':60}
    scenario.update(state='standby', compliance={'pending':False,'values':values})
    b.wait('document.querySelector("#state").textContent==="待命"')
    b.fill('[data-compliance=translational_stiffness]', '1150', 'input')
    b.fill('[data-compliance=rotational_stiffness]', '70', 'input')
    b.click('#viewConfig'); b.key('\ue00c')
    b.wait('!document.querySelector("#configDrawer").open')
    b.wait('document.querySelector("#complianceStatus").textContent.includes("未应用")')
    assert b.js('return document.querySelector("[data-compliance=translational_stiffness]").value') == '1150'
    b.click('#applyCompliance')
    b.wait('!!document.querySelector("#compliancePanel").dataset.pendingRequest')
    cid = b.js('return document.querySelector("#compliancePanel").dataset.pendingRequest')
    b.wait('document.querySelector("#complianceStatus").textContent.includes("应用中")')
    b.fill('[data-compliance=translational_stiffness]', '1170', 'input')
    scenario['compliance']={'values':values,'pending':True,'request':{'request_id':'other','status':'applied'}}
    # A telemetry refresh and drawer blur must not revert either field.
    assert b.js('return document.querySelector("#compliancePanel").dataset.pendingRequest') == cid
    scenario['compliance']={'values':{**values,'translational_stiffness':1150,'rotational_stiffness':70},
                            'pending':False,'request':{'request_id':cid,'status':'applied'}}
    scenario['command_results']=[{'command_id':cid,'operation':'compliance','status':'completed'}]
    b.wait('!document.querySelector("#compliancePanel").dataset.pendingRequest')
    assert b.js('return document.querySelector("[data-compliance=translational_stiffness]").value') == '1170'
    assert b.js('return document.querySelector("#compliancePanel").dataset.dirtyFields') == 'translational_stiffness'
    b.click('#applyCompliance'); b.wait('!!document.querySelector("#compliancePanel").dataset.pendingRequest')
    cid = b.js('return document.querySelector("#compliancePanel").dataset.pendingRequest')
    scenario['compliance']={**scenario['compliance'], 'request':{'request_id':cid,'status':'failed','error':'mock failure'}}
    scenario['command_results'].append({'command_id':cid,'operation':'compliance','status':'failed','error':'mock failure'})
    b.wait('document.querySelector("#complianceStatus").textContent.includes("mock failure")')
    assert b.js('return document.querySelector("[data-compliance=translational_stiffness]").value') == '1170'
    scenario['disconnect']()
    b.wait('document.querySelector("#connection").textContent==="已连接"')
    assert b.js('return document.querySelector("[data-compliance=translational_stiffness]").value') == '1170'
    scenario.update(state='stopped', final=True)
    scenario['record']={**scenario['record'],'alive':False}
    phase(b,'finished'); b.click('#nextSession'); phase(b,'configure')
    assert b.js('return document.querySelector("[data-compliance=translational_stiffness]").value') == '1170'
    b.click('#resetCompliance')
    assert b.js('return document.querySelector("[data-compliance=translational_stiffness]").value') == '1150'
    b.wait('!document.getAnimations().some(a=>a.playState==="running")')
    b.shot(root/'compliance-draft.png')
    b.call('WebDriver:SetWindowRect', width=390, height=900)
    b.js('document.documentElement.dataset.theme="light"')
    b.wait('document.documentElement.scrollWidth<=innerWidth')
    b.shot(root/'compliance-narrow-light.png')


@pytest.mark.parametrize('width', [1440, 390])
def test_restart_retains_confirmed_compliance_and_reload_keeps_draft(session_browser, width):
    b, scenario, root = session_browser
    def editor_values():
        return b.js('return Object.fromEntries(Array.from(document.querySelectorAll("[data-compliance]"),'
                    'input=>[input.dataset.compliance,Number(input.value)]))')
    b.call('WebDriver:SetWindowRect', width=width, height=1000)
    phase(b, 'configure')
    b.click('#start'); phase(b, 'running')
    # Runtime settings differ from the immutable initial session config.
    values = dict(asdict(Compliance()), translational_stiffness=1000, rotational_stiffness=50,
                  nullspace_stiffness=20, translational_clip=.05, rotational_clip=.3,
                  max_step=.05, max_step_rad=.2, max_delta_tau=.5, gains_time_constant=.1)
    scenario.update(state='standby', compliance={'pending': False, 'values': values})
    b.wait('document.querySelector("[data-compliance=rotational_clip]").value==="0.3"')
    scenario.update(state='stopped', final=True)
    scenario['record'] = {**scenario['record'], 'alive': False}
    phase(b, 'finished')
    # Recover a session tuned by the previous UI, which never stored preferences.
    b.js('localStorage.removeItem("vla.compliance.v1")')
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
    b.fill('[data-compliance=rotational_stiffness]', '60', 'input')
    b.call('WebDriver:Navigate', url=url); phase(b, 'finished')
    b.wait('document.querySelector("[data-compliance=rotational_stiffness]").value==="60"')
    b.wait('document.querySelector("[data-effective=rotational_stiffness]").textContent.endsWith("50")')
    b.click('#nextSession'); phase(b, 'configure')
    b.fill('[name=arms]', 'dual')
    b.wait('document.querySelector("[data-compliance=rotational_stiffness]").value==="65"')
    b.fill('[name=arms]', 'single_left')
    assert editor_values() == {**values, 'rotational_stiffness': 60}
    b.fill('[data-compliance=rotational_stiffness]', '', 'input')
    b.call('WebDriver:Navigate', url=url); phase(b, 'finished')
    b.click('#nextSession'); phase(b, 'configure')
    assert b.js('return document.querySelector("[data-compliance=rotational_stiffness]").value') == ''
    b.fill('[data-compliance=rotational_stiffness]', '60', 'input')
    b.click('#preflight')
    b.wait('!document.querySelector("#preflight").disabled')
    assert scenario['checks'][-1]['compliance'] == {**values, 'rotational_stiffness': 60}
    b.click('#start'); phase(b, 'running')
    assert scenario['posts'][-1]['compliance'] == scenario['checks'][-1]['compliance']
    b.shot(root/'compliance-restart.png')


def test_reloading_dual_session_does_not_save_uninitialized_single_values(session_browser):
    b, scenario, _ = session_browser
    phase(b, 'configure')
    b.fill('[name=arms]', 'dual')
    b.click('#start'); phase(b, 'running')
    b.js('localStorage.removeItem("vla.compliance.v1")')
    b.call('WebDriver:Refresh'); phase(b, 'running')
    scenario.update(state='stopped', final=True)
    scenario['record'] = {**scenario['record'], 'alive': False}
    phase(b, 'finished')
    b.click('#nextSession'); phase(b, 'configure')
    b.fill('[name=arms]', 'single_left')
    b.wait('document.querySelector("[data-compliance=rotational_stiffness]").value==="65"')
    assert b.js('return Array.from(document.querySelectorAll("[data-compliance]"),'
                'input=>input.value!==""&&input.checkValidity()).every(Boolean)')
