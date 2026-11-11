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


@pytest.mark.parametrize('session_browser', [False, True], indirect=True)
def test_browser_draft_and_correlated_ack(session_browser):
    b, scenario, root = session_browser
    phase(b,'configure')
    b.wait('document.querySelector("[data-compliance=translational_stiffness]").value==="1000"')
    b.fill('[data-compliance=translational_stiffness]', '1100', 'input')
    b.fill('[data-compliance=rotational_stiffness]', '60', 'input')
    b.click('#preflight')
    b.wait('!document.querySelector("#preflight").disabled')
    assert scenario['checks'][-1]['compliance']['translational_stiffness'] == 1100
    b.click('#start'); phase(b,'running')
    assert scenario['posts'][-1]['compliance']['rotational_stiffness'] == 60
    values = {**asdict(Compliance()), 'translational_stiffness':1100, 'rotational_stiffness':60}
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
