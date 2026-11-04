from dataclasses import replace
import math

import pytest
from fastapi.testclient import TestClient

from vla.web.app import create_app
from vla.web.config import SessionConfig, workspace_defaults
from vla.web.backends import BaseBackend
from vla.datacollection.start_teleop.gamepad_record import DEFAULT_CONFIG
from vla.datacollection.start_teleop.gamepad_recording import load_recording_config, dataset_contract


BOUNDS = {'left': {'min':[-.1,-.6,.02], 'max':[.85,.6,.8]}}


@pytest.mark.parametrize('bounds', [[], {}, {'left':{}}, {'left':{'min':[0,0,0], 'max':[0,1,1]}},
    {'left':{'min':[0,0], 'max':[1,1,1]}}, {'left':{'min':[False,0,0], 'max':[1,1,1]}},
    {'left':{'min':[0,0,0], 'max':[1,1,math.inf]}}])
def test_invalid_workspace_rejected(bounds):
    with pytest.raises(ValueError, match='工作空间'):
        SessionConfig.parse({'task':'test', 'workspace_bounds':bounds})


def test_workspace_applies_to_robot_and_recording_contract():
    config = SessionConfig.parse({'task':'test', 'workspace_bounds':BOUNDS})
    control, recording = load_recording_config(DEFAULT_CONFIG)
    assert SessionConfig().apply_robot_workspace(control.robot) is control.robot
    robot = config.apply_robot_workspace(control.robot)
    assert robot.workspace_min == tuple(BOUNDS['left']['min'])
    assert robot.workspace_max == tuple(BOUNDS['left']['max'])
    contract = dataset_contract(replace(control, robot=robot), recording, 'test')
    assert contract['settings']['robot']['workspace_min'] == BOUNDS['left']['min']


def test_dual_workspace_reaches_real_rollout_config(tmp_path):
    bounds = {**BOUNDS, 'right':{'min':[.1,-.4,.05], 'max':[.8,.4,.7]}}
    config = SessionConfig.parse({'task':'test', 'arms':'dual','workspace_bounds':bounds})
    backend = object.__new__(BaseBackend)
    backend.config, backend.directory = config, tmp_path
    result = backend.rollout_config()
    for arm in result.arms:
        assert arm.workspace_min == tuple(bounds[arm.name]['min'])
        assert arm.workspace_max == tuple(bounds[arm.name]['max'])


def test_workspace_endpoint_defaults_and_token(tmp_path):
    with TestClient(create_app(tmp_path, demo=True), base_url='http://127.0.0.1') as client:
        data={'task':'test'}
        assert client.post('/api/workspace', json=data).status_code == 403
        token=client.get('/api/bootstrap').json()['token']
        headers={'X-VLA-Token':token}
        response=client.post('/api/workspace', json=data, headers=headers)
        assert response.status_code == 200
        assert response.json()['bounds'] == workspace_defaults(SessionConfig())
        for arms in ('single_left','dual'):
            response=client.post('/api/workspace',json={**data,'arms':arms},headers=headers)
            assert set(response.json()['bounds']) == ({'left','right'} if arms=='dual' else {'left'})
        invalid={**data, 'workspace_bounds':{'left':{'min':[1,0,0],'max':[0,1,1]}}}
        assert client.post('/api/preflight', json=invalid, headers=headers).status_code == 422


@pytest.mark.parametrize('extra', [{'source':'gello'}, {'workflow':'deploy'}])
def test_unsupported_route_cannot_silently_ignore_workspace(extra):
    with pytest.raises(ValueError, match='gamepad 数据采集'):
        SessionConfig.parse({'task':'test','workspace_bounds':BOUNDS,**extra})
