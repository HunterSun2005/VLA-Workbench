from dataclasses import replace
import math

import pytest
from fastapi.testclient import TestClient

from vla.web.app import create_app
from vla.web.config import SessionConfig, workspace_defaults
from vla.web.backends import BaseBackend


BOUNDS = {'left': {'min':[-.1,-.6,.02], 'max':[.85,.6,.8]}}


@pytest.mark.parametrize('bounds', [[], {}, {'left':{}}, {'left':{'min':[0,0,0], 'max':[0,1,1]}},
    {'left':{'min':[0,0], 'max':[1,1,1]}}, {'left':{'min':[False,0,0], 'max':[1,1,1]}},
    {'left':{'min':[0,0,0], 'max':[1,1,math.inf]}}])
def test_invalid_workspace_rejected(bounds):
    with pytest.raises(ValueError, match='工作空间'):
        SessionConfig.parse({'task':'test', 'workspace_bounds':bounds})




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
def test_all_routes_apply_workspace(extra, tmp_path):
    config = SessionConfig.parse({'task':'test','workspace_bounds':BOUNDS,'workspace_limits_enabled':False,**extra})
    backend = object.__new__(BaseBackend)
    backend.config, backend.directory = config, tmp_path
    rollout = backend.rollout_config()
    assert not rollout.workspace_limits_enabled
    assert rollout.arms[0].workspace_min == tuple(BOUNDS['left']['min'])


@pytest.mark.parametrize('recorded_enabled',[True,False])
def test_resume_rejects_old_bounded_data_without_rewriting(tmp_path,ros_ready,recorded_enabled):
    import json
    from vla.web.config import preflight
    from vla.datacollection.franka_robotiq_single_left.schema import STATE_FIELDS
    config=ros_ready(tmp_path,SessionConfig(task='test',dataset_root=str(tmp_path/'data/../data'),resume=True))
    root=config.dataset_path(tmp_path)
    bounds=workspace_defaults(config)['left']
    info={'total_episodes':3,'gripper_recording_version':2,'workspace_limits_enabled':recorded_enabled,
          'execution_projection':{'lower':bounds['min'],'upper':bounds['max'],'workspace_limits_enabled':recorded_enabled},
          'features':{'observation.state':{'shape':[50],'names':STATE_FIELDS},
                      **{f'observation.images.{r}':{} for r in config.cameras}}}
    path=root/'meta/info.json';path.parent.mkdir(parents=True);path.write_text(json.dumps(info))
    before=path.read_bytes()
    result=preflight(config,root=tmp_path)
    assert result['ok'] is (not recorded_enabled)
    assert result['existing_episodes']==3
    if recorded_enabled:
        assert any('工作空间开关' in e for e in result['errors'])
    assert result['dataset_root']==str(root)
    result=preflight(replace(config,resume=False),root=tmp_path)
    assert not result['ok'] and any('非空' in e for e in result['errors'])
    assert path.read_bytes()==before


def test_current_sessions_cannot_reenable_xyz_limits(tmp_path):
    import yaml
    from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
    raw=yaml.safe_load(SessionConfig().default_config().read_text())
    raw['control']['workspace_limits_enabled']=False
    path=tmp_path/'config.yaml';path.write_text(yaml.safe_dump(raw))
    c=SessionConfig(task='test',config_path=str(path))
    assert c.effective_workspace_limits() is False
    assert c.apply_rollout_workspace(load_rollout_config(path)).workspace_limits_enabled is False
    with pytest.raises(ValueError, match='workspace limits have been removed'):
        replace(c,workspace_limits_enabled=True).apply_rollout_workspace(load_rollout_config(path))
    raw['control'].pop('workspace_limits_enabled');path.write_text(yaml.safe_dump(raw))
    assert c.effective_workspace_limits() is False
    assert load_rollout_config(path).workspace_limits_enabled is False
    raw['control']['workspace_limits_enabled']=True
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='workspace limits have been removed'):
        load_rollout_config(path)
    with pytest.raises(ValueError, match='workspace limits have been removed'):
        SessionConfig.parse({'task':'test', 'workspace_limits_enabled':True})
