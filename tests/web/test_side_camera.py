"""Optional collection view: contracts and real synthetic video, no hardware."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from vla.web.config import SessionConfig, collection_camera_serials, preflight
from vla.web.demo import DemoBackend
from vla.web.hardware import preset
from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter


@pytest.mark.parametrize('arms,source', [('single_left','gamepad'), ('single_left','gello'), ('dual','gamepad'), ('dual','gello')])
@pytest.mark.parametrize('serial', ['', 'SIDE123'])
def test_optional_camera_session_and_demo(arms, source, serial, tmp_path):
    cfg = SessionConfig.parse(dict(task='test', arms=arms, source=source, side_camera_serial=serial))
    roles = ['base', 'left_wrist'] + (['right_wrist'] if arms == 'dual' else []) + (['side'] if serial else [])
    assert cfg.cameras == roles
    assert list(collection_camera_serials(cfg)) == roles
    assert set(DemoBackend(cfg, tmp_path).images()) == set(roles)
    assert cfg.default_dataset_root().endswith('_side' if serial else '_full')


@pytest.mark.parametrize('arms,source', [('single_left','gamepad'), ('single_left','gello'), ('dual','gamepad'), ('dual','gello')])
def test_reject_duplicate_side_camera(arms, source):
    cfg = SessionConfig(arms=arms, source=source, side_camera_serial=preset(arms)['BASE_CAMERA_SERIAL'])
    with pytest.raises(ValueError, match='同一设备'):
        collection_camera_serials(cfg)


def test_side_camera_is_collection_only():
    with pytest.raises(ValueError, match='only supported for collection'):
        SessionConfig.parse(dict(task='test', workflow='deploy', side_camera_serial='SIDE123'))


def test_side_frame_participates_in_synchronized_snapshot():
    from vla.sensors.realsense import RealSenseCapture
    capture = RealSenseCapture({'base':'B', 'left_wrist':'W', 'side':'S'}, width=640, height=480,
                               fps=30, image_size=224, timeout_s=1., max_pair_skew_s=.025,
                               max_frame_age_s=.075, clock=lambda:1.)
    for i, role in enumerate(capture.serials):
        capture._frames[role].append((.99-i*.001, np.zeros((224,224,3), np.uint8)))
    assert set(capture.snapshot().images) == {'base','left_wrist','side'}
    capture._frames['side'].clear()
    capture._frames['side'].append((.5, np.zeros((224,224,3), np.uint8)))
    with pytest.raises(TimeoutError, match='no synchronized'):
        capture.snapshot()


@pytest.mark.parametrize('initial_side', [False, True])
def test_resume_refuses_adding_or_removing_side(tmp_path, initial_side, ros_ready):
    from vla.datacollection.franka_robotiq_single_left.schema import STATE_FIELDS
    cfg = SessionConfig(task='test', dataset_root=str(tmp_path/'data'), resume=True,
                        side_camera_serial='SIDE123' if initial_side else '')
    cfg = ros_ready(tmp_path,cfg)
    path = cfg.dataset_path(tmp_path)/'meta/info.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'gripper_recording_version': 2, 'features': {'observation.state': {'shape':[50], 'names':STATE_FIELDS},
                                            **{f'observation.images.{r}': {} for r in cfg.cameras}}}))
    assert preflight(cfg, root=tmp_path)['ok']
    changed = replace(cfg, side_camera_serial='' if initial_side else 'SIDE123')
    assert any('续采相机视角' in e for e in preflight(changed, root=tmp_path)['errors'])
    # The writer also checks this independently of the web preflight.
    root = tmp_path/'writer'
    V30DatasetWriter(root, 30, 'test', tuple(f'observation.images.{r}' for r in cfg.cameras), 224, 224)
    with pytest.raises(RuntimeError, match='camera schema'):
        V30DatasetWriter(root, 30, 'test', tuple(f'observation.images.{r}' for r in changed.cameras), 224, 224)






@pytest.mark.parametrize('extra', [
    {'camera_serials':[]}, {'camera_serials':{'right_wrist':'R'}},
    {'camera_serials':{'base':''}}, {'camera_serials':{'base':None}},
    {'camera_serials':{'base':'B','left_wrist':'B'}},
    {'camera_serials':{'base':'B'},'side_camera_serial':'B'},
    {'camera_serials':{'base':'B'},'workflow':'deploy'},
])
def test_camera_override_validation(extra):
    with pytest.raises(ValueError):
        SessionConfig.parse({'task':'test',**extra})


def test_camera_default_api_uses_yaml_and_requires_token(tmp_path):
    from fastapi.testclient import TestClient
    from vla.web.app import create_app
    with TestClient(create_app(tmp_path,demo=True),base_url='http://127.0.0.1') as client:
        data={'task':'test','camera_serials':{'base':'OVERRIDE'}}
        assert client.post('/api/camera-defaults',json=data).status_code == 403
        headers={'x-vla-token':client.get('/api/bootstrap').json()['token']}
        result=client.post('/api/camera-defaults',json=data,headers=headers)
        assert result.status_code == 200
        assert result.json()['serials'] == collection_camera_serials(SessionConfig())
        invalid={**data,'camera_serials':{'base':'B','left_wrist':'B'}}
        assert client.post('/api/preflight',json=invalid,headers=headers).status_code == 422
