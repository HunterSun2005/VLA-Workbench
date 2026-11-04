"""Optional collection view: contracts and real synthetic video, no hardware."""
from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from vla.web.config import SessionConfig, collection_camera_serials, preflight
from vla.web.demo import DemoBackend
from vla.web.hardware import preset
from vla.datacollection.start_teleop.gamepad_record import DEFAULT_CONFIG
from vla.datacollection.start_teleop.gamepad_recording import load_recording_config, RecordingStore
from vla.datacollection.franka_robotiq_single_left.v30_writer import V30DatasetWriter
from tests.rlt.test_gamepad_recording import snapshot, command


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
    from vla.robots.franka_robotiq.camera import RealSenseCapture
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
def test_resume_refuses_adding_or_removing_side(tmp_path, initial_side):
    from vla.datacollection.franka_robotiq_single_left.schema import STATE_FIELDS
    cfg = SessionConfig(task='test', dataset_root=str(tmp_path/'data'), resume=True,
                        side_camera_serial='SIDE123' if initial_side else '')
    path = cfg.dataset_path(tmp_path)/'meta/info.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'features': {'observation.state': {'shape':[50], 'names':STATE_FIELDS},
                                            **{f'observation.images.{r}': {} for r in cfg.cameras}}}))
    assert preflight(cfg, root=tmp_path)['ok']
    changed = replace(cfg, side_camera_serial='' if initial_side else 'SIDE123')
    assert any('续采相机视角' in e for e in preflight(changed, root=tmp_path)['errors'])
    # The writer also checks this independently of the web preflight.
    root = tmp_path/'writer'
    V30DatasetWriter(root, 30, 'test', tuple(f'observation.images.{r}' for r in cfg.cameras), 224, 224)
    with pytest.raises(RuntimeError, match='camera schema'):
        V30DatasetWriter(root, 30, 'test', tuple(f'observation.images.{r}' for r in changed.cameras), 224, 224)


def test_side_video_timestamps_and_offline_preview_roundtrip(tmp_path):
    from vla.web.processing_service import ProcessingService
    from vla.web.manager import Manager
    control, recording = load_recording_config(DEFAULT_CONFIG)
    control = replace(control, robot=replace(control.robot, camera_serials={**control.robot.camera_serials, 'side':'SIDE123'}))
    def observed(at):
        value = snapshot(at)
        return replace(value, images={**value.images, 'side':np.full((224,224,3), 170, np.uint8)},
                       image_timestamps={**value.image_timestamps, 'side':at-.004})
    store = RecordingStore(tmp_path/'raw', control, recording, 'test')
    try:
        with pytest.raises(ValueError, match='camera roles'):
            store.start(snapshot(), command())
        worker = store.start(observed(1.), command())
        for i in range(14):
            start = 1+i*.04
            for offset in (.01,.02,.03):
                worker.put('command', command(start+offset))
            worker.put('observation', (observed(start+.04), False))
        worker.put('finish', None)
        assert worker.done.wait(20)
        assert worker.error is None
        assert store.writer.video_path(0, 'observation.images.side').is_file()
        rows = pd.read_parquet(store.writer.data_path(0))
        assert len(rows) == 14
        assert rows['complementary_info.side_image_at_ns'].iloc[0] == 996000000
    finally:
        store.close()
    service = ProcessingService(Manager(tmp_path, demo=True))
    summary = service.open('raw')
    assert 'observation.images.side' in summary['cameras']
    jpeg = service.jpeg(summary['draft']['dataset_id'], 0, 0, 'observation.images.side')
    assert jpeg[:2] == b'\xff\xd8'
    from vla.datacollection.processing import DatasetIndex, export_selection
    source = DatasetIndex(store.root)
    output = export_selection(store.root, tmp_path/'selected', {
        'version':1, 'source_fingerprint':source.fingerprint,
        'state_fields':source.names[:10], 'episodes':[0],
    })
    assert 'observation.images.side' in DatasetIndex(output).cameras
