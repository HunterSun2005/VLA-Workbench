"""Hardware setup boundaries; no robot, serial or camera device is opened."""
import json
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from vla.web.app import create_app
from vla.web.config import SessionConfig, preflight
from vla.web.hardware import inventory, preset, profiles, validate


@pytest.mark.parametrize('arms', ['single_left', 'dual'])
def test_presets_match_repository_calibration(arms):
    import yaml
    values = validate(preset(arms), arms, complete=True)
    root = Path(__file__).resolve().parents[2]
    for field, package in [('GELLO_CONFIG', 'franka_gello_state_publisher'),
                           ('ROBOT_CONFIG', 'franka_fr3_arm_controllers'),
                           ('GRIPPER_CONFIG', 'franka_gripper_manager')]:
        config = yaml.safe_load((root / 'ros_ws/src' / package / 'config' / values[field]).read_text())
        assert set(config) == ({'LEFT', 'RIGHT'} if arms == 'dual' else {'LEFT'})
        for arm in config:
            expected = config[arm]
            key = {'GELLO_CONFIG': '_GELLO_SERIAL_ID', 'ROBOT_CONFIG': '_ROBOT_IP',
                   'GRIPPER_CONFIG': '_ROBOTIQ_PORT'}[field]
            assert values[arm + key] == expected['robot_ip' if field == 'ROBOT_CONFIG' else 'com_port']


def test_passive_inventory_deduplicates_usb_and_keeps_stable_ports(tmp_path):
    serials, usb = tmp_path / 'serial', tmp_path / 'usb'
    serials.mkdir(); usb.mkdir()
    node = tmp_path / 'ttyUSB1'; node.touch()
    (serials / 'stable-gello').symlink_to(node)
    (serials / 'disconnected').symlink_to(tmp_path / 'missing')
    for name, vendor, product, serial in [('1', '8086', 'Intel RealSense D435i', '123'),
                                         ('2', '8086', 'Intel RealSense D435i', '123'),
                                         ('3', '0403', 'USB Serial', '456')]:
        device = usb / name; device.mkdir()
        for field, value in [('idVendor', vendor), ('product', product), ('serial', serial)]:
            (device / field).write_text(value)
    result = inventory(serial_root=serials, usb_root=usb)
    assert [s['value'] for s in result['serials']] == [str(serials / 'stable-gello')]
    assert [c['value'] for c in result['cameras']] == ['123']
    assert not result['warnings']
    assert inventory(serial_root=tmp_path / 'missing', usb_root=usb)['warnings']


@pytest.mark.parametrize('key,value', [
    ('RIGHT_ROBOT_IP', '172.16.0.2'),
    ('RIGHT_WRIST_CAMERA_SERIAL', '260722303785'),
    ('RIGHT_GELLO_SERIAL_ID', '/dev/serial/by-id/usb-FTDI_USB_TO_RS-485_DABC0KC0-if00-port0'),
    ('LEFT_ROBOT_IP', '999.1.1.1'), ('GELLO_CONFIG', '$(bad)'), ('UNDECLARED', 'value')])
def test_invalid_or_conflicting_assignment_refused(key, value):
    values = preset('dual'); values[key] = value
    with pytest.raises(ValueError):
        validate(values, 'dual', complete=True)


def test_save_api_preserves_old_references_and_protects_writes(tmp_path, monkeypatch):
    monkeypatch.setattr('vla.web.hardware.inventory', lambda: {'serials': [], 'cameras': [], 'warnings': []})
    with TestClient(create_app(tmp_path, demo=True), base_url='http://127.0.0.1') as client:
        info = client.get('/api/hardware').json()
        values = info['profiles']['single_left']['values']
        payload = {'arms': 'single_left', 'values': values}
        assert client.post('/api/hardware', json=payload).status_code == 403
        headers = {'X-VLA-Token': client.get('/api/bootstrap').json()['token']}
        first = client.post('/api/hardware', json=payload, headers=headers).json()
        assert not Path(first['path']).is_absolute()
        assert client.post('/api/hardware', json=payload, headers=headers).json() == first
        values['LEFT_ROBOT_IP'] = '172.16.2.2'
        second = client.post('/api/hardware', json=payload, headers=headers).json()
        assert first['path'] != second['path']
        assert json.loads((tmp_path / first['path']).read_text())['LEFT_ROBOT_IP'] == '172.16.0.2'
        assert profiles(tmp_path)['single_left']['values']['LEFT_ROBOT_IP'] == '172.16.2.2'
        assert profiles(tmp_path)['dual']['values'] == preset('dual')
        payload['values']['RIGHT_ROBOT_IP'] = '1.2.3.4'
        assert client.post('/api/hardware', json=payload, headers=headers).status_code == 422
        assert client.get('/static/hardware.js').status_code == 200
    (tmp_path / 'runtime/web/hardware/single_left.json').write_text('{bad')
    assert profiles(tmp_path)['single_left']['warning']


def test_relative_custom_config_preflight_and_session_snapshot(tmp_path, monkeypatch):
    import vla.web.manager as manager_module
    from types import SimpleNamespace
    source = tmp_path / 'hardware.json'
    values = preset('single_left')
    # Use repository calibration for static preflight; never launch the workspace.
    root = Path(__file__).resolve().parents[2]
    values['GELLO_CONFIG'] = str(root / 'ros_ws/src/franka_gello_state_publisher/config' / values['GELLO_CONFIG'])
    source.write_text(json.dumps(values))
    monkeypatch.chdir('/')
    cfg = SessionConfig(task='test', source='gello', ros_config='hardware.json')
    result = preflight(cfg, root=tmp_path)
    assert not any('硬件配置' in e or 'namespace' in e or 'hardware.json' in e for e in result['errors'])
    monkeypatch.setattr(manager_module, 'preflight', lambda *a, **k: {'ok': True, 'dataset_root': str(tmp_path / 'data')})
    monkeypatch.setattr(manager_module, 'process_identity', lambda pid: 'mock-start-time')
    calls = []
    def popen(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(pid=123, wait=lambda: None)
    monkeypatch.setattr(manager_module.subprocess, 'Popen', popen)
    manager = manager_module.Manager(tmp_path)
    record = manager.create({'task': 'test', 'source': 'gello', 'ros_config': 'hardware.json'})
    saved = json.loads((Path(record['directory']) / 'config.json').read_text())
    frozen = Path(saved['ros_config'])
    assert frozen.parent == Path(record['directory'])
    assert json.loads(frozen.read_text()) == values
    source.write_text('{}')
    assert json.loads(frozen.read_text()) == values
    assert record['config']['ros_config'] == 'hardware.json'
    assert len(calls) == 1


def test_session_profile_uses_frozen_values_not_latest_editor_save(tmp_path):
    from vla.web.hardware import session_profile, save, snapshot
    from dataclasses import asdict, replace
    values = preset('dual')
    original = save(tmp_path, {'arms':'dual','values':values})
    directory = tmp_path/'runtime/web/session'
    directory.mkdir()
    config = SessionConfig(task='test',arms='dual',source='gello',ros_config=snapshot(tmp_path,original['path'],directory))
    (directory/'config.json').write_text(json.dumps(asdict(config)))
    updated = {**values,'BASE_CAMERA_SERIAL':'OTHER_CAMERA'}
    save(tmp_path,{'arms':'dual','values':updated})
    record={'directory':str(directory),'session_id':'session'}
    assert session_profile(tmp_path,record)['values'] == values
    app=create_app(tmp_path)
    app.state.manager.active=lambda: record
    with TestClient(app,base_url='http://127.0.0.1') as client:
        result=client.get('/api/hardware/session')
        assert result.status_code == 200
        assert result.json()['session_id']=='session'
        assert result.json()['values']['BASE_CAMERA_SERIAL']==values['BASE_CAMERA_SERIAL']
    (directory/'config.json').write_text(json.dumps(asdict(replace(config,arms='single_left',source='gamepad'))))
    assert session_profile(tmp_path,record) is None
