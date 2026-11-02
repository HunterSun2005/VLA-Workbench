"""Passive hardware inventory and immutable web hardware profiles. No device SDKs."""
from __future__ import annotations

import hashlib
from importlib.resources import files
import ipaddress
import json
import os
from pathlib import Path
import tempfile

from .processes import hardware_environment

ARMS = ('single_left', 'dual')


def preset(arms):
    if arms not in ARMS:
        raise ValueError('invalid arms')
    return json.loads(files('vla.web').joinpath('presets', arms + '.json').read_text())


def serial_path(value):
    path = Path(value)
    return path if path.is_absolute() else Path('/dev/serial/by-id') / path


def validate(values, arms, *, complete=False):
    values = hardware_environment(values)
    defaults = preset(arms)
    if complete and set(values) != set(defaults):
        raise ValueError('预设字段不完整或包含其他机械臂字段')
    prefixes = ('LEFT', 'RIGHT') if arms == 'dual' else ('LEFT',)
    addresses, ports = [], []
    for prefix in prefixes:
        value = values.get(prefix + '_ROBOT_IP')
        if value:
            try:
                address = str(ipaddress.IPv4Address(value))
            except ValueError as error:
                raise ValueError(f'{prefix}_ROBOT_IP 需要有效 IPv4 地址') from error
            addresses.append(address)
        for suffix in ('_GELLO_SERIAL_ID', '_ROBOTIQ_PORT'):
            value = values.get(prefix + suffix)
            if value:
                ports.append(str(serial_path(value).resolve()))
    cameras = [v for k, v in values.items() if k.endswith('_CAMERA_SERIAL')]
    for label, items in [('机械臂 IP', addresses), ('串口（GELLO / 夹爪）', ports), ('相机序列号', cameras)]:
        if len(items) != len(set(items)):
            raise ValueError(f'{label} 不能重复分配')
    return values


def inventory(*, serial_root=Path('/dev/serial/by-id'), usb_root=Path('/sys/bus/usb/devices')):
    """Read OS metadata only; never open serial ports, camera streams or robot sockets."""
    serials, cameras, warnings = [], [], []
    try:
        entries = sorted(serial_root.iterdir())
    except OSError as error:
        entries = []
        warnings.append(f'串口目录不可见或不可读：{error.strerror}')
    for path in entries:
        if path.exists():
            serials.append({'value': str(path), 'label': path.name,
                            'accessible': os.access(path, os.R_OK | os.W_OK)})
    try:
        devices = sorted(usb_root.iterdir())
    except OSError as error:
        devices = []
        warnings.append(f'USB 信息不可读：{error.strerror}')
    seen = set()
    for device in devices:
        try:
            vendor = (device / 'idVendor').read_text().strip().lower()
            product = (device / 'product').read_text().strip()
            serial = (device / 'serial').read_text().strip()
        except OSError:
            continue
        if vendor == '8086' and 'realsense' in product.lower() and serial and serial not in seen:
            seen.add(serial)
            cameras.append({'value': serial, 'label': product + ' · ' + serial})
    return {'serials': serials, 'cameras': cameras, 'warnings': warnings,
            'note': '仅枚举 USB / 串口信息；不连接机械臂，不打开设备。检测到不代表可用或未被占用。'}


def profiles(root):
    result = {}
    for arms in ARMS:
        path = Path(root) / 'runtime/web/hardware' / (arms + '.json')
        default = preset(arms)
        values, warning = default, ''
        if path.exists():
            try:
                values = validate(json.loads(path.read_text()), arms, complete=True)
            except (ValueError, OSError, TypeError) as error:
                warning = f'已保存配置无法读取，显示仓库预设：{error}'
        result[arms] = {'values': values, 'defaults': default, 'warning': warning}
    return result


def _atomic_json(path, content):
    fd, temporary = tempfile.mkstemp(prefix='hardware-', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def save(root, data):
    if not isinstance(data, dict) or set(data) != {'arms', 'values'}:
        raise ValueError('invalid hardware profile')
    values = validate(data['values'], data['arms'], complete=True)
    content = json.dumps(values, ensure_ascii=False, sort_keys=True, indent=2) + '\n'
    directory = Path(root) / 'runtime/web/hardware'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    # Every browser gets its own content-addressed reference, so another tab's save
    # cannot change what this tab preflights or launches.
    path = directory / (data['arms'] + '-' + hashlib.sha256(content.encode()).hexdigest() + '.json')
    _atomic_json(path, content)
    _atomic_json(directory / (data['arms'] + '.json'), content)
    return {'path': str(path.relative_to(root)), 'values': values}


def config_path(root, value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(root) / path


def snapshot(root, value, directory):
    values = hardware_environment(json.loads(config_path(root, value).read_text()))
    path = Path(directory) / 'hardware.json'
    path.write_text(json.dumps(values, indent=2) + '\n')
    return str(path)
