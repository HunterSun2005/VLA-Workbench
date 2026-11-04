"""Session settings and static preflight. Never opens a device."""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields, replace
from pathlib import Path
import json
import math
import os
import sys

from vla.paths import resource


@dataclass(frozen=True)
class SessionConfig:
    workflow: str = 'collect'
    arms: str = 'single_left'
    source: str = 'gamepad'
    schema: str = 'full'
    config_path: str = ''
    dataset_root: str = ''
    task: str = ''
    resume: bool = False
    policy_kind: str = 'rlt'
    policy_path: str = ''
    tokenizer_path: str = ''
    handover_path: str = ''
    mode: str = 'vla'
    device: str = 'cuda'
    model_python: str = sys.executable
    ros_workspace: str = ''
    ros_config: str = ''
    side_camera_serial: str = ''
    workspace_bounds: dict | None = None

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise ValueError('unknown session settings')
        result = cls(**data)
        for name in ('workflow','arms','source','schema','config_path','dataset_root','task',
                     'policy_kind','policy_path','tokenizer_path','handover_path','mode','device',
                     'model_python','ros_workspace','ros_config','side_camera_serial'):
            value = getattr(result, name)
            if not isinstance(value, str) or len(value) > 4096 or '\x00' in value:
                raise ValueError(f'invalid {name}')
        import re
        if result.side_camera_serial and not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', result.side_camera_serial):
            raise ValueError('invalid side camera serial')
        if result.side_camera_serial and result.workflow != 'collect':
            raise ValueError('side camera is only supported for collection')
        if type(result.resume) is not bool:
            raise ValueError('resume must be boolean')
        for name, choices in {'workflow': ('collect','deploy'), 'arms': ('single_left','dual'),
                              'source': ('gamepad','gello'), 'schema': ('full',),
                              'policy_kind': ('sft','rlt'), 'mode': ('vla','rlt','auto')}.items():
            if getattr(result, name) not in choices:
                raise ValueError(f'invalid {name}')
        if result.workflow == 'deploy' and result.source != 'gamepad':
            raise ValueError('deployment intervention uses gamepad')
        if not result.task.strip():
            raise ValueError('task is required')
        if result.workflow == 'deploy' and result.policy_kind == 'sft' and result.mode != 'vla':
            raise ValueError('SFT supports VLA only')
        result.validate_workspace()
        return result

    def validate_workspace(self):
        if self.workspace_bounds is None:
            return
        if self.workflow != 'collect' or self.source != 'gamepad':
            raise ValueError('面板自定义工作空间仅用于 gamepad 数据采集')
        expected = {'left', 'right'} if self.arms == 'dual' else {'left'}
        if not isinstance(self.workspace_bounds, dict) or set(self.workspace_bounds) != expected:
            raise ValueError('工作空间必须包含所选机械臂的完整范围')
        for arm, bounds in self.workspace_bounds.items():
            if not isinstance(bounds, dict) or set(bounds) != {'min', 'max'}:
                raise ValueError(f'{arm} 工作空间需要 min / max')
            for name in ('min', 'max'):
                values = bounds[name]
                if not isinstance(values, (list, tuple)) or len(values) != 3 or any(
                    isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values
                ):
                    raise ValueError(f'{arm} 工作空间 {name} 必须为三个有限数值（米）')
            if any(lo >= hi for lo, hi in zip(bounds['min'], bounds['max'])):
                raise ValueError(f'{arm} 工作空间各轴最小值必须小于最大值')

    def apply_robot_workspace(self, robot):
        self.validate_workspace()
        if self.workspace_bounds is None:
            return robot
        bounds = self.workspace_bounds['left']
        return replace(robot, workspace_min=tuple(bounds['min']), workspace_max=tuple(bounds['max']))

    def apply_rollout_workspace(self, rollout):
        self.validate_workspace()
        if self.workspace_bounds is None:
            return rollout
        return replace(rollout, arms=tuple(replace(arm,
            workspace_min=tuple(self.workspace_bounds[arm.name]['min']),
            workspace_max=tuple(self.workspace_bounds[arm.name]['max'])) for arm in rollout.arms))

    def default_dataset_root(self):
        if self.workflow == 'collect':
            return f'datasets/raw/franka_{self.arms}_{self.source}_{self.schema}' + ('_side' if self.side_camera_serial else '')
        return f'datasets/rollouts/franka_{self.arms}_{self.policy_kind}'

    def dataset_path(self, root):
        value = self.dataset_root if self.dataset_root.strip() else self.default_dataset_root()
        path = Path(value).expanduser()
        return (path if path.is_absolute() else Path(root) / path).resolve()

    @property
    def ros(self):
        return self.arms == 'dual' or self.source == 'gello'

    @property
    def cameras(self):
        return ['base','left_wrist'] + (['right_wrist'] if self.arms == 'dual' else []) + (['side'] if self.workflow == 'collect' and self.side_camera_serial else [])

    def default_config(self):
        if self.config_path:
            return Path(self.config_path).expanduser().resolve()
        if self.workflow == 'collect' and self.source == 'gamepad' and not self.ros:
            return resource('datacollection/start_teleop/configs/single_left_gamepad_record.yaml')
        suffix = 'dual' if self.arms == 'dual' else 'single_left_sft_takeover'
        return resource(f'datacollection/franka_robotiq_rollout/configs/{suffix}.yaml')

    def modes(self, demo=False):
        if self.workflow != 'deploy':
            return []
        if self.policy_kind == 'sft':
            return ['vla']
        return ['vla','rlt'] + (['auto'] if demo or self.handover_path else [])


def workspace_defaults(config):
    """Read configured Cartesian bounds without loading robot/model libraries."""
    import yaml
    if config.workflow != 'collect' or config.source != 'gamepad':
        return {}
    raw = yaml.safe_load(config.default_config().read_text())
    if config.arms == 'single_left':
        bounds = raw['robot']['workspace']
        result = {'left': {'min': bounds['min'], 'max': bounds['max']}}
    else:
        result = {arm['name']: {'min':arm['workspace_min'], 'max':arm['workspace_max']} for arm in raw['arms']}
    replace(config, workspace_bounds=result).validate_workspace()
    return result


def collection_camera_serials(config, hardware=None):
    """Resolve the selected collection cameras without opening any device."""
    if config.ros:
        from .hardware import preset
        values = {**preset(config.arms), **(hardware or {})}
        keys = {'base':'BASE_CAMERA_SERIAL', 'left_wrist':'LEFT_WRIST_CAMERA_SERIAL',
                'right_wrist':'RIGHT_WRIST_CAMERA_SERIAL'}
        serials = {role:values[keys[role]] for role in config.cameras if role != 'side'}
    else:
        import yaml
        raw = yaml.safe_load(config.default_config().read_text())
        serials = {role:str(raw['robot']['cameras'][role]['serial']) for role in ('base','left_wrist')}
    if config.side_camera_serial:
        serials['side'] = config.side_camera_serial
    if len(set(serials.values())) != len(serials):
        raise ValueError('Side 相机与 base / wrist 相机不能使用同一设备；相机序列号必须互不相同')
    return serials


def preflight(config, *, demo=False, root=None):
    checks, errors = [], []
    config.validate_workspace()
    root = Path(root or os.environ.get('VLA_ROOT', Path.cwd())).expanduser().resolve()
    target = config.dataset_path(root)
    if demo:
        return {'ok': True, 'checks': ['模拟模式：不访问硬件、不写数据集'], 'errors': [],
                'modes': config.modes(True), 'cameras': config.cameras, 'dataset_root': str(target)}
    for label, path, directory in [('配置', str(config.default_config()), False),
                                   ('模型 Python', config.model_python, False)]:
        p = Path(path).expanduser()
        (checks if p.is_file() else errors).append(f'{label}: {p}')
    checks.append(f'数据集输出目录: {target}')
    if config.resume and not (target / 'meta/info.json').is_file():
        errors.append('续采需要有效的已有数据集')
    if config.resume and (target / 'meta/info.json').is_file():
        try:
            from vla.datacollection.franka_robotiq_single_left.schema import STATE_FIELDS
            names = (STATE_FIELDS if config.arms == 'single_left' else
                     [f'{arm}_{field}' for arm in ('left', 'right') for field in STATE_FIELDS])
            info = json.loads((target / 'meta/info.json').read_text())
            recorded_cameras = {k for k in info['features'] if k.startswith('observation.images.')}
            if recorded_cameras != {f'observation.images.{r}' for r in config.cameras}:
                errors.append('续采相机视角必须与原数据集一致；新增或移除 side 相机请使用新目录')
            state = info['features']['observation.state']
            if state.get('shape') != [len(names)] or state.get('names') != names:
                errors.append(f'续采需要相同的完整 {len(names)}D 状态字段；旧 29D/31D/58D 数据请使用新目录')
        except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
            errors.append(f'无法校验续采字段: {error}')
    if not config.resume and target.exists() and (not target.is_dir() or any(target.iterdir())):
        errors.append('输出目录非空；请选择续采或新目录')
    if config.ros:
        ws = Path(config.ros_workspace or root / 'ros_ws').expanduser().resolve()
        if ws != (root / 'ros_ws').resolve():
            errors.append('面板仅使用 VLA_ROOT/ros_ws，避免继承外部 ROS overlay')
        if not (ws / 'install/local_setup.bash').is_file():
            errors.append(f'ROS 工作空间未构建: {ws}')
        if not Path('/opt/ros/humble/setup.bash').is_file():
            errors.append('需要系统 ROS 2 Humble')
        from .hardware import config_path, validate
        hardware_path = config_path(root, config.ros_config)
        if not config.ros_config or not hardware_path.is_file():
            errors.append('ROS 路线需要硬件配置：请使用预设与设备选择，或指定自定义 JSON')
        else:
            try:
                hardware = validate(json.loads(hardware_path.read_text()), config.arms)
                for arm in (('LEFT','RIGHT') if config.arms == 'dual' else ('LEFT',)):
                    for key in (f'{arm}_ROBOT_IP', f'{arm}_ROBOTIQ_PORT'):
                        if not hardware.get(key):
                            errors.append(f'ROS 硬件配置缺少 {key}')
                if config.source == 'gello':
                    if not hardware.get('GELLO_CONFIG'):
                        errors.append('GELLO 采集需要显式 GELLO_CONFIG 配置路径')
                    else:
                        import yaml
                        source = Path(hardware['GELLO_CONFIG'])
                        if not source.is_absolute():
                            source = ws/'src/franka_gello_state_publisher/config'/source
                        entries = yaml.safe_load(source.read_text())
                        expected = {'left','right'} if config.arms == 'dual' else {'left'}
                        if not isinstance(entries,dict) or not expected.issubset({v.get('namespace') for v in entries.values() if isinstance(v,dict)}):
                            errors.append('GELLO 配置缺少所选机械臂 namespace')
            except (ValueError, OSError, TypeError) as error:
                errors.append(str(error))
    if config.workflow == 'collect':
        try:
            hardware = None
            if config.ros and config.ros_config:
                from .hardware import config_path
                hardware = json.loads(config_path(root, config.ros_config).read_text())
            collection_camera_serials(config, hardware)
            if config.source == 'gamepad':
                bounds = config.workspace_bounds if config.workspace_bounds is not None else workspace_defaults(config)
                checks.append('TCP 工作空间（各臂基座坐标系，米）: ' + json.dumps(bounds, ensure_ascii=False))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f'无法配置采集相机: {error}')
    if config.workflow == 'deploy':
        for label, value in [('模型', config.policy_path), ('tokenizer', config.tokenizer_path)]:
            if not value or not Path(value).expanduser().is_dir():
                errors.append(f'{label} 必须为已有本地目录')
        if config.mode not in config.modes():
            errors.append('自动模式需要分类头目录')
        if config.handover_path and not (Path(config.handover_path).expanduser() / 'handover_config.json').is_file():
            errors.append('分类头缺少 handover_config.json')
        try:
            cfg = json.loads((Path(config.policy_path).expanduser() / 'config.json').read_text())
            kind = cfg.get('type')
            if kind != ('pi05' if config.policy_kind == 'sft' else 'rlt_ac'):
                errors.append('模型类型与选择的 SFT/RLT 不匹配')
            dim = 20 if config.arms == 'dual' else 10
            for group, key in [('input_features','observation.state'), ('output_features','action')]:
                if cfg.get(group, {}).get(key, {}).get('shape') != [dim]:
                    errors.append(f'{key} 需要 {dim} 维')
            expected = {f'observation.images.{r}' for r in config.cameras}
            actual = {k for k in cfg.get('input_features', {}) if k.startswith('observation.images.')}
            if actual != expected:
                errors.append('模型相机字段与机械臂配置不匹配')
        except (OSError, ValueError, TypeError) as error:
            errors.append(f'无法校验模型配置: {error}')
    return {'ok': not errors, 'checks': checks, 'errors': errors,
            'modes': config.modes(), 'cameras': config.cameras, 'dataset_root': str(target)}


def dataset_defaults():
    """Shared defaults for the form and non-browser callers."""
    result = {}
    for arms in ('single_left', 'dual'):
        for source in ('gamepad', 'gello'):
            for schema in ('full',):
                config = SessionConfig(arms=arms, source=source, schema=schema)
                result[f'collect/{arms}/{source}/{schema}'] = config.default_dataset_root()
        for kind in ('sft', 'rlt'):
            config = SessionConfig(workflow='deploy', arms=arms, policy_kind=kind)
            result[f'deploy/{arms}/{kind}'] = config.default_dataset_root()
    return result
