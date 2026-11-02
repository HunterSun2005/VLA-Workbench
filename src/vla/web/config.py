"""Session settings and static preflight. Never opens a device."""
from __future__ import annotations
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import json
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

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise ValueError('unknown session settings')
        result = cls(**data)
        for name in ('workflow','arms','source','schema','config_path','dataset_root','task',
                     'policy_kind','policy_path','tokenizer_path','handover_path','mode','device',
                     'model_python','ros_workspace','ros_config'):
            value = getattr(result, name)
            if not isinstance(value, str) or len(value) > 4096 or '\x00' in value:
                raise ValueError(f'invalid {name}')
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
        return result

    def default_dataset_root(self):
        if self.workflow == 'collect':
            return f'datasets/raw/franka_{self.arms}_{self.source}_{self.schema}'
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
        return ['base','left_wrist'] + (['right_wrist'] if self.arms == 'dual' else [])

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


def preflight(config, *, demo=False, root=None):
    checks, errors = [], []
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
        if not config.ros_config or not Path(config.ros_config).expanduser().is_file():
            errors.append('ROS 路线需要显式硬件 JSON 配置（IP、夹爪、GELLO、相机及配置文件路径）')
        else:
            try:
                from .processes import hardware_environment
                hardware = hardware_environment(json.loads(Path(config.ros_config).expanduser().read_text()))
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
