"""Session settings and static preflight. Never opens a device."""
from __future__ import annotations
from vla.core.ros_workspace import ros_workspace
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
    action_space: str = 'tcp'
    config_path: str = ''
    dataset_root: str = ''
    task: str = ''
    tasks: tuple[str, ...] = ()
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
    camera_serials: dict | None = None
    workspace_bounds: dict | None = None
    workspace_limits_enabled: bool | None = None
    velocity_control: dict | None = None
    lingbot_execution_mode: str = 'rtc'
    lingbot_inference_slots: int = 14
    lingbot_guard_slots: int = 1
    model_backend: str = 'pi05'
    policy_endpoint: str = 'tcp://127.0.0.1:8007'
    policy_timeout_s: float = 5.0

    @classmethod
    def parse(cls, data):
        if not isinstance(data, dict) or set(data) - {f.name for f in fields(cls)}:
            raise ValueError('unknown session settings; Tracker compliance removed, use velocity_control')
        result = cls(**data)
        for name in ('workflow','arms','source','schema','action_space','config_path','dataset_root','task',
                     'policy_kind','policy_path','tokenizer_path','handover_path','mode','device',
                     'model_python','ros_workspace','ros_config','side_camera_serial','model_backend','policy_endpoint','lingbot_execution_mode'):
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
                              'source': ('gamepad','gello'), 'schema': ('full',), 'action_space': ('tcp',),
                              'policy_kind': ('sft','rlt'), 'mode': ('vla','rlt','auto')}.items():
            if getattr(result, name) not in choices:
                raise ValueError(f'invalid {name}')
        if (not isinstance(result.tasks, (list, tuple)) or any(
                not isinstance(task, str) or not task.strip() or len(task) > 4096 or '\x00' in task
                for task in result.tasks)):
            raise ValueError('invalid task prompts')
        result = replace(result, tasks=tuple(dict.fromkeys(result.tasks)))
        if result.tasks and result.task not in result.tasks:
            raise ValueError('tasks must include the current task')
        if not result.task.strip():
            raise ValueError('task is required')
        if result.workflow == 'deploy' and result.policy_kind == 'sft' and result.mode != 'vla':
            raise ValueError('SFT supports VLA only')
        result.validate_model_backend()
        result.validate_camera_serials()
        result.validate_workspace()
        result.validate_velocity_control()
        return result

    def validate_model_backend(self):
        if self.model_backend not in ('pi05', 'lingbot_v2'):
            raise ValueError('unknown model backend')
        if (isinstance(self.policy_timeout_s, bool) or not isinstance(self.policy_timeout_s, (int, float))
                or not math.isfinite(self.policy_timeout_s) or not 0 < self.policy_timeout_s <= 300):
            raise ValueError('policy_timeout_s must be in (0, 300]')
        if self.model_backend == 'lingbot_v2':
            from vla_rollout_protocol.external import endpoint_address
            endpoint_address(self.policy_endpoint)
            if self.lingbot_execution_mode not in ('rtc', 'serial'):
                raise ValueError('invalid LingBot execution mode')
            if self.lingbot_execution_mode == 'rtc':
                from vla_rollout_protocol.external import validate_rtc_budget
                validate_rtc_budget(self.lingbot_inference_slots, self.lingbot_guard_slots)
            if (self.workflow != 'deploy' or self.arms != 'single_left' or self.action_space != 'tcp'
                    or self.mode != 'vla' or self.policy_kind != 'sft' or self.handover_path):
                raise ValueError('LingBot requires single-left TCP SFT/VLA deployment, without handover')

    def validate_velocity_control(self):
        if self.velocity_control is not None:
            if not isinstance(self.velocity_control, dict):
                raise ValueError('velocity_control must be an object')
            from vla.control.franky.worker import VelocityControl
            VelocityControl.from_mapping(self.velocity_control)

    def effective_velocity_control(self):
        from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
        rollout = load_rollout_config(self.default_config())
        return self.apply_rollout_velocity_control(rollout).franky_velocity_control

    def apply_rollout_velocity_control(self, rollout):
        self.validate_velocity_control()
        from vla.control.franky.worker import VelocityControl
        value = VelocityControl.from_mapping(self.velocity_control, VelocityControl.from_mapping(rollout.franky_velocity_control))
        return replace(rollout, franky_velocity_control=asdict(value))

    def validate_camera_serials(self):
        if self.camera_serials is None:
            return
        if self.workflow != 'collect':
            raise ValueError('独立相机角色设置仅用于数据采集；部署请使用硬件配置')
        import re
        if not isinstance(self.camera_serials, dict) or set(self.camera_serials) - set(self.cameras):
            raise ValueError('相机角色必须与所选机械臂及侧视相机配置一致')
        for value in self.camera_serials.values():
            if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]{1,128}', value):
                raise ValueError('相机序列号不能为空或包含非法字符')
        serials = list(self.camera_serials.values()) + ([self.side_camera_serial] if self.side_camera_serial else [])
        if len(serials) != len(set(serials)):
            raise ValueError('各相机角色不能使用同一设备')

    def validate_workspace(self):
        if self.workspace_limits_enabled is not None and type(self.workspace_limits_enabled) is not bool:
            raise ValueError('workspace_limits_enabled must be boolean')
        if self.workspace_bounds is None:
            return
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

    def effective_workspace_limits(self):
        self.validate_workspace()
        if self.workspace_limits_enabled is not None:
            return self.workspace_limits_enabled
        from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
        return load_rollout_config(self.default_config()).workspace_limits_enabled

    def apply_rollout_workspace(self, rollout):
        self.validate_workspace()
        if self.workspace_limits_enabled is not None:
            rollout = replace(rollout, workspace_limits_enabled=self.workspace_limits_enabled)
        if self.workspace_bounds is None:
            return rollout
        return replace(rollout, arms=tuple(replace(arm,
            workspace_min=tuple(self.workspace_bounds[arm.name]['min']),
            workspace_max=tuple(self.workspace_bounds[arm.name]['max'])) for arm in rollout.arms))

    def default_dataset_root(self):
        if self.workflow == 'collect':
            return f'datasets/raw/franka_{self.arms}_{self.source}_{self.schema}' + ('_side' if self.side_camera_serial else '')
        suffix = 'lingbot_v2' if self.model_backend == 'lingbot_v2' else self.policy_kind
        return f'datasets/rollouts/franka_{self.arms}_{suffix}'

    def dataset_path(self, root):
        value = self.dataset_root if self.dataset_root.strip() else self.default_dataset_root()
        path = Path(value).expanduser()
        return (path if path.is_absolute() else Path(root) / path).resolve()

    @property
    def ros(self):
        return True

    @property
    def cameras(self):
        return ['base','left_wrist'] + (['right_wrist'] if self.arms == 'dual' else []) + (['side'] if self.workflow == 'collect' and self.side_camera_serial else [])

    def default_config(self):
        if self.config_path:
            return Path(self.config_path).expanduser().resolve()
        suffix = 'dual' if self.arms == 'dual' else 'single_left_sft_takeover'
        return resource(f'datacollection/franka_robotiq_rollout/configs/{suffix}.yaml')

    def modes(self, demo=False):
        if self.workflow != 'deploy':
            return []
        if self.model_backend == 'lingbot_v2' or self.policy_kind == 'sft':
            return ['vla']
        return ['vla','rlt'] + (['auto'] if demo or self.handover_path else [])


def workspace_defaults(config):
    """Read configured Cartesian bounds without loading robot/model libraries."""
    import yaml
    raw = yaml.safe_load(config.default_config().read_text())
    result = {arm['name']: {'min':arm['workspace_min'], 'max':arm['workspace_max']} for arm in raw['arms']}
    replace(config, workspace_bounds=result).validate_workspace()
    return result


def collection_camera_serials(config, hardware=None):
    """Resolve the selected collection cameras without opening any device."""
    config.validate_camera_serials()
    from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
    rollout = load_rollout_config(config.default_config())
    serials = {camera.name: camera.serial for camera in rollout.cameras}
    keys = {'base':'BASE_CAMERA_SERIAL', 'left_wrist':'LEFT_WRIST_CAMERA_SERIAL',
            'right_wrist':'RIGHT_WRIST_CAMERA_SERIAL'}
    for role, key in keys.items():
        if role in serials and hardware and hardware.get(key):
            serials[role] = hardware[key]
    if config.camera_serials:
        serials.update(config.camera_serials)
    if config.side_camera_serial:
        serials['side'] = config.side_camera_serial
    if len(set(serials.values())) != len(serials):
        raise ValueError('Side 相机与 base / wrist 相机不能使用同一设备；相机序列号必须互不相同')
    return serials


def preflight(config, *, demo=False, root=None):
    checks, errors = [], []
    existing_episodes = None
    workspace_enabled = config.effective_workspace_limits()
    config.validate_model_backend()
    if config.action_space != 'tcp':
        errors.append('新会话仅支持 Franky TCP 动作空间')
    effective_velocity_control = None
    try:
        effective_velocity_control = config.effective_velocity_control()
    except (ValueError, TypeError, KeyError, OSError) as error:
        errors.append(f'速度控制配置无效: {error}')
    config.validate_workspace()
    config.validate_camera_serials()
    root = Path(root or os.environ.get('VLA_ROOT', Path.cwd())).expanduser().resolve()
    target = config.dataset_path(root)
    if demo:
        return {'ok': not errors, 'checks': ['模拟模式：不访问硬件、不写数据集'], 'errors': errors, 'velocity_control': effective_velocity_control,
                'modes': config.modes(True), 'cameras': config.cameras, 'dataset_root': str(target), 'workspace_limits_enabled':workspace_enabled}
    try:
        from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
        rollout = load_rollout_config(config.default_config())
        if rollout.backend == 'franky':
            from vla.core.franky_runtime import check_worker
            python, binding = check_worker(root, rollout.worker_python)
            checks.append(f'Franky Python: {python}; libfranka {binding["libfranka"]}; velocity/Affine passed')
    except (RuntimeError, ValueError, OSError) as error:
        errors.append(f'Franky 环境检查失败: {error}')
    local_paths = [('配置', str(config.default_config()), False)]
    if config.model_backend == 'pi05':
        local_paths.append(('模型 Python', config.model_python, False))
    for label, path, directory in local_paths:
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
            from vla.dataset_metadata import load_info
            info = load_info(target)
            existing_episodes = info.get('total_episodes')
            if type(existing_episodes) is not int or existing_episodes < 0:
                errors.append('续采元数据缺少有效的已有回合数')
                existing_episodes = None
            else:
                checks.append(f'已有回合数: {existing_episodes}')
            rollout = config.apply_rollout_workspace(load_rollout_config(config.default_config()))
            if info.get('workspace_limits_enabled', True) != rollout.workspace_limits_enabled:
                errors.append('续采工作空间开关与原数据集不一致；请使用原配置或新目录')
            recorded_projection = info.get('execution_projection', {})
            expected_bounds = {
                'lower': [v for a in rollout.arms for v in a.workspace_min],
                'upper': [v for a in rollout.arms for v in a.workspace_max]}
            if any(recorded_projection.get(k) != v for k, v in expected_bounds.items()):
                errors.append('续采工作空间范围与原数据集不一致；启动时还将严格校验控制指纹')
            if config.ros:
                from vla.datacollection.gripper_contract import ROS_GRIPPER_RECORDING_VERSION
                from vla.dataset_metadata import load_info
                if load_info(target).get('gripper_recording_version') != ROS_GRIPPER_RECORDING_VERSION:
                    errors.append('ROS 夹爪动作记录契约已更新；旧夹爪标签数据请使用新目录，不能续采')
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
        try:
            ws = ros_workspace(root, config.ros_workspace)
        except ValueError as error:
            errors.append(str(error))
            ws = root/'runtime/ros/current'
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
            bounds = config.workspace_bounds if config.workspace_bounds is not None else workspace_defaults(config)
            checks.append('TCP 工作空间限制: ' + ('开启' if workspace_enabled else '关闭') + '；范围（米）: ' + json.dumps(bounds, ensure_ascii=False))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f'无法配置采集相机: {error}')
    if config.workflow == 'deploy' and config.model_backend == 'lingbot_v2':
        try:
            from vla.deployment.external import inspect_service
            from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
            rollout = load_rollout_config(config.default_config())
            health = inspect_service(config.policy_endpoint, config.policy_timeout_s,
                                     tasks=config.tasks or (config.task,), image_size=rollout.image_size,
                                     execution_mode=config.lingbot_execution_mode,
                                     inference_slots=config.lingbot_inference_slots, guard_slots=config.lingbot_guard_slots)
            checks.append(f'LingBot {config.lingbot_execution_mode} 服务就绪: {config.policy_endpoint}; 模型 {health["model_version"]}')
            checks.append('外部部署契约通过；实际 TCP 变换将在控制启用前核对')
        except (OSError, EOFError, ValueError, TypeError, KeyError) as error:
            errors.append(f'LingBot 服务检查失败: {error}')
    if config.workflow == 'deploy' and config.model_backend == 'pi05':
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
            if kind not in (('pi05', 'vla_pi05') if config.policy_kind == 'sft' else ('rlt_ac',)):
                errors.append('模型类型与选择的 SFT/RLT 不匹配')
            from vla.actions.metadata import checkpoint_contract
            from vla.actions import ActionSpec
            contract = checkpoint_contract(config.policy_path)
            spec = ActionSpec.from_dict(contract['action_spec'])
            if spec.space != 'tcp':
                errors.append('新会话仅支持 Franky TCP checkpoint')
            expected_arms = ('left', 'right') if config.arms == 'dual' else ('left',)
            if spec.arms != expected_arms:
                errors.append('checkpoint 机械臂顺序不匹配')
            checks.append(f'动作空间: {spec.space}; 训练表示: {spec.representation}; 传输目标: absolute')
            if cfg.get('output_features', {}).get('action', {}).get('shape') != [spec.dim]:
                errors.append('动作维度与 checkpoint 契约不一致')
            expected = {f'observation.images.{r}' for r in config.cameras}
            actual = {k for k in cfg.get('input_features', {}) if k.startswith('observation.images.')}
            if actual != expected:
                errors.append('模型相机字段与机械臂配置不匹配')
        except (OSError, ValueError, TypeError) as error:
            errors.append(f'无法校验模型配置: {error}')
    return {'ok': not errors, 'checks': checks, 'errors': errors, 'velocity_control': effective_velocity_control,
            'modes': config.modes(), 'cameras': config.cameras, 'dataset_root': str(target),
            'existing_episodes': existing_episodes, 'workspace_limits_enabled':workspace_enabled}


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
