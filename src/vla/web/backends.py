"""Managed adapters invoke original coordinators; no keystroke emulation."""
from __future__ import annotations
from contextlib import ExitStack
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import time
import numpy as np
from .processes import Children, clean_environment, hardware_environment, interpreter_command
from .telemetry import descriptors, extra_signals, vector, ros_values


def make_backend(config, directory, root, *, cancelled=lambda:False):
    cls = ROSBackend
    result = cls.__new__(cls)
    result.resources = ExitStack()
    result.cancelled = cancelled
    try:
        result.__init__(config, directory, root)
    except BaseException:
        result.resources.close()
        raise
    return result


class BaseBackend:
    def setup(self, config, directory, root):
        self.config, self.directory, self.root = config, Path(directory), Path(root)
        self.children = Children(directory)
        self.resources.callback(self.children.close)
        self.arms = ['left','right'] if config.arms == 'dual' else ['left']
        self.enabled = False
        self.started = time.monotonic()
        self.record_started = None
        self.mode = config.mode
        self._next_control = 0.
        self._next_health = 0.
        self.events = []
        from .pose_control import PoseActions
        self.poses = PoseActions(self, root)

    def report(self, text):
        print(text, flush=True)
        self.events = (self.events + [str(text)])[-10:]

    def check_children(self):
        if getattr(self,'cancelled',lambda:False)():
            raise RuntimeError('session startup cancelled')
        now = time.monotonic()
        if now >= self._next_health:
            self._next_health = now + .5
            self.children.check()

    def start_policy(self, rollout):
        config = self.config
        socket_path = self.directory/'policy.sock'
        args = ['-m','vla.web.policy_server','--kind',config.policy_kind,
                '--policy-path',config.policy_path,'--tokenizer-path',config.tokenizer_path,
                '--socket-path',str(socket_path),'--task',config.task,'--device',config.device,
                '--arms',config.arms]
        if len(rollout.tasks) > 1:
            args += ['--tasks', *rollout.tasks]
        if config.handover_path:
            args += ['--handover-path',config.handover_path]
        if rollout.rtc_deployment_config is not None and config.policy_kind == 'rlt':
            args += ['--rtc-deployment-config',str(rollout.rtc_deployment_config)]
        env = clean_environment(self.root, config.model_python)
        self.children.start('policy', interpreter_command(config.model_python,args), env)
        from vla_rollout_protocol import recv_packet, send_packet
        import socket
        deadline = time.monotonic()+300
        while time.monotonic() < deadline:
            self.check_children()
            if socket_path.exists():
                try:
                    with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
                        sock.settimeout(2)
                        sock.connect(str(socket_path))
                        send_packet(sock, {'kind':'health','request_id':-1})
                        health,_ = recv_packet(sock)
                    if health.get('kind') == 'health':
                        return replace(rollout, socket_path=socket_path)
                except (OSError, RuntimeError):
                    pass
            time.sleep(.2)
        raise TimeoutError('model startup timed out; see policy.log')

    def rollout_config(self):
        from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
        c = load_rollout_config(self.config.default_config())
        expected = ('left','right') if self.config.arms == 'dual' else ('left',)
        if tuple(a.name for a in c.arms) != expected:
            raise ValueError('arm configuration mismatch')
        c = self.config.apply_rollout_workspace(c)
        space = self.config.action_space
        if self.config.workflow == 'deploy' and self.config.policy_path:
            from vla.actions.metadata import checkpoint_contract
            space = checkpoint_contract(self.config.policy_path)['action_spec']['space']
        c = replace(c, action_space=space, input_source=self.config.source)
        return replace(c, dataset_root=Path(self.config.dataset_root), task=self.config.task,
                       tasks=self.config.tasks or tuple(dict.fromkeys((self.config.task, *c.tasks))), joy_socket_path=self.directory/'joy.sock')

    def collector_command(self, op, args):
        c = self.collector
        active = c.episode_active if hasattr(c,'episode_active') else c._recording
        if op == 'enable':
            if self.config.workflow == 'collect' or hasattr(c,'_setup_teleop'):
                c.enable_teleop()
                self.enabled = bool(c._teleop_enabled if hasattr(c,'_teleop_enabled') else c._setup_teleop.mode == 'active')
                if not self.enabled:
                    raise RuntimeError('enable rejected; release controls and check device health')
            else:
                problems = c._preflight_problems()
                if problems:
                    raise RuntimeError('; '.join(problems))
                self.enabled = True
                if getattr(c, '_waiting_resume', False):
                    c.resume_policy()
        elif op == 'next_task':
            if active:
                raise RuntimeError('task selection requires standby')
            c.next_task()
        elif op == 'record':
            if not self.enabled or active:
                raise RuntimeError('enable first; only one active episode')
            c.start_episode()
            active = c.episode_active if hasattr(c,'episode_active') else c._recording
            if not active:
                raise RuntimeError('start rejected; see session log')
            self.record_started = time.monotonic()
        elif op in ('save','discard'):
            if not active and not getattr(c, '_storage_blocked', False) and not getattr(c, '_states', []):
                raise RuntimeError('no active episode')
            if op == 'save' and self.config.workflow == 'collect' and args.get('outcome','success') != 'success':
                raise ValueError('this demonstration backend supports success/save or discard')
            before = c.writer.episode_count
            c.finish_episode(args.get('outcome','success') if op == 'save' else None, discard=op == 'discard')
            if getattr(c, '_pending_finish', None) is not None or getattr(c, '_saving', False):
                return
            if op == 'save' and c.writer.episode_count <= before:
                raise RuntimeError('episode not saved; inspect minimum frame count, stage or storage error')
            if c.episode_active if hasattr(c,'episode_active') else c._recording:
                raise RuntimeError('episode was not finalized; inspect storage/robot error')
            self.enabled = False
        elif op in ('pause','stop'):
            if hasattr(c,'pause_control'):
                if op == 'stop' and active:
                    raise RuntimeError('save or discard the current episode first')
                c.pause_control()
            else:
                if op == 'stop' and active:
                    raise RuntimeError('save or discard the current episode first')
                c.pause_teleop()
            self.enabled = False
        elif op == 'intervention':
            if not active:
                raise RuntimeError('no active deployment')
            before = c._intervention
            c.toggle_intervention()
            if c._intervention == before:
                raise RuntimeError('intervention transition rejected')
        elif op == 'mode':
            c.set_control_mode(args.get('mode'))
            self.mode = args['mode']
        elif op == 'critical' and (self.config.workflow == 'collect' or getattr(c, 'require_online', False)):
            if not active:
                raise RuntimeError('no active episode')
            c.toggle_phase()
        elif op == 'next_subtask' and hasattr(c,'next_subtask'):
            c.next_subtask()
        elif op == 'recover' and hasattr(c,'recover_fault'):
            if not c.recover_fault():
                raise RuntimeError('recovery failed')
        else:
            raise ValueError('operation unavailable for this backend')

    def collector_status(self):
        c = self.collector
        active = c.episode_active if hasattr(c,'episode_active') else c._recording
        result = getattr(c, '_execution_record', None)
        human = c._intervention
        actual = (c.config.input_source if human else 'waiting' if result is None else 'rlt' if result.source == 'actor' else 'vla' if result.source == 'vla' else 'holding') if active else 'idle'
        setup = getattr(c,'_setup_teleop',None)
        controls = getattr(c,'_gamepad_controls',None) if active else setup
        arms = {a: {'source': actual, 'speed': 'fast' if getattr(c, '_high_speed', {}).get(a, False) else 'normal', 'speed_scale': getattr(c, '_speed_scale', {}).get(a, 1.)} for a in self.arms}
        if controls is not None:
            arms['left'].update(speed='fast' if controls.high_speed else 'normal', speed_scale=controls.speed_scale,
                                transition=abs(controls.speed_scale-(controls.config.speed_modes.high_multiplier if controls.high_speed else 1.))>1e-6)
        ops = ['enable','pause','record','save','discard','stop']
        if hasattr(c,'recover_fault'):
            ops.append('recover')
        if self.config.workflow == 'deploy':
            ops += ['mode','intervention']
        else:
            ops.append('critical')
            if getattr(c.config, 'subtasks', ()):
                ops.append('next_subtask')
        return {'execution_state': (c.executor.state.value if getattr(c, 'executor', None) else
                                   'human' if human else 'standby'),
                'state': 'storage_blocked' if getattr(c, '_storage_blocked', False) else
                'waiting_resume' if getattr(c, '_waiting_resume', False) else 'fault' if c._faulted else 'recording' if active else 'active' if self.enabled else 'standby',
                'enabled': self.enabled, 'recording': active, 'episode': c.writer.episode_count,
                'frames': len(c._states), 'elapsed_s': time.monotonic()-self.record_started if active and self.record_started else 0.,
                'saving': bool(getattr(c,'_saving', False) or getattr(c,'_processing',False) or getattr(c,'_pending_finish',None) is not None), 'error': c._fault_reason,
                'requested_mode': self.mode, 'actual_source': actual,
                'switch_pending': bool(getattr(c,'_web_switch_pending',False)),
                'probability': getattr(result,'probability',None),
                'inference_ms': getattr(c, '_plan_latency_ms', None),
                'arms': arms, 'available_commands': ops, 'modes': self.config.modes(),
                'outcomes': ['success','failure'] if self.config.workflow == 'deploy' else ['success'],
                'critical': c._phase == 'rl' if self.config.workflow == 'collect' else None,
                'subtask': ({'index':c._subtask_index, 'total':len(c.config.subtasks),
                             'name':c.config.subtasks[c._subtask_index]}
                            if self.config.workflow == 'collect' and getattr(c.config, 'subtasks', ()) else None),
                'rtc': c.policy_client.contract.get('rtc_enabled',False) if c.policy_client else False}

    def close(self):
        if self.poses.moving:
            self.poses.finish("cancelled", "会话结束")
        self.resources.close()


class ROSBackend(BaseBackend):
    def __init__(self, config, directory, root):
        self.setup(config,directory,root)
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from vla.paths import resource
        self.rclpy = rclpy
        self.collector = self.recorder = None
        self.arbiters = []
        self.storage_fault = None
        from vla.datacollection.franka_robotiq_rollout.safety import FrankaHealthMonitor
        self.health = FrankaHealthMonitor(.90, 3)
        self._health_stamps = {}
        self.hardware = hardware_environment(json.loads(Path(config.ros_config).expanduser().read_text()))
        if config.workflow == 'collect':
            from .config import collection_camera_serials
            self.camera_serials = collection_camera_serials(config, self.hardware)
        from vla.core.hardware_lease import HardwareLease
        for arm in self.arms:
            lease = HardwareLease(self.hardware[f'{arm.upper()}_ROBOT_IP'],
                                  self.hardware[f'{arm.upper()}_ROBOTIQ_PORT'])
            self.resources.callback(lease.close)
        self.resources.callback(self.children.close)
        legacy = self.root / 'runtime' / f'fr3_fine_teleop_{os.getuid()}' / 'launcher.pid'
        if legacy.exists():
            from .manager import process_identity
            old_pid = int(legacy.read_text().strip())
            if process_identity(old_pid) is not None:
                raise RuntimeError('an existing ROS launcher owns the hardware; stop it before starting a Web session')
        rollout = self.rollout_config()
        if rollout is not None:
            keys = {'base':'BASE_CAMERA_SERIAL','left_wrist':'LEFT_WRIST_CAMERA_SERIAL','right_wrist':'RIGHT_WRIST_CAMERA_SERIAL'}
            rollout = replace(rollout,cameras=tuple(replace(cam,serial=self.hardware.get(keys.get(cam.name, ''),cam.serial))
                                                    for cam in rollout.cameras))
            if len({cam.serial for cam in rollout.cameras}) != len(rollout.cameras):
                raise ValueError('camera serials must be distinct')
        if rollout is not None and config.workflow == 'collect':
            from vla.datacollection.franka_robotiq_rollout.config import CameraConfig
            rollout = replace(rollout, cameras=tuple(CameraConfig(role, serial) for role, serial in self.camera_serials.items()))
        if config.workflow == 'deploy' and config.policy_path:
            rollout = self.start_policy(rollout)
        # Both arm/gripper command routes go through managed coordinators, including GELLO.
        env = clean_environment(root,'/usr/bin/python3',ros=True)
        env.update(self.hardware, VLA_RUN_MODE='managed', CONTROL_OWNER='rollout', TELEOP_MODE='duo' if config.arms == 'dual' else 'single_left',
                   ROS2_WS=str(config.ros_workspace or Path(root)/'ros_ws'),
                   VLA_WEB_READY_FILE=str(self.directory/'hardware.ready'),
                   LOG_DIR=str(self.directory/'ros'), STATE_DIR=str(self.directory/'ros-state'), START_CAMERAS='0')
        self.children.start('hardware',['/bin/bash',str(resource('datacollection/start_teleop/start_teleop.sh'))],env)
        deadline = time.monotonic()+180
        while not (self.directory/'hardware.ready').exists():
            self.check_children()
            if time.monotonic() > deadline:
                raise TimeoutError('ROS hardware startup timed out; see hardware.log')
            time.sleep(.1)
        rclpy.init(args=[])
        self.resources.callback(rclpy.shutdown)
        self.executor = SingleThreadedExecutor()
        self.resources.callback(self.executor.shutdown)
        if config.source == 'gello':
            self._start_gello(env)
        from vla.datacollection.franka_robotiq_rollout.collect_ros import RolloutCollector
        self.collector = RolloutCollector(rollout, demonstration=config.workflow == 'collect')
        self.executor.add_node(self.collector)
        self.resources.callback(self.collector.destroy_node)
        self.resources.callback(self.collector.close)
        if config.workflow == 'deploy' and not self.collector.require_online:
            self.collector.set_control_mode(config.mode)

    def _start_gello(self, env):
        import yaml
        from rclpy.parameter import Parameter
        from vla.paths import resource
        ws = Path(self.config.ros_workspace or self.root/'ros_ws')
        source = Path(self.hardware.get('GELLO_CONFIG','franka_gello_duo.yaml'))
        if not source.is_absolute():
            source = ws/'src/franka_gello_state_publisher/config'/source
        configs = yaml.safe_load(source.read_text())
        selected = {k:v for k,v in configs.items() if v['namespace'] in self.arms}
        if {v['namespace'] for v in selected.values()} != set(self.arms):
            raise ValueError('GELLO configuration is missing an arm')
        for item in selected.values():
            port = self.hardware.get(f"{item['namespace'].upper()}_GELLO_SERIAL_ID")
            if port:
                item['com_port'] = port.removeprefix('/dev/serial/by-id/')
        gello_path = self.directory/'gello.yaml'
        gello_path.write_text(yaml.safe_dump(selected))
        launch = resource('datacollection/start_teleop/gello_duo_fine.launch.py')
        self.children.start('gello',interpreter_command('/usr/bin/python3',
            [str(launch),'--config-path',str(gello_path)],ros=True,root=self.root,workspace=ws),env)
    def command(self, op, args):
        if self.poses.command(op, args):
            return
        return self.collector_command(op, args)

    def tick(self):
        if not self.collector._recording and not self.collector._teleop_enabled:
            self.enabled = False
        self.check_children()
        for _ in range(16):
            self.executor.spin_once(timeout_sec=0)
        self.poses.tick()

    def sample(self):
        node = self.collector
        with node._lock:
            latest = dict(node._latest)
        status = self.collector_status()
        status['action_space'] = node.config.action_space
        status['training_representation'] = (node.policy_client.contract['action_spec']['representation']
                                               if node.policy_client else 'absolute')
        status['control_generation'] = node.coordinator.generation
        status['controller_state'] = node.coordinator.mode
        self.poses.decorate(status)
        values = ros_values(latest, self.arms)
        command = node._last_command_snapshot
        if command is not None and time.monotonic()-command.published_at_ns/1e9 <= .5:
            for arm, target in command.arms.items():
                vector(values, arm, 'target_joint', [f'q{i}' for i in range(1, 8)], target.joints)
        status.update(values=values, signals=descriptors(self.arms)+extra_signals(self.arms),
                      sample_time=time.monotonic())
        return status

    def images(self):
        with self.collector._lock:
            latest = dict(self.collector._latest)
        return {role: (latest['camera.'+role][0]/1e9, latest['camera.'+role][1])
                for role in self.config.cameras if 'camera.'+role in latest}
