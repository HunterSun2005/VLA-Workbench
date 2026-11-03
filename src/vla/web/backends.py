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
from .telemetry import descriptors, extra_signals, vector, franky_values, ros_values


def make_backend(config, directory, root, *, cancelled=lambda:False):
    cls = ROSBackend if config.ros else FrankyBackend
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
        return replace(c, dataset_root=Path(self.config.dataset_root), task=self.config.task,
                       tasks=(self.config.task,), joy_socket_path=self.directory/'joy.sock')

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
        elif op == 'record':
            if not self.enabled or active:
                raise RuntimeError('enable first; only one active episode')
            c.start_episode()
            active = c.episode_active if hasattr(c,'episode_active') else c._recording
            if not active:
                raise RuntimeError('start rejected; see session log')
            self.record_started = time.monotonic()
        elif op in ('save','discard'):
            if not active:
                raise RuntimeError('no active episode')
            if op == 'save' and self.config.workflow == 'collect' and args.get('outcome','success') != 'success':
                raise ValueError('this demonstration backend supports success/save or discard')
            before = c.writer.episode_count
            c.finish_episode(args.get('outcome','success') if op == 'save' else None, discard=op == 'discard')
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
        elif op == 'critical' and self.config.workflow == 'collect':
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
        result = c._latest_policy_result
        human = c._intervention
        actual = ('gamepad' if human else 'waiting' if result is None else 'rlt' if result.phase > .5 else 'vla') if active else 'idle'
        setup = getattr(c,'_setup_teleop',None)
        controls = getattr(c,'_gamepad_controls',None) if active else setup
        arms = {a: {'source': actual, 'speed': 'fixed', 'speed_scale': None} for a in self.arms}
        if controls is not None:
            arms['left'].update(speed='fast' if controls.high_speed else 'normal', speed_scale=controls.speed_scale,
                                transition=abs(controls.speed_scale-(controls.config.speed_modes.high_multiplier if controls.high_speed else 1.))>1e-6)
        ops = ['enable','pause','record','save','discard','stop']
        if hasattr(c,'recover_fault'):
            ops.append('recover')
        if self.config.workflow == 'deploy':
            ops += ['mode','intervention']
        else:
            ops += ['critical','next_subtask']
        return {'state': 'fault' if c._faulted else 'recording' if active else 'active' if self.enabled else 'standby',
                'enabled': self.enabled, 'recording': active, 'episode': c.writer.episode_count,
                'frames': len(c._states), 'elapsed_s': time.monotonic()-self.record_started if active and self.record_started else 0.,
                'saving': bool(getattr(c,'_processing',False)), 'error': c._fault_reason,
                'requested_mode': self.mode, 'actual_source': actual,
                'switch_pending': bool(getattr(c,'_web_switch_pending',False)),
                'probability': getattr(result,'probability',None),
                'inference_ms': result.latency_s*1000 if result else None,
                'arms': arms, 'available_commands': ops, 'modes': self.config.modes(),
                'outcomes': ['success','failure'] if self.config.workflow == 'deploy' else ['success'],
                'rtc': c.policy_client.contract.get('rtc_enabled',False) if c.policy_client else False}

    def close(self):
        self.resources.close()


class FrankyBackend(BaseBackend):
    def __init__(self, config, directory, root):
        self.setup(config,directory,root)
        from vla.datacollection.start_teleop.gamepad_teleop import JoyReceiver, JoySidecars
        if config.workflow == 'collect':
            module = importlib.import_module('vla.datacollection.start_teleop.gamepad_record')
            control, recording = module.load_recording_config(config.default_config())
            module.recording_preflight(control,recording)
            # Preserve configured realtime core for Franky's control thread.
            if control.robot.realtime_cpu is not None:
                os.sched_setaffinity(0, os.sched_getaffinity(0)-{control.robot.realtime_cpu})
            store = module.RecordingStore(config.dataset_root,control,recording,config.task,resume=config.resume)
            self.resources.callback(store.close)
            receiver = JoyReceiver(self.directory/'joy.sock')
            self.resources.callback(receiver.close)
            self.sidecars = self.resources.enter_context(JoySidecars(control,receiver.path,self.directory))
            robot_class = module.FrankaRobotiqRobot
            self.robot = robot_class(replace(control.robot,calibration_dir=self.directory/'calibration'))
            self.resources.callback(self.robot.disconnect)
            self.check_children()
            self.robot.connect()
            self.session = module.RecordingSession(control,recording,self.robot,receiver,store,report=self.report)
            self.resources.callback(self._abort_recording)
            self.collector = None
        else:
            from vla.datacollection.franka_robotiq_rollout.collect_franky import FrankyRolloutCollector, _configure_affinity, _load_deployment
            from vla.robots.franka_robotiq import load_franka_robotiq_config
            from vla.robots.franka_robotiq.robot import FrankaRobotiqRobot
            from vla.datacollection.start_teleop.gamepad_recording import load_recording_config
            from vla.datacollection.start_teleop.rollout_gamepad import PROFILE
            rollout = self.rollout_config()
            if rollout.rtc_deployment_config is None:
                raise ValueError('single-arm deployment requires control.rtc_deployment_config')
            rollout = self.start_policy(rollout)
            control,_ = load_recording_config(PROFILE)
            robot_cfg = load_franka_robotiq_config(rollout.rtc_deployment_config)
            robot_cfg = replace(robot_cfg,human_linear_velocity_control=True,human_angular_velocity_control=True,
                                human_gripper_immediate_hold=True)
            _configure_affinity(_load_deployment(rollout.rtc_deployment_config).get('deployment',{}),robot_cfg.realtime_cpu)
            receiver = JoyReceiver(rollout.joy_socket_path)
            self.resources.callback(receiver.close)
            self.sidecars = self.resources.enter_context(JoySidecars(control,receiver.path,self.directory))
            self.robot = FrankaRobotiqRobot(robot_cfg)
            self.resources.callback(self.robot.disconnect)
            self.collector = FrankyRolloutCollector(replace(rollout,gamepad_profile='sft_record'),self.robot,joy_receiver=receiver)
            self.resources.callback(self.collector.close)
            self.collector.set_control_mode(config.mode)
            self.check_children()
            self.robot.connect()
            self.collector._gamepad_controls.capture_initial_pose(self.robot.get_control_state())
            self.session = None
        self._state_time = None
        self._state_seen = 0.

    def _abort_recording(self):
        if self.session.worker:
            self._finish(discard=True)

    def _finish(self, *, discard=False, outcome='success'):
        self.session.finish(not discard and outcome == 'success', 'web operator',
                            save_failure=not discard and outcome == 'failure')

    def command(self, op, args):
        if self.collector:
            return self.collector_command(op,args)
        s = self.session
        if op == 'enable':
            if s.saving or s.storage_fault:
                raise RuntimeError('save/cleanup pending or storage fault')
            s.controller.enable()
            if s.controller.mode != 'active':
                raise RuntimeError('enable rejected: release inputs and check fresh state')
        elif op == 'record':
            if s.worker is not None or s.storage_fault:
                raise RuntimeError('recording/save already pending or storage fault')
            s.begin()
            if s.worker is None:
                raise RuntimeError('recording rejected: release inputs and check session log')
        elif op in ('save','discard'):
            if s.worker is None or s.saving:
                raise RuntimeError('no active recording or save already pending')
            self._finish(discard=op == 'discard',outcome=args.get('outcome','success'))
            def finalized():
                if s.storage_fault:
                    raise RuntimeError('recording storage fault; inspect diagnostic files before restarting')
                return s.worker is None and not s.saving
            return finalized
        elif op == 'pause':
            if s.worker:
                self._finish(discard=True)
            else:
                s.controller.pause('web pause')
        elif op == 'recover':
            if s.worker or s.storage_fault:
                raise RuntimeError('finish episode first; storage faults require restart with resume')
            s.controller.recover()
            if s.controller.mode == 'fault':
                raise RuntimeError(s.controller.reason)
        elif op == 'stop':
            if s.worker or s.saving or s.storage_fault:
                raise RuntimeError('save/discard must finish successfully before stopping')
            s.controller.pause('web stop')
        elif op == 'critical':
            if not s.worker or s.saving:
                raise RuntimeError('no active episode')
            # Same state used by physical RB edges.
            if s.phase.state == 'after':
                raise RuntimeError('one contiguous critical interval per episode')
            s.phase.state = 'after' if s.phase.active else 'critical'
        else:
            raise ValueError('operation unavailable')

    def tick(self):
        self.check_children()
        self.sidecars.check()
        if time.monotonic() < self._next_control:
            return
        if self.session:
            self.session.tick()
            self._next_control = time.monotonic()+1/self.session.controller.config.control.control_rate
        else:
            self.collector.gamepad_tick()
            if time.monotonic() >= getattr(self,'_next_observation',0.):
                self.collector.tick()
                self._next_observation = time.monotonic()+1/self.collector.config.fps
            self._next_control = time.monotonic()+1/self.collector.config.gamepad_control_rate

    def sample(self):
        if self.collector:
            status = self.collector_status()
            width = None
        else:
            s = self.session
            width = s.last_snapshot.gripper_width if s.last_snapshot is not None else None
            c = s.controller
            status = {'state': 'fault' if s.storage_fault else 'saving' if s.saving else 'recording' if s.worker else c.mode,
                      'enabled': c.mode == 'active', 'recording': bool(s.worker), 'saving': s.saving,
                      'episode': s.worker.index if s.worker else s.store.writer.episode_count,
                      'frames': s.worker.frames if s.worker else 0,
                      'elapsed_s': time.monotonic()-s.started_at if s.worker else 0.,
                      'critical': s.phase.active, 'error': c.reason if c.mode == 'fault' else None,
                      'arms': {'left': {'source':'gamepad','speed':'fast' if c.high_speed else 'normal',
                                        'speed_scale':c.speed_scale,
                                        'transition':abs(c.speed_scale-(c.config.speed_modes.high_multiplier if c.high_speed else 1.))>1e-6}},
                      'available_commands':['enable','pause','record','save','discard','recover','stop','critical'],
                      'modes':[], 'actual_source':'gamepad',
                      'outcomes':['success','failure']}
        state = getattr(self.robot,'_telemetry_state',None)
        stamp = getattr(state,'time',None)
        if stamp != self._state_time:
            self._state_time, self._state_seen = stamp,time.monotonic()
        try:
            from vla.robots.franka_robotiq.robot import GRIPPER_OPEN_WIDTH_M
            width = self.robot._gripper.position() * GRIPPER_OPEN_WIDTH_M
        except (RuntimeError, TimeoutError):
            width = None
        values = franky_values(state,width) if time.monotonic()-self._state_seen <= .5 else {}
        debug = self.robot.command_debug_snapshot()
        command_stamp = debug.get('human_command_at_ns',0)/1e9 or debug.get('command_completed_monotonic_s',0)
        if time.monotonic()-command_stamp <= .5:
            for field,group in [('human_target_xyz','target_tcp'), ('commanded_linear_velocity','target_linear'),
                                ('commanded_angular_velocity','target_angular')]:
                vector(values,'left',group,['x','y','z'],debug.get(field))
        command = self.collector._command if self.collector else None
        if command is not None and time.monotonic()-command.applied_at <= .5:
            vector(values,'left','target_tcp',['x','y','z'],command.action[:3])
        status.update(values=values, signals=descriptors(self.arms)+extra_signals(self.arms), sample_time=time.monotonic(),
                      events=self.events)
        return status

    def images(self):
        return self.robot.get_camera_preview_frames()


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
        rollout = self.rollout_config() if config.source == 'gamepad' else None
        if rollout is not None:
            keys = {'base':'BASE_CAMERA_SERIAL','left_wrist':'LEFT_WRIST_CAMERA_SERIAL','right_wrist':'RIGHT_WRIST_CAMERA_SERIAL'}
            rollout = replace(rollout,cameras=tuple(replace(cam,serial=self.hardware.get(keys[cam.name],cam.serial))
                                                    for cam in rollout.cameras))
            if len({cam.serial for cam in rollout.cameras}) != len(rollout.cameras):
                raise ValueError('camera serials must be distinct')
        if config.workflow == 'deploy':
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
        else:
            from vla.datacollection.franka_robotiq_rollout.collect_ros import RolloutCollector
            self.collector = RolloutCollector(rollout,demonstration=config.workflow == 'collect')
            self.executor.add_node(self.collector)
            self.resources.callback(self.collector.destroy_node)
            self.resources.callback(self.collector.close)
            if config.workflow == 'deploy':
                self.collector.set_control_mode(config.mode)

    def _start_gello(self, env):
        import yaml
        from rclpy.parameter import Parameter
        from vla.paths import resource
        from vla.datacollection.start_teleop.fine_teleop import FineTeleopNode
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
        # Configure each existing arbiter; right arm has independent topics and remains GELLO by default.
        for arm in self.arms:
            params = {'base_link':f'{arm}_fr3_link0','tip_link':f'{arm}_fr3_link8',
                      'robot_state_publisher':f'/{arm}/robot_state_publisher',
                      'raw_joint_topic':f'/{arm}/gello/joint_states',
                      'raw_gripper_topic':f'/{arm}/gello/gripper_width_percent',
                      'measured_joint_topic':f'/{arm}/franka_robot_state_broadcaster/measured_joint_states',
                      'measured_gripper_topic':f'/{arm}/gripper/joint_state_broadcaster_robotiq/joint_states',
                      'joy_topic':f'/{arm}/gamepad/joy','command_joint_topic':f'/{arm}/teleop/joint_states',
                      'command_gripper_topic':f'/{arm}/teleop/gripper_width_percent',
                      'action_gripper_command_topic':f'/{arm}/teleop/gripper_action_width_percent',
                      'gripper_speed_topic':f'/{arm}/gripper/{arm}_gripper_speed_controller/commands',
                      'gripper_action_topic':f'/{arm}/gripper/robotiq_gripper_controller/gripper_cmd'}
            node = FineTeleopNode(managed=True,namespace=arm,parameter_overrides=[Parameter(k,value=v) for k,v in params.items()])
            self.arbiters.append(node)
            self.executor.add_node(node)
            self.resources.callback(node.destroy_node)
            self.resources.callback(node.managed_pause)
        prefix = 'DUAL' if self.config.arms == 'dual' else 'LEFT'
        os.environ[f'{prefix}_FRANKA_DATASET_ROOT'] = self.config.dataset_root
        os.environ[f'{prefix}_FRANKA_TASK'] = self.config.task
        os.environ.update({k:v for k,v in self.hardware.items() if 'CAMERA_SERIAL' in k})
        module = importlib.import_module('vla.datacollection.franka_robotiq_' +
                                        ('dual' if self.config.arms == 'dual' else 'single_left') + '.auto_record')
        cls = module.DualLeRobotRecorder if self.config.arms == 'dual' else module.DirectLeRobotRecorder
        self.recorder = cls()
        self.executor.add_node(self.recorder)
        self.resources.callback(self.recorder.destroy_node)
        self.resources.callback(self.recorder.close,save_active=False)

    def command(self, op, args):
        if self.collector:
            return self.collector_command(op,args)
        r = self.recorder
        if op == 'enable':
            if self.storage_fault:
                raise RuntimeError(self.storage_fault)
            try:
                for node in self.arbiters:
                    node.managed_enable()
            except Exception:
                for node in self.arbiters:
                    node.managed_pause()
                raise
            self.enabled = True
        elif op == 'record':
            if not self.enabled or r.is_recording:
                raise RuntimeError('enable first; only one active episode')
            r.start_recording()
            if not r.is_recording:
                raise RuntimeError('recording preflight rejected; see runner.log')
            self.record_started = time.monotonic()
        elif op in ('save','discard'):
            if not r.is_recording:
                raise RuntimeError('no active episode')
            if op == 'save' and args.get('outcome','success') != 'success':
                raise ValueError('GELLO recording supports save or discard; failure labels are unavailable')
            for node in self.arbiters:
                node.managed_pause()
            self.enabled = False
            before = r.dataset.episode_count
            try:
                r.stop_recording(discard=op == 'discard')
                if op == 'save' and r.dataset.episode_count <= before:
                    raise RuntimeError('episode not saved (insufficient frames or storage error); see runner.log')
            except Exception as error:
                self.storage_fault = str(error)
                raise
        elif op in ('pause','stop'):
            if op == 'stop' and (r.is_recording or self.storage_fault):
                raise RuntimeError('save/discard must finish successfully before stopping')
            for node in self.arbiters:
                node.managed_pause()
            self.enabled = False
            if r.is_recording:
                if op == 'stop':
                    raise RuntimeError('save or discard the current episode first')
                r.stop_recording(discard=True)
            if self.storage_fault and op == 'stop':
                raise RuntimeError(self.storage_fault)
        elif op == 'recover':
            if r.is_recording or self.storage_fault:
                raise RuntimeError('storage fault needs inspection; recovery cannot discard data')
            for node in self.arbiters:
                node.managed_fault = None
            self.enabled = False
        else:
            raise ValueError('operation unavailable for GELLO collection')

    def tick(self):
        self.check_children()
        for _ in range(16):
            self.executor.spin_once(timeout_sec=0)
        if self.recorder:
            with self.recorder._lock:
                latest = dict(self.recorder._latest)
            for arm, node in zip(self.arms, self.arbiters):
                entry = latest.get(f'{arm}.robot_state') or (latest.get('robot_state') if arm == 'left' else None)
                if entry and entry[0] != self._health_stamps.get(arm):
                    self._health_stamps[arm] = entry[0]
                    try:
                        self.health.check(arm, entry[1])
                    except RuntimeError as error:
                        node.managed_fault = str(error)
        if self.recorder and any(n.managed_fault for n in self.arbiters):
            self.enabled = False
            for node in self.arbiters:
                if node.managed_enabled:
                    node.managed_pause()
            if self.recorder.is_recording:
                self.recorder.stop_recording(discard=True)

    def sample(self):
        node = self.collector or self.recorder
        with node._lock:
            latest = dict(node._latest)
        if self.collector:
            status = self.collector_status()
        else:
            r = self.recorder
            status = {'state':'fault' if self.storage_fault or any(n.managed_fault for n in self.arbiters) else
                      'recording' if r.is_recording else 'active' if self.enabled else 'standby',
                      'enabled':self.enabled, 'recording':r.is_recording,'episode':r.dataset.episode_count,
                      'frames':len(r._states),'saving':False,
                      'elapsed_s':time.monotonic()-self.record_started if r.is_recording and self.record_started else 0.,
                      'error':self.storage_fault or next((n.managed_fault for n in self.arbiters if n.managed_fault),None),
                      'actual_source':'gello','arms':{a:{'source':n.mode.value,'speed':'fixed','speed_scale':None,'transition':n.mode.value=='returning_to_gello'}
                                                      for a,n in zip(self.arms,self.arbiters)},
                      'available_commands':['enable','pause','record','save','discard','recover','stop'],'modes':[], 'outcomes':['success']}
        values = ros_values(latest,self.arms)
        if self.collector:
            command = self.collector._last_command_snapshot
            if command is not None and time.monotonic()-command.published_at_ns/1e9 <= .5:
                for arm, target in command.arms.items():
                    vector(values,arm,'target_joint',[f'q{i}' for i in range(1,8)],target.joints)
        else:
            for arm in self.arms:
                entry = latest.get(f'{arm}.command_joint') or (latest.get('arm_command') if arm=='left' else None)
                if entry and time.monotonic()-entry[0]/1e9 <= .5:
                    vector(values,arm,'target_joint',[f'q{i}' for i in range(1,8)],entry[1].position)
        status.update(values=values,signals=descriptors(self.arms)+extra_signals(self.arms),sample_time=time.monotonic())
        return status

    def images(self):
        node = self.collector or self.recorder
        with node._lock:
            latest = dict(node._latest)
        prefix = 'camera.' if self.collector else ''
        return {role:(latest[prefix+role][0]/1e9,latest[prefix+role][1])
                for role in self.config.cameras if prefix+role in latest}
