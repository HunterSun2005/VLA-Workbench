#!/usr/bin/env python3
"""Actual DDS/controller_manager/production controller, fixed-state mock hardware only.

Run via validate_ros_control.sh. No robot launch scripts, devices, weights or camera APIs.
Artifacts stay in a unique /tmp directory. Only child process groups are terminated.
"""
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import yaml
import xacro
import rclpy
from rclpy.duration import Duration
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import UInt64
from ament_index_python.packages import get_package_share_directory
from vla.actions import ActionSpec
from vla.control.ros import ControlCoordinator, ArmMeasurement
from vla.control.ros.cartesian_servo import CartesianServo
from vla.control.ros.transport import joint_command, controller_identity
from vla.datacollection.start_teleop.franka_kinematics import build_kinematic_model
from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config

ROOT = Path(__file__).resolve().parents[1]
Q = np.array([0., -.4, .1, -1.8, .1, 1.4, .5])


def distribution(values):
    v = np.asarray(values, dtype=float)
    return {'count': len(v), **({k: float(np.quantile(v, q)) for k, q in
        [('p50', .5), ('p95', .95), ('p99', .99), ('max', 1.)]} if len(v) else {})}


def intervals(stamps):
    v = np.asarray(stamps, dtype=np.int64)
    return {'hz': float((len(v)-1)*1e9/(v[-1]-v[0])) if len(v)>1 else 0.,
            'period_ms': distribution(np.diff(v)/1e6)}


def domain_lease():
    # Refuse the caller's domain; advisory lease prevents concurrent validation runs.
    original = int(os.environ.get('ROS_DOMAIN_ID', '0'))
    for domain in range(170, 200):
        if domain == original:
            continue
        stream = open(f'/tmp/vla-validation-domain-{domain}-{os.getuid()}.lock', 'a')
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.environ.update(ROS_DOMAIN_ID=str(domain), ROS_LOCALHOST_ONLY='1')
            return stream, domain
        except BlockingIOError:
            stream.close()
    raise RuntimeError('no isolated validation domain available')


def require_empty_domain():
    """Fail closed if an unrelated process already uses the leased domain."""
    probe = Node('vla_validation_domain_probe')
    try:
        end = time.monotonic()+1.5
        while time.monotonic() < end:
            rclpy.spin_once(probe, timeout_sec=.05)
            peers = [name for name in probe.get_node_names()
                     if name != probe.get_name()]
            if peers:
                raise RuntimeError(f'validation domain already contains ROS nodes: {peers}')
    finally:
        probe.destroy_node()


def description(arm, directory):
    path = Path(get_package_share_directory('franka_description'))/'robots/fr3/fr3.urdf.xacro'
    xml = xacro.process_file(str(path), mappings={
        'hand':'false', 'arm_prefix':arm, 'ros2_control':'false'}).toxml()
    model = build_kinematic_model(xml, f'{arm}_fr3_link0', f'{arm}_fr3_link8', .05)
    root = ET.fromstring(xml)
    assert not root.findall('ros2_control')
    control = ET.SubElement(root, 'ros2_control', name=f'{arm}_timing_mock', type='system')
    hardware = ET.SubElement(control, 'hardware')
    ET.SubElement(hardware, 'plugin').text = 'vla_control_validation/TimingMockSystem'
    ET.SubElement(hardware, 'param', name='timing_output').text = str(directory/f'{arm}-updates.bin')
    for name, initial in zip(model.joint_names, Q):
        joint = ET.SubElement(control, 'joint', name=name)
        ET.SubElement(joint, 'command_interface', name='effort')
        for interface, value in [('position', initial), ('velocity', 0.)]:
            state = ET.SubElement(joint, 'state_interface', name=interface)
            ET.SubElement(state, 'param', name='initial_value').text = str(value)
    assert [p.text for p in root.findall('ros2_control/hardware/plugin')] == ['vla_control_validation/TimingMockSystem']
    return ET.tostring(root, encoding='unicode'), model


class Session:
    def __init__(self, directory):
        self.directory, self.children, self.streams = directory, {}, []
        self.nodes, self.diag, self.last_seen = {}, {}, {}
        self.command_pubs, self.reset_pubs, self.models = {}, {}, {}
        self.drop_publish, self.drop_reset, self.drop_ack = set(), set(), set()
        self.drop_observation = False
        self.sequence = 0
        self.sent, self.received, self.delays = {}, {}, []
        self.tick_times, self.accept_times = [], []
        self.c = None
        self.health_errors = []
        self.executor = SingleThreadedExecutor()
        self.node = Node('vla_validation_bridge')
        self.executor.add_node(self.node)

    def initialize(self):
        directory = self.directory
        config = load_rollout_config(ROOT/'src/vla/datacollection/franka_robotiq_rollout/configs/dual.yaml')
        self.config = config
        controller_params = yaml.safe_load((ROOT/'ros_ws/src/franka_fr3_arm_controllers/config/controllers.yaml').read_text())['/**']['joint_impedance_controller']['ros__parameters']
        for arm in ('left','right'):
            xml, self.models[arm] = description(arm, directory)
            params = {
                f'/{arm}/controller_manager': {'ros__parameters': {
                    'robot_description': xml, 'update_rate': 1000,
                    'joint_impedance_controller': {'type':'franka_fr3_arm_controllers/JointImpedanceController'},
                    'joint_state_broadcaster': {'type':'joint_state_broadcaster/JointStateBroadcaster'}}},
                f'/{arm}/joint_impedance_controller': {'ros__parameters': {**controller_params,'arm_id':'fr3'}},
                f'/{arm}/joint_state_broadcaster': {'ros__parameters': {'update_rate':30, 'use_local_topics': True}}
            }
            file = directory/f'{arm}.yaml'; file.write_text(yaml.safe_dump(params))
            provider = Node('robot_state_publisher', namespace=arm)
            provider.declare_parameter('robot_description', xml)
            self.nodes[arm] = provider; self.executor.add_node(provider)
            self.command_pubs[arm] = self.node.create_publisher(JointState, f'/{arm}/gello/joint_states', 1)
            self.reset_pubs[arm] = self.node.create_publisher(UInt64, f'/{arm}/teleop/reset', 10)
            self.node.create_subscription(JointState, f'/{arm}/teleop/controller_state',
                                          lambda msg, arm=arm: self.on_diagnostic(arm, msg), 10)
            self.node.create_subscription(JointState, f'/{arm}/joint_state_broadcaster/joint_states',
                                          lambda msg, arm=arm: self.on_state(arm, msg), 10)
            self.start(arm, ['/opt/ros/humble/lib/controller_manager/ros2_control_node', '--ros-args',
                             '-r', f'__ns:=/{arm}', '--params-file', str(file)])
        self.servos = {a: CartesianServo(m, config) for a,m in self.models.items()}

    def start(self, name, argv):
        stream = open(self.directory/f'{name}.log', 'w'); self.streams.append(stream)
        child = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.children[name] = child
        return child

    def spin(self, seconds):
        end = time.monotonic()+seconds
        while time.monotonic()<end:
            self.executor.spin_once(timeout_sec=min(.002, max(0,end-time.monotonic())))

    def wait(self, predicate, timeout=5.):
        end = time.monotonic()+timeout
        while not predicate():
            if time.monotonic()>end:
                raise TimeoutError(f'validation deadline; logs: {self.directory}')
            self.spin(.005)

    def configure(self):
        for arm in ('left','right'):
            child = self.start(f'{arm}-spawner', ['/opt/ros/humble/lib/controller_manager/spawner',
                'joint_state_broadcaster', 'joint_impedance_controller', '-c', f'/{arm}/controller_manager',
                '--controller-manager-timeout', '20'])
            self.wait(lambda: child.poll() is not None, 30)
            if child.returncode:
                raise RuntimeError(f'{arm} spawner failed; see logs')
        self.wait(lambda: len(self.diag)==2 and len(self.last_seen)==2)

    def on_state(self, arm, message):
        now = time.monotonic_ns()
        joints = dict(zip(message.name, message.position))
        q = np.array([joints[n] for n in self.models[arm].joint_names])
        self.last_seen[arm] = now, q
        if self.c is not None and not self.drop_observation:
            self.c.observe(arm, ArmMeasurement(q, .45, np.array([.015,-.012,.12]), np.eye(3), now))

    def on_diagnostic(self, arm, message):
        now = time.monotonic_ns()
        generation, command_id, fault = controller_identity(message)
        self.diag[arm] = generation, command_id, fault, np.array(message.position), now
        key = arm, generation, command_id
        if key in self.sent and key not in self.received:
            self.received[key] = now
            self.delays.append((now-self.sent[key])/1e6)
        if self.c is None or arm in self.drop_ack or generation != self.c.generation:
            return
        if fault and self.c.mode != 'fault':
            self.health_errors.append(f'{arm}: watchdog')
            self.c.transition(fault=True, reason=f'{arm} watchdog')
        else:
            self.c.acknowledge(arm, generation, command_id, message.position, now)

    def publish(self, arm, q, gripper, generation, command_id):
        if arm in self.drop_publish:
            return
        self.sequence += 1
        self.command_pubs[arm].publish(joint_command(self.models[arm].joint_names, q, generation,
            command_id, self.sequence, self.node.get_clock().now().to_msg()))
        self.sent.setdefault((arm, generation, command_id), time.monotonic_ns())

    def reset(self, generation):
        for arm, pub in self.reset_pubs.items():
            if arm not in self.drop_reset:
                pub.publish(UInt64(data=generation))

    def coordinator(self, space='joint'):
        self.c = ControlCoordinator(ActionSpec(space, ('left','right')), self.servos,
            {a:(np.array([-.8,-.8,-.1]),np.array([.8,.8,1.2])) for a in self.models}, self.publish, self.reset)
        self.spin(.08)
        self.c.transition('human')
        self.wait(lambda: self.c.mode == 'active')

    def run_for(self, seconds, *, commands=True, record=False):
        end = time.monotonic()+seconds
        deadline = time.monotonic()
        while time.monotonic()<end:
            now = time.monotonic()
            if now < deadline:
                self.executor.spin_once(timeout_sec=min(.002, deadline-now)); continue
            try:
                if self.c.mode == 'active':
                    if commands:
                        goals = {a: self.last_seen[a][1]+.004*np.sin(now*2) for a in self.models}
                        target = self.c.gello_target(goals, {a:.45 for a in self.models})
                        accepted = self.c.accept(target, source=self.c.owner, generation=self.c.generation)
                        if accepted is not None and record:
                            self.accept_times.append(accepted.accepted_at_ns)
                    self.c.tick(.01)
                    if record: self.tick_times.append(time.monotonic_ns())
            except (KeyError, ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                # Same exception-to-fault boundary as RolloutCollector's timers.
                if self.c.mode != 'fault':
                    self.c.transition(fault=True, reason=str(error))
                raise
            deadline += .01
            if time.monotonic()-deadline > .05: deadline=time.monotonic()+.01
            self.executor.spin_once(timeout_sec=0)

    def raw(self, arm, generation, command_id, sequence, *, age=0, q=None):
        stamp = (self.node.get_clock().now()-Duration(seconds=age)).to_msg()
        self.command_pubs[arm].publish(joint_command(self.models[arm].joint_names,
            Q if q is None else q, generation, command_id, sequence, stamp))

    def faults(self):
        checks = []
        self.coordinator('tcp'); self.run_for(.15)
        assert self.c.mode == 'active'; checks.append('tcp production bridge')
        self.coordinator()
        self.run_for(.1)
        old = self.c.generation
        self.drop_ack.add('right')
        self.c.transition('policy'); self.spin(.12)
        assert self.c.mode == 'resetting' and self.c.owner is None
        checks.append('delayed right reset blocks both arms')
        self.drop_ack.clear(); self.wait(lambda:self.c.mode=='active')
        self.run_for(.1)
        # Hold coordinator ticks while malformed DDS packets are injected, <watchdog duration.
        baseline = self.diag['left'][1]
        for gen, ident, seq, age in [(old,9999,9999,0), (self.c.generation,9999,9999,1),
                                    (self.c.generation,9999,9999,-1), (self.c.generation,0,9999,0)]:
            self.raw('left',gen,ident,seq,age=age); self.spin(.025)
            assert self.diag['left'][1] == baseline
        checks.append('revoked stale future and out-of-order commands rejected')
        self.run_for(.08)
        generation, ident, _, before, _ = self.diag['left']
        # A first packet is accepted; a duplicate sequence with another target must not mutate it.
        sequence = self.sequence+10
        self.raw('left',generation,ident+1,sequence,q=Q+.006); self.spin(.04)
        before = self.diag['left'][3].copy()
        self.raw('left',generation,ident+1,sequence,q=Q+.1); self.spin(.04)
        np.testing.assert_array_equal(self.diag['left'][3],before)
        self.sequence = sequence+1
        checks.append('duplicate sequence cannot change target')
        self.c.transition('human'); self.wait(lambda:self.c.mode=='active')
        self.run_for(.08)
        self.drop_reset.add('right')
        self.c.transition('policy'); self.spin(1.05)
        try: self.c.tick()
        except RuntimeError: pass
        assert self.c.mode == 'fault'
        checks.append('missing reset acknowledgement latches fault')
        self.drop_reset.clear(); self.c.transition('human'); self.wait(lambda:self.c.mode=='active')
        self.run_for(.08)
        self.drop_observation = True
        try: self.run_for(.65, commands=False)
        except RuntimeError: pass
        assert self.c.mode == 'fault'; checks.append('stale observation resets all arms')
        self.drop_observation = False; self.spin(.08)
        self.c.transition('human'); self.wait(lambda:self.c.mode=='active'); self.run_for(.08)
        # Abrupt end of command source, while controller_manager continues running.
        self.spin(.65)
        assert any('watchdog' in e for e in self.health_errors)
        assert self.c.mode == 'fault'; checks.append('lost command source triggers watchdog')
        self.c.transition('human'); self.wait(lambda:self.c.mode=='active')
        self.run_for(.08)
        original = self.c.publish
        def partial(*args):
            if args[0]=='right': raise RuntimeError('injected right publisher failure')
            original(*args)
        self.c.publish = partial
        try: self.c.tick()
        except RuntimeError: pass
        assert self.c.mode=='fault' and self.c.accepted is None
        self.c.publish = original; checks.append('partial publish revokes both arms')
        self.c.transition('human'); self.wait(lambda:self.c.mode=='active')
        return checks

    def resources(self):
        result = {}
        for name, pid in [('bridge', os.getpid()), *[(n,p.pid) for n,p in self.children.items() if n in ('left','right')]]:
            try:
                fields = Path(f'/proc/{pid}/stat').read_text().split()
                result[name]={'cpu_seconds': (int(fields[13])+int(fields[14]))/os.sysconf('SC_CLK_TCK'),
                              'rss_mb': int(fields[23])*os.sysconf('SC_PAGE_SIZE')/1024**2}
            except FileNotFoundError: pass
        return result

    def close(self):
        for child in self.children.values():
            if child.poll() is None: os.killpg(child.pid, signal.SIGINT)
        for child in self.children.values():
            try: child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait(timeout=3)
        self.executor.shutdown()
        for node in [self.node,*self.nodes.values()]: node.destroy_node()
        for stream in self.streams: stream.close()


def main():
    if not __debug__:
        raise RuntimeError('validation requires Python assertions; unset PYTHONOPTIMIZE')
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration', type=float, default=600)
    args = parser.parse_args()
    if not 0 < args.duration <= 900: parser.error('duration must be in (0,900]')
    directory = Path(tempfile.mkdtemp(prefix='vla-ros-control-'))
    os.environ.update(ROS_HOME=str(directory/'ros'), ROS_LOG_DIR=str(directory/'ros/log'))
    lease, domain = domain_lease()
    print(f'ARTIFACTS={directory} ROS_DOMAIN_ID={domain}', flush=True)
    rclpy.init()
    session = None
    report = {'domain':domain, 'hardware':'fixed-state GenericSystem subclass; no dynamics', 'checks':[]}
    try:
        require_empty_domain()
        session = Session(directory); session.initialize(); session.configure()
        report['checks'] = session.faults()
        print('Fault checks passed:', report['checks'], flush=True)
        session.sent.clear(); session.received.clear(); session.delays.clear()
        start = time.monotonic_ns(); report['start_ns'] = start
        report['resource_samples'] = [{'elapsed_s':0., **session.resources()}]
        while (time.monotonic_ns()-start)/1e9 < args.duration:
            remaining = args.duration-(time.monotonic_ns()-start)/1e9
            session.run_for(min(10.,remaining), record=True)
            report['resource_samples'].append({'elapsed_s':(time.monotonic_ns()-start)/1e9, **session.resources()})
            if session.c.mode != 'active': raise RuntimeError(f'unexpected benchmark fault: {session.c.reason}')
            print(f"progress {(time.monotonic_ns()-start)/1e9:.1f}/{args.duration}s", flush=True)
        end = time.monotonic_ns(); report['end_ns'] = end
        report['accept'] = intervals(session.accept_times)
        report['tracking'] = intervals(session.tick_times)
        report['first_diagnostic_delay_ms'] = distribution(session.delays)
        skew=[]
        for arm,gen,ident in session.received:
            if arm=='left' and ('right',gen,ident) in session.received:
                skew.append(abs(session.received[('left',gen,ident)]-session.received[('right',gen,ident)])/1e6)
        report['first_diagnostic_observation_skew_ms'] = distribution(skew)
        # Exit one controller_manager only after the normal performance window.
        os.killpg(session.children['right'].pid, signal.SIGINT)
        try: session.run_for(.7)
        except RuntimeError: pass
        assert session.c.mode=='fault'
        report['checks'].append('controller process exit faults remaining arm')
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        if session: session.close()
        rclpy.shutdown(); lease.close()
        for arm in ('left','right'):
            path=directory/f'{arm}-updates.bin'
            if path.exists() and 'end_ns' in report:
                values=np.fromfile(path,dtype=np.int64)
                values=values[(values>=report['start_ns']) & (values<=report['end_ns'])]
                report[arm+'_controller']=intervals(values)
                report[arm+'_mock']=json.loads(Path(str(path)+'.json').read_text())
        if 'tracking' in report and all(a+'_controller' in report for a in ('left','right')):
            report['frequency_target_passed']=all(abs(report[key]['hz']/hz-1)<=.03 for key,hz in
                [('accept',30),('tracking',100),('left_controller',1000),('right_controller',1000)])
        (directory/'report.json').write_text(json.dumps(report,indent=2))
        print('REPORT='+str(directory/'report.json'),flush=True)
    if not report.get('frequency_target_passed', False):
        raise SystemExit('30/100/1000 Hz frequency acceptance failed; see report')

if __name__=='__main__': main()
