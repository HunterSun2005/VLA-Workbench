"""Synthetic backend: never imports or opens robot devices."""
from __future__ import annotations
import math
import time
import numpy as np


def descriptors(arms):
    result = []
    for arm in arms:
        for group, names, unit in [('tcp', ('x','y','z'), 'm'), ('orientation', ('rx','ry','rz'), 'rad'),
                                   ('joint', tuple(f'q{i}' for i in range(1,8)), 'rad'),
                                   ('velocity', tuple(f'dq{i}' for i in range(1,8)), 'rad/s'),
                                   ('torque', tuple(f'tau{i}' for i in range(1,8)), 'Nm'),
                                   ('tcp_force', ('x','y','z'), 'N'),
                                   ('tcp_torque', ('x','y','z'), 'N·m'),
                                   ('gripper', ('width',), 'm')]:
            for name in names:
                result.append({'key': f'{arm}.{group}.{name}', 'arm': arm, 'group': group,
                               'label': name, 'unit': unit,
                               'source': 'estimated' if group in ('tcp_force','tcp_torque') else 'measured',
                               'default': group in ('tcp','gripper')})
    return result


class DemoBackend:
    def __init__(self, config, directory, root=None):
        self.config = config
        from pathlib import Path
        from .pose_control import PoseActions
        self._demo_poses = {}
        self.poses = PoseActions(self, root or Path(directory), demo=True)
        self.started = time.monotonic()
        self.enabled = self.recording = self.human = self.fault = False
        self.episode = self.frames = 0
        self.mode = config.mode
        self.actual = 'vla'
        self.probability = None
        self.speed = 1.
        self.critical = False
        self.saved = 0
        self.closed = False
        self.arms = ['left','right'] if config.arms == 'dual' else ['left']
        self.signals = descriptors(self.arms)
        self._images = {}
        self._image_index = -1
        self._next_auto = 0.
        self._votes = 0
        self._last = time.monotonic()
        self._recorded = 0.

    def tick(self):
        self.poses.tick()
        now = time.monotonic()
        if self.recording and self.enabled:
            self._recorded += now - self._last
            self.frames = int(self._recorded * 30)
        self._last = now
        if self.config.workflow == 'deploy':
            self.probability = .5 + .49 * math.sin((now-self.started)/3)
            if self.mode == 'auto' and self.enabled and not self.human and now >= self._next_auto:
                self._next_auto = now + .3
                transition = self.probability > .8 if self.actual == 'vla' else self.probability < .2
                self._votes = self._votes + 1 if transition else 0
                if self._votes >= 2:
                    self.actual = 'rlt' if self.actual == 'vla' else 'vla'
                    self._votes = 0
            elif self.mode != 'auto':
                self.actual = self.mode

    def command(self, op, args):
        if self.poses.command(op, args):
            return
        if op == 'enable':
            if self.fault:
                raise RuntimeError('recover first')
            self.enabled = True
        elif op == 'pause':
            self.enabled = False
        elif op == 'record':
            if not self.enabled or self.recording:
                raise RuntimeError('enable first; only one active episode')
            self.recording = True
            self.frames = 0
            self._recorded = 0.
        elif op in ('save','discard'):
            if not self.recording:
                raise RuntimeError('no active episode')
            self.recording = self.enabled = False
            if op == 'save':
                self.episode += 1
                self.saved += 1
        elif op == 'mode':
            if args.get('mode') not in self.config.modes(True):
                raise ValueError('mode unavailable')
            self.mode = args['mode']
            self._votes = 0
        elif op == 'intervention':
            if self.config.workflow != 'deploy' or not self.recording:
                raise RuntimeError('intervention requires running deployment')
            self.human = not self.human
        elif op == 'critical':
            if not self.recording:
                raise RuntimeError('no active episode')
            self.critical = not self.critical
        elif op == 'recover':
            if self.recording:
                raise RuntimeError('finish episode first')
            self.fault = False
            self.enabled = False
        elif op == 'stop':
            self.enabled = False
            if self.recording:
                raise RuntimeError('save or discard the current episode first')
        else:
            raise ValueError('unsupported operation')

    def sample(self):
        now = time.monotonic()
        t = now - self.started
        values = {}
        for i, item in enumerate(self.signals):
            group = item['group']
            value = math.sin(t*.8+i*.2)
            values[item['key']] = (.4 + .04*value if group == 'tcp' else
                                  .04 + .02*value if group == 'gripper' else
                                  2*value if group == 'torque' else .2*value)
        for arm, pose in self._demo_poses.items():
            from scipy.spatial.transform import Rotation
            for axis, value in zip('xyz', pose['position']):
                values[f'{arm}.tcp.{axis}'] = value
            for axis, value in zip('xyz', Rotation.from_matrix(pose['rotation']).as_rotvec()):
                values[f'{arm}.orientation.r{axis}'] = float(value)
        source = ('human' if self.human else self.actual) if self.config.workflow == 'deploy' else self.config.source
        status = {'state': 'fault' if self.fault else 'recording' if self.recording else 'active' if self.enabled else 'standby',
                'enabled': self.enabled, 'recording': self.recording, 'episode': self.episode,
                'frames': self.frames, 'elapsed_s': self._recorded, 'saving': False, 'critical': self.critical,
                'requested_mode': self.mode, 'actual_source': source, 'switch_pending': False,
                'probability': self.probability, 'inference_ms': 35. if self.config.workflow == 'deploy' else None,
                'arms': {a: {'source': self.config.source if source == 'human' else source,
                            'speed': 'normal' if self.config.source == 'gamepad' else 'fixed',
                            'speed_scale': 1. if self.config.source == 'gamepad' else None} for a in self.arms},
                'signals': self.signals, 'values': values, 'sample_time': now,
                'available_commands': ['enable','pause','record','save','discard','recover','stop','critical'] +
                    (['mode','intervention'] if self.config.workflow == 'deploy' else []),
                'modes': self.config.modes(True), 'outcomes':['success','failure'] if self.config.workflow == 'deploy' or (self.config.arms == 'single_left' and self.config.source == 'gamepad') else ['success'], 'demo': True}
        return self.poses.decorate(status)

    def images(self):
        now = time.monotonic()
        fps = 30 if self.config.workflow == 'collect' else 10
        index = int((now-self.started) * fps)
        if index != self._image_index:
            self._image_index = index
            captured_at = self.started + index/fps
            yy, xx = np.mgrid[:240, :320]
            self._images = {role: (captured_at, np.stack(((xx+int(now*30)+i*40)%256,
                              (yy*2+i*60)%256, ((xx+yy)//2+i*90)%256), axis=-1).astype(np.uint8))
                            for i, role in enumerate(self.config.cameras)}
        return self._images

    def close(self):
        if self.poses.moving:
            self.poses.finish("cancelled", "会话结束")
        self.closed = True
