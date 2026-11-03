"""Cached wrench channels use the recording coordinate frame, without hardware IO."""
import time
from types import SimpleNamespace as NS

import numpy as np
import pytest

from vla.web.demo import descriptors
from vla.web.telemetry import franky_values, ros_values


def ros_state():
    return NS(
        o_t_ee=NS(pose=NS(position=NS(x=.4, y=0., z=.3),
                         orientation=NS(x=0., y=0., z=0., w=1.))),
        measured_joint_state=NS(name=[], position=[0.]*7, velocity=[0.]*7, effort=[0.]*7),
        o_f_ext_hat_k=NS(wrench=NS(force=NS(x=1., y=2., z=3.), torque=NS(x=4., y=5., z=6.))),
        k_f_ext_hat_k=NS(wrench=NS(force=NS(x=99., y=99., z=99.), torque=NS(x=99., y=99., z=99.))),
    )


def assert_wrench(values, arm):
    assert [values[f'{arm}.{group}.{axis}']
            for group in ('tcp_force', 'tcp_torque') for axis in 'xyz'] == [1., 2., 3., 4., 5., 6.]


@pytest.mark.parametrize('arms', [['left'], ['left', 'right']])
def test_ros_wrench_uses_base_frame_and_disappears_when_stale(arms):
    state = ros_state()
    latest = {f'{arm}.robot_state': (time.monotonic_ns(), state) for arm in arms}
    if len(arms) == 1:
        latest['robot_state'] = latest.pop('left.robot_state')  # Legacy single-arm cache.
    values = ros_values(latest, arms)
    for arm in arms:
        assert_wrench(values, arm)
    assert ros_values({key: (0, state) for key in latest}, arms) == {}
    del state.o_f_ext_hat_k
    assert not any('.tcp_force.' in key or '.tcp_torque.' in key for key in ros_values(latest, arms))


def test_franky_wrench_preserves_units_and_leaves_unavailable_channels_empty():
    state = NS(O_T_EE=np.eye(4), O_F_ext_hat_K=[1., 2., 3., 4., 5., 6.], K_F_ext_hat_K=[99.]*6)
    assert_wrench(franky_values(state), 'left')
    state.O_F_ext_hat_K = [np.nan, 2., 3., 4., np.inf, 6.]
    values = franky_values(state)
    assert 'left.tcp_force.x' not in values
    assert 'left.tcp_torque.y' not in values
    assert values['left.tcp_force.y'] == 2.
    del state.O_F_ext_hat_K
    assert not any('.tcp_force.' in key or '.tcp_torque.' in key for key in franky_values(state))


def test_wrench_descriptors_separate_force_and_torque_without_changing_defaults():
    signals = descriptors(['left', 'right'])
    wrench = [s for s in signals if s['group'] in ('tcp_force', 'tcp_torque')]
    assert len(wrench) == 12
    assert all(s['source'] == 'estimated' and not s['default'] for s in wrench)
    assert {s['unit'] for s in wrench if s['group'] == 'tcp_force'} == {'N'}
    assert {s['unit'] for s in wrench if s['group'] == 'tcp_torque'} == {'N·m'}
    assert {s['group'] for s in signals if s['default']} == {'tcp', 'gripper'}
