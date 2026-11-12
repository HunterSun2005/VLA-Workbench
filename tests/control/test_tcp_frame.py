"""Installation offset belongs at the hardware boundary, for all TCP commands."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from vla.control.franky.tcp_frame import TCPFrame


@pytest.mark.parametrize('angles', [[0., 0., -45.], [20., -15., 70.]])
def test_task_frame_round_trip_keeps_tcp_origin_and_full_orientation(angles):
    mount = Rotation.from_euler('xyz', angles, degrees=True)
    frame = TCPFrame([.01, -.02, .174, *mount.as_quat()])
    for r in [np.diag([1., -1., -1.]), Rotation.from_euler('xyz', [.3, -.8, 1.2]).as_matrix()]:
        logical = np.r_[[.42, -.12, .31], Rotation.from_matrix(r).as_quat()]
        hardware = frame.to_hardware(logical)
        np.testing.assert_array_equal(hardware[:3], logical[:3])
        np.testing.assert_allclose(Rotation.from_quat(hardware[3:]).as_matrix(), r @ mount.as_matrix(), atol=1e-12)
        result = frame.to_logical(hardware)
        np.testing.assert_allclose(Rotation.from_quat(result[3:]).as_matrix(), r, atol=1e-12)
        np.testing.assert_array_equal(result[:3], logical[:3])
    np.testing.assert_allclose(frame.logical_tool, [.01, -.02, .174, 0, 0, 0, 1])


def test_downward_action_has_zero_yaw_and_hardware_has_installation_bias():
    frame = TCPFrame([0, 0, .174, *Rotation.from_euler('z', -45, degrees=True).as_quat()])
    logical = [0.4, 0., .3, 1., 0., 0., 0.]
    hardware = frame.to_hardware(logical)
    rotation = Rotation.from_quat(hardware[3:]).as_matrix()
    np.testing.assert_allclose(rotation[:, 2], [0, 0, -1], atol=1e-12)
    assert np.arctan2(rotation[1, 0], rotation[0, 0]) == pytest.approx(np.pi/4)
    np.testing.assert_allclose(frame.to_logical(hardware), logical, atol=1e-12)


def test_installation_change_requires_new_session_but_quaternion_sign_does_not():
    pose = np.r_[[0, 0, .174], Rotation.from_euler('z', -.7).as_quat()]
    frame = TCPFrame(pose)
    frame.check_tool(np.r_[pose[:3], -pose[3:]])
    for changed in (pose + [0, 0, .01, 0, 0, 0, 0], [0, 0, .174, 0, 0, 0, 1]):
        with pytest.raises(RuntimeError, match='tool transform changed'):
            frame.check_tool(changed)


def test_calibrated_physical_gamepad_rotation_keeps_its_direction():
    mount = Rotation.from_euler('z', -45, degrees=True).as_matrix()
    hardware_rotation = Rotation.from_euler('xyz', [.4, -.3, 1.1]).as_matrix()
    logical_rotation = hardware_rotation @ mount.T
    physical_angular = np.array([.12, -.08, .03])
    logical_angular = mount @ physical_angular
    np.testing.assert_allclose(logical_rotation @ logical_angular,
                               hardware_rotation @ physical_angular, atol=1e-12)


def test_missing_installation_transform_cannot_start_tracker():
    from types import SimpleNamespace
    from vla.control.franky.worker import FrankyArm
    arm = FrankyArm('left', 'mock')
    arm.robot = SimpleNamespace(join_motion=lambda: None, state=SimpleNamespace(q=np.zeros(7)))
    arm.franky = SimpleNamespace(CartesianImpedanceTracker=lambda *a, **kw: pytest.fail('tracker started without tool calibration'))
    with pytest.raises(RuntimeError, match='measured F_T_EE is required'):
        arm.reset(1)
    assert arm.tracker is None


def test_real_wheel_receives_offset_once_and_feedback_is_logical():
    import json
    import time
    from tests.control.test_franky_process import process_worker, request
    from vla.control.franky.tcp_frame import TCP_FRAME
    mount = Rotation.from_euler('z', -45, degrees=True)
    tool = [0, 0, .174, *mount.as_quat()]
    logical = [.4, -.1, .3, 1., 0., 0., 0.]
    with process_worker(tool_pose=tool) as (root, _, _):
        (root/'capture-targets').touch()
        (root/'follow-targets').touch()
        assert request(root)['tcp_frame'] == TCP_FRAME
        assert not request(root, 'reset', 1)['fault']
        for seq in range(1, 51):
            reply = request(root, 'target', 1, sequence=seq, slot_id=seq,
                            timestamp_ns=time.monotonic_ns(), arms={'left': {'pose': logical}})
            assert not reply['fault']
        state = reply['states'][0]
        for pose in (reply['references']['left'], state['tcp_pose'], state['desired_tcp_pose']):
            np.testing.assert_allclose(pose[:3], logical[:3], atol=1e-12)
            np.testing.assert_allclose(Rotation.from_quat(pose[3:]).as_matrix(), np.diag([1, -1, -1]), atol=1e-12)
        physical = [json.loads(x) for x in (root/'targets.jsonl').read_text().splitlines()][-1]
        expected = np.diag([1, -1, -1]) @ mount.as_matrix()
        np.testing.assert_allclose(Rotation.from_quat(physical['quaternion']).as_matrix(), expected, atol=1e-12)
        np.testing.assert_allclose(state['hardware_tcp_pose'][3:], physical['quaternion'], atol=1e-12)
        np.testing.assert_allclose(state['hardware_reference_tcp_pose'][3:], physical['quaternion'], atol=1e-12)
        np.testing.assert_allclose(state['hardware_flange_to_tcp_pose'], tool, atol=1e-12)
        np.testing.assert_allclose(state['flange_to_tcp_pose'], [0, 0, .174, 0, 0, 0, 1], atol=1e-12)
        # Changing the installation cannot silently reinterpret a running plan.
        (root/'hardware-tool.json').write_text(json.dumps([0, 0, .174, 0, 0, 0, 1]))
        assert 'tool transform changed' in request(root)['fault']
