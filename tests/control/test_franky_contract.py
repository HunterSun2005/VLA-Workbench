"""Franky recordings keep their own identity; legacy data cannot join a run."""
import pytest
from vla.actions import ActionSpec
from vla.actions.metadata import RecordingContract, validate_mixture
from vla.control.ros import ControlCoordinator


def test_worker_contract_cannot_mix_with_legacy_ros_contract():
    current = RecordingContract(ActionSpec('tcp', ('left',)), 'device', 'tool', 'fk', 'control',
                                backend='franky_cartesian_velocity').to_dict()
    legacy = dict(current, backend='ros2_joint_impedance')
    with pytest.raises(ValueError, match='do not mix control chains'):
        validate_mixture([current, legacy])
    assert validate_mixture([current, current])['sources'] == [current, current]


def test_franky_cannot_enable_joint_action_space():
    with pytest.raises(ValueError, match='TCP'):
        ControlCoordinator(ActionSpec('joint', ('left',)), {}, {}, lambda *a: None,
                           lambda *a: None, backend='franky')
