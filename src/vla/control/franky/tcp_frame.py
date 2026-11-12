"""Logical TCP orientation at the physical TCP origin; no device access."""
from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


TCP_FRAME = 'flange_aligned_tcp_v1'


class TCPFrame:
    """Remove the measured tool mounting rotation, without moving the TCP.

    XYZ is always the physical TCP position. The logical axes are parallel to
    the flange axes, so R_base_hardware = R_base_logical @ R_flange_hardware.
    This is a change of orientation convention, not a move to the flange origin.
    """

    def __init__(self, flange_to_hardware):
        self.hardware_tool = self._pose(flange_to_hardware)
        self.mount = Rotation.from_quat(self.hardware_tool[3:])

    @staticmethod
    def _pose(value):
        pose = np.array(value, dtype=np.float64, copy=True)
        if pose.shape != (7,) or not np.isfinite(pose).all() or np.linalg.norm(pose[3:]) < 1e-9:
            raise ValueError('TCP frame requires finite xyz and a nonzero quaternion')
        pose[3:] /= np.linalg.norm(pose[3:])
        return pose

    def check_tool(self, flange_to_hardware):
        other = self._pose(flange_to_hardware)
        if (not np.allclose(other[:3], self.hardware_tool[:3], atol=1e-7, rtol=0)
                or (Rotation.from_quat(other[3:]) * self.mount.inv()).magnitude() > 1e-7):
            raise RuntimeError('TCP tool transform changed during the session; restart explicitly')

    @property
    def logical_tool(self):
        return np.r_[self.hardware_tool[:3], [0., 0., 0., 1.]]

    def to_hardware(self, logical):
        result = self._pose(logical)
        result[3:] = (Rotation.from_quat(result[3:]) * self.mount).as_quat()
        return result

    def to_logical(self, hardware):
        result = self._pose(hardware)
        result[3:] = (Rotation.from_quat(result[3:]) * self.mount.inv()).as_quat()
        return result
