"""Wire v2 must preserve empty candidate arrays alongside nonempty payloads."""
import socket

import numpy as np
import pytest

from vla_rollout_protocol import recv_packet, send_packet


@pytest.mark.parametrize('shape', [(0,), (0, 10), (2, 0, 3)])
def test_empty_arrays_roundtrip_without_shifting_other_payloads(shape):
    arrays = {'anchor': np.arange(10, dtype=np.float32),
              'empty': np.empty(shape, dtype=np.float32),
              'suffix': np.arange(6, dtype=np.int64).reshape(2, 3)}
    sender, receiver = socket.socketpair()
    with sender, receiver:
        sender.settimeout(1)
        receiver.settimeout(1)
        send_packet(sender, {'type': 'candidate'}, arrays)
        header, decoded = recv_packet(receiver)
    assert header == {'type': 'candidate', 'protocol_version': 2}
    assert set(decoded) == set(arrays)
    for name, expected in arrays.items():
        assert decoded[name].dtype == expected.dtype
        assert decoded[name].shape == expected.shape
        np.testing.assert_array_equal(decoded[name], expected)
