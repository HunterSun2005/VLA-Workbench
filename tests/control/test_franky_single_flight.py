"""Deterministic IPC backpressure and receipt tests, without device access."""
import threading
import time

import pytest

from vla.control.franky.ipc import FrankyWorkerClient
from tests.control.test_franky_process import wait_for

POSE = [0., 0., 0., 0., 0., 0., 1.]


class DelayedClient(FrankyWorkerClient):
    def __init__(self, path, mutate=lambda r:r):
        self.entered, self.release = threading.Event(), threading.Event()
        self.sent, self.mutate = [], mutate
        super().__init__(path)

    def _request(self, message):
        self.sent.append(message.copy())
        reply = dict(kind='state', session_id=message['session_id'],
            generation=message['generation'], request_id=message['request_id'],
            request_status='applied', states=[])
        if message['kind'] == 'target':
            self.entered.set()
            assert self.release.wait(2)
            reply.update(sequence=message['sequence'], slot_id=message['slot_id'],
                applied_at_ns=time.monotonic_ns(), references={'left': POSE})
            reply = self.mutate(reply)
        return reply


def test_single_outstanding_request_has_no_target_backlog(tmp_path):
    client = DelayedClient(tmp_path/'unused')
    try:
        first = client.target('s', 1, 1, {'left':POSE}, slot_id=10)
        assert client.entered.wait(1)
        assert not client.target_available
        for seq in range(2, 8):
            with pytest.raises(RuntimeError, match='busy'):
                client.target('s', 1, seq, {'left':POSE}, slot_id=seq+10)
        assert not client._queue
        client.release.set()
        wait_for(lambda: client.result(first))
        wait_for(lambda: client.target_available)
        second = client.target('s', 1, 8, {'left':POSE}, slot_id=20)
        wait_for(lambda: client.result(second))
        assert [m['slot_id'] for m in client.sent] == [10, 20]
    finally:
        client.release.set(); client.close()


@pytest.mark.parametrize('field,value', [('session_id','other'), ('generation',2),
    ('sequence',999), ('request_id','other'), ('slot_id',99), ('references',{}),
    ('request_status','ignored'), ('applied_at_ns',0)])
def test_wrong_ack_latches_fault_and_does_not_reopen_target_channel(tmp_path,field,value):
    client = DelayedClient(tmp_path/'unused', lambda r:{**r,field:value})
    try:
        key = client.target('s',1,1,{'left':POSE},slot_id=0)
        assert client.entered.wait(1)
        client.release.set()
        wait_for(lambda:client.fault)
        assert client.result(key) and not client.target_available
        with pytest.raises(RuntimeError):
            client.target('s',1,2,{'left':POSE},slot_id=1)
        assert len(client.sent) == 1  # Unknown outcomes are never retransmitted.
    finally:
        client.release.set(); client.close()


def test_stop_revokes_already_in_sdk_target_and_late_ack_is_cancelled(tmp_path):
    client = DelayedClient(tmp_path/'unused')
    try:
        key = client.target('s',1,1,{'left':POSE},slot_id=0)
        assert client.entered.wait(1)
        stop = client.stop('s',2)
        assert [m['kind'] for m in client.sent] == ['target']
        client.release.set()
        wait_for(lambda:client.result(stop))
        assert client.result(key)['request_status'] == 'cancelled'
        assert [m['kind'] for m in client.sent] == ['target','stop']
        assert not client.states()
    finally:
        client.release.set(); client.close()


def test_real_socket_target_response_deadline_is_100ms(tmp_path):
    import socket
    from vla.control.franky.ipc import decode_packet
    path=tmp_path/'delayed.sock'
    listener=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
    listener.bind(str(path)); listener.listen()
    entered=threading.Event()
    def server():
        with listener:
            conn,_=listener.accept()
            with conn:
                assert decode_packet(conn)['kind']=='target'
                entered.set()
                time.sleep(.18)  # No ACK within the target deadline.
    thread=threading.Thread(target=server,daemon=True);thread.start()
    client=FrankyWorkerClient(path,timeout_s=.5,target_timeout_s=.1)
    try:
        started=time.monotonic()
        key=client.target('s',1,1,{'left':POSE},slot_id=0)
        assert entered.wait(1)
        wait_for(lambda:client.fault)
        assert time.monotonic()-started < .18
        assert client.transport_failed and not client.target_available
        assert client.result(key)['request_status']=='failed'
    finally:
        client.close();thread.join(1)
