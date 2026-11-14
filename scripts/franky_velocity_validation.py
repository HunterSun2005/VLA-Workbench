"""Official 2.0.0 SDK boundary on a fixed-state fake Robot, never hardware."""
from pathlib import Path
import json
import subprocess
import time
import uuid
import socket
import numpy as np
from scipy.spatial.transform import Rotation
from vla.actions import rot6d_to_matrix
from vla.control.franky.ipc import encode_packet, decode_packet
from vla.core.franky_runtime import worker_environment, worker_python


class VelocityValidation:
    def __init__(self, root, directory):
        self.directory = directory/'velocity'
        self.directory.mkdir()
        cfg = self.directory/'config.json'
        cfg.write_text(json.dumps({'session_id':'velocity-validation',
            'arms':{a:{'robot_ip':a} for a in ('left','right')},
            'target_timeout_s':.1, 'state_timeout_s':.5}))
        self.log = (self.directory/'worker.log').open('w')
        python = worker_python(root)
        self.child = subprocess.Popen([str(python), str(root/'tests/control/franky_fake_process.py'),
            '--config',str(cfg),'--socket-path',str(self.directory/'worker.sock')],
            env=worker_environment(root,python),stdout=self.log,stderr=self.log)
        self.ack = 0
        self.stamps = {'left':[],'right':[]}
        self.accepted = 0
        deadline = time.monotonic()+20
        while not (self.directory/'worker.sock').exists():
            if self.child.poll() is not None or time.monotonic()>deadline:
                self.close()
                raise RuntimeError('velocity validation worker failed; see worker.log')
            time.sleep(.01)
        self.exchange('reset', generation=1)

    def exchange(self, kind, generation=1, **values):
        message = dict(kind=kind,session_id='velocity-validation',generation=generation,
                       request_id=uuid.uuid4().hex,audit_ack=self.ack,**values)
        with socket.socket(socket.AF_UNIX,socket.SOCK_STREAM) as sock:
            sock.settimeout(.5)
            sock.connect(str(self.directory/'worker.sock'))
            sock.sendall(encode_packet(message))
            response=decode_packet(sock)
        if response.get('fault') or response.get('request_status')!='applied':
            raise RuntimeError(f'velocity validation: {response}')
        for event in response['servo_events']:
            if event['audit_id']<=self.ack:continue
            self.stamps[event['arm']].append(event['applied_at_ns'])
            self.ack=event['audit_id']
        return response

    def submit(self, accepted):
        targets={}
        for index,arm in enumerate(('left','right')):
            block=accepted.target[10*index:10*index+10]
            pose=np.r_[block[:3],Rotation.from_matrix(rot6d_to_matrix(block[3:9])).as_quat()]
            targets[arm]={'pose':pose.tolist(),'context':{'source':'policy','dt':1/30}}
        self.accepted+=1
        return self.exchange('target', sequence=self.accepted, slot_id=accepted.command_id,
                             timestamp_ns=time.monotonic_ns(),arms=targets)

    def report(self):
        from validate_ros_control import intervals
        self.exchange('stop',generation=2)
        return {'sdk':'official Franky 2.0.0+libfranka.0.21.2',
                'boundary':'real SDK motion constructors; fixed-state fake Robot; no FCI',
                'accepted':self.accepted, 'servo':{arm:intervals(stamps) for arm,stamps in self.stamps.items()}}

    def close(self):
        if self.child.poll() is None:
            self.child.terminate()
            self.child.wait(timeout=5)
        self.log.close()
