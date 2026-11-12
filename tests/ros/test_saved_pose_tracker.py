"""Saved reference uses coordinator -> Unix IPC -> real wheel, with fake Robot."""
from dataclasses import replace
from types import SimpleNamespace as NS
import time

import numpy as np
from scipy.spatial.transform import Rotation

from tests.ros.test_teleop_inputs import rig
from tests.control.test_franky_process import process_worker, request
from vla.control.ros import ArmMeasurement
from vla.web.pose_control import ROSPoseDriver, pose_value


def test_saved_pose_synchronized_reference_through_real_tracker(monkeypatch):
    c=rig('dual')
    c.config=replace(c.config,workspace_limits_enabled=False)
    c.coordinator.workspace_limits_enabled=False
    c._faulted=False
    c._neutral_gamepads=lambda:True
    c._observe_control=lambda:None
    c.compliance_status=lambda:{'values':c.config.franky_compliance}
    poses={name:pose_value(name,name,'base/EE',np.zeros(3),np.eye(3),np.zeros(3),np.eye(3))
           for name in ('left','right')}
    with process_worker(('left','right')) as (root,_,_):
        def boundary(owner=None,**kw):
            generation=c.coordinator.transition(owner,**kw)
            for name in ('left','right'):
                c.coordinator.observe(name,ArmMeasurement(np.zeros(7),.5,np.zeros(3),np.eye(3),time.monotonic_ns()))
            reply=request(root,'reset' if owner else 'stop',generation)
            assert reply['request_status']=='applied'
            for name in ('left','right'):
                c.coordinator.acknowledge(name,generation,0,np.zeros(7),time.monotonic_ns())
        c._control_boundary=boundary
        targets={}; replies=[]
        def publish(name,pose,gripper,generation,command_id):
            targets[name]={'pose':pose.tolist()}
            if name=='right':
                replies.append(request(root,'target',generation,sequence=command_id,
                    slot_id=command_id-1,timestamp_ns=time.monotonic_ns(),arms=dict(targets)))
        c.coordinator.publish=publish
        driver=ROSPoseDriver(NS(collector=c,enabled=False))
        monkeypatch.setattr(driver,'read',lambda name:poses[name])
        goal={**poses['left'],'position':[.06,.025,.012],
              'rotation':Rotation.from_rotvec([.4,-.3,.6]).as_matrix().tolist()}
        driver.start(goal)
        for slot in range(120):
            for name in ('left','right'):
                c.coordinator.observe(name,ArmMeasurement(np.zeros(7),.5,np.zeros(3),np.eye(3),time.monotonic_ns()))
            action=driver.advance(1/30)
            c.coordinator.accept(action,source='pose',generation=c.coordinator.generation,slot_id=slot)
            c.coordinator.tick(1/30)
            reply=replies[-1]
            assert reply['request_status']=='applied'
            np.testing.assert_allclose(reply['references']['left'][:3],action[:3],atol=1e-12)
            np.testing.assert_allclose(reply['references']['right'],[0,0,0,0,0,0,1],atol=1e-12)
            expected=Rotation.from_matrix(np.asarray(goal['rotation'])).as_rotvec()*driver.plan.progress
            np.testing.assert_allclose(Rotation.from_quat(reply['references']['left'][3:]).as_rotvec(),expected,atol=1e-12)
        assert driver.plan.done
        c._last_control_tick=time.monotonic()
        assert driver.tick(goal) is None  # Fixed-state fake Robot has not physically arrived.
        assert driver.progress_status['waiting_measured']
        driver.stop()
