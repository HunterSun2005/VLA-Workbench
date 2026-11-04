"""Status fields consumed by the panel; no devices or ROS nodes."""
from types import SimpleNamespace as NS

import pytest
from vla.web.backends import BaseBackend
from vla.web.config import SessionConfig


@pytest.mark.parametrize('subtasks', [(), ('approach','place')])
def test_collection_reports_annotation_and_only_configured_stage_controls(subtasks):
    backend = object.__new__(BaseBackend)
    backend.config = SessionConfig(arms='dual',source='gamepad')
    backend.arms = ['left','right']
    backend.enabled,backend.mode,backend.record_started = True,'vla',None
    backend.collector = NS(episode_active=True,_latest_policy_result=None,_intervention=True,
        writer=NS(episode_count=3),_states=[1],_faulted=False,_fault_reason=None,
        policy_client=None,config=NS(subtasks=subtasks),_subtask_index=0,_phase='rl')
    status = backend.collector_status()
    assert status['critical'] is True
    assert ('next_subtask' in status['available_commands']) == bool(subtasks)
    assert status['subtask'] == ({'index':0,'total':2,'name':'approach'} if subtasks else None)
