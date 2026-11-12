import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from vla.web.config import SessionConfig
from vla.web.backends import BaseBackend


def config(**kw):
    return SessionConfig.parse(dict(workflow='deploy', model_backend='lingbot_v2', policy_kind='sft',
                                    task='Pick and place.', **kw))


def test_external_config_needs_no_local_model_assets():
    value = config()
    assert not value.policy_path and not value.tokenizer_path
    assert value.modes() == ['vla']
    assert value.default_dataset_root().endswith('lingbot_v2')
    assert value.policy_endpoint == 'tcp://127.0.0.1:8007'


@pytest.mark.parametrize('kw', [{'arms': 'dual'}, {'mode': 'auto'}, {'handover_path': '/head'},
                               {'policy_endpoint': 'http://localhost:8007'}, {'policy_timeout_s': float('nan')}])
def test_external_unsupported_modes_fail(kw):
    with pytest.raises(ValueError):
        config(**kw)


def test_external_start_connects_without_managing_model_process(monkeypatch):
    from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
    import vla.deployment.external as external
    value = config()
    backend = BaseBackend()
    backend.config = value
    backend.children = SimpleNamespace(start=lambda *a: pytest.fail('must not launch model process'))
    calls = []
    monkeypatch.setattr(external, 'inspect_service', lambda *a, **k: calls.append((a, k)))
    rollout = replace(load_rollout_config(value.default_config()), tasks=(value.task,))
    result = backend.start_policy(rollout)
    assert result.model_backend == 'lingbot_v2' and result.policy_timeout_s == 5
    assert calls and 'lingbotvla' not in sys.modules


def test_default_pi05_selection_is_unchanged():
    value = SessionConfig.parse(dict(workflow='deploy', policy_kind='sft', task='task'))
    assert value.model_backend == 'pi05'
    assert value.default_dataset_root().endswith('_sft')
