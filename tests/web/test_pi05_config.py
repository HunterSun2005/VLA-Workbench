"""PI0.5 branch boundaries fail closed before starting any session or device."""
from pathlib import Path

import pytest
import yaml

from vla.cli.ros_session import main
from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
from vla.web.config import SessionConfig


@pytest.mark.parametrize('settings', [
    {'model_backend': 'lingbot_v2'},
    {'model_backend': 'pi05'},
    {'policy_endpoint': 'tcp://127.0.0.1:8007'},
])
def test_web_rejects_external_model_settings(settings):
    with pytest.raises(ValueError, match='unknown session settings'):
        SessionConfig.parse({'task': 'test', **settings})


@pytest.mark.parametrize('flag,value', [
    ('--model-backend', 'lingbot_v2'),
    ('--policy-endpoint', 'tcp://127.0.0.1:8007'),
])
def test_cli_rejects_external_model_flags(flag, value, capsys):
    with pytest.raises(SystemExit) as error:
        main([flag, value, '--check-only'])
    assert error.value.code == 2
    assert 'unrecognized arguments' in capsys.readouterr().err


@pytest.mark.parametrize('section', ['root', 'control'])
@pytest.mark.parametrize('key,value', [('model_backend', 'lingbot_v2'),
                                      ('policy_endpoint', 'tcp://127.0.0.1:8007')])
def test_yaml_rejects_external_model_settings(tmp_path, section, key, value):
    raw = yaml.safe_load(Path(SessionConfig(task='test').default_config()).read_text())
    (raw if section == 'root' else raw['control'])[key] = value
    path = tmp_path/'external.yaml'
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(ValueError, match='external model settings are not supported'):
        load_rollout_config(path)


def test_generic_policy_timeout_is_preserved(tmp_path):
    raw = yaml.safe_load(Path(SessionConfig(task='test').default_config()).read_text())
    raw['control']['policy_timeout_s'] = 3.5
    path = tmp_path/'pi05.yaml'
    path.write_text(yaml.safe_dump(raw))
    assert load_rollout_config(path).policy_timeout_s == 3.5
