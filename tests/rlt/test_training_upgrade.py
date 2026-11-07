"""New execution semantics deliberately reject legacy training assets."""
import copy
import pytest
from vla.core.training_contract import (
    REFERENCE_POLICY, TRAINING_SEMANTICS_VERSION, validate_cache_upgrade, validate_training_cache,
)
from vla.core.online_replay import PersistentReplayBuffer


def cache_contract():
    return {'cache': {'training_semantics_version': TRAINING_SEMANTICS_VERSION,
                      'annotation_policy': {'ref_chunk': REFERENCE_POLICY}}}


@pytest.mark.parametrize('version', [None, 1, 2])
def test_old_cache_requires_rebuild(version):
    value = cache_contract()
    value['cache']['training_semantics_version'] = version
    with pytest.raises(ValueError, match='rebuild'):
        validate_training_cache(value)
    validate_training_cache(cache_contract())
    with pytest.raises(ValueError, match='conversion is not supported'):
        validate_cache_upgrade(value, cache_contract())


def test_old_replay_refuses_implicit_upgrade(tmp_path):
    replay = PersistentReplayBuffer(tmp_path, capacity=10, contract={'v': 3}, resume=False, seed=7)
    replay._set_state('format_version', '3')
    replay._connection.commit()
    replay.close()
    with pytest.raises(ValueError, match='Replay'):
        PersistentReplayBuffer(tmp_path, capacity=10, contract={'v': 3}, resume=True, seed=7)
