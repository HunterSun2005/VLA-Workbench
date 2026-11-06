"""Sparse model targets retain physical reward time and request ownership."""
import numpy as np
import pytest
import torch

from vla.core.losses import critic_loss
from vla.cli.build_transition_cache_v2 import _encoded_to_transitions, validate_chunk_settings
from vla.core.online_replay import ReplayTransitionRecord
from tests.rlt.test_online_training import _replay, _transition
from tests.rlt.test_franka_rollout_collection import _mock_rollout_server


def test_sparse_cache_bootstraps_after_physical_horizon():
    encoded = {frame: {'state_vec': torch.tensor([frame]),
                      'exec_chunk': torch.tensor([[10.], [11.]]),
                      'ref_chunk': torch.tensor([[20.], [21.]])} for frame in (0, 6)}
    annotations = {0: {'collector_policy_id': 1}, 3: {'collector_policy_id': 0}}
    item, = _encoded_to_transitions(encoded, [0], 6, 2, True, 0, annotations, action_stride=3)
    assert item['next_state_vec'].item() == 6
    assert item['ref_chunk'][:, 0].tolist() == [20., 11.]
    assert item['reward_offsets'].tolist() == [2, 5]
    assert item['elapsed_steps'].item() == 6
    assert item['actual_steps'].item() == 2
    assert item['reward_seq'].tolist() == [0, 1]
    validate_chunk_settings(2, 3, 4, action_stride=3)


def test_sparse_discount_and_bootstrap_hand_computed():
    class Actor:
        def forward(self, state, ref):
            return ref, None
    class Critic:
        def __call__(self, state, actions):
            return torch.zeros(1, 1), torch.zeros(1, 1)
        def min_q(self, state, actions):
            return torch.full((1, 1), 2.)
    batch = dict(state_vec=torch.zeros(1, 1), exec_chunk_flat=torch.zeros(1, 3),
                 next_state_vec=torch.zeros(1, 1), next_ref_flat=torch.zeros(1, 3),
                 reward_seq=torch.tensor([[1., 2., 999.]]), done=torch.zeros(1),
                 actual_steps=torch.tensor([2]), elapsed_steps=torch.tensor([5]),
                 reward_offsets=torch.tensor([[2, 4, 4]]))
    # Two valid targets cover three and two control frames; padding contributes nothing.
    target = .5**2 + 2*.5**4 + 2*.5**5
    assert critic_loss(Critic(), Critic(), Actor(), batch, .5, 3).item() == pytest.approx(2*target**2)
    for key in ('actual_steps', 'elapsed_steps', 'reward_offsets'):
        invalid = {k: v for k, v in batch.items() if k != key}
        with pytest.raises(ValueError, match='physical timing'):
            critic_loss(Critic(), Critic(), Actor(), invalid, .5, 3)


def test_sparse_replay_tracks_union_of_control_frames(tmp_path):
    replay = _replay(tmp_path)
    try:
        item = _transition()
        item['elapsed_steps'] = torch.tensor(8)
        item['reward_offsets'] = torch.tensor([1, 3, 5, 7])
        replay.commit_episode(episode_key='sparse', episode_id=0, outcome='success', warmup=False,
            warmup_env_steps=0, records=[ReplayTransitionRecord(0, item), ReplayTransitionRecord(4, item)], utd_ratio=2)
        assert replay.trainable_env_steps == 12
        assert replay.produced_credits == 4  # UTD remains per transition.
    finally:
        replay.close()


def test_sparse_server_holds_absolute_target_and_discards_hold_on_boundary():
    server = _mock_rollout_server()
    server.action_stride = 3
    calls = []
    def predict(**kwargs):
        calls.append(kwargs['observation']['observation.anchor'].copy())
        return np.full(10, len(calls), dtype=np.float32)
    server._predict_action = predict
    arrays = {'observation.state': np.zeros(10, np.float32), 'observation.anchor': np.zeros(10),
              **{key: np.zeros((224, 224, 3), np.uint8) for key in server.camera_keys}}
    header = {'kind': 'infer', 'task': server.task, 'phase': 'rl'}
    outputs = []
    for index in range(4):
        arrays['observation.anchor'][:] = index
        response, values = server.handle({**header, 'request_id': index}, arrays)
        assert response['request_id'] == index
        outputs.append(values['action'][0])
    assert outputs == [1, 1, 1, 2]
    _, values = server.handle({**header, 'reset_episode': True}, arrays)
    assert values['action'][0] == 3
    _, values = server.handle({**header, 'phase': 'vla'}, arrays)
    assert values['action'][0] == 4
    assert len(calls) == 4
