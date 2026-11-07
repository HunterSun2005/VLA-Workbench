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
    from tests.rlt.execution_fixtures import human_rows, encoded_anchors
    encoded_anchors(encoded)
    annotations = human_rows(7)
    item, = _encoded_to_transitions(encoded, [0], 6, 2, True, 0, annotations, action_stride=3)
    assert item['next_state_vec'].item() == 6
    assert item['ref_chunk'][:, 0].tolist() == [10., 11.]
    assert item['reward_offsets'].tolist() == pytest.approx(list(range(6)), abs=1e-6)
    assert item['elapsed_steps'].item() == 6
    assert item['actual_steps'].item() == 2
    assert item['reward_seq'].tolist() == [0, 0, 0, 0, 0, 1]
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
        item['reward_seq'] = torch.zeros(8)
        item['reward_valid_mask'] = torch.ones(8, dtype=torch.bool)
        item['reward_offsets'] = torch.arange(8)
        item['interval_slot_ids'] = torch.arange(8)
        item['reward_interval_start_ns'] = 1_000_000_000+(torch.arange(8).double()*1e9/30).round().long()
        item['endpoint_ns'] = torch.tensor(1_000_000_000+round(8e9/30))
        following = {**item, 'interval_slot_ids': torch.arange(4,12)}
        replay.commit_episode(episode_key='sparse', episode_id=0, outcome='success', warmup=False,
            warmup_env_steps=0, records=[ReplayTransitionRecord(0, item), ReplayTransitionRecord(4, following)], utd_ratio=2)
        assert replay.trainable_env_steps == 12
        assert replay.produced_credits == 24  # UTD uses unique physical intervals.
    finally:
        replay.close()


def test_sparse_server_holds_absolute_target_and_discards_hold_on_boundary():
    from tests.control.test_execution import executor, request, response
    from dataclasses import replace
    e = executor(stride=3)
    req = request(e)
    candidate = response(req)
    actions = candidate.actions.copy()
    actions[:, 0] = np.arange(len(actions))*.01
    assert e.install(replace(candidate, actions=actions), 1)
    values = [e.advance(s, e.control_clock.deadline(s)).accepted[0] for s in range(36)]
    np.testing.assert_array_equal(values[req.replace_slot:], np.repeat(actions[req.start_idx:,0], 3))
    e.invalidate(reason='takeover')
    assert not e.install(candidate, 2)
