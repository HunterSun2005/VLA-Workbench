from types import SimpleNamespace
import copy
import numpy as np
import pytest
import torch
from vla.actions import ActionSpec
from vla.actions.projection import ActionProjection
from vla.core.losses import actor_loss, critic_loss
from vla.core.execution_replay import build_execution_replay, validate_execution_transition
from vla.core.online_replay import PersistentReplayBuffer, ReplayTransitionRecord


def contract(space='joint', arms=('left',), representation='absolute'):
    spec = ActionSpec(space, arms, representation)
    n = len(arms)*(7 if space == 'joint' else 3)
    low, high = np.full(spec.dim, -1.), np.ones(spec.dim)
    low[spec.width-1::spec.width] = 0
    return {'version': 2, 'action_spec': spec.to_dict(), 'statistics': {'q01': low.tolist(), 'q99': high.tolist()},
            'execution_projection': {'lower': [-.4]*n, 'upper': [.4]*n}}


@pytest.mark.parametrize('space', ['joint', 'tcp'])
@pytest.mark.parametrize('arms', [('left',), ('left', 'right')])
@pytest.mark.parametrize('representation', ['absolute', 'chunk_relative'])
def test_numpy_torch_projection_and_suffix_gradient(space, arms, representation):
    from tests.actions.test_contract import sample
    c = contract(space, arms, representation)
    spec = ActionSpec.from_dict(c['action_spec'])
    anchor = sample(spec, 2)
    values = np.random.default_rng(9).uniform(-.8, .8, (2, 4, spec.dim))
    projection = ActionProjection(c)
    mu = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    actual = projection.torch(mu, torch.tensor(anchor))
    expected = projection.numpy(values, anchor)
    np.testing.assert_allclose(actual.detach().numpy(), expected, atol=1e-10, rtol=1e-10)
    actual.square().sum().backward()
    assert torch.isfinite(mu.grad).all() and mu.grad.abs().sum() > 0


class Actor(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.mu = torch.nn.Parameter(torch.tensor([values], dtype=torch.float32))
    def forward(self, state, ref, training=False):
        return self.mu.expand(len(state), -1), None


class SumCritic:
    def min_q(self, state, action):
        self.seen = action
        return action.sum(-1, keepdim=True)
    def __call__(self, state, action):
        z = torch.zeros(len(state), 1)
        return z, z


def batch(d, valid=4):
    mask = torch.arange(4)[None] < valid
    prefix = mask & (torch.arange(4)[None] < d)
    return dict(state_vec=torch.zeros(1, 1), exec_chunk_flat=torch.full((1, 4), .6),
                ref_chunk_flat=torch.zeros(1, 4), valid_mask=mask, prefix_mask=prefix, actor_mask=mask & ~prefix,
                reward_seq=torch.zeros(1, 4), actual_steps=torch.tensor([valid]))


@pytest.mark.parametrize('d,valid', [(0,4), (2,4), (3,4), (1,2)])
def test_q_prefix_stop_gradient_and_independent_prefix_bc(d, valid):
    actor, critic, data = Actor([.2]*4), SumCritic(), batch(d, valid)
    actor_loss(actor, critic, data, beta=0, prefix_bc_weight=0).backward()
    assert actor.mu.grad[0, :d].eq(0).all()
    assert actor.mu.grad[0, d:valid].eq(-1).all()
    assert actor.mu.grad[0, valid:].eq(0).all()
    actor.zero_grad()
    metrics = {}
    loss = actor_loss(actor, critic, data, beta=0, prefix_bc_weight=2, metrics=metrics)
    loss.backward()
    np.testing.assert_allclose(actor.mu.grad[0, :d], [-1.6]*d, atol=1e-6)
    expected = -(d*.6+(valid-d)*.2)+2*d*(.2-.6)**2
    assert loss.item() == pytest.approx(expected, abs=1e-6)
    assert metrics['prefix_dims'] == d and metrics['actor_dims'] == valid-d


def test_next_prefix_and_fractional_physical_discount():
    actor, critic = Actor([.2]*4), SumCritic()
    data = batch(2)
    data.update(next_state_vec=torch.zeros(1,1), next_ref_flat=torch.zeros(1,4),
                next_exec_chunk_flat=torch.full((1,4), .7), next_valid_mask=torch.ones(1,4,dtype=torch.bool),
                next_prefix_mask=torch.tensor([[1,1,0,0]]), next_actor_mask=torch.tensor([[0,0,1,1]]),
                reward_seq=torch.tensor([[1.,2.,999.,999.]]), reward_valid_mask=torch.tensor([[1,1,0,0]]),
                reward_offsets=torch.tensor([[0.,1.5,0.,0.]]), elapsed_steps=torch.tensor([2.5]),
                done=torch.zeros(1))
    target = 1+2*.5**1.5+(2*.7+2*.2)*.5**2.5
    assert critic_loss(critic, critic, actor, data, .5, 4).item() == pytest.approx(2*target**2)
    np.testing.assert_allclose(critic.seen, [[.7,.7,.2,.2]])


def recorded_plan(*, end=10, terminal=True):
    H, d = 10, 5
    hold = np.r_[np.zeros(7), .5]
    action = np.r_[np.full(7, .2), .5]
    req = dict(start_slot=0, frame_stride=1, prefix=np.tile(hold, (d,1)).tolist(), anchor=hold.tolist(),
               plan_version=1, generation=1, model_version='model', session_id='s', episode_id='e')
    plan = dict(request=req, actions=np.tile(action, (H,1)).tolist(), sources=['actor']*H, phases=[1]*H,
                vla_reference=np.tile(hold, (H,1)).tolist(), context={'state_vec': [[.1,.2]]})
    rows = [dict(slot_id=i, segment_id=0, action_valid=int(i<end), observation_valid=1,
                 task_terminal=int(i==end and terminal), collector_policy_id=-1 if i<d else 2, phase=int(i>=d), plan_version=1 if i>=d else -1,
                 action_index=i if i>=d else -1, action=hold if i<d else action, anchor=hold,
                 observed_at_ns=1_000_000_000+i*33_333_333,
                 command_at_ns=1_000_100_000+i*33_333_333) for i in range(end+1)]
    return rows, [('install', plan)]


@pytest.mark.parametrize('end', [6, 10])
def test_replay_uses_committed_execution_and_real_terminal(end):
    rows, events = recorded_plan(end=end)
    records = build_execution_replay(rows, events, contract(), episode_id=0, success=True,
                                     encode_endpoint=lambda *_: pytest.fail('terminal must not bootstrap'))
    assert len(records) == 1
    tx = records[0].transition
    validate_execution_transition(tx)
    assert tx['prefix_mask'].sum() == 5 and tx['actor_mask'].sum() == end-5
    assert tx['absolute_exec_chunk'][:5,:7].eq(0).all()
    assert tx['reward_seq'].sum() == 1 and tx['done'] == 1
    rows[-1]['task_terminal'] = 0
    assert build_execution_replay(rows, events, contract(), episode_id=0, success=True, encode_endpoint=None) == []
    assert build_execution_replay(rows, [('reject', events[0][1])], contract(), episode_id=0,
                                  success=True, encode_endpoint=None) == []


def test_replay_unique_credits_and_recovery(tmp_path):
    rows, events = recorded_plan()
    records = build_execution_replay(rows, events, contract(), episode_id=0, success=True, encode_endpoint=None)
    duplicate_window = ReplayTransitionRecord(1, copy.deepcopy(records[0].transition))
    replay = PersistentReplayBuffer(tmp_path, capacity=10, contract={'v':4}, resume=False, seed=5)
    kwargs = dict(episode_key='e', episode_id=0, outcome='success', warmup=False, warmup_env_steps=0,
                  records=[*records, duplicate_window], utd_ratio=2)
    assert replay.commit_episode(**kwargs) == 2
    assert replay.produced_credits == 20 and replay.trainable_env_steps == 10
    assert replay.commit_episode(**kwargs) == 0
    replay.close()
    replay = PersistentReplayBuffer(tmp_path, capacity=10, contract={'v':4}, resume=True, seed=5)
    assert replay.produced_credits == 20
    assert replay.sample(1)['start_idx'].item() == 5
    replay.close()


def test_nonterminal_replay_bootstraps_from_recorded_commitment():
    rows, events = recorded_plan(terminal=False)
    context = dict(slot_id=10, generation=1, model_version='model', horizon=10,
                   start_idx=5, prefix=np.tile(rows[5]['action'], (5,1)), anchor=rows[10]['anchor'])
    seen = []
    def encode(row, recorded):
        seen.append((row['slot_id'], recorded))
        return np.array([.3,.4]), np.zeros((10,8))
    result = build_execution_replay(rows, [*events, ('execution_context', context)], contract(),
                                    episode_id=0, success=False, encode_endpoint=encode)
    tx, = [r.transition for r in result]
    validate_execution_transition(tx)
    assert seen[0][0] == 10 and seen[0][1] is context
    assert tx['next_prefix_mask'].sum() == 5 and tx['next_actor_mask'].sum() == 5
    expected = ActionProjection(contract()).normalizer.normalize(context['prefix'])
    np.testing.assert_allclose(tx['next_exec_chunk'][:5], expected)
    assert not tx['done'] and tx['reward_seq'].sum() == 0
    context['generation'] += 1
    assert build_execution_replay(rows, [*events, ('execution_context', context)], contract(),
                                  episode_id=0, success=False, encode_endpoint=encode) == []


def test_projected_small_network_training_checkpoint_and_resume(tmp_path):
    from tests.rlt.test_online_consistency import session_at
    from vla.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
    from vla.core.actor import ChunkActor
    from vla.core.critic import TwinCritic
    from vla.core.online_training import OnlineUpdateScheduler
    session = session_at(tmp_path)
    policy = object.__new__(ChunkACPolicy)
    torch.nn.Module.__init__(policy)
    policy.config = SimpleNamespace(device='cpu', gamma=.99, beta=.3, prefix_bc_weight=.4,
                                     tau=.005, chunk_length=10, target_q_clip=100.,
                                     actor_update_interval=2, action_contract=contract())
    policy.actor = ChunkActor(2,80,hidden_dim=16,num_layers=1)
    policy.critic = TwinCritic(2,80,hidden_dim=16,num_layers=1)
    policy.target_critic = copy.deepcopy(policy.critic)
    policy.register_buffer('_critic_step',torch.zeros((),dtype=torch.long))
    object.__setattr__(policy, '_rl_token_policy', None)
    session.policy = policy
    session.scheduler = OnlineUpdateScheduler(policy=policy,replay=session.replay,config=session.config)
    rows, events = recorded_plan()
    records = build_execution_replay(rows,events,contract(),episode_id=0,success=True,encode_endpoint=None)
    try:
        session._checkpoint_scheduler_progress()
        session.replay.commit_episode(episode_key='projected',episode_id=0,outcome='success',warmup=False,
                                      warmup_env_steps=0,records=records,utd_ratio=1)
        before = copy.deepcopy(session._model_state())
        metrics = session.scheduler.drain()
        assert metrics['critic_updates'] == 10 and metrics['actor_updates'] == 5
        assert policy.loss_metrics['prefix_bc_weight'] == .4
        assert policy.loss_metrics['prefix_dims'] == 40
        assert any(not torch.equal(before[k],v) for k,v in session._model_state().items())
        session._checkpoint_scheduler_progress()
        saved = copy.deepcopy(session._model_state())
        optimizers = (copy.deepcopy(session.scheduler.actor_optimizer.state_dict()),
                      copy.deepcopy(session.scheduler.critic_optimizer.state_dict()))
        checkpoint = session.replay.active_checkpoint
        with torch.no_grad():
            for param in policy.actor.parameters():
                param.add_(100)
        session._restore_checkpoint(checkpoint)
        for k,v in saved.items():
            torch.testing.assert_close(v,session._model_state()[k],rtol=0,atol=0)
        torch.testing.assert_close(optimizers[0], session.scheduler.actor_optimizer.state_dict(),rtol=0,atol=0)
        torch.testing.assert_close(optimizers[1], session.scheduler.critic_optimizer.state_dict(),rtol=0,atol=0)
        assert session.replay.consumed_credits == 10
        assert session.scheduler.drain()['critic_updates'] == 0
    finally:
        session.replay.close()
