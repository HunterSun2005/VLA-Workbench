import numpy as np
import pytest
import torch
from lerobot.lerobot_types import TransitionKey
from vla.actions import ActionSpec, SamplingSpec, compute_statistics
from vla.adapters.lerobot.action_processor import DeltaActionsProcessor
from vla.adapters.lerobot.action_runtime import ActionRuntime
from test_contract import sample


@pytest.mark.parametrize('space', ['joint', 'tcp'])
@pytest.mark.parametrize('arms', [('left',), ('left', 'right')])
@pytest.mark.parametrize('representation', ['absolute', 'chunk_relative'])
def test_real_processor_and_runtime(space, arms, representation):
    spec = ActionSpec(space, arms, representation)
    values = sample(spec)
    stats = compute_statistics([(values, values, np.ones(7, bool))], spec, SamplingSpec(3), 'data')
    runtime = ActionRuntime({'action_spec': spec.to_dict(), 'statistics': stats})
    anchor = torch.tensor(values[1:2], dtype=torch.float32)
    targets = torch.tensor(values[2:5][None], dtype=torch.float32)
    transition = {TransitionKey.OBSERVATION: {'observation.anchor': anchor,
                    'observation.state': torch.ones(1, 2)}, TransitionKey.ACTION: targets}
    processor = DeltaActionsProcessor(spec.to_dict())
    result = processor(transition)
    assert transition[TransitionKey.ACTION] is targets
    normalized = torch.tensor(runtime.normalizer.normalize(result[TransitionKey.ACTION].numpy()), dtype=torch.float32)
    decoded = runtime.decode(normalized, {'observation.anchor': anchor})
    torch.testing.assert_close(decoded, targets, atol=1e-5, rtol=1e-5)
    with pytest.raises(ValueError, match='anchor'):
        processor({TransitionKey.OBSERVATION: {'observation.state': anchor}, TransitionKey.ACTION: targets})


def test_vla_policy_config_sampler_and_serialization(tmp_path):
    from vla.adapters.lerobot.policies.configuration_vla_pi05 import VLAPI05Config
    from vla.adapters.lerobot.policies.modeling_rlt_token import _load_pi05_config_from_dir
    spec = ActionSpec('joint', representation='chunk_relative')
    values = sample(spec)
    stats = compute_statistics([(values, values, np.ones(7, bool))], spec, SamplingSpec(3, 2), 'data')
    contract = {'version': 1, 'action_spec': spec.to_dict(), 'statistics': stats, 'sources': []}
    config = VLAPI05Config(chunk_size=3, n_action_steps=3, action_contract=contract)
    assert config.action_delta_indices == [0, 2, 4]
    config.save_pretrained(tmp_path)
    loaded = _load_pi05_config_from_dir(str(tmp_path))
    assert loaded.action_contract == contract
    with pytest.raises(ValueError, match='chunk_size'):
        VLAPI05Config(chunk_size=4, n_action_steps=3, action_contract=contract)


def test_conversion_rejects_missing_commands_and_marks_derivation():
    from vla.actions.dataset import export_actions
    spec = ActionSpec('joint')
    table = {'action': sample(spec, 3)}
    with pytest.raises(ValueError, match='allow_derived'):
        export_actions(table, spec, 'tcp')
    with pytest.raises(ValueError, match='missing command.fk_tcp'):
        export_actions(table, spec, 'tcp', allow_derived=True)
    # Real dataframe provides row count, not a column-count mapping.
    import pandas as pd
    table = pd.DataFrame({'action': list(sample(spec, 3)), 'command.fk_tcp': list(sample(ActionSpec(), 3))})
    values, new, source = export_actions(table, spec, 'tcp', allow_derived=True)
    assert values.shape == (3, 10) and new.space == 'tcp' and source == 'joint_command_fk'


def test_training_masks_episode_padding_and_invalid_segments():
    from types import SimpleNamespace
    from datasets import Dataset
    from vla.adapters.lerobot.action_dataset import ActionDataset
    table = Dataset.from_dict({'episode_index':[0,0,0,0,1,1],
        'complementary_info.action_valid':[1,0,1,1,1,1],
        'complementary_info.segment_id':[0,0,1,1,2,2]})
    class Raw:
        hf_dataset=table
        def __getitem__(self,i):
            return {'action':torch.zeros(3,8),'action_is_pad':torch.zeros(3,dtype=torch.bool)}
    wrapped=ActionDataset(Raw(),{'chunk_length':3,'frame_stride':1})
    assert len(wrapped)==5
    assert wrapped[0]['action_is_pad'].tolist()==[False,True,True]
    assert wrapped[1]['action_is_pad'].tolist()==[False,False,True]
    assert wrapped[4]['action_is_pad'].tolist()==[False,True,True]


def test_vla_pi05_loss_excludes_padded_targets_without_loading_model():
    from types import SimpleNamespace as NS
    from vla.adapters.lerobot.policies.modeling_vla_pi05 import VLAPI05Policy
    from lerobot.utils.constants import OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK
    fake=NS(_preprocess_images=lambda batch:([],[]), prepare_action=lambda batch:batch['action'],
        config=NS(output_features={'action':NS(shape=(2,))}),
        model=NS(sample_noise=lambda shape,device:None,sample_time=lambda size,device:None,
                 forward=lambda *args:torch.tensor([[[2.,4.],[1e6,1e6]]])))
    loss,metrics=VLAPI05Policy.forward(fake,{'action':torch.zeros(1,2,2),
        'action_is_pad':torch.tensor([[False,True]]), OBS_LANGUAGE_TOKENS:None, OBS_LANGUAGE_ATTENTION_MASK:None})
    assert loss.item()==3 and metrics['loss_per_dim']==[2.,4.]


def test_statistics_and_training_share_segment_starts_across_parquet_shards(tmp_path):
    import json
    import pandas as pd
    from datasets import Dataset
    from vla.actions.dataset import dataset_statistics
    from vla.adapters.lerobot.action_dataset import ActionDataset
    spec = ActionSpec('joint', representation='chunk_relative')
    raw = np.zeros((6, 8))
    raw[:, 0] = np.arange(6)
    rows = pd.DataFrame({'episode_index': [0]*6, 'frame_index': list(range(6)),
        'complementary_info.segment_id': [[0]]*3+[[1]]*3,
        'action': list(raw), 'observation.anchor': list(np.zeros_like(raw))})
    (tmp_path/'meta').mkdir()
    (tmp_path/'data').mkdir()
    (tmp_path/'meta/info.json').write_text(json.dumps({'action_spec': ActionSpec('joint').to_dict()}))
    rows.iloc[:2].to_parquet(tmp_path/'data/a.parquet')
    rows.iloc[2:].to_parquet(tmp_path/'data/b.parquet')
    stats = dataset_statistics(tmp_path, spec, SamplingSpec(2, start_stride=2))
    assert stats['count'] == 6
    assert stats['mean'][0] == pytest.approx(2.5)
    class Raw:
        hf_dataset = Dataset.from_pandas(rows)
        def __getitem__(self, index):
            return {'action': torch.zeros(2, 8)}
    wrapped = ActionDataset(Raw(), {'chunk_length': 2, 'frame_stride': 1, 'start_stride': 2})
    assert wrapped.indices.tolist() == [0, 2, 3, 5]
    assert wrapped[1]['action_is_pad'].tolist() == [False, True]
