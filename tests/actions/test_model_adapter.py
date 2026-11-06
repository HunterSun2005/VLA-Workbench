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
