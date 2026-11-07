"""PI0.5 with a serialized VLA action and chunk-sampling contract."""
from dataclasses import dataclass
from pathlib import Path
from lerobot.configs.policies import PreTrainedConfig
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from vla.actions import ActionSpec, SamplingSpec
from vla.actions.metadata import load_model_contract
from vla.actions.statistics import validate_statistics


@PreTrainedConfig.register_subclass('vla_pi05')
@dataclass
class VLAPI05Config(PI05Config):
    action_contract_path: str | None = None
    action_contract: dict | None = None

    def __post_init__(self):
        super().__post_init__()
        if self.action_contract_path and self.action_contract is None:
            _, _, self.action_contract = load_model_contract(Path(self.action_contract_path))
        if self.action_contract is None:
            raise ValueError('VLA PI05 requires action_contract_path or a saved action_contract')
        if self.action_contract.get('version') != 2:
            raise ValueError('VLA PI05 requires model action contract v2')
        spec = ActionSpec.from_dict(self.action_contract['action_spec'])
        stats = self.action_contract['statistics']
        sampling = SamplingSpec(**stats['contract']['sampling'])
        validate_statistics(stats, spec, sampling, stats['contract']['data_fingerprint'])
        if sampling.chunk_length != self.chunk_size or sampling.start_stride != 1:
            raise ValueError('checkpoint statistics must match chunk_size and every-frame training sampler')
        # Generic elementwise LeRobot deltas are never applied by this policy.
        self.use_relative_actions = False
        self.action_feature_names = list(spec.action_fields)

    @property
    def action_delta_indices(self):
        stride = self.action_contract['statistics']['contract']['sampling']['frame_stride']
        return [i * stride for i in range(self.chunk_size)]
