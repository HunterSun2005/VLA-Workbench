"""Project-owned training hooks; no files in the installed LeRobot are modified."""
from copy import deepcopy
from pathlib import Path
from vla.actions import ActionSpec
from vla.actions.dataset import dataset_fingerprint
from vla.adapters.lerobot.action_processor import adapt_processors


def install_action_training(trainer):
    if getattr(trainer, '_vla_action_training', False):
        return
    trainer._vla_action_training = True
    original_datasets = trainer.make_train_eval_datasets
    original_processors = trainer.make_pre_post_processors

    def datasets(cfg):
        contract = getattr(cfg.policy, 'action_contract', None)
        if contract:
            if getattr(cfg.dataset, 'eval_split', 0):
                raise ValueError('materialize the training split before computing action statistics; set dataset.eval_split=0')
            root = Path(cfg.dataset.root or cfg.dataset.repo_id)
            from vla.dataset_metadata import load_info
            info = load_info(root)
            spec = ActionSpec.from_dict(contract['action_spec'])
            spec.require_compatible(ActionSpec.from_dict(info['action_spec']), training=False)
            if getattr(cfg.dataset, 'episodes', None):
                raise ValueError('materialize selected training episodes before computing action statistics')
            if cfg.policy.type == 'vla_pi05' and dataset_fingerprint(root) != contract['statistics']['contract']['data_fingerprint']:
                raise ValueError('training dataset changed since action statistics were computed')
            if tuple(info['features']['action']['names']) != spec.action_fields:
                raise ValueError('dataset action names disagree with action contract')
            from vla.actions.metadata import require_recording_source
            recording = info.get('recording_contract') or info.get('processing', {}).get('source_contract', {}).get('recording_contract')
            require_recording_source(contract, recording)
            if info.get('recording_schema_version') != 5:
                raise ValueError('training requires schema-5 recordings')
            if cfg.policy.type == 'rlt_token':
                cfg.policy.training_data_fingerprint = dataset_fingerprint(root)
            sampling = contract['statistics']['contract']['sampling']
            if cfg.policy.type in ('vla_pi05', 'rlt_token') and sampling['chunk_length'] != cfg.policy.chunk_size:
                raise ValueError('reference model sampling does not match statistics')
            if 'observation.anchor' not in info['features']:
                raise ValueError('training dataset lacks independent raw anchors')
        result = original_datasets(cfg)
        if contract and cfg.policy.type in ('vla_pi05', 'rlt_token'):
            from vla.adapters.lerobot.action_dataset import ActionDataset
            sampling = contract['statistics']['contract']['sampling']
            return tuple(ActionDataset(ds, sampling) if ds is not None else None for ds in result)
        return result

    def processors(policy_cfg, **kwargs):
        contract = getattr(policy_cfg, 'action_contract', None)
        if contract and policy_cfg.type == 'vla_pi05':
            action_stats = {k: contract['statistics'][k] for k in ('min', 'max', 'mean', 'std', 'q01', 'q99')}
            kwargs = deepcopy(kwargs)
            if 'dataset_stats' in kwargs:
                kwargs['dataset_stats']['action'] = action_stats
            for key, normalizer in [('preprocessor_overrides', 'normalizer_processor'),
                                    ('postprocessor_overrides', 'unnormalizer_processor')]:
                overrides = kwargs.get(key, {})
                for legacy in ('relative_actions_processor', 'absolute_actions_processor'):
                    overrides.pop(legacy, None)
                if normalizer in overrides and 'stats' in overrides[normalizer]:
                    overrides[normalizer]['stats']['action'] = action_stats
            pre, post = original_processors(policy_cfg=policy_cfg, **kwargs)
            return adapt_processors(pre, post, ActionSpec.from_dict(contract['action_spec']))
        return original_processors(policy_cfg=policy_cfg, **kwargs)

    trainer.make_train_eval_datasets = datasets
    trainer.make_pre_post_processors = processors
