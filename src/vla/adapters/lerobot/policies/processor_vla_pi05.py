from copy import deepcopy
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from vla.actions import ActionSpec
from vla.adapters.lerobot.action_processor import adapt_processors


def make_vla_pi05_pre_post_processors(config, dataset_stats=None):
    stats = deepcopy(dataset_stats or {})
    contract = config.action_contract
    stats['action'] = {k: contract['statistics'][k] for k in ('min', 'max', 'mean', 'std', 'q01', 'q99')}
    pre, post = make_pi05_pre_post_processors(config, stats)
    return adapt_processors(pre, post, ActionSpec.from_dict(contract['action_spec']))
