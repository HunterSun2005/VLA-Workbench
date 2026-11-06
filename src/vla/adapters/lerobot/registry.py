"""Public LeRobot policy registration plus VLA-owned cache dataset routing."""
from pathlib import Path


def register() -> None:
    from vla.adapters.lerobot.policies import configuration_rlt, configuration_rlt_ac, configuration_rlt_token
    from vla.adapters.lerobot.policies import configuration_vla_pi05
    from vla.adapters.lerobot import action_processor
    for cls in (configuration_rlt.RLTPretrainedConfig, configuration_rlt_ac.ChunkACConfig, configuration_rlt_token.RLTokenConfig):
        cls.ensure_registered()


def make_train_eval_datasets(cfg):
    """Preserve pre-split transition caches and defer ordinary data to LeRobot."""
    from lerobot.datasets.factory import make_train_eval_datasets as upstream
    from vla.adapters.lerobot.policies.configuration_rlt_ac import ChunkACConfig
    if isinstance(cfg.policy, ChunkACConfig):
        if getattr(cfg.dataset, "eval_split", 0) != 0:
            raise ValueError("Transition caches are already split; set dataset.eval_split=0")
        from vla.adapters.lerobot.policies.dataset_rlt_ac import ChunkTransitionDataset
        return ChunkTransitionDataset(Path(cfg.dataset.repo_id), split="train"), None
    return upstream(cfg)
