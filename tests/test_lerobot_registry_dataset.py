from types import SimpleNamespace

import torch

import vla.adapters.lerobot.registry as registry
from vla.adapters.lerobot.policies.dataset_rlt_ac import ChunkTransitionDataset


def test_register_loads_chunk_transition_dataset_from_cache_dir(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    from tests.rlt.test_cache_manifest_ac import _make_cache
    _make_cache(cache_dir)

    from vla.adapters.lerobot.policies.configuration_rlt_ac import ChunkACConfig

    registry.register()

    cfg = SimpleNamespace(policy=ChunkACConfig(device="cpu"), dataset=SimpleNamespace(repo_id=str(cache_dir), eval_split=0))
    dataset, eval_dataset = registry.make_train_eval_datasets(cfg)
    assert eval_dataset is None

    assert isinstance(dataset, ChunkTransitionDataset)
    assert len(dataset) == 3
    assert dataset[0]["state_vec"].shape == (42,)
