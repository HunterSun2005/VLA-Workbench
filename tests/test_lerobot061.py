"""Offline integration checks against the real pinned LeRobot factories/trainer."""
from types import SimpleNamespace
import json
import multiprocessing
import sys
import subprocess
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader


def _spawn_registry(queue):
    from vla.adapters.lerobot import register
    from lerobot.policies import get_policy_class
    register()
    queue.put([get_policy_class(name).__name__ for name in ("rlt", "rlt_token", "rlt_ac")])


def test_public_factories_and_spawn():
    # Import facade first: registration must work without monkeypatching stale exports.
    from lerobot.policies import get_policy_class, make_policy_config
    from vla.adapters.lerobot import register
    register()
    for kind, name in [("rlt", "RLTPretrainedPolicy"), ("rlt_token", "RLTokenPolicy"), ("rlt_ac", "ChunkACPolicy")]:
        cfg = make_policy_config(kind, device="cpu")
        assert cfg.type == kind
        assert get_policy_class(kind).__name__ == name
    ctx = multiprocessing.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_spawn_registry, args=(queue,))
    process.start()
    try:
        assert queue.get(timeout=45) == ["RLTPretrainedPolicy", "RLTokenPolicy", "ChunkACPolicy"]
        process.join(timeout=10)
        assert process.exitcode == 0
    finally:
        if process.is_alive():
            process.terminate()
            process.join()
        queue.close()


@pytest.mark.parametrize("kind", ["rlt_token", "rlt_ac"])
def test_official_processor_discovery(kind, monkeypatch):
    from lerobot.policies import make_policy_config, make_pre_post_processors
    from vla.adapters.lerobot import register
    from vla.adapters.lerobot.policies.processor_rlt import make_rlt_pre_post_processors
    register()
    cfg = make_policy_config(kind, device="cpu")
    identity = make_rlt_pre_post_processors(cfg)
    module = f"vla.adapters.lerobot.policies.processor_{kind}"
    monkeypatch.setattr(module + ".load_sft_pi05_processors", lambda *_: identity)
    pre, post = make_pre_post_processors(cfg)
    batch = {"state_vec": torch.zeros(1, 4)} if kind == "rlt_ac" else {"observation.state": torch.zeros(1, 10)}
    result = pre(batch)
    assert set(batch).issubset(result)
    assert torch.equal(post(torch.ones(1, 10)), torch.ones(1, 10))


def test_cache_sampler_spawn_and_resume(tmp_path):
    from tests.rlt.test_cache_manifest_ac import _make_cache
    from vla.adapters.lerobot.policies.dataset_rlt_ac import ChunkTransitionDataset
    from lerobot.datasets import EpisodeAwareSampler
    _make_cache(tmp_path)
    dataset = ChunkTransitionDataset(tmp_path)
    def sampler():
        return EpisodeAwareSampler(dataset.meta.episodes["dataset_from_index"], dataset.meta.episodes["dataset_to_index"], episode_indices_to_use=dataset.episodes, absolute_to_relative_idx=dataset.absolute_to_relative_idx, shuffle=True, seed=42)
    first = sampler()
    order = list(first)
    assert sorted(order) == [0, 1, 2]
    resumed = sampler()
    resumed.load_state_dict({"epoch": 0, "start_index": 1})
    assert list(resumed) == order[1:]
    loader = DataLoader(dataset, batch_size=2, sampler=sampler(), num_workers=1, multiprocessing_context="spawn")
    values = torch.cat([batch["state_vec"][:, 0] for batch in loader])
    assert sorted(values.tolist()) == [1.0, 2.0, 5.0]
    assert not dataset.meta.has_language_columns


def test_sidecar_survives_typed_info_roundtrip(tmp_path):
    from lerobot.datasets.utils import DatasetInfo
    from vla.dataset_metadata import load_info, write_metadata_sidecar
    info = {"codebase_version": "v3.0", "fps": 30, "features": {}, "collection_kind": "human_demonstration", "recording_contract": {"version": 1}, "evo_rlt": {"actions_stored_as": "absolute"}}
    write_metadata_sidecar(tmp_path, info)
    (tmp_path / "meta/info.json").write_text(json.dumps(DatasetInfo.from_dict(info).to_dict()))
    recovered = load_info(tmp_path)
    for key in ("collection_kind", "recording_contract", "evo_rlt"):
        assert recovered[key] == info[key]
    (tmp_path / "meta/vla_metadata.json").write_text('{"fps": 999}')
    with pytest.raises(ValueError, match="standard"):
        load_info(tmp_path)


def test_native_update_metrics_once():
    from accelerate import Accelerator
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.logging_utils import MetricsTracker, AverageMeter
    from vla.cli.train_franka_pi05_sft import install_pi05_loss_metrics_patch
    class TinyPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(10))
            self.config = SimpleNamespace(type="pi05", action_feature_names=[f"a{i}" for i in range(10)])
        def forward(self, batch):
            per_dim = (self.weight - batch["action"]).square().mean(0)
            loss = per_dim.mean()
            return loss, {"loss": loss.item(), "loss_per_dim": per_dim.detach().tolist()}
    policy = TinyPolicy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    metrics = {name: AverageMeter(name) for name in ("loss", "grad_norm", "lr", "update_s", "gpu_mem_gb")}
    tracker = MetricsTracker(2, 4, 1, metrics)
    trainer = SimpleNamespace(update_policy=update_policy)
    install_pi05_loss_metrics_patch(trainer)
    accelerator = Accelerator(cpu=True)
    for _ in range(2):
        trainer.update_policy(tracker, policy, {"action": torch.zeros(2, 10)}, optimizer, 1.0, accelerator=accelerator)
    assert torch.all(policy.weight < 1)
    assert tracker.metrics["loss"].count == 2
    assert tracker.metrics["flow_loss/dim/a0"].count == 2


def test_package_import_is_lightweight():
    result = subprocess.run(["/usr/bin/python3", "-c", "import vla, vla_rollout_protocol, vla.core.writer_lock, vla.core.online_contract, sys; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_ac_resume_processor_keeps_cache_bypass():
    from vla.cli.train_actor_critic_native import install_ac_processor_hook
    from vla.adapters.lerobot.policies.configuration_rlt_ac import ChunkACConfig
    from vla.adapters.lerobot.policies.processor_rlt import make_rlt_pre_post_processors
    from lerobot.processor import PolicyProcessorPipeline
    cfg = ChunkACConfig(device="cpu")
    pre, post = make_rlt_pre_post_processors(cfg)
    trainer = SimpleNamespace(make_pre_post_processors=lambda *a, **kw: (pre, post))
    install_ac_processor_hook(trainer)
    wrapped, _ = trainer.make_pre_post_processors(cfg, pretrained_path="synthetic-resume")
    cache_batch = {"state_vec": torch.zeros(2, 42), "exec_chunk": torch.zeros(2, 4, 10)}
    assert wrapped(cache_batch) is cache_batch
    assert wrapped.steps == pre.steps


def test_old_automatic_home_config_requires_explicit_ros_migration(tmp_path):
    import yaml
    from vla.datacollection.franka_robotiq_rollout.config import load_rollout_config
    path=tmp_path/'old-home.yaml'
    path.write_text(yaml.safe_dump({'robot':{'type':'franka_robotiq'},'deployment':{'home':{'enabled':True}}}))
    with pytest.raises(ValueError,match='Legacy Franky configuration'):
        load_rollout_config(path)


def test_all_lerobot_imports_resolve_including_lazy_entrypoints():
    import ast
    import importlib
    from vla.paths import resource
    imports = set()
    for path in resource("").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("lerobot"):
                imports.update((node.module, alias.name) for alias in node.names if alias.name != "*")
    for module, name in sorted(imports):
        imported = importlib.import_module(module)
        if not hasattr(imported, name):
            importlib.import_module(module + "." + name)


def test_single_step_inference_image_scale_and_call_order():
    import numpy as np
    from vla.adapters.lerobot.inference import predict_action
    image = np.full((4, 6, 3), 255, dtype=np.uint8)
    observation = {"observation.images.base": image, "observation.state": np.zeros(10, dtype=np.float32)}
    calls = []
    def pre(batch):
        calls.append("pre")
        assert batch["observation.images.base"].shape == (1, 3, 4, 6)
        assert torch.all(batch["observation.images.base"] == 1)
        assert batch["task"] == "test"
        assert not torch.is_grad_enabled()
        return batch
    class Policy:
        def select_action(self, batch):
            calls.append("policy")
            return torch.ones(1, 10)
    def post(action):
        calls.append("post")
        return action * 2
    action = predict_action(observation, Policy(), "cpu", pre, post, False, "test", "franka_robotiq")
    assert action.shape == (1, 10)
    assert calls == ["pre", "policy", "post"]
    assert observation["observation.images.base"] is image
    assert isinstance(observation["observation.state"], np.ndarray)

@pytest.mark.parametrize("kind", ["rlt", "rlt_token", "rlt_ac"])
def test_public_make_policy_constructs_real_custom_classes(kind, monkeypatch):
    from lerobot.policies import make_policy, make_policy_config
    from vla.adapters.lerobot import register
    from vla.adapters.lerobot.policies.configuration_rlt_token import RLTokenConfig
    from vla.adapters.lerobot.policies.modeling_rlt_token import RLTokenPolicy
    from vla.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
    register()
    token_cfg = RLTokenConfig(device="cpu", action_dim=10, proprio_dim=10, camera_keys=["base", "left_wrist"], image_resolution=(16, 16), rl_token_dim=32, rl_token_nhead=2, rl_token_enc_layers=1, rl_token_dec_layers=1, rl_token_ff_dim=64, rl_token_num_rl_tokens=1, token_pool_size=8)
    monkeypatch.setattr(RLTokenPolicy, "_load_pi05_backbone", lambda self: torch.nn.Linear(8, 16))
    monkeypatch.setattr(RLTokenPolicy, "_compute_num_image_tokens", lambda *_: 0)
    monkeypatch.setattr(ChunkACPolicy, "_load_rl_token_config", lambda _: token_cfg)
    if kind == "rlt_token":
        cfg = token_cfg
    elif kind == "rlt_ac":
        cfg = make_policy_config(kind, device="cpu", action_dim=10, proprio_dim=10, camera_keys=["base", "left_wrist"], image_resolution=(16, 16), rl_token_dim=32, rl_token_num_rl_tokens=1, token_pool_size=8, chunk_length=4, actor_hidden_dim=16, critic_hidden_dim=16)
    else:
        cfg = make_policy_config(kind, device="cpu", rl_token_dim=32, rl_token_nhead=2, rl_token_enc_layers=1, rl_token_dec_layers=1, rl_token_ff_dim=64, actor_hidden_dim=16)
    features = {
        "observation.state": {"dtype": "float32", "shape": (10,), "names": [f"s{i}" for i in range(10)]},
        "action": {"dtype": "float32", "shape": (10,), "names": [f"a{i}" for i in range(10)]},
        "observation.images.base": {"dtype": "video", "shape": (16, 16, 3), "names": ["height", "width", "channels"]},
        "observation.images.left_wrist": {"dtype": "video", "shape": (16, 16, 3), "names": ["height", "width", "channels"]},
    }
    policy = make_policy(cfg, ds_meta=SimpleNamespace(features=features, stats={}))
    assert policy.config.type == kind
    assert next(policy.parameters()).device.type == "cpu"
    if kind == "rlt_ac":
        from accelerate import Accelerator
        from lerobot.scripts.lerobot_train import update_policy
        from lerobot.utils.logging_utils import MetricsTracker, AverageMeter
        from vla.cli.train_actor_critic_native import install_ac_update_hook
        from tests.rlt.test_cache_manifest_ac import _transition
        batch = {key: torch.stack([value, value]) for key, value in _transition(0.1).items()}
        optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad], lr=1e-3)
        meters = {name: AverageMeter(name) for name in ("loss", "grad_norm", "lr", "update_s", "gpu_mem_gb")}
        tracker = MetricsTracker(2, 2, 1, meters)
        trainer = SimpleNamespace(update_policy=update_policy)
        install_ac_update_hook(trainer)
        before = next(policy.critic.parameters()).detach().clone()
        trainer.update_policy(tracker, policy, batch, optimizer, 1.0, accelerator=Accelerator(cpu=True))
        assert int(policy._critic_step) == 1
        assert not torch.equal(next(policy.critic.parameters()), before)
