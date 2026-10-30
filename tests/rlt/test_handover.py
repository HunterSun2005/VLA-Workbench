from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import Dataset
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from vla.adapters.lerobot.checkpoint_contract import sha256_file
from vla.adapters.lerobot.policies.configuration_rlt_ac import ChunkACPolicyConfig
from vla.adapters.lerobot.policies.modeling_rlt_ac import ChunkACPolicy
from vla.adapters.lerobot.policies.modeling_rlt_token import extract_pi05_prefix_hidden
from vla.cli.build_handover_cache import (
    HANDOVER_CACHE_KIND,
    _encode_episode,
    load_episode_phase_labels,
)
from vla.cli.train_handover import train_handover
from vla.core.handover_checkpoint import (
    HANDOVER_CONFIG_NAME,
    HANDOVER_METRICS_NAME,
    HANDOVER_MODEL_NAME,
    load_handover_classifier,
    save_handover_checkpoint,
)
from vla.core.phase_controller import HandoverClassifier, Phase, PhaseController


class _LogitClassifier(nn.Module):
    def __init__(self, probabilities: list[float]) -> None:
        super().__init__()
        self.probabilities = iter(probabilities)

    def forward(self, z_rl: torch.Tensor) -> torch.Tensor:
        probability = next(self.probabilities)
        logit = torch.logit(torch.tensor(probability, dtype=z_rl.dtype))
        return logit.expand(z_rl.shape[0], 1)


def test_prefix_only_encoder_does_not_sample_actions():
    batch_size = 2
    prefix = torch.arange(batch_size * 3 * 4, dtype=torch.float32).reshape(batch_size, 3, 4)

    class PrefixModel:
        def __init__(self):
            config = SimpleNamespace(_attn_implementation="unset")
            language_model = SimpleNamespace(config=config)
            model = SimpleNamespace(language_model=language_model)
            paligemma = SimpleNamespace(model=model)
            self.paligemma_with_expert = SimpleNamespace(
                paligemma=paligemma,
                forward=self.forward,
            )
            self.forward_calls = 0

        @staticmethod
        def embed_prefix(images, image_masks, tokens, token_masks):
            del images, image_masks, tokens, token_masks
            pad = torch.ones(batch_size, 3, dtype=torch.bool)
            attention = torch.zeros(batch_size, 3, dtype=torch.bool)
            return prefix, pad, attention

        @staticmethod
        def _prepare_attention_masks_4d(attention):
            return attention[:, None]

        def forward(self, **kwargs):
            self.forward_calls += 1
            assert kwargs["use_cache"] is False
            assert kwargs["inputs_embeds"][0] is prefix
            assert kwargs["inputs_embeds"][1] is None
            return [prefix + 1, None], None

        @staticmethod
        def sample_actions(*args, **kwargs):
            raise AssertionError("handover-only extraction must not sample actions")

    model = PrefixModel()
    pi05 = SimpleNamespace(
        _preprocess_images=lambda batch: ([batch["image"]], [torch.ones(batch_size)]),
        model=model,
    )
    batch = {
        "image": torch.zeros(batch_size, 3, 2, 2),
        OBS_LANGUAGE_TOKENS: torch.ones(batch_size, 3, dtype=torch.int64),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(batch_size, 3, dtype=torch.bool),
    }

    hidden = extract_pi05_prefix_hidden(pi05, batch)

    assert torch.equal(hidden, prefix + 1)
    assert model.forward_calls == 1


def test_handover_cache_episode_contains_only_z_rl_labels_and_location():
    class SyntheticDataset(Dataset):
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return {
                "observation.state": torch.tensor([float(index)]),
                "task": "synthetic task",
            }

    class Policy:
        config = SimpleNamespace(rl_token_dim=2)

        @staticmethod
        def encode_observation(batch):
            state = batch["observation.state"].float()
            return torch.cat([state, state + 10], dim=1)

    samples = _encode_episode(
        policy=Policy(),
        preprocessor=lambda batch: batch,
        dataset=SyntheticDataset(),
        frame_indices=[0, 2],
        episode_start=0,
        episode_id=7,
        frame_annotations={0: {"phase": 0.0}, 2: {"phase": 1.0}},
        batch_size=2,
        num_workers=0,
        device="cpu",
        empty_cache_every=0,
        task_override=None,
    )

    assert [set(sample) for sample in samples] == [
        {"z_rl", "is_critical", "episode_id", "frame_index"},
        {"z_rl", "is_critical", "episode_id", "frame_index"},
    ]
    assert torch.equal(samples[0]["z_rl"], torch.tensor([0.0, 10.0]))
    assert samples[1]["is_critical"].item() == 1.0
    assert samples[1]["frame_index"].item() == 2


def test_handover_cache_reads_phase_labels_recorded_during_collection():
    class SelectedDataset:
        def __init__(self, values):
            self.values = values

        def __getitem__(self, key):
            assert key == "complementary_info.phase"
            return self.values

    class HFDataset:
        column_names = ["index", "complementary_info.phase"]

        @staticmethod
        def select(indices):
            assert list(indices) == [3, 4, 5]
            return SelectedDataset([torch.tensor([0.0]), [1.0], 0.0])

    labels = load_episode_phase_labels(
        SimpleNamespace(hf_dataset=HFDataset()),
        episode_start=3,
        episode_stop=6,
    )

    assert labels == {
        3: {"phase": 0.0},
        4: {"phase": 1.0},
        5: {"phase": 0.0},
    }


def test_handover_cache_returns_none_without_recorded_phase_column():
    dataset = SimpleNamespace(hf_dataset=SimpleNamespace(column_names=["index"]))
    assert load_episode_phase_labels(dataset, 0, 1) is None


def test_phase_controller_uses_hysteresis_debounce_and_manual_cooldown():
    classifier = _LogitClassifier([0.9, 0.9, 0.5, 0.1, 0.1, 0.9, 0.9])
    controller = PhaseController(
        mode="learned",
        classifier=classifier,
        enter_threshold=0.8,
        exit_threshold=0.2,
        debounce_chunks=2,
        manual_cooldown_chunks=3,
    )
    z_rl = torch.zeros(1, 4)

    assert controller.update(z_rl) == Phase.VLA_PHASE
    assert controller.update(z_rl) == Phase.CRITICAL_PHASE
    assert controller.update(z_rl) == Phase.CRITICAL_PHASE
    assert controller.update(z_rl) == Phase.CRITICAL_PHASE
    assert controller.update(z_rl) == Phase.VLA_PHASE

    controller.trigger_vla()
    assert controller.manual_cooldown_remaining == 3
    for remaining in (2, 1, 0):
        assert controller.update(z_rl) == Phase.VLA_PHASE
        assert controller.manual_cooldown_remaining == remaining
    assert controller.update(z_rl) == Phase.VLA_PHASE
    assert controller.update(z_rl) == Phase.CRITICAL_PHASE
    controller.reset()
    assert controller.phase == Phase.VLA_PHASE
    assert controller.last_probability is None


def test_chunk_ac_config_requires_checkpoint_only_for_learned_handover():
    assert ChunkACPolicyConfig(device="cpu", phase_mode="manual").phase_mode == "manual"
    assert ChunkACPolicyConfig(device="cpu", phase_mode="always_rl").phase_mode == "always_rl"
    with pytest.raises(ValueError, match="requires handover_pretrained_path"):
        ChunkACPolicyConfig(device="cpu", phase_mode="learned")

    config = ChunkACPolicyConfig(
        device="cpu",
        phase_mode="learned",
        handover_pretrained_path="/tmp/handover",
    )
    assert config.handover_enter_threshold == 0.8
    assert config.handover_exit_threshold == 0.2
    assert config.handover_debounce_chunks == 2
    assert config.handover_manual_cooldown_chunks == 3


def test_chunk_ac_learned_handover_reuses_z_rl_for_action_chunk():
    policy = object.__new__(ChunkACPolicy)
    policy.config = SimpleNamespace(action_dim=2, proprio_dim=2)
    policy._rtc_config = None
    policy._last_phase_changed = False
    object.__setattr__(policy, "eval", lambda: policy)

    class FakePi05:
        @staticmethod
        def predict_action_chunk(batch, **kwargs):
            return torch.zeros(1, 4, 2)

    class FakeRLToken:
        def __init__(self):
            self.calls = 0

        def encode(self, prefix):
            self.calls += 1
            return torch.full((1, 4), float(self.calls))

    encoder = FakeRLToken()
    controller = PhaseController(
        mode="learned",
        classifier=_LogitClassifier([0.9, 0.9]),
        enter_threshold=0.8,
        exit_threshold=0.2,
        debounce_chunks=2,
    )
    seen_z_rl = []

    def compute_chunk(vla_chunk, proprio, prefix_tokens, *, z_rl=None):
        seen_z_rl.append(z_rl)
        return vla_chunk

    modifier = SimpleNamespace(
        phase_ctrl=controller,
        rl_token=encoder,
        compute_chunk=compute_chunk,
    )
    policy._rl_token_policy = SimpleNamespace(_pi05=FakePi05())
    policy._prefix_capture = SimpleNamespace(consume=lambda: torch.zeros(1, 2, 4))
    object.__setattr__(policy, "_ensure_modifier", lambda: modifier)
    batch = {"observation.state": torch.zeros(1, 2)}

    policy.predict_action_chunk(batch)
    assert controller.phase == Phase.VLA_PHASE
    policy.predict_action_chunk(batch)

    assert controller.phase == Phase.CRITICAL_PHASE
    assert encoder.calls == 2
    assert [value[0, 0].item() for value in seen_z_rl] == [1.0, 2.0]
    assert policy._last_phase_changed is True


def _fingerprint(digest: str) -> dict:
    return {"kind": "rlt_token", "digest": digest, "files": {}}


def test_handover_checkpoint_round_trip_and_contract_validation(tmp_path: Path):
    torch.manual_seed(0)
    classifier = HandoverClassifier(4, hidden_dim=3)
    config = {
        "format_version": 1,
        "input_kind": "z_rl",
        "input_dim": 4,
        "hidden_dim": 3,
        "rl_token_contract": _fingerprint("matching"),
    }
    output = save_handover_checkpoint(
        tmp_path / "handover",
        classifier=classifier,
        config=config,
        metrics={"best_val_loss": 0.1},
    )
    assert (output / HANDOVER_MODEL_NAME).is_file()
    assert (output / HANDOVER_CONFIG_NAME).is_file()
    assert (output / HANDOVER_METRICS_NAME).is_file()

    loaded, loaded_config = load_handover_classifier(
        output,
        expected_input_dim=4,
        expected_rl_token_contract=_fingerprint("matching"),
        device="cpu",
    )
    assert loaded_config == config
    for expected, actual in zip(classifier.parameters(), loaded.parameters(), strict=True):
        assert torch.equal(expected, actual)
        assert actual.requires_grad is False

    with pytest.raises(ValueError, match="fingerprint mismatch"):
        load_handover_classifier(
            output,
            expected_input_dim=4,
            expected_rl_token_contract=_fingerprint("different"),
            device="cpu",
        )


def _transition(z_rl: list[float], label: int, episode_id: int) -> dict[str, torch.Tensor]:
    return {
        "state_vec": torch.tensor([*z_rl, 99.0, 100.0]),
        "exec_chunk": torch.zeros(2, 1),
        "ref_chunk": torch.zeros(2, 1),
        "reward_seq": torch.zeros(2),
        "next_state_vec": torch.zeros(4),
        "next_ref_chunk": torch.zeros(2, 1),
        "done": torch.tensor(0.0),
        "intervention": torch.tensor(0.0),
        "actual_steps": torch.tensor(2),
        "source": torch.tensor(0),
        "episode_id": torch.tensor(episode_id),
        "is_critical": torch.tensor(float(label)),
    }


def _make_handover_cache(root: Path, *, handover_only: bool = True) -> dict:
    contract = _fingerprint("cache-token")
    shards = {"train": [], "val": []}
    values = {
        "train": [
            _transition([-2.0, -2.0], 0, 0),
            _transition([-1.0, -2.0], 0, 0),
            _transition([2.0, 2.0], 1, 1),
            _transition([1.0, 2.0], 1, 1),
        ],
        "val": [
            _transition([-3.0, -2.0], 0, 2),
            _transition([-2.0, -3.0], 0, 2),
            _transition([3.0, 2.0], 1, 3),
            _transition([2.0, 3.0], 1, 3),
        ],
    }
    for split, transitions in values.items():
        if handover_only:
            transitions = [
                {
                    "z_rl": transition["state_vec"][:2],
                    "is_critical": transition["is_critical"],
                    "episode_id": transition["episode_id"],
                    "frame_index": torch.tensor(index, dtype=torch.int64),
                }
                for index, transition in enumerate(transitions)
            ]
        path = root / "shards" / split / "episode_000000.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(transitions, path)
        shards[split].append(
            {
                "episode_id": 0,
                "path": path.relative_to(root).as_posix(),
                "transitions": len(transitions),
                "sha256": sha256_file(path),
            }
        )
    manifest = {
        "format_version": 1,
        "status": "complete",
        "contract": {
            "dataset": {"fps": 30, "features": {}},
            "policy": {"rl_token_dim": 2, "rl_token_contract": contract},
            "cache": {"kind": HANDOVER_CACHE_KIND if handover_only else "full_transition"},
        },
        "shards": shards,
    }
    (root / "cache_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return contract


def test_train_handover_uses_cached_z_rl_and_writes_loadable_checkpoint(tmp_path: Path):
    cache = tmp_path / "cache"
    contract = _make_handover_cache(cache)
    output = tmp_path / "handover"
    result = train_handover(
        SimpleNamespace(
            cache_dir=str(cache),
            output_dir=str(output),
            epochs=30,
            batch_size=4,
            lr=0.03,
            weight_decay=0.0,
            hidden_dim=8,
            patience=10,
            device="cpu",
            seed=0,
            num_workers=0,
        )
    )

    assert result["metrics"]["train_counts"] == {"negative": 2, "positive": 2}
    classifier, config = load_handover_classifier(
        output,
        expected_input_dim=2,
        expected_rl_token_contract=contract,
        device="cpu",
    )
    probabilities = torch.sigmoid(
        classifier(torch.tensor([[-2.0, -2.0], [2.0, 2.0]])).squeeze(-1)
    )
    assert probabilities[0] < probabilities[1]
    assert config["label"] == "is_critical"
