from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch.utils.data import Dataset

from vla.cli.build_transition_cache_v2 import (
    _encoded_to_transitions,
    _encode_episode,
    _prepare_manifest,
    resolve_episode_success,
    split_episode_ids,
    validate_chunk_settings,
)


def test_cache_transition_uses_expert_action_and_c_step_bootstrap() -> None:
    encoded = {
        frame: {
            "state_vec": torch.tensor([float(frame)]),
            "exec_chunk": torch.full((2, 1), 100.0 + frame),
            "ref_chunk": torch.full((2, 1), 200.0 + frame),
        }
        for frame in range(5)
    }

    transitions = _encoded_to_transitions(
        encoded,
        frame_indices=list(range(5)),
        episode_last_frame=4,
        chunk_length=2,
        episode_success=True,
        episode_id=7,
    )

    assert len(transitions) == 3
    assert transitions[0]["state_vec"].item() == 0
    assert transitions[0]["next_state_vec"].item() == 2
    assert transitions[0]["exec_chunk"][0, 0].item() == 100
    assert transitions[0]["ref_chunk"][0, 0].item() == 200
    assert transitions[-1]["done"].item() == 1
    assert torch.equal(transitions[-1]["reward_seq"], torch.tensor([0.0, 1.0]))


def test_cache_transition_failure_has_no_reward() -> None:
    encoded = {
        frame: {
            "state_vec": torch.tensor([float(frame)]),
            "exec_chunk": torch.zeros(2, 1),
            "ref_chunk": torch.ones(2, 1),
        }
        for frame in range(3)
    }
    transitions = _encoded_to_transitions(encoded, [0, 1, 2], 2, 2, False, 0)
    assert len(transitions) == 1
    assert transitions[0]["reward_seq"].sum().item() == 0


def test_encode_episode_keeps_dataset_action_separate_from_vla_reference() -> None:
    class SyntheticDataset(Dataset):
        def __len__(self) -> int:
            return 5

        def __getitem__(self, index: int) -> dict:
            return {
                "observation.state": torch.tensor([float(index)]),
                "action": torch.full((4, 1), 10.0 + index),
                "task": "synthetic task",
            }

    class Capture:
        count = 0

        def consume(self) -> torch.Tensor:
            return torch.zeros(self.count, 1, 2)

    capture = Capture()

    class Pi05:
        def predict_action_chunk(self, batch: dict) -> torch.Tensor:
            count = batch["action"].shape[0]
            capture.count = count
            return torch.full((count, 4, 1), 99.0)

    class RLToken:
        def encode(self, prefix: torch.Tensor) -> torch.Tensor:
            return torch.zeros(prefix.shape[0], 2)

    transitions = _encode_episode(
        pi05=Pi05(),
        rl_token=RLToken(),
        preprocessor=lambda batch: batch,
        capture=capture,
        dataset=SyntheticDataset(),
        frame_indices=[0, 1, 2, 3, 4],
        episode_last_frame=4,
        chunk_length=2,
        action_dim=1,
        proprio_dim=1,
        batch_size=2,
        num_workers=0,
        device="cpu",
        empty_cache_every=0,
        task_override=None,
        episode_success=True,
        episode_id=0,
    )

    assert transitions[0]["exec_chunk"][0, 0].item() == 10.0
    assert transitions[0]["ref_chunk"][0, 0].item() == 99.0
    assert transitions[0]["next_state_vec"][-1].item() == 2.0


@pytest.mark.parametrize(
    ("chunk", "stride", "horizon", "message"),
    [(0, 1, 50, "between 1"), (51, 1, 50, "between 1"), (10, 3, 50, "divisible")],
)
def test_validate_chunk_settings_rejects_invalid_values(
    chunk: int, stride: int, horizon: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        validate_chunk_settings(chunk, stride, horizon)


def test_episode_split_is_deterministic_and_nonempty() -> None:
    first = split_episode_ids(list(range(5)), 0.9, 42)
    second = split_episode_ids(list(range(5)), 0.9, 42)
    assert first == second
    assert len(first[0]) == 4
    assert len(first[1]) == 1


def test_one_episode_requires_train_only() -> None:
    with pytest.raises(ValueError, match="train-ratio=1.0"):
        split_episode_ids([0], 0.9, 42)
    assert split_episode_ids([0], 1.0, 42) == ([0], [])


def test_missing_episode_success_requires_explicit_default() -> None:
    episodes = {"episode_index": [0]}
    with pytest.raises(ValueError, match="default-episode-success"):
        resolve_episode_success(episodes, 0, None)
    assert resolve_episode_success(episodes, 0, "success") is True
    assert resolve_episode_success(episodes, 0, "failure") is False


def test_huggingface_dataset_missing_success_column_uses_explicit_default() -> None:
    class ArrowLikeEpisodes:
        column_names = ["episode_index"]

        def __getitem__(self, key):
            raise AssertionError(f"missing column {key!r} should be detected before lookup")

    assert resolve_episode_success(ArrowLikeEpisodes(), 0, "success") is True


def test_manifest_resume_requires_identical_contract(tmp_path: Path) -> None:
    contract = {"dataset": {"fps": 30}, "policy": {}, "cache": {}}
    manifest = _prepare_manifest(tmp_path, contract, resume=False)
    assert manifest["status"] == "building"
    resumed = _prepare_manifest(tmp_path, contract, resume=True)
    assert resumed == manifest

    with pytest.raises(ValueError, match="does not match"):
        _prepare_manifest(
            tmp_path,
            {"dataset": {"fps": 20}, "policy": {}, "cache": {}},
            resume=True,
        )
    assert json.loads((tmp_path / "cache_manifest.json").read_text())["contract"] == contract
