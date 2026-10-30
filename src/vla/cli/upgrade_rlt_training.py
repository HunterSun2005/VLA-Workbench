"""Explicit, offline, copy-on-upgrade migration of schema-4 online runs."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

import pandas as pd
import torch
from safetensors.torch import load_file

from vla.core.online_contract import (
    RECORDING_SCHEMA_VERSION, RUN_MANIFEST, TIMING_FIELDS, durable_json,
    file_digest, fsync_directory, load_run, model_digest,
)
from vla.core.online_replay import ONLINE_REPLAY_FORMAT_VERSION, PersistentReplayBuffer
from vla.core.online_training import OnlineRLTSession, _scalar, build_critical_transition_specs
from vla.core.training_contract import TRAINING_SEMANTICS_VERSION
from vla.core.writer_lock import WriterLease


class _ReplayView:
    def __init__(self, connection):
        self.episodes = {row[0]: row[1:] for row in connection.execute(
            "SELECT episode_key, episode_id, transition_count, episode_digest FROM episodes"
        )}

    def episode_keys(self):
        return set(self.episodes)

    def has_episode(self, key):
        return key in self.episodes

    def episode_digest(self, key):
        return self.episodes[key][2] if key in self.episodes else None


def _require_offline(*roots):
    # Legacy servers do not take writer leases. Detect their open resources too.
    for process in Path("/proc").iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        try:
            for descriptor in (process / "fd").iterdir():
                try:
                    target = descriptor.resolve()
                except OSError:
                    continue
                if any(target == root or target.is_relative_to(root) for root in roots):
                    raise RuntimeError(f"stop process {process.name} before upgrade; open resource: {target}")
        except (PermissionError, FileNotFoundError, ProcessLookupError):
            continue


def _tree_digest(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"upgrade refuses symlinked source entries: {path}")
        if path.is_file() and path.name != "replay.sqlite3-shm":
            result[path.relative_to(root).as_posix()] = file_digest(path)
    return result


def _checkpoint(root, name):
    if not name or Path(name).name != name:
        raise ValueError("invalid checkpoint reference")
    directory = root / "checkpoints" / name
    metadata = json.loads((directory / "metadata.json").read_text())
    if metadata.get("format_version") != 2 or metadata.get("complete") is not True:
        raise ValueError(f"incomplete checkpoint: {name}")
    for filename in ("model.safetensors", "training_state.pt"):
        if metadata.get("sha256", {}).get(filename) != file_digest(directory / filename):
            raise ValueError(f"checkpoint checksum mismatch: {name}/{filename}")
    state = torch.load(directory / "training_state.pt", map_location="cpu", weights_only=False)
    required = {"actor_optimizer", "critic_optimizer", "consumed_credits", "critic_updates", "actor_updates",
        "torch_rng_state", "cuda_rng_state", "python_rng_state", "replay_rng_state"}
    if not isinstance(state, dict) or not required <= state.keys():
        raise ValueError(f"checkpoint lacks complete optimizer/progress/RNG state: {name}")
    for key in ("actor_optimizer", "critic_optimizer"):
        if not isinstance(state[key], dict) or not {"param_groups", "state"} <= state[key].keys():
            raise ValueError(f"checkpoint lacks optimizer state: {name}/{key}")
    return state


def _trainable_steps(dataset_root, rows, contract, view, connection):
    from vla.datacollection.franka_robotiq_single_left.v30_writer import DATA_PATH

    count = 0
    for row in rows:
        key = f"{row['dataset_uuid']}:{row['episode_uuid']}"
        if not view.has_episode(key):
            raise ValueError(f"Dataset episode {row['episode_index']} is missing from Replay; recover the old run first")
        path = dataset_root / DATA_PATH.format(
            chunk_index=int(row["data/chunk_index"]), file_index=int(row["data/file_index"]),
        )
        fields = {**TIMING_FIELDS, "phase": "float", "is_intervention": "float", "collector_policy_id": "int"}
        frame = pd.read_parquet(path, columns=["index", "episode_index", *(f"complementary_info.{key}" for key in fields)])
        start, stop = int(row["dataset_from_index"]), int(row["dataset_to_index"])
        if frame["index"].tolist() != list(range(start, stop)) or not (frame["episode_index"] == int(row["episode_index"])).all():
            raise ValueError("episode frame indices disagree with committed metadata")
        annotations = {}
        for item in frame.to_dict("records"):
            annotations[int(item["index"])] = {
                key: _scalar(item[f"complementary_info.{key}"], integer=dtype != "float")
                for key, dtype in fields.items()
            }
        specs = build_critical_transition_specs(annotations, episode_start=start, episode_stop=stop,
            chunk_length=contract["chunk_length"], stride=contract["stride"], fps=contract["fps"])
        stored_id, stored_count, _ = view.episodes[key]
        if stored_id != int(row["episode_index"]) or len(specs) != stored_count:
            raise ValueError("Replay transition count disagrees with Dataset timing; recover before upgrade")
        expected = {spec.anchor_frame: spec.actual_steps for spec in specs}
        for anchor, payload in connection.execute("SELECT anchor_frame, payload FROM transitions WHERE episode_key = ?", (key,)):
            transition = PersistentReplayBuffer._deserialize(payload)
            if expected.get(anchor) != int(transition["actual_steps"].item()):
                raise ValueError("retained Replay transition disagrees with Dataset timing")
        covered = set()
        for spec in specs:
            covered.update(range(spec.anchor_frame, spec.next_frame))
        count += len(covered)
    return count


def upgrade(dataset_root, replay_root, output_root, *, dry_run=False):
    dataset_root, replay_root, output_root = [Path(p).expanduser().resolve() for p in (dataset_root, replay_root, output_root)]
    roots = [dataset_root, replay_root, output_root]
    if any(a == b or a.is_relative_to(b) or b.is_relative_to(a) for i, a in enumerate(roots) for b in roots[i + 1:]):
        raise ValueError("source Dataset, Replay and output roots must be separate, non-nested directories")
    if output_root.exists():
        raise FileExistsError("upgrade output must not exist")
    if not dataset_root.is_dir() or not (replay_root / "replay.sqlite3").is_file():
        raise FileNotFoundError("source Dataset or Replay is missing")
    with WriterLease(*roots):
        _require_offline(dataset_root, replay_root)
        before = {"dataset": _tree_digest(dataset_root), "replay": _tree_digest(replay_root)}
        snapshot = sqlite3.connect(":memory:")
        try:
            # Even mode=ro may create WAL/SHM files. Read a private copy instead.
            with tempfile.TemporaryDirectory(prefix="rlt-upgrade-snapshot-") as snapshot_dir:
                snapshot_root = Path(snapshot_dir)
                for filename in ("replay.sqlite3", "replay.sqlite3-wal"):
                    if (replay_root / filename).exists():
                        shutil.copy2(replay_root / filename, snapshot_root / filename)
                source = sqlite3.connect((snapshot_root / "replay.sqlite3").as_uri() + "?mode=ro", uri=True)
                try:
                    source.backup(snapshot)
                finally:
                    source.close()
            state = dict(snapshot.execute("SELECT key, value FROM state"))
            if int(state["format_version"]) != 2:
                raise ValueError("upgrade expects legacy Replay format 2; source may already be upgraded")
            run = load_run(dataset_root)
            contract = run["contract"]
            if contract.get("training_semantics_version") not in (None, 1):
                raise ValueError("unsupported previous online training semantics")
            if run.get("recording_schema_version") != RECORDING_SCHEMA_VERSION:
                raise ValueError("only causally aligned schema-4 runs can be upgraded")
            if json.loads(state["contract"]) != run:
                raise ValueError("Dataset/Replay run contract mismatch")
            for prefix in ("rl_token", "vla"):
                if model_digest(contract[f"{prefix}_pretrained_path"]) != contract[f"{prefix}_sha256"]:
                    raise ValueError(f"{prefix} model content changed")
            view = _ReplayView(snapshot)
            session = object.__new__(OnlineRLTSession)
            session.dataset = type("DatasetRoot", (), {"root": dataset_root})()
            session.run, session.replay = run, view
            rows = session._saved_rows()
            trainable = _trainable_steps(dataset_root, rows, contract, view, snapshot)
            active = state["active_checkpoint"]
            saved = _checkpoint(replay_root, active)
            for name in {row["policy_revision"] for row in rows} - {active}:
                _checkpoint(replay_root, name)
            counters = {key: int(state[key]) for key in ("consumed_credits", "critic_updates", "actor_updates")}
            if any(saved[key] != value for key, value in counters.items()):
                raise ValueError("checkpoint/Replay progress mismatch")
            produced = int(state["produced_credits"])
            if (counters["critic_updates"] != counters["consumed_credits"]
                    or counters["actor_updates"] != counters["critic_updates"] // 2
                    or not 0 <= counters["consumed_credits"] <= produced
                    or produced != sum(value[1] for value in view.episodes.values()) * contract["utd_ratio"]):
                raise ValueError("incomplete Actor/Critic update counters")
            tensors = load_file(replay_root / "checkpoints" / active / "model.safetensors")
            if not all(torch.isfinite(tensor).all() for tensor in tensors.values()):
                raise ValueError("checkpoint contains non-finite model weights")
            if int(tensors["critic_step"].item()) != counters["critic_updates"]:
                raise ValueError("model/checkpoint Critic counter mismatch")
            batch_size = int(contract["training"]["batch_size"])
            if int(state["capacity"]) < batch_size:
                raise ValueError("Replay capacity is smaller than batch_size")
            entries = int(snapshot.execute("SELECT COUNT(*) FROM transitions").fetchone()[0])
            enabled = (trainable >= int(contract["training"]["warmup_steps"])
                and entries >= batch_size and counters["actor_updates"] > 0
                and counters["consumed_credits"] == produced)
            report = {"source_dataset": str(dataset_root), "source_replay": str(replay_root),
                "dataset_uuid": run["dataset_uuid"], "run_uuid": run["run_uuid"],
                "training_semantics_version": TRAINING_SEMANTICS_VERSION,
                "trainable_env_steps": trainable, "collected_warmup_steps": int(state["warmup_env_steps"]),
                "actor_enabled": enabled, "produced_credits": produced, **counters,
                "episodes": len(rows), "retained_transitions": entries}
            if before != {"dataset": _tree_digest(dataset_root), "replay": _tree_digest(replay_root)}:
                raise RuntimeError("source changed during preflight; stop all writers and retry")
            if dry_run:
                return {**report, "dry_run": True}
            stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}-upgrade-", dir=output_root.parent))
            try:
                shutil.copytree(dataset_root, stage / "dataset")
                shutil.copytree(replay_root, stage / "replay", ignore=shutil.ignore_patterns("replay.sqlite3", "replay.sqlite3-wal", "replay.sqlite3-shm"))
                upgraded_run = copy.deepcopy(run)
                upgraded_run["contract"]["training_semantics_version"] = TRAINING_SEMANTICS_VERSION
                upgraded_run["upgrade_publication"] = {
                    "dataset_root": str(output_root / "dataset"),
                    "replay_root": str(output_root / "replay"),
                }
                durable_json(stage / "dataset" / RUN_MANIFEST, upgraded_run)
                target_checkpoint = stage / "replay" / "checkpoints" / active
                saved.update(training_semantics_version=TRAINING_SEMANTICS_VERSION, actor_enabled=enabled)
                torch.save(saved, target_checkpoint / "training_state.pt")
                metadata = json.loads((target_checkpoint / "metadata.json").read_text())
                metadata["sha256"]["training_state.pt"] = file_digest(target_checkpoint / "training_state.pt")
                durable_json(target_checkpoint / "metadata.json", metadata)
                updates = {"format_version": str(ONLINE_REPLAY_FORMAT_VERSION),
                    "contract": json.dumps(upgraded_run, sort_keys=True, separators=(",", ":")),
                    "trainable_env_steps": str(trainable), "actor_enabled": str(int(enabled))}
                with snapshot:
                    snapshot.executemany("INSERT OR REPLACE INTO state(key, value) VALUES(?, ?)", updates.items())
                target = sqlite3.connect(stage / "replay" / "replay.sqlite3")
                try:
                    snapshot.backup(target)
                    if target.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                        raise ValueError("upgraded Replay integrity check failed")
                finally:
                    target.close()
                after = {"dataset": _tree_digest(dataset_root), "replay": _tree_digest(replay_root)}
                if before != after:
                    raise RuntimeError("source changed during upgrade; stop all writers and retry")
                report["source_sha256"] = before
                report["complete"] = True
                durable_json(stage / "upgrade_report.json", report)
                for path in stage.rglob("*"):
                    if path.is_file():
                        with path.open("rb") as stream:
                            os.fsync(stream.fileno())
                for directory in sorted((p for p in stage.rglob("*") if p.is_dir()), reverse=True):
                    fsync_directory(directory)
                fsync_directory(stage)
                if output_root.exists():
                    raise FileExistsError("upgrade output appeared during staging; publication refused")
                os.rename(stage, output_root)
                fsync_directory(output_root.parent)
            except BaseException as error:
                # Leave a clearly incomplete staging directory for inspection, never publish it.
                raise RuntimeError(f"upgrade not published; inspect staging directory {stage}: {error}") from error
            return {**report, "output_root": str(output_root)}
        finally:
            snapshot.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    report = upgrade(args.dataset_root, args.replay_root, args.output_root, dry_run=args.dry_run)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
