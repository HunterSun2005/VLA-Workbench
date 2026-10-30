"""Report operator-labelled stage completion and intervention rates without videos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(root: Path) -> dict:
    root = Path(root).expanduser().resolve()
    info = json.loads((root / "meta/info.json").read_text())
    stages = info.get("subtasks", [])
    if not stages:
        raise ValueError("dataset has no subtask annotations; set dataset.subtasks before collecting")
    rows = [pd.read_parquet(path) for path in sorted((root / "meta/episodes").glob("*/*.parquet"))]
    if not rows:
        raise ValueError("dataset has no saved episodes")
    episodes = pd.concat(rows, ignore_index=True)
    outcomes = dict(zip(episodes.episode_index, episodes.episode_success, strict=True))
    task_table = pd.read_parquet(root / "meta/tasks.parquet")
    task_by_index = {int(row.task_index): str(index) for index, row in task_table.iterrows()}
    episode_tasks: dict[int, str] = {}
    for _, row in episodes.iterrows():
        values = row.tasks.tolist() if hasattr(row.tasks, "tolist") else row.tasks
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, (list, tuple)) or len(values) != 1 or str(values[0]) not in task_table.index:
            raise ValueError(f"episode {row.episode_index} has an invalid task reference: {values!r}")
        episode_tasks[int(row.episode_index)] = str(values[0])
    tables = [pd.read_parquet(path) for path in sorted((root / "data").glob("*/*.parquet"))]
    frames = pd.concat(tables, ignore_index=True)
    result = {"episodes": len(outcomes), "successes": sum(value == "success" for value in outcomes.values()),
              "tasks": [],
              "completion_source": "operator Tab progression and final episode outcome", "stages": []}
    for task_index, task_name in task_by_index.items():
        selected = [episode for episode, name in episode_tasks.items() if name == task_name]
        successes = sum(outcomes[episode] == "success" for episode in selected)
        result["tasks"].append({"task_index": task_index, "task": task_name,
                                 "episodes": len(selected), "successes": successes,
                                 "success_rate": successes / len(selected) if selected else None})
    demo = info.get("collection_kind") == "human_demonstration"
    for index, name in enumerate(stages):
        entered = completed = count = human = 0
        for episode_id, episode in frames.groupby("episode_index"):
            labels = np.asarray(episode["complementary_info.subtask_index"].tolist()).reshape(-1)
            if not np.isin(labels, np.arange(len(stages))).all() or np.any(np.diff(labels) < 0):
                raise ValueError(f"invalid ordered subtask labels in episode {episode_id}")
            selected = episode.loc[labels == index]
            if selected.empty:
                continue
            entered += 1
            count += len(selected)
            completed += int(labels.max() > index or (index == len(stages)-1 and outcomes[episode_id] == "success"))
            if not demo:
                human += int((np.asarray(selected["complementary_info.is_intervention"].tolist()).reshape(-1) > .5).sum())
        result["stages"].append({"name": name, "entered_episodes": entered,
                                 "completed_episodes": completed,
                                 "completion_rate": completed / entered if entered else None,
                                 "frames": count,
                                 "intervention_frame_rate": human / count if count and not demo else None})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    print(json.dumps(summarize(parser.parse_args().dataset_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
