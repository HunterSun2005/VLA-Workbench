from __future__ import annotations

import argparse
import logging
from pathlib import Path

from vla.diagnostics.pi05_sft import run_diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare Franka PI0.5 SFT checkpoints on a fixed seen or holdout subset."
    )
    parser.add_argument("--dataset-repo-id", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, default=Path("models/paligemma"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--episode-count", type=int, default=15)
    parser.add_argument("--anchors-per-episode", type=int, default=32)
    parser.add_argument("--flow-repeats", type=int, default=4)
    parser.add_argument("--inference-repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--video-backend", default="torchcodec")
    parser.add_argument("--dataset-role", choices=("seen", "val"), default="seen")
    parser.add_argument(
        "--split-manifest",
        type=Path,
        default=None,
        help="Required for dataset-role=val; proves the checkpoint train/validation split is disjoint.",
    )
    parser.add_argument("--transition-anchors-per-episode", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    report = run_diagnostics(
        dataset_repo_id=args.dataset_repo_id,
        dataset_root=args.dataset_root,
        checkpoints=args.checkpoint,
        manifest_path=args.manifest,
        output_path=args.output,
        tokenizer_path=args.tokenizer_path,
        device=args.device,
        batch_size=args.batch_size,
        episode_count=args.episode_count,
        anchors_per_episode=args.anchors_per_episode,
        flow_repeats=args.flow_repeats,
        inference_repeats=args.inference_repeats,
        seed=args.seed,
        video_backend=args.video_backend,
        dataset_role=args.dataset_role,
        split_manifest_path=args.split_manifest,
        transition_anchors_per_episode=args.transition_anchors_per_episode,
    )
    selection = report.get("selection") or {}
    print(f"Diagnostic report: {args.output.resolve()}")
    print(f"Conclusion: {selection.get('conclusion', 'unavailable')}")
    print(f"Recommended checkpoint: {selection.get('recommended_checkpoint')}")


if __name__ == "__main__":
    main()
