from __future__ import annotations

import argparse
import json
from pathlib import Path

from vla.cli.common import configure_logging
from vla.diagnostics.franka_sft_split import (
    load_split_manifest,
    prepare_franka_sft_split,
    training_schedule,
)


log = configure_logging(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit and materialize deterministic Franka PI0.5 SFT train/validation splits."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-repo-id", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--validation-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--candidate-count", type=int, default=10_000)
    parser.add_argument(
        "--skip-video-verification",
        action="store_true",
        help="Skip ffprobe integrity checks. Intended only for synthetic tests.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest_path = prepare_franka_sft_split(
        args.source_root,
        args.output_root,
        source_repo_id=args.source_repo_id,
        validation_episodes=args.validation_episodes,
        seed=args.seed,
        candidate_count=args.candidate_count,
        verify_videos=not args.skip_video_verification,
    )
    manifest = load_split_manifest(manifest_path)
    schedule = training_schedule(manifest)
    log.info("SFT split manifest: %s", manifest_path)
    print(json.dumps(schedule, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
