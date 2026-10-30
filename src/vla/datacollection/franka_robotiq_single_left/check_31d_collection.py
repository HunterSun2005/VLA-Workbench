"""Read-only preflight for the independent 31D Franky collection setup."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "datacollection/franka_robotiq_rollout/configs/single_left_31d.yaml"
        ),
    )
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    root = (config_path.parents[3] / config["dataset"]["root"]).resolve()
    print(f"config={config_path}")
    print(f"backend={config['backend']}")
    print(f"dataset_root={root}")
    print(f"dataset_root_exists={root.exists()}")
    print(f"dataset_root_is_empty={not any(root.iterdir()) if root.exists() else True}")
    print("required_state=10D+tau_J[7]+tau_ext_hat_filtered[7]+dq[7]")
    print("required_shape=(31,)")
    print("hardware_connection=not_attempted")
    print("WARNING: existing collect_franky.py still writes the legacy schema;")
    print("use only a 31D-aware collector after its offline validation.")


if __name__ == "__main__":
    main()
