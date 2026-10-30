"""Run LeRobot's native trainer with VLA policies registered."""

from __future__ import annotations


def main() -> None:
    """Delegate argument parsing and training to the pinned LeRobot trainer."""
    from vla.adapters.lerobot import register

    register()

    from lerobot.scripts.lerobot_train import main as lerobot_train_main

    lerobot_train_main()


if __name__ == "__main__":
    main()
