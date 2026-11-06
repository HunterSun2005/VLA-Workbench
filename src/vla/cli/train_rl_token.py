"""Run LeRobot's native trainer with VLA policies registered."""

from __future__ import annotations


def main() -> None:
    """Delegate argument parsing and training to the pinned LeRobot trainer."""
    from vla.adapters.lerobot import register

    register()

    from lerobot.scripts import lerobot_train
    from vla.adapters.lerobot.action_training import install_action_training
    install_action_training(lerobot_train)
    lerobot_train_main = lerobot_train.main

    import sys
    if not any(arg.startswith("--dataset.eval_split=") for arg in sys.argv):
        sys.argv.append("--dataset.eval_split=0")
    lerobot_train_main()


if __name__ == "__main__":
    main()
