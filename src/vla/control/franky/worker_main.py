"""Command line entry point for the isolated Franky worker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal

from .worker import FrankyWorker, FrankyWorkerServer, check_binding


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--socket-path", required=True, type=Path)
    args = parser.parse_args()
    check_binding()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    worker = FrankyWorker(
        config["arms"],
        session_id=str(config["session_id"]),
        compliance=config.get("compliance"),
    )
    server = FrankyWorkerServer(worker, args.socket_path)
    # Managed ROS shutdown sends SIGTERM to child process groups. Convert it
    # into the same orderly tracker stop and lease release as Ctrl-C.
    signal.signal(signal.SIGTERM, lambda _signum, _frame: server.close())
    signal.signal(signal.SIGINT, lambda _signum, _frame: server.close())
    try:
        server.serve()
    finally:
        server.close()


if __name__ == "__main__":
    main()
