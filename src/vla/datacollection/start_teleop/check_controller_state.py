#!/usr/bin/env python3
"""Wait for a ros2_control controller using the structured ListControllers service."""

from __future__ import annotations

import argparse
import time
from collections.abc import Iterable


def controller_is_active(controllers: Iterable[object], target: str) -> bool:
    return any(
        str(getattr(controller, "name", "")).strip("/").split("/")[-1] == target
        and str(getattr(controller, "state", "")).strip().lower() == "active"
        for controller in controllers
    )


def format_controllers(controllers: Iterable[object]) -> str:
    rows = []
    for controller in controllers:
        rows.append(
            " ".join(
                (
                    str(getattr(controller, "name", "<unnamed>")),
                    str(getattr(controller, "type", "<unknown-type>")),
                    str(getattr(controller, "state", "<unknown-state>")),
                )
            )
        )
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manager", required=True)
    parser.add_argument("--controller", required=True)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    return parser.parse_args()


def wait_for_controller(manager: str, controller_name: str, timeout_s: float) -> int:
    if timeout_s <= 0:
        raise ValueError("timeout-s must be positive")

    import rclpy
    from controller_manager_msgs.srv import ListControllers

    rclpy.init()
    node = rclpy.create_node("vla_controller_state_check")
    service_name = f"{manager.rstrip('/')}/list_controllers"
    client = node.create_client(ListControllers, service_name)
    deadline = time.monotonic() + timeout_s
    latest = []
    try:
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if not client.wait_for_service(timeout_sec=min(1.0, remaining)):
                continue
            future = client.call_async(ListControllers.Request())
            rclpy.spin_until_future_complete(node, future, timeout_sec=min(2.0, remaining))
            if future.done() and future.exception() is None:
                response = future.result()
                latest = list(response.controller)
                if controller_is_active(latest, controller_name):
                    print(format_controllers(latest))
                    return 0
            time.sleep(0.2)
        print(format_controllers(latest))
        return 1
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


def main() -> None:
    args = parse_args()
    raise SystemExit(wait_for_controller(args.manager, args.controller, args.timeout_s))


if __name__ == "__main__":
    main()
