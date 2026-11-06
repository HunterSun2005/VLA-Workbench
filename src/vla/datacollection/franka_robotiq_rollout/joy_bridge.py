"""Forward one ROS2 sensor_msgs/Joy topic to the rollout input process."""

from __future__ import annotations

import argparse
import socket
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Joy

from .joy_transport import JoyPacket, encode_joy_packet


class JoyBridge(Node):
    def __init__(self, topic: str, socket_path: str, node_name: str = "vla_rollout_joy_bridge") -> None:
        super().__init__(node_name)
        self.socket_path = socket_path
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        # Policy control must not be able to block this ROS callback and make
        # all subsequently produced Joy samples stale.
        self.socket.setblocking(False)
        self.sequence = 0
        self.last_warning = 0.0
        self.create_subscription(Joy, topic, self._callback, 10)
        self.get_logger().info(f"forwarding {topic} to {socket_path}")

    def _callback(self, message: Joy) -> None:
        packet = JoyPacket(
            sequence=self.sequence,
            monotonic_ns=time.monotonic_ns(),
            axes=tuple(float(item) for item in message.axes),
            buttons=tuple(int(item) for item in message.buttons),
        )
        self.sequence += 1
        try:
            self.socket.sendto(encode_joy_packet(packet), self.socket_path)
        except OSError as error:
            now = time.monotonic()
            if now - self.last_warning >= 2.0:
                self.get_logger().warning(f"collector Joy socket unavailable: {error}")
                self.last_warning = now

    def destroy_node(self) -> bool:
        self.socket.close()
        return super().destroy_node()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--topic", default="/left/gamepad/joy")
    parser.add_argument("--node-name", default="vla_rollout_joy_bridge")
    parser.add_argument(
        "--socket-path",
        default="/tmp/vla_franka_single_left_joy.sock",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rclpy.init()
    bridge = JoyBridge(args.topic, args.socket_path, args.node_name)
    try:
        rclpy.spin(bridge)
    finally:
        bridge.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
