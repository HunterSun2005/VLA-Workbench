"""Dependency-light wire protocol shared by ROS2 and VLA runtimes."""

from .protocol import PROTOCOL_VERSION, recv_packet, send_packet

__all__ = ["PROTOCOL_VERSION", "recv_packet", "send_packet"]
