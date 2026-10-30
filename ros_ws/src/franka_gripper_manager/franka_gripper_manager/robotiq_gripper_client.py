from control_msgs.action import GripperCommand
import rclpy
from rclpy.action import ActionClient
from rclpy.node import Node
from std_msgs.msg import Float32

from franka_gripper_manager.robotiq_command import (
    snap_width_percent_to_endpoints,
    width_percent_to_joint_position,
)

DEFAULT_GRIPPER_COMMAND_TOPIC = "gripper_client/target_gripper_width_percent"
DEFAULT_MOVE_ACTION_TOPIC = "robotiq_gripper_controller/gripper_cmd"
DEFAULT_OPEN_POSITION = 0.0
DEFAULT_CLOSED_POSITION = 0.8


class RobotiqGripperClient(Node):
    def __init__(self):
        super().__init__("robotiq_gripper_client")

        self.declare_parameter("gripper_command_topic", DEFAULT_GRIPPER_COMMAND_TOPIC)
        self.declare_parameter("move_action_topic", DEFAULT_MOVE_ACTION_TOPIC)
        self.declare_parameter("open_position", DEFAULT_OPEN_POSITION)
        self.declare_parameter("closed_position", DEFAULT_CLOSED_POSITION)
        self.declare_parameter("max_effort", 50.0)
        self.declare_parameter("command_deadband", 0.02)
        self.declare_parameter("endpoint_snap_threshold", 0.25)
        self.declare_parameter("action_server_timeout_sec", 90.0)

        gripper_command_topic = self.get_parameter("gripper_command_topic").value
        move_action_topic = self.get_parameter("move_action_topic").value
        self.open_position = float(self.get_parameter("open_position").value)
        self.closed_position = float(self.get_parameter("closed_position").value)
        self.max_effort = float(self.get_parameter("max_effort").value)
        self.command_deadband = float(self.get_parameter("command_deadband").value)
        self.endpoint_snap_threshold = float(
            self.get_parameter("endpoint_snap_threshold").value
        )
        action_server_timeout_sec = float(
            self.get_parameter("action_server_timeout_sec").value
        )

        if not 0.0 <= self.command_deadband < 0.5:
            raise ValueError("command_deadband must be in the range [0.0, 0.5)")
        if not 0.0 <= self.endpoint_snap_threshold < 0.5:
            raise ValueError("endpoint_snap_threshold must be in the range [0.0, 0.5)")
        if self.max_effort < 0.0:
            raise ValueError("max_effort must be non-negative")

        self.get_logger().info("Starting Robotiq Gripper Client")
        self.gripper_state_sub = self.create_subscription(
            Float32,
            gripper_command_topic,
            self.gripper_state_callback,
            10,
        )
        self.action_client = ActionClient(self, GripperCommand, move_action_topic)
        if not self.action_client.wait_for_server(timeout_sec=action_server_timeout_sec):
            raise RuntimeError(
                f"Robotiq action server '{move_action_topic}' was not available within "
                f"{action_server_timeout_sec:.1f}s"
            )
        self.last_width = None
        self._goal_futures = set()
        self._minimum_input_width = float("inf")
        self._reported_minimum_input_width = float("inf")
        self._width_diagnostic_timer = self.create_timer(
            1.0, self.report_width_input_minimum
        )
        self.get_logger().info(
            "Gripper action server is ready "
            f"(open={self.open_position:.3f}, closed={self.closed_position:.3f}, "
            f"deadband={self.command_deadband:.3f}, "
            f"endpoint_snap={self.endpoint_snap_threshold:.3f})"
        )

    def gripper_state_callback(self, msg: Float32) -> None:
        raw_width = float(msg.data)
        self._minimum_input_width = min(self._minimum_input_width, raw_width)
        gripper_width = snap_width_percent_to_endpoints(
            raw_width, self.endpoint_snap_threshold
        )
        if (
            self.last_width is not None
            and abs(gripper_width - self.last_width) < self.command_deadband
        ):
            return

        if gripper_width != raw_width:
            self.get_logger().info(
                f"Snapped gripper width command {raw_width:.3f} to {gripper_width:.3f}"
            )
        self.send_gripper_command(gripper_width)

    def report_width_input_minimum(self) -> None:
        if self._minimum_input_width >= self._reported_minimum_input_width:
            return
        self._reported_minimum_input_width = self._minimum_input_width
        self.get_logger().info(
            "Minimum GELLO width input observed: "
            f"{self._minimum_input_width:.3f} "
            f"(close snap threshold={self.endpoint_snap_threshold:.3f})"
        )

    def send_gripper_command(self, gripper_width: float) -> None:
        self.last_width = gripper_width
        goal_msg = GripperCommand.Goal()
        goal_msg.command.position = width_percent_to_joint_position(
            gripper_width, self.open_position, self.closed_position
        )
        goal_msg.command.max_effort = self.max_effort
        future = self.action_client.send_goal_async(goal_msg)
        self._goal_futures.add(future)
        future.add_done_callback(self.gripper_response_callback)

    def gripper_response_callback(self, future: rclpy.task.Future) -> None:
        self._goal_futures.discard(future)
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Failed to send Robotiq goal: {exc}")
            self.last_width = None
            return

        if not goal_handle.accepted:
            self.get_logger().error(f"Robotiq goal rejected (status={goal_handle.status})")
            self.last_width = None
            return

        result_future = goal_handle.get_result_async()
        self._goal_futures.add(result_future)
        result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future: rclpy.task.Future) -> None:
        self._goal_futures.discard(future)
        try:
            wrapped_result = future.result()
        except Exception as exc:
            self.get_logger().error(f"Robotiq action failed: {exc}")
            return

        self.get_logger().debug(f"Robotiq action finished (status={wrapped_result.status})")


def main(args=None):
    rclpy.init(args=args)
    gripper_client = None
    try:
        gripper_client = RobotiqGripperClient()
        rclpy.spin(gripper_client)
    except KeyboardInterrupt:
        pass
    finally:
        if gripper_client is not None:
            gripper_client.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
