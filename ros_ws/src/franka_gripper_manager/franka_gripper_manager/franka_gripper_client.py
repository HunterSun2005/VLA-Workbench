import rclpy
import time
from rclpy.node import Node
from franka_msgs.action import Move, Homing, Grasp  # 引入 Grasp Action
from sensor_msgs.msg import JointState
from rclpy.action import ActionClient
from std_msgs.msg import Float32

DEFAULT_MOVE_ACTION_TOPIC = "franka_gripper/move"
DEFAULT_GRASP_ACTION_TOPIC = "franka_gripper/grasp" # 新增 Grasp Topic
DEFAULT_HOMING_ACTION_TOPIC = "franka_gripper/homing"
DEFAULT_JOINT_STATES_TOPIC = "franka_gripper/joint_states"
DEFAULT_GRIPPER_COMMAND_TOPIC = "gripper/gripper_client/target_gripper_width_percent"


class GripperClient(Node):
    def __init__(self):
        super().__init__("gripper_client")

        self.declare_parameter("move_action_topic", DEFAULT_MOVE_ACTION_TOPIC)
        self.declare_parameter("grasp_action_topic", DEFAULT_GRASP_ACTION_TOPIC)
        self.declare_parameter("homing_action_topic", DEFAULT_HOMING_ACTION_TOPIC)
        self.declare_parameter("gripper_command_topic", DEFAULT_GRIPPER_COMMAND_TOPIC)
        self.declare_parameter("joint_states_topic", DEFAULT_JOINT_STATES_TOPIC)
        self.declare_parameter("homing_max_attempts", 5)
        self.declare_parameter("homing_retry_delay_sec", 3.0)

        move_action_topic = self.get_parameter("move_action_topic").get_parameter_value().string_value
        grasp_action_topic = self.get_parameter("grasp_action_topic").get_parameter_value().string_value
        homing_action_topic = self.get_parameter("homing_action_topic").get_parameter_value().string_value
        gripper_command_topic = self.get_parameter("gripper_command_topic").get_parameter_value().string_value
        joint_states_topic = self.get_parameter("joint_states_topic").get_parameter_value().string_value
        self._homing_max_attempts = self.get_parameter("homing_max_attempts").value
        self._homing_retry_delay_sec = self.get_parameter("homing_retry_delay_sec").value

        self._ACTION_SERVER_TIMEOUT = 10.0
        self._MIN_GRIPPER_WIDTH_PERCENT = 0.0
        self._MAX_GRIPPER_WIDTH_PERCENT = 1.0

        # --- 新增的抓取逻辑参数 ---
        # 假设 0.0 是完全闭合，1.0 是完全张开
        self._grasp_trigger_percent = 0.3   # GELLO 闭合到 30% 以下时，触发抓取
        self._release_trigger_percent = 0.6 # GELLO 松开到 60% 以上时，触发张开
        self._grasp_force = 100.0            # 设定目标抓取力 (N)
        self._is_grasping = False           # 状态机：当前是否处于抓取状态
        self._action_busy = False           # 状态机：是否正在等待 Action Server 返回
        # -------------------------

        self._max_width = 0.0

        self.get_logger().info("Initializing gripper client...")
        self._home_gripper(homing_action_topic)
        self._get_max_gripper_width(joint_states_topic)

        self.get_logger().info("Subscribing to gripper commands...")
        self._gripper_command_subscription = self.create_subscription(
            Float32, gripper_command_topic, self._gripper_command_callback, 10
        )

        # 创建 Move 和 Grasp 两个 Client
        self._move_client = ActionClient(self, Move, move_action_topic)
        self._grasp_client = ActionClient(self, Grasp, grasp_action_topic)

        self.get_logger().info("Waiting for gripper action servers...")
        if not self._move_client.wait_for_server(timeout_sec=self._ACTION_SERVER_TIMEOUT):
            raise RuntimeError("Move action server not available!")
        if not self._grasp_client.wait_for_server(timeout_sec=self._ACTION_SERVER_TIMEOUT):
            raise RuntimeError("Grasp action server not available!")

        self.get_logger().info("Gripper client initialized!")

    def _home_gripper(self, homing_action_topic: str) -> None:
        homing_client = ActionClient(self, Homing, homing_action_topic)
        if not homing_client.wait_for_server(timeout_sec=self._ACTION_SERVER_TIMEOUT):
            raise RuntimeError("Homing action server not available!")

        for attempt in range(1, self._homing_max_attempts + 1):
            self.get_logger().info(
                f"Starting gripper homing (attempt {attempt}/{self._homing_max_attempts})..."
            )
            goal_msg = Homing.Goal()
            future = homing_client.send_goal_async(goal_msg)
            rclpy.spin_until_future_complete(self, future)
            goal_handle = future.result()
            if goal_handle is not None and goal_handle.accepted:
                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future)
                result = result_future.result()
                if result is not None and result.result.success:
                    time.sleep(2)
                    self.get_logger().info("Gripper homing successful!")
                    return

            if attempt < self._homing_max_attempts:
                self.get_logger().warning(
                    f"Gripper homing failed; retrying in {self._homing_retry_delay_sec:.1f}s."
                )
                time.sleep(self._homing_retry_delay_sec)

        raise RuntimeError(
            f"Gripper homing failed after {self._homing_max_attempts} attempts!"
        )

    def _get_max_gripper_width(self, joint_states_topic: str) -> None:
        # [代码保持不变，复用你原来的 _get_max_gripper_width 逻辑]
        self.get_logger().info("Readout maximum gripper width...")
        future = rclpy.task.Future()
        def joint_state_callback(msg):
            _INDEX_FINGER_LEFT = 0
            self._max_width = 2 * msg.position[_INDEX_FINGER_LEFT]
            self.get_logger().info(f"Maximum gripper width determined: {self._max_width}")
            future.set_result(True)
        gripper_subscription = self.create_subscription(JointState, joint_states_topic, joint_state_callback, 10)
        rclpy.spin_until_future_complete(self, future)
        self.destroy_subscription(gripper_subscription)

    def _gripper_command_callback(self, msg: Float32) -> None:
        if self._action_busy:
            return  # 如果当前正在执行动作，忽略新的输入，防止指令堆叠

        current_percent = msg.data

        try:
            # 逻辑 1：触发抓取 (Grasp)
            if current_percent <= self._grasp_trigger_percent and not self._is_grasping:
                self.get_logger().info(f"Triggering GRASP (percent: {current_percent:.2f})")
                self._send_grasp_command()
                self._is_grasping = True
                self._action_busy = True

            # 逻辑 2：触发松开 (Move)
            elif current_percent >= self._release_trigger_percent and self._is_grasping:
                self.get_logger().info(f"Triggering RELEASE (percent: {current_percent:.2f})")
                self._send_move_command()
                self._is_grasping = False
                self._action_busy = True
        except Exception as exc:
            self.get_logger().error(f"Failed to send gripper command: {exc}")
            self._action_busy = False

    def _send_grasp_command(self) -> None:
        goal_msg = Grasp.Goal()
        goal_msg.width = 0.0             # 目标宽度设为0，让它一直夹紧
        goal_msg.speed = 0.1             # 抓取速度
        goal_msg.force = float(self._grasp_force)

        # 宽容差设定：这是防止卸力的关键！
        goal_msg.epsilon.inner = 0.08
        goal_msg.epsilon.outer = 0.08

        self._future = self._grasp_client.send_goal_async(goal_msg)
        self._future.add_done_callback(self._gripper_response_callback)

    def _send_move_command(self) -> None:
        goal_msg = Move.Goal()
        goal_msg.width = self._max_width  # 松开到最大宽度
        goal_msg.speed = 0.5              # 松开可以稍微快一点

        self._future = self._move_client.send_goal_async(goal_msg)
        self._future.add_done_callback(self._gripper_response_callback)

    def _gripper_response_callback(self, future: rclpy.task.Future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"Gripper goal request failed: {exc}")
            self._action_busy = False
            return

        if not goal_handle.accepted:
            self.get_logger().error("Goal rejected!")
            self._action_busy = False
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self._get_result_callback)

    def _get_result_callback(self, future: rclpy.task.Future) -> None:
        try:
            result = future.result().result
        except Exception as exc:
            self.get_logger().error(f"Gripper action result failed: {exc}")
            self._action_busy = False
            return

        self.get_logger().info(f"Action finished with success: {result.success}")
        self._action_busy = False  # 动作执行完毕，释放锁


def main(args=None):
    rclpy.init(args=args)
    gripper_client = GripperClient()
    rclpy.spin(gripper_client)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
