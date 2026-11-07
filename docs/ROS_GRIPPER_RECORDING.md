# ROS 夹爪目标与记录

单／双臂、Web／终端采集共用 [RolloutCollector](../src/vla/datacollection/franka_robotiq_rollout/collect_ros.py) 与协调器。动作记录点遵循 [ROS 动作契约](ROS_ACTION_CONTRACT.md)，不再使用独立旧 recorder 的订阅路由作为采集定义。

## 数值与记录点

| 字段／对象 | 含义 |
| --- | --- |
| `observation.state` 中夹爪实测字段 | 测量开口，单位为米 |
| `observation.anchor` 中夹爪维度 | 原始测量转换后的 0～1 开度比例 |
| `action` 中夹爪维度 | 当前接受目标，0～1 绝对开度；不由后续测量漂移改写 |
| `command.joint` 中夹爪维度 | 与本次关节命令一起记录的提交目标 |

Gamepad 持续开／关时使用端点目标；松手时锁定当时实测开度作为 hold。GELLO 使用输入参考开度，策略使用候选动作的夹爪目标。共同规则是区分目标和实测值，不能用测量差分替代示教动作。

## 提交与故障

公共 `_coordinated_publish` 在目标改变时经 `gripper_action_topic` 提交 ActionClient 请求，并在 `command_gripper_topic` 发布记录目标；pose 控制权不驱动夹爪。夹爪开／闭位置换算由每臂配置决定，不假设所有设备方向相同。

提交请求不等于 ActionServer 已接受，也不保证物理到达。同步提交失败不更新成功目标缓存；异步拒绝／执行异常进入故障处理。预检要求对应 ActionServer 和必要消费者可用，不能用记录话题的订阅数代替执行服务就绪状态。

## 数据契约与验证

当前采集写入 recording schema 5、`gripper_recording_version=2` 和 `action_gripper_source=command_goal_or_latched_hold`，并绑定实际动作／控制来源。旧数据不能通过补写一个版本字段成为新格式；当前采集、训练与回放规则见 [执行闭环](EXECUTION_RTC_RLT.md)。

相关离线用例位于 [夹爪命令测试](../tests/rlt/test_ros_gripper_commands.py) 和 [模拟采集恢复测试](../tests/rlt/test_dual_gamepad_record.py)。当前通过情况由 [自动化验收](ACCEPTANCE.md) 生成；真实 ActionClient 接受／抢占、夹爪到达和连续采集仍需现场验证。

旧话题拆分、旧 recorder 和旧格式识别的背景保存在 [2026-11-06 历史记录](archive/ROS_GRIPPER_RECORDING_2026-11-06.md)。
