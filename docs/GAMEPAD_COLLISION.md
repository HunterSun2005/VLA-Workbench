# ROS 碰撞阈值与配置边界

Gamepad、GELLO 和策略控制共用 ROS 外设层及 Franky TCP 控制链路。Franky worker 在连接时显式设置 collision behavior；旧配置中的 `robot.collision_behavior_profile` 仍不是当前入口。

旧 ROS backend 的启动器 [start_teleop.sh](../src/vla/datacollection/start_teleop/start_teleop.sh) 在 `SET_FULL_COLLISION_BEHAVIOR` 启用时，对各活动机械臂调用 `set_full_collision_behavior`。Franky backend 跳过 ROS arm controller，改由 worker 在 FCI 连接后设置阈值；任一路线调用超时或失败都会中止启动。

参数分为关节力矩／笛卡尔 wrench、加速／nominal、上／下阈值。具体数值维护在上述代码中，此处不复制一份可能漂移的表格。没有启用设置时，不能仅凭仓库默认值推断机器实际生效的阈值；刷新网页也不会更新运行会话的参数。

碰撞阈值、关节／力矩限速、阻抗参数和工具负载含义不同。调整阈值不提高额定载荷，也不代替工具／负载标定。当前控制边界与故障行为见 [ROS 动作契约](ROS_ACTION_CONTRACT.md)，现场接触稳定性须单独验收，开发测试不连接硬件。

旧 Franky 路线的背景和当时验证保留在 [2026-11-04 历史记录](archive/GAMEPAD_COLLISION_2026-11-04.md)，不能将其中的 YAML 示例直接用于当前 ROS 配置。
