# 采集数据格式

当前 ROS 采集保存单臂 50D／双臂 100D 绝对实测状态、相机、任务、critical／阶段标注。joint action 为每臂 8D；TCP action 为每臂 10D；双臂按左、右拼接。状态字段与单位定义在 `src/vla/datacollection/franka_robotiq_single_left/schema.py`。

原生 action、独立实测锚点、实际提交关节命令、FK 辅助命令、时间与代际、诊断字段，以及派生数据限制，统一见 [ROS 动作契约](ROS_ACTION_CONTRACT.md)。磁盘不保存训练用相对动作；夹爪实测单位为米，action 与 anchor 中为 0～1 开度。

旧 29D／31D／58D 数据只读保留，不能续采为新格式。没有必要命令字段时不能可靠转换到另一动作空间；不得以实测关节或任意 IK 解补造标签。旧 `_31d` 采集入口现在也调用公共 ROS 会话，保存完整状态。
