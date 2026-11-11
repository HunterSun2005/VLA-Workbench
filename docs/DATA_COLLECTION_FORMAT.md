# 采集数据格式

新增时隙、有效性、传感器时间、执行审计和 staging 格式见 [统一执行闭环](EXECUTION_RTC_RLT.md)。

当前 ROS 采集保存单臂 50D／双臂 100D 绝对实测状态、相机、任务、critical／阶段标注。joint action 为每臂 8D；TCP action 为每臂 10D；双臂按左、右拼接。状态字段与单位定义在 `src/vla/datacollection/franka_robotiq_single_left/schema.py`。

原生 action、独立实测锚点、实际提交关节命令、FK 辅助命令、时间与代际、诊断字段，以及派生数据限制，统一见 [ROS 动作契约](ROS_ACTION_CONTRACT.md)。磁盘不保存训练用相对动作；夹爪实测单位为米，action 与 anchor 中为 0～1 开度。

当前只接受 schema 5；旧数据保留作历史资料，不能接入新采集／训练链路。没有必要命令字段时不能可靠转换到另一动作空间；不得以实测关节或任意 IK 解补造标签。旧 `_31d` 采集入口现在也调用公共 ROS 会话，保存完整状态。

### 30 Hz 提交确认与缺口

新会话仅采集 Franky 绝对 TCP。每次提交冻结该 slot 的观察和接受目标，匹配 worker ACK 后才记录；`action` 是接受目标，实际 tracker 参考／应用时间在 `target_applied` 审计中。未知、失败或撤销的目标不能生成有效训练标签。IPC 等待或没有新相机曝光时跳过整行，不补帧。保留真实 slot、时间戳、segment；SFT 窗口、RLT 奖励区间和 Replay 均不得跨缺口（包括最后动作到终点观察）。结束无新曝光时不复制旧图；schema 保持 5。
