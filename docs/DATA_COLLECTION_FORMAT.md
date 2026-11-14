# 采集数据格式

新增时隙、有效性、传感器时间、执行审计和 staging 格式见 [统一执行闭环](EXECUTION_RTC_RLT.md)。

当前 ROS 采集保存单臂 50D／双臂 100D 绝对实测状态、相机、任务、critical／阶段标注。joint action 为每臂 8D；TCP action 为每臂 10D；双臂按左、右拼接。状态字段与单位定义在 `src/vla/datacollection/franka_robotiq_single_left/schema.py`。

原生 action、独立实测锚点、实际提交关节命令、FK 辅助命令、时间与代际、诊断字段，以及派生数据限制，统一见 [ROS 动作契约](ROS_ACTION_CONTRACT.md)。磁盘不保存训练用相对动作；夹爪实测单位为米，action 与 anchor 中为 0～1 开度。

当前只接受 schema 5；旧数据保留作历史资料，不能接入新采集／训练链路。没有必要命令字段时不能可靠转换到另一动作空间；不得以实测关节或任意 IK 解补造标签。旧 `_31d` 采集入口现在也调用公共 ROS 会话，保存完整状态。

### 30 Hz 提交确认与缺口

新会话仅采集 Franky 绝对 TCP。每次提交冻结该 slot 的观察和接受目标，匹配 worker ACK 后才记录；`action` 是接受目标，实际 tracker 参考／应用时间在 `target_applied` 审计中。未知、失败或撤销的目标不能生成有效训练标签。IPC 等待或没有新相机曝光时跳过整行，不补帧。保留真实 slot、时间戳、segment；SFT 窗口、RLT 奖励区间和 Replay 均不得跨缺口（包括最后动作到终点观察）。结束无新曝光时不复制旧图；schema 保持 5。

工作空间开关实际值记录为 `workspace_limits_enabled` 以及 `execution_projection.workspace_limits_enabled`，参与控制指纹；未声明时按 true 解释。关闭只跳过 TCP XYZ 范围约束，保留有限数值、姿态与夹爪校验及有限范围元数据。训练投影跟随同一模式。续采沿用 schema 5，必须保持模式、范围、字段及控制契约一致，只追加新回合；禁止改写旧元数据绕过检查。

### 逻辑 TCP 与硬件安装朝向

Franky implementation 6 使用 `flange_aligned_tcp_v1`：TCP 原点仍为实测物理 TCP，逻辑坐标轴与法兰平行。`action`、`observation.state` 中 TCP 朝向、`command.target_tcp` 和 `diagnostic.actual_tcp` 使用相同逻辑约定；worker 根据锁定的实测 `F_T_EE` 旋转转换硬件目标并逆变换反馈。朝下时逻辑 yaw 为 0，−45° 安装偏置只出现在硬件目标层。`tool_definition` 保留逻辑工具定义、坐标标识及 `hardware_flange_to_tcp_pose`；worker 状态另含 `hardware_tcp_pose`、`hardware_reference_tcp_pose`，执行审计另记 `hardware_target_applied`。控制指纹与工具指纹随语义改变，既有数据、统计或 checkpoint 不得通过改写元数据混用。姿态库保存新逻辑框架标识，旧记录保留但不自动转换。

## Franky 速度 backend

官方 Franky 2.0.0/libfranka 0.21.2 控制使用 `franky_cartesian_velocity` 身份，recording schema 仍为 5。动作仍是接受的绝对逻辑 TCP 目标；100 Hz 派生 Twist、实测及实际 SDK 提交时间以 `velocity_servo` 审计关联 generation、sequence、slot。历史 `diagnostic.tracker` 列在此 backend 表示速度伺服就绪。控制指纹绑定 worker 实现、SDK/库哈希和速度参数，拒绝与旧 Tracker 数据续采或混合冒充同一控制链。历史文件不转换。
