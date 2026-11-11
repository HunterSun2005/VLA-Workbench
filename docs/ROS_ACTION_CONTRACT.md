# ROS 控制与动作契约

本仓库的统一采集／部署路线使用独立 Franky worker 和 `CartesianImpedanceTracker`。ROS 2 Humble 继续负责 GELLO、Gamepad、夹爪、相机和采集器；worker 独占 Franka FCI 控制连接。旧 ROS 关节阻抗路线保留为隔离的 legacy backend，不能与 Franky 数据混采。

## 数据与训练

| 空间 | 单臂 | 双臂（左、右） | 磁盘 action |
|---|---:|---:|---|
| joint | 7 关节角 + 夹爪，8D | 16D | 公共关节入口接受、裁剪后的绝对目标 |
| tcp | XYZ + rot6d + 夹爪，10D | 20D | 公共 TCP 入口接受、裁剪后，平滑和 IK 前的绝对目标 |

所有夹爪动作都是 0～1 绝对开度。完整实测状态仍为每臂 50D，夹爪实测字段单位为米。`observation.anchor` 独立保存原始实测动作空间状态，夹爪在锚点中转换为开度比例，不受模型输入字段筛选影响。

训练配置通过 `action_contract` 选择 `absolute` 或 `chunk_relative`。相对化针对同一 chunk 起始实测状态：

- 关节：`dq = q_target - q_measured_start`。
- TCP：`dp = p_target - p_measured_start`；`dR = R_target @ R_measured_start.T`。
- 逆变换：`p_target = p_start + dp`，`R_target = dR @ R_start`。rot6d 使用旋转矩阵的前两列，不能逐元素相减；夹爪不相对化。

转换发生在归一化之前。统计量按实际 chunk 长度、目标帧间隔和有效帧计算，排除回合外 padding；绑定动作空间、表示、锚点、旋转约定和数据内容指纹。模型适配器屏蔽 padding 和无效动作的训练损失，不修改已安装的 LeRobot。训练集应先物化再计算统计量，契约训练拒绝非零 `dataset.eval_split`，防止统计量绑定后再隐式改变训练样本集合。

统计计算逐回合读取必要的 Parquet 字段，重叠 chunk 分批转换并按维度写入临时文件，随后逐维计算精确分位数。重叠采样的权重、chunk 起始锚点和 padding 排除规则保持不变。临时目录可在启动 CLI／Web 前通过 `TMPDIR` 指定；内存、磁盘开销与实测结果见 [统计内存优化](ACTION_STATISTICS_MEMORY.md)。

## 控制与记录

Franky 路线只接受绝对 TCP 目标。GELLO 通过 FK、Gamepad 和策略都更新同一个 TCP 参考位姿，worker 使用 `CartesianImpedanceTracker.set_target()` 持续跟踪；Gamepad 松杆后保持最后参考位姿。旧 ROS backend 才使用 CartesianServo 和关节目标。

Franky 默认阻抗参数为平移刚度 1000 N/m、旋转刚度 50 Nm/rad、平移误差裁剪 0.008 m、旋转误差裁剪 0.04 rad、单次目标步长 0.03 m/0.10 rad。Web 支持启动前覆盖初值和运行中调参。参数草稿与 worker 确认值分离，仅实际变化的裁剪、力矩变化率和增益时间常数会停止并重建 tracker；刚度调用 `set_gains`，不变参数不重启控制。刚度上限为 1200 N/m、80 Nm/rad，裁剪下限为 0.005 m、0.02 rad。

学习目标最多按 30 Hz 接受，上层跟踪按 100 Hz 更新；Franky 由 tracker 执行 FCI 控制，旧 ROS backend 的 C++ 控制器保持 1000 Hz。Normal/Fast、工作空间、关节速度及力矩变化率保护沿用 ROS 基线。停止、接管、回合边界和故障通过代际撤销命令，重置发生在实时循环中；所有机械臂确认新代际后才允许新目标。确认超时保持故障。诊断快照使用序列号避免读到混合代际的状态。

以下 CartesianServo／关节前瞻阈值仅用于旧 ROS backend。CartesianServo 从复位后的实测关节初始化，随后从上一周期的关节目标积分 IK，使用真实跟踪周期（包括 joint 空间的 Gamepad IK）。平移／旋转误差和速度上限均定义在 TCP，KDL Jacobian 的参考点由法兰平移到 TCP，避免工具俯仰所需的法兰补偿被 TCP 线速度上限截断。阻尼 IK 保留原有参数，仍存在小量近似误差；关节速度、位置及跟随误差限幅统一缩放整个关节增量，不逐关节裁剪而改变运动方向。

轨迹目标的前瞻以已有命令位姿为基准，实测位姿的跟随误差约束作用于最终提交的轨迹，避免两次限制叠加跟踪延迟、压低可实现的匀速运动。TCP 平移／旋转软限幅仍为对应最大速度乘 `velocity_tracking_tau_s`，关节软限幅仍为 `max_joint_speed × velocity_tracking_tau_s`。若实测值变化使旧目标稍超软限幅，保持旧目标或向范围内移动，不允许继续增加该误差；通过统一缩放和有界二分检查防止目标累积或瞬间跳回测量位置。

关节硬故障阈值在软限幅外保留一个动作周期的移动量 `max_joint_speed / fps`：默认软限幅 0.045 rad，硬阈值 0.055 rad。该容差不增加主动发送目标的前瞻范围，用于避免限幅边缘的微小测量波动直接触发故障。超出硬阈值仍撤销控制权并要求手动恢复，错误报告具体关节、实测值、目标和阈值；代际切换清空积分状态。跟踪实现标识 `tcp_reference_coordinated_limits_v2` 和夹爪绝对目标速度一起绑定控制指纹，不改变动作／记录协议版本，不允许把旧控制契约伪装成新契约。

待命不发送无主关节命令，C++ 控制器保持激活／复位姿态。GELLO 在控制权已确认且所有臂输入有效后恢复 `policy_gripper_speed`，再提交绝对夹爪目标；暂停、故障保持零速度，动作拒绝／执行失败仍触发故障。

Gamepad 默认使用 SDL 标准布局，姿态增量为 `Exp(R_measured · ω_tool · dt) · R_target`，平移参考不随姿态旋转。输入布局、工具轴积分实现及 worker 实现版本绑定控制指纹。GELLO 驱动缓存含成功采样时间和序号；collector 以源采样时间作为有序消息身份，重复／倒序／过期消息不刷新新鲜度，关节和夹爪同帧传递。

调参等待期间停止积分、接受和发送新目标，清空旧待发送目标；当前 generation 的 request ID 确认后，等待新的接受目标再继续跟踪。暂停、停止或切换控制权撤销等待，迟到确认不恢复运动。回合起始 journal identity 保存有效参数；回合内已确认变化以 `compliance` 审计事件保存请求 ID、generation、worker sequence、时间和参数，不增加 schema 字段。

独立字段包括：

| 字段 | 含义 |
|---|---|
| `action` | 当前原生空间的接受目标，始终绝对值 |
| `observation.anchor` | 当前原始实测锚点 |
| `command.joint` | Franky 路线记录的实测关节（旧 ROS 路线为提交关节）与夹爪 |
| `command.fk_tcp` | 提交目标 TCP（旧 ROS 路线由关节目标 FK 得到）与夹爪 |
| `command.target_tcp` | Franky worker 接受的绝对 TCP 目标与夹爪 |
| `command.identity` | 代际、命令 ID、接受时间、观测时间（纳秒） |
| `command.source` | 示教／PI05／RLT 来源 |
| `diagnostic.controller` | 每臂接收代际、命令 ID、诊断接收时间 |
| `diagnostic.filtered_joint` | 控制器可用的滤波后关节目标 |
| `diagnostic.actual_tcp` | worker 回传的实测 TCP 位姿（xyz + quaternion_xyzw） |
| `diagnostic.target_joint` | Franky tracker 回传的目标关节（不可用时为零） |
| `diagnostic.tracker` | tracker 运行标志、代际、序号及故障标志 |

诊断未返回时使用 `-1` 标识缺失身份；诊断不是到达确认，不能替换训练标签。元数据包含设备、TCP、运动学和控制参数指纹。

导出另一空间时，只使用 `command.joint` 或 `command.fk_tcp`，必须显式允许派生表示并记录来源。没有对应命令字段时拒绝转换，不用实测值或任意 IK 解补造标签。混合数据和训练默认拒绝未声明的派生数据及不一致的控制契约。

## 推理、RTC 与在线回放

推理请求保留独立的原始锚点。模型返回后一次性还原绝对 chunk，执行队列存绝对目标。ROS 执行器固定承诺前缀，模型侧按新锚点重新编码，结果仅安装固定边界后的后缀。接管、归还、故障和回合结束使在途结果失效。

RLT reference、actor 和 executed chunk 使用相同表示及该请求锚点；next-state 分支保留自己的锚点。回放同时保存绝对 executed chunk 和锚点，缓存／checkpoint 校验嵌入的动作契约。在线保存先取得终点观测、持久化回合，再等待控制器停止确认后通知既有在线训练服务；Web 不提供训练任务管理。

SFT 与 RLT 支持不同 chunk 长度和目标帧间隔；稀疏目标部署按间隔保持绝对目标。RLT 将有效模型动作数 `actual_steps` 与实际控制帧数 `elapsed_steps` 分开记录，按 `reward_offsets` 计算奖励折扣，bootstrap 使用实际经过的控制帧数；末尾不足一个 chunk 时屏蔽 padding。缓存／在线命令中的 `frame_stride` 是训练起点间隔，与动作契约中的目标帧间隔不同。回放奖励按真实纳秒区间折扣；UTD 按唯一有效控制区间计数。完整协议、状态、schema 5、Replay 4、损失与恢复说明见 [统一执行闭环](EXECUTION_RTC_RLT.md)。

## 离线与实机边界

离线检查涵盖数学往返、模型适配器、命令入口、模拟采集、Web、CLI、wheel、ROS 构建与导入。没有运行真实大模型训练，没有连接机器人或开启相机。

仍须在获授权的现场分别验收：ROS 30/100/1000 Hz 时序、诊断延迟、单／双臂同步、原生标签实机重放、接触和跟踪表现、夹爪松手保持、Normal/Fast 与手柄手感、GELLO 跟随、姿态恢复、RTC 延迟与接管、在线训练中断恢复、模型效果。离线通过不能替代这些结果。
