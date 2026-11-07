# ROS 控制与动作契约

本仓库只提供 ROS 2 Humble + C++ 关节阻抗控制路线。Franky 驱动、采集器、独占串口夹爪路线及依赖已删除；旧实现仅在 Git 历史中保留。旧 Franky YAML 会报迁移错误。

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

关节模式下，GELLO 直接提交关节目标；Gamepad 笛卡尔操纵经过 IK 后进入同一入口。TCP 模式下，GELLO 经过 FK、Gamepad 和策略直接提交 TCP 目标，随后使用同一个跟踪器和 IK。策略重放关节动作不会再次经过 Gamepad IK。

学习目标最多按 30 Hz 接受，上层跟踪按 100 Hz 更新，C++ 控制器保持 1000 Hz。Normal/Fast、工作空间、关节速度及力矩变化率保护沿用 ROS 基线。停止、接管、回合边界和故障通过代际撤销命令，重置发生在实时循环中；所有机械臂确认新代际后才允许新目标。确认超时保持故障。诊断快照使用序列号避免读到混合代际的状态。

独立字段包括：

| 字段 | 含义 |
|---|---|
| `action` | 当前原生空间的接受目标，始终绝对值 |
| `observation.anchor` | 当前原始实测锚点 |
| `command.joint` | 实际提交的绝对关节与夹爪目标 |
| `command.fk_tcp` | 提交关节目标对应的 FK TCP 与夹爪 |
| `command.identity` | 代际、命令 ID、接受时间、观测时间（纳秒） |
| `command.source` | 示教／PI05／RLT 来源 |
| `diagnostic.controller` | 每臂接收代际、命令 ID、诊断接收时间 |
| `diagnostic.filtered_joint` | 控制器可用的滤波后关节目标 |

诊断未返回时使用 `-1` 标识缺失身份；诊断不是到达确认，不能替换训练标签。元数据包含设备、TCP、运动学和控制参数指纹。

导出另一空间时，只使用 `command.joint` 或 `command.fk_tcp`，必须显式允许派生表示并记录来源。没有对应命令字段时拒绝转换，不用实测值或任意 IK 解补造标签。混合数据和训练默认拒绝未声明的派生数据及不一致的控制契约。

## 推理、RTC 与在线回放

推理请求保留独立的原始锚点。模型返回后一次性还原绝对 chunk，执行队列存绝对目标。ROS 执行器固定承诺前缀，模型侧按新锚点重新编码，结果仅安装固定边界后的后缀。接管、归还、故障和回合结束使在途结果失效。

RLT reference、actor 和 executed chunk 使用相同表示及该请求锚点；next-state 分支保留自己的锚点。回放同时保存绝对 executed chunk 和锚点，缓存／checkpoint 校验嵌入的动作契约。在线保存先取得终点观测、持久化回合，再等待控制器停止确认后通知既有在线训练服务；Web 不提供训练任务管理。

SFT 与 RLT 支持不同 chunk 长度和目标帧间隔；稀疏目标部署按间隔保持绝对目标。RLT 将有效模型动作数 `actual_steps` 与实际控制帧数 `elapsed_steps` 分开记录，按 `reward_offsets` 计算奖励折扣，bootstrap 使用实际经过的控制帧数；末尾不足一个 chunk 时屏蔽 padding。缓存／在线命令中的 `frame_stride` 是训练起点间隔，与动作契约中的目标帧间隔不同。回放奖励按真实纳秒区间折扣；UTD 按唯一有效控制区间计数。完整协议、状态、schema 5、Replay 4、损失与恢复说明见 [统一执行闭环](EXECUTION_RTC_RLT.md)。

## 离线与实机边界

离线检查涵盖数学往返、模型适配器、命令入口、模拟采集、Web、CLI、wheel、ROS 构建与导入。没有运行真实大模型训练，没有连接机器人或开启相机。

仍须在获授权的现场分别验收：ROS 30/100/1000 Hz 时序、诊断延迟、单／双臂同步、原生标签实机重放、接触和跟踪表现、夹爪松手保持、Normal/Fast 与手柄手感、GELLO 跟随、姿态恢复、RTC 延迟与接管、在线训练中断恢复、模型效果。离线通过不能替代这些结果。
