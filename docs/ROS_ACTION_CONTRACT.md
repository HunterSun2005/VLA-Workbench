# ROS 控制与动作契约

本仓库的统一采集／部署路线使用独立 Franky worker 和官方 Franky 2.0.0 `CartesianVelocityMotion`。ROS 2 Humble 继续负责 GELLO、Gamepad、夹爪、相机和采集器；worker 独占 Franka FCI 控制连接。旧 ROS 关节阻抗路线保留为隔离的 legacy backend，不能与 Franky 数据混采。

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

`lingbot-dev` 的外部 RTC／串行部署使用明确标识的 external deployment v2，允许 LingBot 原生统计。服务必须声明物理单位、10D 字段顺序、基座/TCP、相机角色、50 步/30 Hz 采样及资产指纹；Workbench 校验这些字段并保留执行投影。此例外不生成训练来源，不放宽 PI05/SFT/RLT 的模型统计契约。记录仍为 schema 5，等待保持标为 `system_hold` 并排除监督，原观测时间与实际安装时间分别保存。详见 [LingBot 部署](LINGBOT_DEPLOYMENT.md)。

Franky 路线只接受绝对逻辑 TCP 位姿（`flange_aligned_tcp_v1`）。逻辑与硬件 TCP 原点相同，安装旋转只换算朝向；Twist 使用 base 轴，SDK `ee_frame` 为 identity，禁止重复施加 45° 偏移。

ROS 独占 30 Hz 接受、队列、提交和记录；官方 Franky 2.0.0/libfranka 0.21.2 worker 在固定 100 Hz 时隙将最新绝对目标和实测转换为速度。没有第二动作队列或外部速度/关节动作入口。首次 SDK 提交成功才 ACK；等待 ACK 冻结人工积分和记录，不补算。reset 只确认就绪，recover 后仍须手动启用。

手柄每时隙从实测重锚并取消释放通道追赶；策略和 GELLO 持续跟踪目标。目标与速度平滑默认关闭、独立可选，仅用于策略/GELLO。保存点/快捷回位重复提交绝对最终目标，由 worker 反馈减速，以实测 0.5 mm/2° 判断到位。

目标 stream 100 ms、状态 500 ms watchdog 不被 health 或内部伺服续期。故障、断连、队列耗尽、存储阻塞和迟到结果均不能自动恢复运动；代次撤销同时保护 RTC 前缀与安装后缀。serial 最终目标仍按提交 ACK 门控。

记录保持 schema 5，backend 改为 `franky_cartesian_velocity`。保留 `target_applied`、`hardware_target_applied`，新增 `velocity_servo` 审计关联实际观测、SDK 提交时间、派生 Twist、代次和 slot。有界审计缓冲满时停止，未提交事件保留重试；控制线程不写盘。原 `diagnostic.tracker` 列保留字段名，在新 backend 表示速度伺服就绪诊断。

旧 `compliance`、Tracker 裁剪、静态积分补偿配置已移除；`velocity_control` 更新必须等待对应 request ID 的 worker 确认。更新取消旧目标，等待新的接受目标；迟到确认不能恢复旧代次。参数、原生库身份、算法版本和实际 SDK 滤波设置均参与控制指纹。

安装、参数、迁移与无硬件验收见 [Franky 环境](FRANKY_ENVIRONMENT.md)。

IPC v2 的目标包含 session、generation、sequence、request ID、slot ID 和单调时间戳；只有身份全部匹配的成功响应才确认执行。响应区分接受目标、平滑参考与实测 TCP，返回首次 SDK 提交时间。双臂先校验整个包，任一执行失败停止全部臂；审计保留部分应用事实，不声称物理原子性。普通目标不能启用或恢复速度伺服，SDK 阻塞不能被排队命令抢占；父进程有界关闭本会话 worker。

记录在提交时冻结观察、候选、接受目标和身份，匹配 ACK 后才入记录流程。相机偶发缺帧跳过整行记录，连续过期超过 `sync_tolerance_ms`（默认 100 ms）故障。IPC 等待、缺帧、漏时隙形成真实 slot 缺口和 segment 边界，SFT／RLT／Replay 不跨越；没有新终点图像时不得复制旧图构造终止帧。观测帧只关联本时隙实际创建的待确认目标，不能附加到已撤销的旧目标。

Gamepad 默认使用旧单臂 `raw_joy` 布局（`joy_node`），显式选择时才用 SDL 布局。工具轴增量以实测朝向转换到 base，安装旋转与右摇杆校准分别处理。GELLO 缓存成功采样时间和序号；重复／倒序／过期消息不刷新新鲜度，关节和夹爪同帧传递。调参等待暂停积分、接受和发送；暂停、停止或切换控制权撤销等待。已确认参数变化保存 `velocity_control` 审计事件和请求身份。

独立字段包括：

| 字段 | 含义 |
|---|---|
| `action` | 当前原生空间的接受目标，始终绝对值 |
| `observation.anchor` | 当前原始实测锚点 |
| `command.joint` | Franky 路线记录的实测关节（旧 ROS 路线为提交关节）与夹爪 |
| `command.fk_tcp` | 提交目标 TCP（旧 ROS 路线由关节目标 FK 得到）与夹爪 |
| `command.target_tcp` | coordinator 接受的绝对 TCP 目标与夹爪（实际速度伺服参考见审计） |
| `command.identity` | 代际、命令 ID、接受时间、观测时间（纳秒） |
| `command.source` | 示教／PI05／RLT 来源 |
| `diagnostic.controller` | 每臂接收代际、命令 ID、诊断接收时间 |
| `diagnostic.filtered_joint` | 控制器可用的滤波后关节目标 |
| `diagnostic.actual_tcp` | worker 回传的实测逻辑 TCP 位姿（xyz + quaternion_xyzw） |
| `diagnostic.target_joint` | Franky 原生状态回传的目标关节（不可用时为零） |
| `diagnostic.tracker` | 速度伺服就绪标志、代际、序号及故障标志 |

诊断未返回时使用 `-1` 标识缺失身份；诊断不是到达确认，不能替换训练标签。元数据包含设备、TCP、运动学和控制参数指纹。

导出另一空间时，只使用 `command.joint` 或 `command.fk_tcp`，必须显式允许派生表示并记录来源。没有对应命令字段时拒绝转换，不用实测值或任意 IK 解补造标签。混合数据和训练默认拒绝未声明的派生数据及不一致的控制契约。

## 推理、RTC 与在线回放

推理请求保留独立的原始锚点。模型返回后一次性还原绝对 chunk，执行队列存绝对目标。ROS 执行器固定承诺前缀，模型侧按新锚点重新编码，结果仅安装固定边界后的后缀。接管、归还、故障和回合结束使在途结果失效。

RLT reference、actor 和 executed chunk 使用相同表示及该请求锚点；next-state 分支保留自己的锚点。回放同时保存绝对 executed chunk 和锚点，缓存／checkpoint 校验嵌入的动作契约。在线保存先取得终点观测、持久化回合，再等待控制器停止确认后通知既有在线训练服务；Web 不提供训练任务管理。

SFT 与 RLT 支持不同 chunk 长度和目标帧间隔；稀疏目标部署按间隔保持绝对目标。RLT 将有效模型动作数 `actual_steps` 与实际控制帧数 `elapsed_steps` 分开记录，按 `reward_offsets` 计算奖励折扣，bootstrap 使用实际经过的控制帧数；末尾不足一个 chunk 时屏蔽 padding。缓存／在线命令中的 `frame_stride` 是训练起点间隔，与动作契约中的目标帧间隔不同。回放奖励按真实纳秒区间折扣；UTD 按唯一有效控制区间计数。完整协议、状态、schema 5、Replay 4、损失与恢复说明见 [统一执行闭环](EXECUTION_RTC_RLT.md)。

## 离线与实机边界

离线检查涵盖数学往返、模型适配器、命令入口、模拟采集、Web、CLI、wheel、ROS 构建与导入。没有运行真实大模型训练，没有连接机器人或开启相机。

仍须在获授权的现场分别验收：30 Hz 接受／提交与 worker 100 Hz 与原生 Franky FCI 底层时序、诊断延迟、单／双臂同步、原生标签实机重放、接触和跟踪表现、夹爪松手保持、Normal/Fast 与手柄手感、GELLO 跟随、姿态恢复、RTC 延迟与接管、在线训练中断恢复、模型效果。离线通过不能替代这些结果。

### 工作空间与恢复参考

会话配置 `control.workspace_limits_enabled`（默认 true，Web 可显式覆盖后锁定）统一控制所有输入源、快捷键、姿态库及保存时的 TCP XYZ 范围校验／投影。关闭时仍保存有限边界，不改变有限数值、SO(3)、机器人自身限制、碰撞配置、worker 速度限幅和故障保护。实际模式写入元数据及控制指纹，并由 `execution_projection.workspace_limits_enabled` 传递到 NumPy／Torch 动作投影；未声明开关按 true 解释，关节动作仍投影到关节限制。续采独立严格核对模式、范围和控制指纹，不重写旧契约。人工快捷键指纹版本为 8：View／Menu 固定为逻辑 TCP 在基座的 `diag(1,−1,−1)`，R3 仅绕基座 Z 恢复逻辑 yaw=0 并保留倾斜。硬件安装旋转在 worker 内部处理，不写入 action 的逻辑朝向。工具指纹包含逻辑定义与原始硬件安装变换，控制指纹包含坐标约定；旧坐标数据不能续写或静默转换。


姿态库在全部速度伺服准备确认后保存实测起点，沿同一 coordinator → IPC 入口反复提交最终绝对目标。worker 以旧 VLA 的反馈减速接近，平移上限 0.03 m/s、旋转上限 0.15 rad/s；不再用五次曲线进度表示到位。进度按实测残差计算，实测 0.5 mm／2° 内才完成，30 秒超时；等待 ACK 不推进提交，取消和代际变化撤销目标。未选中臂保持起始参考，不改写姿态库的目标朝向。离线反馈测试不保证实机精度或恢复耗时。
