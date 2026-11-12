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

`lingbot-dev` 的外部串行部署使用明确标识的 external deployment v2，允许 LingBot 原生统计。服务必须声明物理单位、10D 字段顺序、基座/TCP、相机角色、50 步/30 Hz 采样及资产指纹；Workbench 校验这些字段并保留执行投影。此例外不生成训练来源，不放宽 PI05/SFT/RLT 的模型统计契约。记录仍为 schema 5，等待保持标为 `system_hold` 并排除监督，原观测时间与实际安装时间分别保存。详见 [LingBot 部署](LINGBOT_DEPLOYMENT.md)。

Franky 路线只接受绝对逻辑 TCP 目标（`flange_aligned_tcp_v1`）。逻辑 TCP 原点仍为物理 TCP，轴向平行于法兰；XYZ 不因朝向换算移动。设实测安装旋转为 `R_F_E`，则 worker 提交 `R_base_hardware = R_base_logical · R_F_E`，反馈使用 `R_base_logical = R_base_hardware · R_F_Eᵀ`。安装 −45° 时逻辑竖直 yaw=0 对应硬件 yaw=+45°。action、observation、coordinator、GELLO FK、策略、姿态库和实际参考统一使用逻辑坐标；硬件 TCP／参考及原始安装变换单独保留在 worker 状态和审计中。worker implementation 6 显式声明该坐标约定，collector 拒绝不匹配的 worker；工具变换在会话内锁定，变化后故障停止。GELLO 通过 FK、Gamepad 和策略都更新同一个 TCP 参考位姿，worker 使用 `CartesianImpedanceTracker.set_target()` 持续跟踪；Gamepad 松杆仅将刚释放方向的参考收回到实测位姿一次，随后固定保持。新会话拒绝旧 ROS backend、关节动作空间和旧 tracking rate／IK 调参字段；CartesianServo 只供隔离的历史验证工具使用。

Franky 默认阻抗参数为平移刚度 1100 N/m、旋转刚度 65 Nm/rad、平移误差裁剪 0.008 m、旋转误差裁剪 0.04 rad、单次目标步长 0.03 m/0.10 rad。Web 支持启动前覆盖初值和运行中调参。参数草稿与 worker 确认值分离，仅实际变化的裁剪、力矩变化率和增益时间常数会停止并重建 tracker；刚度调用 `set_gains`，不变参数不重启控制。刚度上限为 1200 N/m、80 Nm/rad，裁剪下限为 0.005 m、0.02 rad。

应用层只有一个 30 Hz 回调，输入更新、动作接受和提交共用时隙。独立 IPC 消费线程最多处理一条在途目标，没有待发目标积压队列。上一目标未确认时，当前时隙不积分人工输入、不接受动作、不产生记录；确认后使用当前输入，策略取当前真实时隙，不补发过期动作。worker 以独立 1000 Hz FCI 阻抗环跟踪参考。`max_step` / `max_step_rad` 仍是每次调用的参考变化上限。

IPC v2 的目标包含 session、generation、sequence、request ID、slot ID 和单调时间戳；只有全部身份匹配的成功响应才确认执行。响应区分请求目标、实际提交给 tracker 的参考、实测 TCP，并提供应用时间。双臂先校验整个包，任一执行失败停止全部 tracker；审计保留已执行部分，不声称物理原子性。旧 worker 协议被拒绝。

tracker 在显式启用准备中从实测 TCP／关节初始化，所有臂准备确认后才允许目标。普通目标不能创建或恢复 tracker；停止／恢复保持待命。代际切换撤销旧请求，生命周期命令优先。默认目标请求及目标流期限 100 ms、通信／状态期限 500 ms，健康轮询不延长目标流。SDK 阻塞不能由排队命令抢占；通信失败时父进程有界关闭本会话 worker，故障锁存且不自动重建恢复。

记录在提交时冻结观察、候选、接受目标和身份，匹配 ACK 后才入记录流程。相机偶发缺帧跳过整条图像／观察／动作记录，控制继续；连续过期超过 `sync_tolerance_ms`（默认 100 ms）故障，不再用额外 1.5 帧门限停控。未录制人工摆位不依赖单次相机曝光。IPC 等待、缺帧、漏时隙形成真实 slot 缺口和 segment 边界，SFT／RLT／Replay 不跨越；没有新终点图像时不得复制旧图构造终止帧。1.5 帧仅用于区间标签有效性，不是停控阈值。

`target_applied` 审计关联 generation、sequence、slot、worker 应用时间及实际参考；失败保留 `target_failure` 与部分应用事实。控制回调只提交有界内存任务，编码、文件写入和 fsync 留在后台。存储恢复不会恢复控制。新单循环实现标识、IPC v2、worker 版本、binding 二进制哈希与阻抗参数组成控制指纹，旧指纹数据和 checkpoint 不能混用。

Gamepad 默认使用旧单臂已验证的 `raw_joy` 布局（`joy_node`），显式选择时才使用 SDL 标准布局，姿态增量为 `Exp(R_measured · ω_tool · dt) · R_target`，平移参考不随姿态旋转。输入布局、工具轴积分实现及 worker 实现版本绑定控制指纹。GELLO 驱动缓存含成功采样时间和序号；collector 以源采样时间作为有序消息身份，重复／倒序／过期消息不刷新新鲜度，关节和夹爪同帧传递。

调参等待期间停止积分、接受和发送新目标，清空旧待发送目标；当前 generation 的 request ID 确认后，等待新的接受目标再继续跟踪。暂停、停止或切换控制权撤销等待，迟到确认不恢复运动。回合起始 journal identity 保存有效参数；回合内已确认变化以 `compliance` 审计事件保存请求 ID、generation、worker sequence、时间和参数，不增加 schema 字段。

独立字段包括：

| 字段 | 含义 |
|---|---|
| `action` | 当前原生空间的接受目标，始终绝对值 |
| `observation.anchor` | 当前原始实测锚点 |
| `command.joint` | Franky 路线记录的实测关节（旧 ROS 路线为提交关节）与夹爪 |
| `command.fk_tcp` | 提交目标 TCP（旧 ROS 路线由关节目标 FK 得到）与夹爪 |
| `command.target_tcp` | coordinator 接受的绝对 TCP 目标与夹爪（实际 tracker 参考见审计） |
| `command.identity` | 代际、命令 ID、接受时间、观测时间（纳秒） |
| `command.source` | 示教／PI05／RLT 来源 |
| `diagnostic.controller` | 每臂接收代际、命令 ID、诊断接收时间 |
| `diagnostic.filtered_joint` | 控制器可用的滤波后关节目标 |
| `diagnostic.actual_tcp` | worker 回传的实测逻辑 TCP 位姿（xyz + quaternion_xyzw） |
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

仍须在获授权的现场分别验收：30 Hz 接受／提交与 Franky 1000 Hz 底层时序、诊断延迟、单／双臂同步、原生标签实机重放、接触和跟踪表现、夹爪松手保持、Normal/Fast 与手柄手感、GELLO 跟随、姿态恢复、RTC 延迟与接管、在线训练中断恢复、模型效果。离线通过不能替代这些结果。

### 工作空间与恢复参考

会话配置 `control.workspace_limits_enabled`（默认 true，Web 可显式覆盖后锁定）统一控制所有输入源、快捷键、姿态库及保存时的 TCP XYZ 范围校验／投影。关闭时仍保存有限边界，不改变有限数值、SO(3)、机器人自身限制、碰撞配置、worker 单步和故障保护。实际模式写入元数据及控制指纹，并由 `execution_projection.workspace_limits_enabled` 传递到 NumPy／Torch 动作投影；未声明开关按 true 解释，关节动作仍投影到关节限制。续采独立严格核对模式、范围和控制指纹，不重写旧契约。人工快捷键指纹版本为 8：View／Menu 固定为逻辑 TCP 在基座的 `diag(1,−1,−1)`，R3 仅绕基座 Z 恢复逻辑 yaw=0 并保留倾斜。硬件安装旋转在 worker 内部处理，不写入 action 的逻辑朝向。工具指纹包含逻辑定义与原始硬件安装变换，控制指纹包含坐标约定；旧坐标数据不能续写或静默转换。

姿态库在全部 tracker 准备确认后，从实测起点建立一次直线位置／最短旋转参考，共用五次进度 `s=10u³−15u⁴+6u⁵`。时长为 `max(0.3, 1.875×距离/v, 1.875×角度/ω)`，速度默认最多 0.10 m/s、0.45 rad/s，且受会话上限与 worker 单步×30 限制。进度只按可提交的 30 Hz 时隙推进，等待 ACK 时冻结；仍经 coordinator → IPC → tracker.set_target()，不新增控制线程或 SDK 模式。参考终点与实测 3 mm／2°共同决定完成，60 秒超时；取消和代际变化撤销计划。未选中臂保持起始参考，完整目标朝向和姿态库既有记录不改写。参考几何保证不等于实机轨迹连续性或恢复耗时已验收。
