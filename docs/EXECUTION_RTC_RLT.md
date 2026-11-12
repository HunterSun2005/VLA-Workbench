# 统一执行、RTC 与 RLT 数据闭环

PI05 主线在 `ros-dev` 维护；`lingbot-dev` 增加 [LingBot 外部串行部署](LINGBOT_DEPLOYMENT.md)。ROS 进程拥有动作时隙、执行队列和运动代次；模型服务只返回候选块。Web 与终端共用 `RolloutCollector`／`ControlCoordinator`。模型 Python 与 ROS Python 分离。当前提交的验证方式见 [自动化验收](ACCEPTANCE.md)，[历史归档](archive/EXECUTION_2026-11-07.md)保留初次验收快照。

## 协议与时间轴

Unix socket 使用 NumPy 数组协议 v2，magic 为 `VLARLT02`。旧 magic、旧数据、旧模型部署契约、旧 Replay 和旧训练语义直接拒绝；不提供转换入口。官方基础权重可以初始化带新契约的新训练。

外部 LingBot 服务通过 TCP 使用同一 wire v2，但采用明确的 `external_deployment` v2 物理契约及原生统计。此例外不绕过 PI05 训练来源校验。下面的 RTC 前缀/预算/裁头规则适用于 PI05；LingBot 的 `SerialExecutionContract` 始终空前缀、从索引 0 执行 50 步，原请求时间不改写，实际安装 slot/time 另写 `install_serial`。最终目标 ACK 后才重新采集观测，正常等待保持最后目标且不作为监督动作。

| 契约 | 版本 |
| --- | ---: |
| wire／ExecutionContract／模型动作契约／统计契约 | 2 |
| recording／episode journal／execution audit | 5 |
| 在线 run／训练语义 | 3 |
| PersistentReplay | 4 |

`health` 返回模型身份、动作契约、策略 horizon、frame_stride 和允许的模式；`plan` 携带完整请求身份及数组；`invalidate` 撤销代次；`progress` 记录控制进展。`begin_episode`、`episode_saved`、`recover_online` 保留，但保存后训练必须附带当前代次、standby 状态和全部控制器复位确认的 `progress`。GPU 推理与训练共用一个工作锁；健康查询和代次撤销由独立连接处理。

| 不可变对象 | 主要字段 |
| --- | --- |
| ExecutionContract | absolute ActionSpec、30 Hz 接受与提交、frame_stride、各策略 horizon、推理／保护预算、控制／URDF／TCP 指纹 |
| PlanRequest | session_id、episode_id、generation、request_id、observation_id、plan_version、model_version、contract_digest、start_slot、replace_slot、deadline_ns、observed_at_ns、anchor、prefix |
| PlanResponse | 原样回显 PlanRequest、完整 absolute actions、逐动作 sources/phases、实际 vla_reference、state_vec 等上下文 |
| ExecutionRecord | slot_id、generation、plan_version、action_index、candidate、accepted、source、phase、accepted_at_ns、submitted_at_ns、outcome、diagnostics |

`FixedDeadline` 以单调时钟计算整数截止时间：一个 30 Hz 回调顺序完成状态读取、人工候选更新、目标接受、提交和记录，不再通过第二个 100 Hz 时钟重复提交目标。错过时隙只处理当前时隙，不补发历史动作；协调器要求唯一 slot_id，同一时隙幂等，倒序拒绝。Franky C++ / libfranka 独立执行 1 kHz 笛卡尔阻抗闭环。worker 的目标步长仍是每次提交的上限，不随频率降低而自动放大；输入、状态、目标流超时仍锁存故障。

默认 `inference_slots=4`、`installation_guard_slots=1`。模型间隔 r 下，最短冻结长度为 `ceil(5/r)`；替换边界对齐动作网格，并受 Actor 已锁定块末限制。d 由实际边界决定，响应提前到达不缩短它。启动时每种策略都必须满足 H ≥ 2d；自动模式发起请求时也按可能返回的最短 horizon 校验。不会自动扩大模型 horizon。

每个会话至多一个在途请求和一个待安装结果。旧代次、重复／倒序、身份／模型／计划／上下文不匹配、过安装截止时间的结果不安装。前缀是已承诺的 absolute 目标，在本次 anchor 下重新编码后交给 RTC。只安装 `[d,H)`，Actor 后缀锁定到块末。首次准备的保持动作来源为 `system_hold`；用户启用、首块安装后才能进入策略运动。

旧 `Pi05RTC`／`DeploymentLoop`、模型 Actor 动作弹出队列和执行游标、`AbsoluteChunkQueue` 及单动作推理辅助函数已删除；自有策略的单动作 `select_action` 明确拒绝。单／双臂 GELLO 的 `auto_record` 入口也统一调用 ROS 会话，旧独立采集器已删除。候选块中的阶段和来源只有在 ROS 安装／执行相应动作时生效。推理异常或代次撤销会回滚分类头状态。自动 VLA／Actor 切换只用于部署；在线服务仅允许完整回合或单关键阶段模式。

## 状态转换

| 状态 | 进入条件 | 离开条件 |
| --- | --- | --- |
| standby | 初始化、回合结束、正常停止 | 用户启用并完成复位确认 |
| preparing | 用户启用／手动恢复，准备首块 | 已安装有效后缀到达执行边界 |
| waiting_inference | LingBot 串行启用或末动作 ACK 后的正常等待 | 当前有效完整 chunk 安装；超时/断连转入人工恢复 |
| human | 人工接管／人工控制 | 用户显式归还；Web 点击具体策略即请求归还，底层仅设置策略模式的接口仍保留人工控制 |
| policy | 执行已安装策略动作 | 接管、暂停、结束、耗尽或故障 |
| waiting_resume | 策略队列耗尽 | 明确手动恢复；迟到结果不能退出此状态 |
| storage_blocked | 队列溢出、编码／磁盘／提交失败 | 保存重试或显式丢弃；成功后仍不自动运动 |
| fault | 健康、控制器、输入或协议错误 | 明确恢复及控制器复位确认，再重新启用 |

接管、暂停、手动模式切换、结束及故障均撤销旧代次。普通 chunk 接续不切控制 segment；中断切段。双臂先整体校验，再顺序提交：一侧发布失败会保留部分提交记录并撤销双臂，**不声称物理原子性**。进程重启只恢复数据与训练，不恢复运动队列。

部署支持启用后的未录制人工摆位，沿用 ROS 30 Hz 输入、接受和提交和同一协调器，尚未开始回合时不创建 episode journal、不写入训练帧、不请求策略块。人工摆位后开始回合会撤销摆位代次并保持人工 owner，之后的人工帧按接管记录；切换策略才申请新策略块。未录制时归还策略先回待命，须再次启用并开始回合。Web 模式选择为显式、幂等请求；故障、存储阻塞、等待手动恢复、保存／训练处理中及控制器复位未确认时拒绝切换，选项本身不能恢复运动。仅选择下一控制来源与「启用控制」分离，模型协议中的模式仍只有原有策略值。

## 记录、有效窗口和持久化

`action` 始终是裁剪后、跟踪／IK 前的 absolute 接受目标；`action.candidate` 保存原始候选。`command.joint`／`command.fk_tcp` 保存实际提交结果。`observation.anchor` 独立于模型输入字段筛选。

所有采集路线都保存 `complementary_info.slot_id`、`observed_at_ns`、`command_at_ns`、`submitted_at_ns`、`segment_id`、`command_id`、`plan_version`、`action_index`、`action_valid`、`observation_valid`、`supervision_valid`、`segment_terminal`、`task_terminal`，以及逐动作 phase、collector_policy_id 和 is_intervention。来源编码：-1 系统保持，0 人工，1 VLA，2 Actor。人工示教不标为接管。

`observation.sensor_frames` 按 `meta/info.json` 中的 `sensor_frame_layout.keys` 排列，每个传感器为 `[frame_id, received_at_ns, age_ns]`。时间来源明确为 host monotonic capture/receive；它不是硬件时钟同步保证。观测时锁存这些字段，提交时不使用最新传感器信息覆盖。视频 `timestamp` 仍按帧号/FPS 定位，物理奖励和执行间隔只用独立纳秒字段。

执行有效性与监督有效性分离：系统保持可以是有效执行，但不能成为 SFT 示教。读取、统计和采样使用相同的首个无效位置窗口规则；不得跨控制中断、缺失时隙或无效观测，r>1 时也检查跳过的控制时隙。统计继续逐列计算精确分位数。

每回合有 `execution/episode-NNNNNN/manifest.json` 和带 schema／校验和的事件文件。事件包括请求、完整候选、安装／拒绝／撤销、承诺前缀、VLA reference、执行／提交、控制失败和 endpoint commitment。manifest 保存执行契约。诊断 JSONL 不作为训练数据源。

控制回调只提交有界队列中的不可变记录；编码、NPZ、Parquet、视频合并和 fsync 在后台完成。`staging/<journal_id>` 中按一秒写独立视频片段、对应数组及事件，再发布带校验和的 segment 清单。完成全部媒体帧数检查后，episode 元数据行是提交点。提交后发布标记失败的重试只重建元数据和标记，不重写视频、不重复回合。

重启恢复只接受连续且校验和、数组长度、视频帧数相符的持久前缀。缺失尾部标记 `incomplete_tail`，不制造 terminal、endpoint、图像或动作；缺少因果终点的窗口不进入 RL。只有显式丢弃会删除未提交回合。正常用户暂停沿用主动丢弃约定，耗尽和存储阻塞不走该入口。

损坏清单、数组或事件，以及媒体检查失败，会终止当前扫描并返回此前的有效前缀；`issues` 给出失败文件和原因。编码写入／结束与帧数检查默认超时 30 秒，最终拼接默认 120 秒。超时保留 staging，并允许停止运动后重试；具体恢复操作见 [工作流](WORKFLOWS.md#执行耗尽与保存恢复)。

实际 TCP、URDF、Python 跟踪参数、C++ 控制器有效参数、执行预算和投影边界绑定契约。Franky 底层闭环设计频率为 1000 Hz；运行中阻抗变化必须通过已有请求确认流程并记录事件。训练与部署比较实际 recording source 和模型声明来源。RLToken 沿用 SFT 归一化并单独记录自己的训练数据指纹；AC 独立校验 Actor horizon。

## Replay、梯度与奖励

已安装 Actor 计划的执行窗口是 `A_exec = [P[0:d], A[d:H]]`。Replay 保存原始 absolute 执行序列、请求 anchor、归一化执行、start_idx、valid/prefix/actor 掩码、实际 VLA reference、来源和模型／计划身份；下一状态也有独立 anchor、承诺前缀和生效边界。人工离线窗口显式 d=0。未安装或未生效的候选仅作审计。

只有真实任务终止且有有效终点观测的短窗口可进入 Critic。接管、故障及缺失终点不能伪造 terminal 或 bootstrap。非终止窗口必须有完整执行和下一执行上下文。

Critic 输入真实 A_exec。Actor Q 分支使用：

```text
A_tilde = M_prefix * stopgrad(A_exec) + M_actor * Projection(mu)
L_actor = -mean Q(x, A_tilde)
          + beta * mean_batch sum_dims M_actor * (mu - ref_clamped)^2
          + prefix_bc_weight * mean_batch sum_dims M_prefix * (mu - stopgrad(A_exec))^2
```

无效 padding 为零。Q 梯度只经有效 Actor 后缀；前缀 BC 作用于 Actor 原始输出。两项 BC 都保持动作维度求和、batch 求平均，不按有效维数放大。`prefix_bc_weight` 默认解析为 beta，但独立保存、报告；日志分别报告两项 BC 及掩码维数。NumPy 执行投影与 Torch 可微投影覆盖 joint/TCP、单/双臂、absolute/chunk_relative 一致性测试，Torch 分支不借助 NumPy 截断梯度。Bootstrap 同样拼接下一承诺前缀与下一 Actor 后缀。

奖励按真实控制区间保存，单位 Δ=1/30 s：

```text
offset_i = (reward_interval_start_ns[i] - transition_start_ns) / Δ
elapsed  = (endpoint_ns - transition_start_ns) / Δ
y = sum_i gamma^offset_i * reward_i + (1-done) * gamma^elapsed * Q_target(x_next, A_next)
```

Replay 校验物理时间与折扣字段一致。窗口可重叠；训练额度为 `UTD * 本回合唯一有效 interval_slot_ids 数`，不是请求或 transition 数。在线训练只在回合提交、控制器停止确认后进行。模型、目标网络、优化器、随机状态、Replay 游标和训练额度在一个恢复边界保存；半次更新异常锁定训练，恢复完整边界后重算。

## 验证与操作入口

当前提交的复现命令、自动化报告和未验收范围统一见 [自动化验收](ACCEPTANCE.md)。保存重试、staging 检查与发布操作见 [工作流](WORKFLOWS.md#执行耗尽与保存恢复)。

[首次验收快照](archive/EXECUTION_2026-11-07.md)保留当时的测试数量、十分钟 ROS mock 测量及产物位置，不代表后续提交已经验证。真实相机／GPU 并行负载、实机跟踪和任务成功率仍需单独验收；Actor 模型输出本身的轨迹连续性不在当前保证范围内。

## 单个应用控制循环

采集、部署和姿态恢复统一使用单个 30 Hz 回调，异步 IPC 最多一条在途目标，无待发目标队列。等待确认时冻结人工积分和记录，恢复后不补算；策略按当前真实时隙取动作。Franky 保持独立 1000 Hz 阻抗闭环，固定 wheel/libfranka 0.19.0。新会话只接受 Franky/TCP，旧独立 tracking rate 和 IK 执行参数明确报错。

启用先准备全部 tracker；停止、故障、恢复后须显式启用。目标请求／流默认 100 ms、通信／状态 500 ms，健康查询不能代替目标确认。运行调参暂停目标，匹配请求确认才结束等待，代际变化使迟到确认失效。worker 通信失败有界关闭，不自动恢复运动。

相机偶发缺帧跳过整条记录，连续过期超过 `sync_tolerance_ms` 才停控。提交时冻结数据，匹配 IPC v2 确认后记录；实际参考和应用时间写入审计。缺帧和 IPC 等待形成 slot／segment 缺口，训练和 Replay 不跨缺口。旧指纹的 checkpoint 需使用匹配的新数据训练，不做静默转换。详见 [动作契约](ROS_ACTION_CONTRACT.md)。
