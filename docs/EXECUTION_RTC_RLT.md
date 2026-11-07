# 统一执行、RTC 与 RLT 数据闭环

本轮在 `ros-dev` 实现，只作本地提交。ROS 进程拥有动作时隙、执行队列和运动代次；模型服务只返回候选块。Web 与终端共用 `RolloutCollector`／`ControlCoordinator`。模型 Python 与 ROS Python 分离。

## 协议与时间轴

Unix socket 使用 NumPy 数组协议 v2，magic 为 `VLARLT02`。旧 magic、旧数据、旧模型部署契约、旧 Replay 和旧训练语义直接拒绝；不提供转换入口。官方基础权重可以初始化带新契约的新训练。

| 契约 | 版本 |
| --- | ---: |
| wire／ExecutionContract／模型动作契约／统计契约 | 2 |
| recording／episode journal／execution audit | 5 |
| 在线 run／训练语义 | 3 |
| PersistentReplay | 4 |

`health` 返回模型身份、动作契约、策略 horizon、frame_stride 和允许的模式；`plan` 携带完整请求身份及数组；`invalidate` 撤销代次；`progress` 记录控制进展。`begin_episode`、`episode_saved`、`recover_online` 保留，但保存后训练必须附带当前代次、standby 状态和全部控制器复位确认的 `progress`。GPU 推理与训练共用一个工作锁；健康查询和代次撤销由独立连接处理。

| 不可变对象 | 主要字段 |
| --- | --- |
| ExecutionContract | absolute ActionSpec、30/100 Hz、frame_stride、各策略 horizon、推理／保护预算、控制／URDF／TCP 指纹 |
| PlanRequest | session_id、episode_id、generation、request_id、observation_id、plan_version、model_version、contract_digest、start_slot、replace_slot、deadline_ns、observed_at_ns、anchor、prefix |
| PlanResponse | 原样回显 PlanRequest、完整 absolute actions、逐动作 sources/phases、实际 vla_reference、state_vec 等上下文 |
| ExecutionRecord | slot_id、generation、plan_version、action_index、candidate、accepted、source、phase、accepted_at_ns、submitted_at_ns、outcome、diagnostics |

`FixedDeadline` 以同一单调时钟 epoch 计算整数截止时间：30 Hz 接受目标，100 Hz 跟踪。错过时隙只处理当前时隙，不补发历史动作；协调器要求唯一 slot_id，同一时隙幂等，倒序拒绝。C++ 1 kHz 关节阻抗、现有跟踪滤波和限速保持原有实现。

默认 `inference_slots=4`、`installation_guard_slots=1`。模型间隔 r 下，最短冻结长度为 `ceil(5/r)`；替换边界对齐动作网格，并受 Actor 已锁定块末限制。d 由实际边界决定，响应提前到达不缩短它。启动时每种策略都必须满足 H ≥ 2d；自动模式发起请求时也按可能返回的最短 horizon 校验。不会自动扩大模型 horizon。

每个会话至多一个在途请求和一个待安装结果。旧代次、重复／倒序、身份／模型／计划／上下文不匹配、过安装截止时间的结果不安装。前缀是已承诺的 absolute 目标，在本次 anchor 下重新编码后交给 RTC。只安装 `[d,H)`，Actor 后缀锁定到块末。首次准备的保持动作来源为 `system_hold`；用户启用、首块安装后才能进入策略运动。

旧 `Pi05RTC`／`DeploymentLoop`、模型 Actor 动作弹出队列和执行游标、`AbsoluteChunkQueue` 及单动作推理辅助函数已删除；自有策略的单动作 `select_action` 明确拒绝。单／双臂 GELLO 的 `auto_record` 入口也统一调用 ROS 会话，旧独立采集器已删除。候选块中的阶段和来源只有在 ROS 安装／执行相应动作时生效。推理异常或代次撤销会回滚分类头状态。自动 VLA／Actor 切换只用于部署；在线服务仅允许完整回合或单关键阶段模式。

## 状态转换

| 状态 | 进入条件 | 离开条件 |
| --- | --- | --- |
| standby | 初始化、回合结束、正常停止 | 用户启用并完成复位确认 |
| preparing | 用户启用／手动恢复，准备首块 | 已安装有效后缀到达执行边界 |
| human | 人工接管／人工控制 | 用户归还；人工期间选策略只更新待恢复模式 |
| policy | 执行已安装策略动作 | 接管、暂停、结束、耗尽或故障 |
| waiting_resume | 策略队列耗尽 | 明确手动恢复；迟到结果不能退出此状态 |
| storage_blocked | 队列溢出、编码／磁盘／提交失败 | 保存重试或显式丢弃；成功后仍不自动运动 |
| fault | 健康、控制器、输入或协议错误 | 明确恢复及控制器复位确认，再重新启用 |

接管、暂停、手动模式切换、结束及故障均撤销旧代次。普通 chunk 接续不切控制 segment；中断切段。双臂先整体校验，再顺序提交：一侧发布失败会保留部分提交记录并撤销双臂，**不声称物理原子性**。进程重启只恢复数据与训练，不恢复运动队列。

## 记录、有效窗口和持久化

`action` 始终是裁剪后、跟踪／IK 前的 absolute 接受目标；`action.candidate` 保存原始候选。`command.joint`／`command.fk_tcp` 保存实际提交结果。`observation.anchor` 独立于模型输入字段筛选。

所有采集路线都保存 `complementary_info.slot_id`、`observed_at_ns`、`command_at_ns`、`submitted_at_ns`、`segment_id`、`command_id`、`plan_version`、`action_index`、`action_valid`、`observation_valid`、`supervision_valid`、`segment_terminal`、`task_terminal`，以及逐动作 phase、collector_policy_id 和 is_intervention。来源编码：-1 系统保持，0 人工，1 VLA，2 Actor。人工示教不标为接管。

`observation.sensor_frames` 按 `meta/info.json` 中的 `sensor_frame_layout.keys` 排列，每个传感器为 `[frame_id, received_at_ns, age_ns]`。时间来源明确为 host monotonic capture/receive；它不是硬件时钟同步保证。观测时锁存这些字段，提交时不使用最新传感器信息覆盖。视频 `timestamp` 仍按帧号/FPS 定位，物理奖励和执行间隔只用独立纳秒字段。

执行有效性与监督有效性分离：系统保持可以是有效执行，但不能成为 SFT 示教。读取、统计和采样使用相同的首个无效位置窗口规则；不得跨控制中断、缺失时隙或无效观测，r>1 时也检查跳过的控制时隙。统计继续逐列计算精确分位数。

每回合有 `execution/episode-NNNNNN/manifest.json` 和带 schema／校验和的事件文件。事件包括请求、完整候选、安装／拒绝／撤销、承诺前缀、VLA reference、执行／提交、控制失败和 endpoint commitment。manifest 保存执行契约。诊断 JSONL 不作为训练数据源。

控制回调只提交有界队列中的不可变记录；编码、NPZ、Parquet、视频合并和 fsync 在后台完成。`staging/<journal_id>` 中按一秒写独立视频片段、对应数组及事件，再发布带校验和的 segment 清单。完成全部媒体帧数检查后，episode 元数据行是提交点。提交后发布标记失败的重试只重建元数据和标记，不重写视频、不重复回合。

重启恢复只接受连续且校验和、数组长度、视频帧数相符的持久前缀。缺失尾部标记 `incomplete_tail`，不制造 terminal、endpoint、图像或动作；缺少因果终点的窗口不进入 RL。只有显式丢弃会删除未提交回合。正常用户暂停沿用主动丢弃约定，耗尽和存储阻塞不走该入口。

实际 TCP、URDF、Python 跟踪参数、C++ 控制器有效参数、执行预算和投影边界绑定契约。控制器要求 1000 Hz；运行中参数变化触发故障。训练与部署比较实际 recording source 和模型声明来源。RLToken 沿用 SFT 归一化并单独记录自己的训练数据指纹；AC 独立校验 Actor horizon。

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

## 复现与验收边界

不安装或升级既有环境，不启动真实机器人或相机。模型回归使用 CPU、小网络、合成数据及 mock；ROS 原生测试用实际 URDF／PyKDL，十分钟通信测量使用隔离 DDS 域和固定状态 mock plugin。

```bash
# 模型回归（脚本启用离线资源模式）
env VLA_PYTHON=/data0/miniconda3/envs/vla/bin/python \
  HF_HOME=/tmp/vla-execution-hf CUDA_VISIBLE_DEVICES='' \
  OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  bash scripts/test.sh -q -p no:cacheprovider --tb=short -rs

# ROS Python 原生回放；显式给测试目录
bash scripts/test_ros.sh tests/ros -q

# 真 ROS 通信，固定状态 mock，禁止替换成真实硬件 plugin
bash scripts/validate_ros_control.sh --duration 600

# 临时干净源码构建 wheel，核对文件清单并在仓库外分别验证 ROS／模型导入
# 不安装依赖或修改既有环境，也不复用可能残留已删除模块的 build/ 缓存
/data0/miniconda3/envs/vla/bin/python scripts/validate_wheel.py

# 停止采集后检查 staging；无 outcome 时只读
/data0/miniconda3/envs/vla/bin/python -m vla.cli.recover_episode \
  --dataset /path/to/new-dataset --staging /path/to/new-dataset/staging/JOURNAL_ID
# 显式发布匹配的持久前缀：追加 --outcome success 或 --outcome failure
```

2026-11-07 的十分钟测量结果：

| 指标 | 实测 | 验收 |
| --- | ---: | --- |
| 动作接受 | 30.000009 Hz | 30 Hz ±3%，通过 |
| 跟踪 | 99.998707 Hz | 100 Hz ±3%，通过 |
| 左／右 C++ 控制器 | 1000.000007 / 1000.000012 Hz | 1000 Hz ±3%，通过 |
| 接受周期 p99／最大 | 34.311 / 34.555 ms | 实测记录 |
| 跟踪周期 p99／最大 | 11.045 / 21.336 ms | 实测记录 |
| 首次诊断延迟 p99／最大 | 10.900 / 30.231 ms | 实测记录 |
| 左／右 mock 记录丢失 | 0 / 0 | 通过 |

报告：`/tmp/vla-ros-control-omhee899/report.json`；日志：`/tmp/vla-execution-ros-timing.log`。故障检查覆盖延迟复位、撤销代次／倒序／重复命令、复位确认丢失、过期观测、命令 watchdog、双臂部分发布失败和控制器进程退出。故障场景按正确停止验收，不要求持续运动。

全量回归、ROS 原生回放与 wheel 最终结果见本文件末尾的完成记录。测量覆盖调度器、协调器和真实 ROS 通信；不把它扩大为带真实相机、编码负载和 GPU 推理的完整采集频率保证。

尚待资源／现场验收：新格式真实 PI0.5 权重和 tokenizer 下的前后向、推理、GPU 延迟；真实相机采集／同步与长回合磁盘吞吐；实机控制连续性、接触／跟踪误差、双臂运动和任务成功率。它们均未计为通过。Actor 模型输出自身的轨迹连续性也不在本轮保证范围内。

## 完成记录（2026-11-07）

- 模型环境全量回归：708 passed、7 skipped，42.67 秒。7 个跳过中，1 个 ROS 专用模块另在 ROS Python 下通过，6 个依赖未提供的真实 PI0.5 资源，未计通过。日志：`/tmp/vla-execution-completion-final.log`。
- 补充候选请求的任务／相机形状拒绝断言并清理旧测试名称后，服务／协议针对性复验：53 passed，3.19 秒。日志：`/tmp/vla-execution-last-targeted.log`。
- ROS Python 原生回放：10 passed，2.38 秒；无硬件启动。日志：`/tmp/vla-execution-completion-native.log`。
- 干净 wheel：176 个 Python 文件与当前源码清单一致；仓库外导入通过。ROS 侧不导入 Torch，模型侧不导入 ROS；未安装 wheel 或改动既有环境。产物：`/tmp/vla-wheel-validation-jzuxqrtz/vla-0.1.0-py3-none-any.whl`；日志：`/tmp/vla-execution-clean-wheel.log`。
- 十分钟 ROS mock 测量沿用上表及原始报告；后续改动为旧入口清理、停止状态校验及测试／文档收尾，没有重新测量真实相机或 GPU 负载。
- 五个本地提交依次覆盖协议与执行器、候选块与 RTC、记录和恢复、Replay 与损失、集成验收。未推送、未合并 `main`；数据、权重、日志及构建产物均不入 Git。
