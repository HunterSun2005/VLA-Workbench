# ROS 采集、训练与部署

先完成 README 中的独立模型环境与 ROS 工作空间安装。开发期间不得启动硬件；下面带 `--execute` 的命令仅供现场授权后使用。

## 采集

```bash
source scripts/model_env.sh
python -m vla.cli.ros_session --arms single_left --source gamepad --action-space joint --check-only
python -m vla.cli.ros_session --arms dual --source gello --action-space tcp --check-only
# 现场执行示例：
python -m vla.cli.ros_session --arms single_left --source gamepad --action-space joint \
  --dataset-root datasets/raw/joint_task --task "Complete the task." --execute
```

终端 `e` 启用、`p` 暂停、`r` 录制、`s` 保存、`d` 丢弃、`c` 恢复、空格接管、`k` critical、Tab 下一阶段、`t` 切换任务、`q` 退出。丢弃和带未保存回合退出使用非阻塞 y/n 确认；停止优先。`--tasks "Task A" "Task B"` 提供多个可选任务；未显式覆盖时保留 YAML 的动作空间、输入来源和数据目录。`--no-preview` 关闭独立预览窗口。硬件配置通过 `--hardware-config` 传入；不指定时使用本会话保存的设备预设。旧通用 shell 名称调用同一入口。工作空间和初始姿态来自 ROS YAML，Web 可配置工作空间并持久记录／恢复姿态。

## 数据和模型契约

Web 的数据处理页可筛选完整回合和命名状态字段，选择 joint／TCP 导出。跨空间导出须确认派生数据；缺少原始命令则拒绝。状态筛选保留独立锚点。absolute／chunk_relative 是模型表示，磁盘仍为 absolute。

先完成训练集划分，再为实际训练集生成统计量与契约：

```bash
vla-action-contract --dataset datasets/sft/task_train --output outputs/sft/task-actions.json \
  --representation chunk_relative --chunk-length 50 --frame-stride 1
vla-train-franka-pi05-sft --policy.type=vla_pi05 \
  --policy.action_contract_path=outputs/sft/task-actions.json --policy.chunk_size=50 \
  --dataset.repo_id=local/task --dataset.root=datasets/sft/task_train \
  --output_dir=outputs/sft/task_run --dataset.eval_split=0
```

其余模型权重、tokenizer、设备与训练超参数按本地训练配置传入。`--allow-derived` 必须显式写入契约生成和数据混合命令。不要复用数据、采样规则或动作空间不同的统计量。

RLToken 使用 `vla-train-rl-token --policy.type=rlt_token`，指定新的 SFT checkpoint。缓存通过 `vla-build-transition-cache-v2` 构建，Actor-Critic 通过 `vla-train-actor-critic` 训练；它们继承 SFT 动作契约，不在磁盘写相对动作。RLT 的稀疏目标使用实际控制帧折扣；缓存起点间隔参数不表示动作目标采样间隔。详见 [动作契约](ROS_ACTION_CONTRACT.md)。

## 部署与在线训练

`lingbot-dev` 另支持独立 LingBot V2 串行服务：通过 `--model-backend lingbot_v2` 选择，CLI/Web 共用，完整 50 步执行后才请求下一块。它不支持 RTC/RLT/在线训练，不要求本地 PI05 模型资产。服务启动、外部物理契约及预检见 [LingBot 部署](LINGBOT_DEPLOYMENT.md)。以下默认 PI05 路线保持不变。

```bash
# 离线配置检查，不启动硬件
vla-deploy-franka-pi05-rtc --arms single_left --policy-path outputs/sft/task_run/pretrained_model \
  --tokenizer-path /absolute/local/tokenizer --check-only
# 现场部署在相同命令上加 --execute，并设置硬件配置、输出目录和任务。
```

部署从 checkpoint 读取动作空间和训练表示，不能在运行中改变解释。终端 RLT 推理可用 `--policy-kind rlt --mode rlt`；自动切换使用 `--mode auto` 并用 `--handover-path` 提供相应 checkpoint。策略服务运行在模型 Python，ROS runner 在系统 Python。`vla-serve-franka-sft` 提供 PI05 RTC；`vla-serve-franka-rollout` 提供 RLT、接管、RTC 和已有 `--online-*` 训练服务。

在线训练仍由终端启动服务；新运行使用独立 dataset/replay 路径，恢复使用服务器 `--online-resume`。客户端用 `vla-collect-franka --config ROS_YAML --online --execute` 连接 YAML 的 socket，要求服务已声明 online 契约。客户端保存回合后等待所有控制器停止确认，再通知训练服务处理；停止失败不能宣称恢复。

所有实机操作的验收范围见 [动作契约末尾](ROS_ACTION_CONTRACT.md#离线与实机边界)。

## 执行耗尽与保存恢复

策略耗尽进入等待恢复，使用终端 `e`／Web 启用明确恢复；迟到结果不会恢复运动。存储阻塞保留回合，先重试保存或显式丢弃，存储恢复不会启动运动。重启后的 staging 可用 `vla-recover-episode` 检查及恢复。只支持 schema 5、新模型契约和 Replay 4；详见 [协议、数据及验证命令](EXECUTION_RTC_RLT.md)。

恢复检查返回 `durable_frames`、`complete`、`incomplete_tail` 以及 `issues`（失败文件和原因）。清单、数组、事件或视频无法验证时，在首个失败分段处停止，保留此前完整的数据；不会跳过坏段拼接后面的样本。损坏的回合身份／schema 仍直接拒绝。只有显式指定 `--outcome` 才发布已验证前缀，且不会补造终点或恢复运动。

后台视频单帧写入、编码结束和 `ffprobe` 帧数检查默认各有 30 秒超时，最终视频拼接为 120 秒。编码器持续读取错误输出并最多保留 64 KiB 诊断。超时进入存储阻塞，保留 staging 和未完成记录；修复负载或存储问题后可重试，不能以超时为由删除回合。这些是媒体子进程的期限，不承诺操作系统故障下的文件系统 I/O 上界。

## 锁文件报错

运行 `python3 scripts/check_locks.py` 查看真实锁状态。Git index 锁确认废弃后可用 `--clear-stale-index` 保守清理；运行时 `flock` 文件保留是正常设计，不应直接删除。完整说明见 [锁文件诊断与恢复](LOCKS.md)。
