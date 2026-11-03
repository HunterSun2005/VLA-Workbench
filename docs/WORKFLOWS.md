# 操作说明

## 环境边界

模型终端先 `conda activate vla`，再 `source scripts/model_env.sh`。也可以显式设置 `VLA_PYTHON` 后 source。ROS 终端只 source `scripts/ros_env.sh`；此脚本清除 Conda/其他 ROS overlay 的路径，只加入 Humble、新工作空间与 `/opt/openrobots`。ROS 的 Python 依赖由 `scripts/install_ros_python.sh` 安装到专用目录，`ros_ws/python/sitecustomize.py` 在该启动环境中排除其他工作空间和全局 editable 导入器。两侧通过 `vla_rollout_protocol` 和 Unix socket 交互。

需要修改配置时，从安装包复制到工作目录，避免修改 site-packages：

```bash
python -c 'from pathlib import Path; from shutil import copyfile; from vla.paths import resource; p=Path("runtime/config"); p.mkdir(parents=True, exist_ok=True); copyfile(resource("configs/deployment/franka_single_left_pi05_rtc.yaml"), p/"rtc.yaml")'
```

迁移不附带 tokenizer、PI0.5 基础模型或 SFT/RL checkpoint。接入时分别提供本地绝对路径；程序会检查缺少的资源。没有对旧资产创建软链接。

## 数据处理和 SFT

单／双臂 GELLO、gamepad 采集和部署回合记录统一保存完整状态：单臂 50D、双臂 100D。动作仍为 10D／20D，部署策略输入仍为 10D／20D。旧 `_31d` 命令转到同一个完整状态采集器，不再生成独立 31D 数据。单臂 gamepad 保留成功保存、失败保存和丢弃（终端 y/f/x）。本项目不附带 force-VLA 训练。

字段、单位和训练提取示例见 [完整采集格式](DATA_COLLECTION_FORMAT.md)。已有 29D/31D/58D 数据保留不变、仍可离线提取；新采集使用新目录，禁止混格式续采。

可在 `vla-web` 的「数据处理」页预览各视角和逐帧数值、剔除完整回合并全局选择状态字段；结果写入新数据集。步骤见 [离线数据处理](DATA_PROCESSING.md)。

也可先运行 `vla-extract-dataset-features --help` 选择状态维度，再运行 `vla-prepare-franka-sft-split --help` 生成外部训练/验证划分。SFT 入口为 `vla-train-franka-pi05-sft`，支持 `--split-manifest`；LeRobot 参数采用 0.6.1 名称，例如 `--env_eval_freq=0 --dataset.eval_split=0`。

已划分数据不再自动划分；SFT 入口拒绝非零 `dataset.eval_split`。所有训练任务显式提供数据根目录、策略路径和输出目录。`scripts/franka/train_franka_pi05_sft_screens.sh` 保留多配置筛选能力，使用前配置新的 `SPLIT_ROOT` 与模型路径。

## RLToken、AC 和在线学习

1. `vla-validate-rl-token-inputs` 校验 SFT 与数据的特征、归一化和动作约定。
2. `vla-train-rl-token` 训练 RLToken，使用策略类型 `rlt_token`。
3. `vla-build-transition-cache-v2` 构建带 manifest、分片校验和 checkpoint 契约的缓存。
4. `vla-train-actor-critic --policy.type=rlt_ac --dataset.repo_id=<缓存绝对路径> --dataset.eval_split=0 ...` 训练 AC。缓存校验及显式语义升级保护保留；不通过普通 LeRobot dataset 工厂隐式识别缓存。
5. `vla-serve-franka-rollout` 提供部署与在线 RLT 服务。在线模式需要新的 `--online-replay-root` 和 `--online-dataset-root`，恢复使用 `--online-resume`，全回合模式和 handover 的参数见 `--help`。

## 部署与遥操作

SFT 服务使用 `vla-serve-franka-sft --checkpoint <SFT目录> --tokenizer-path <tokenizer目录> --socket-path <socket> --task <任务>`。该服务使用 SFT action queue；RTC 部署另用 `vla-deploy-franka-pi05-rtc`，RLT RTC 服务用 `vla-serve-franka-rollout --rtc-deployment-config <配置>`。

单臂 Franky 的手柄入口使用独立 ROS Joy 子进程，模型侧不能加载 ROS Python。双臂/GELLO 使用 `scripts/franka/start_teleop_duo.sh`；可通过 `ROS2_WS` 指向其他明确配置的专用工作空间，默认是当前 VLA 的 `ros_ws`。

手柄 Menu（“恢复竖直”）保持按键时的 XYZ，将末端恢复为竖直向下，并同时复位 X/B 控制的旋转：朝向以本次启用遥操作时记录的初始朝向为准。暂停后重新启用会重新记录参考姿态。View 则同时恢复初始 XYZ。在线接管沿用其已捕获的初始朝向；若显式配置了部署 home 位置或姿态，Menu 仍优先使用对应的 home 配置。自动恢复期间可通过手动运动输入取消。

ROS 运行 YAML 和日志写入 `runtime/`，不会改写包的安装目录。`vla_bringup` 提供 `teleop.launch.py`，接收绝对路径的 `robot_config_file`、`config_file` 和 `control_owner`。

## 终端显示、日志与确认

终端默认保留简洁事件与运行状态：TTY 最多每秒原位刷新一次；重定向输出时只有普通文本事件。任务进入检查、连接、就绪、录制、保存或故障时显示对应事件；高频手柄数值、编码器详情、状态范围及完整堆栈保留在诊断文件中。输出队列有界；输出拥堵可能丢弃低优先级消息并报告数量，控制循环不等待终端或磁盘 IO。

Python CLI 默认创建 `runtime/terminal/<时间-入口-唯一标识>/`，保存 `terminal.log`，及适用的 `joy.log`、`bridge.log`、`rollout.jsonl`、`rtc.jsonl`。ROS 终端启动器同样创建持久目录，其 `launcher.log` 记录启动事件，`ros/` 保留各子进程日志。退出后诊断保留，socket 等临时资源正常清理；可用 `VLA_TERMINAL_LOG_DIR` 指定会话目录，已有显式 trace 路径继续优先使用。

- GELLO 终端保留启动即跟随；不新增启用确认，`s` 仍是切换 fine 模式。
- Gamepad 保留 `--execute` 和待命启用；非 SFT Franky 不再要求重复输入 `ARM`。健康、中立、独占及动作契约检查保留。
- 保存和模式切换直接执行。显式丢弃、删除最近已保存回合、普通退出将丢失未保存回合时，提示一次 `y` 确认、`n/Esc` 取消；10 秒超时或回合状态改变自动取消。等待确认期间控制监控继续。
- `d` 暂停立即执行，原先会丢弃当前回合的采集路线仍会丢弃；本轮没有新增冻结回合。GELLO 正常退出保留原自动保存行为。
- Ctrl-C、EOF、故障与断连不等待确认；ROS 启动器 Ctrl-C 走原有序清理，Web 仍通过 SIGTERM 退出。停止优先级不会被同批按键中的丢弃确认遮挡。

## 后续真实资产验收

提供新的本地模型和数据后，设置 `VLA_TEST_SFT_PI05_DIR` 运行需要 processor 文件的测试；基础 PI0.5 权重测试按测试说明指定本地模型。再比较模型输出、训练恢复、RTC 时序和真实机器人行为。当前模拟/离线验收不能替代这些检查。
