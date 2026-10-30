# 操作说明

## 环境边界

模型终端先 `conda activate vla`，再 `source scripts/model_env.sh`。也可以显式设置 `VLA_PYTHON` 后 source。ROS 终端只 source `scripts/ros_env.sh`；此脚本清除 Conda/其他 ROS overlay 的路径，只加入 Humble、新工作空间与 `/opt/openrobots`。ROS 的 Python 依赖由 `scripts/install_ros_python.sh` 安装到专用目录，`ros_ws/python/sitecustomize.py` 在该启动环境中排除其他工作空间和全局 editable 导入器。两侧通过 `vla_rollout_protocol` 和 Unix socket 交互。

需要修改配置时，从安装包复制到工作目录，避免修改 site-packages：

```bash
python -c 'from pathlib import Path; from shutil import copyfile; from vla.paths import resource; p=Path("runtime/config"); p.mkdir(parents=True, exist_ok=True); copyfile(resource("configs/deployment/franka_single_left_pi05_rtc.yaml"), p/"rtc.yaml")'
```

迁移不附带 tokenizer、PI0.5 基础模型或 SFT/RL checkpoint。接入时分别提供本地绝对路径；程序会检查缺少的资源。没有对旧资产创建软链接。

## 数据处理和 SFT

单臂手柄采集、31D 采集和双臂采集脚本分别保留。31D 是采集接口，本项目不附带 force-VLA 训练。

先运行 `vla-extract-dataset-features --help` 选择状态维度，再运行 `vla-prepare-franka-sft-split --help` 生成外部训练/验证划分。SFT 入口为 `vla-train-franka-pi05-sft`，支持 `--split-manifest`；LeRobot 参数采用 0.6.1 名称，例如 `--env_eval_freq=0 --dataset.eval_split=0`。

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

## 后续真实资产验收

提供新的本地模型和数据后，设置 `VLA_TEST_SFT_PI05_DIR` 运行需要 processor 文件的测试；基础 PI0.5 权重测试按测试说明指定本地模型。再比较模型输出、训练恢复、RTC 时序和真实机器人行为。当前模拟/离线验收不能替代这些检查。
