# VLA：Franka + PI0.5 + RLT

从 Evo-RLT 独立提取的 Franka 采集、训练和部署框架。采用官方 LeRobot 0.6.1，不包含 LeRobot fork、旧数据集、模型、tokenizer、回放数据库或训练结果。

## 本机使用

```bash
conda activate vla
cd /data0/sht/VLA
source scripts/model_env.sh
bash scripts/test.sh -q
```

本机环境已安装 wheel。编辑代码后重新安装 `python -m pip install --no-deps -e .`，或者重新构建 wheel；项目不会自动读取其他 LeRobot checkout。

## 从代码重新安装

```bash
conda env create -f environment.yml
conda activate vla
python -m pip install -r requirements.lock
python -m pip install --no-deps '.[lerobot,franka,test]'
source scripts/model_env.sh
```

`requirements.lock` 固定本次实际验证的 Python 依赖。需要复现本机 Conda 原生包时，用 `conda create -n vla --file conda-linux-64.lock` 替代第一行。模型环境为 Python 3.12；ROS 使用系统 Python 3.10。

## Web 采集与部署面板

```bash
python -m pip install -r requirements-web.lock
python -m pip install --no-deps '.[web]'
vla-web --demo --root /data0/sht/VLA
```

访问 `http://127.0.0.1:8765`。Demo 不连接硬件；实际会话取消 `--demo` 并配置设备和模型。支持单左臂／双臂、相机与可选遥测、采集及 VLA/RLT/自动部署。浏览器退出后任务继续运行。配置、软件暂停语义及限制见 [Web 面板说明](docs/WEB_PANEL.md)，离线证据见 [Web 验收](docs/web/VALIDATION.md)。

## 专用 ROS 工作空间

在单独终端操作，不激活模型环境：

```bash
cd /data0/sht/VLA
/usr/bin/python3 scripts/fetch_ros.py
bash scripts/install_ros_python.sh
bash scripts/build_ros.sh
source scripts/ros_env.sh
CHECK_ONLY=1 bash scripts/franka/start_teleop_single_left.sh
CHECK_ONLY=1 bash scripts/franka/start_teleop_duo.sh
```

系统前提是 ROS 2 Humble、MoveIt/ros2_control、Robotiq description/controllers、Joy、PyKDL，以及本机 `/opt/openrobots` 的原生依赖。Dynamixel SDK、串口库和 ROS 采集用的 NumPy/OpenCV/相机等依赖安装在 `ros_ws/python_deps`，版本固定在 `ros_ws/requirements-python.lock`；不改动系统 Python，也不引入模型 Conda 环境。版本快照见 `ros_ws/system-packages.lock`；第三方源码精确提交见 `ros_ws/dependencies.repos`。本轮没有安装或更改系统 ROS。

`CHECK_ONLY=1` 只校验配置和依赖。实际启动需要按任务配置设备、模型和工作空间；单臂 Franky 独占机械臂、夹爪与相机，ROS 侧只运行 Joy sidecar。双臂/GELLO 使用专用 ROS 控制器。

## 功能入口

| 功能 | 入口 |
| --- | --- |
| 单臂手柄采集 / 31D 采集 | `scripts/franka/start_gamepad_record_single_left.sh` / `start_gamepad_record_single_left_31d.sh` |
| 双臂采集 / GELLO | `scripts/franka/start_gamepad_record_dual.sh` / `start_teleop_duo.sh` |
| 特征处理 / SFT 划分 | `vla-extract-dataset-features` / `vla-prepare-franka-sft-split` |
| PI0.5 SFT / RLToken | `vla-train-franka-pi05-sft` / `vla-train-rl-token` |
| 离线缓存 / Actor-Critic | `vla-build-transition-cache-v2` / `vla-train-actor-critic` |
| Handover | `vla-build-handover-cache` / `vla-train-handover` |
| RTC 部署 / 采集客户端 | `vla-deploy-franka-pi05-rtc` / `vla-collect-franka` |
| SFT 服务 / RLT 服务 | `vla-serve-franka-sft` / `vla-serve-franka-rollout` |
| 在线 RLT / 恢复 | `vla-serve-franka-rollout --online-*`；恢复使用 `--online-resume` |
| 训练语义升级 | `python -m vla.cli.upgrade_rlt_training --help` |

详细用法见 [操作说明](docs/WORKFLOWS.md)、[迁移对照](docs/migration/FEATURES.md) 和 [验收记录](docs/migration/VALIDATION.md)。

## 数据集分层

`datasets/raw/` 存放人工原始采集，`sft/` 存放训练输入，`transitions/` 存放 transition cache，`rollouts/` 存放部署回合，`online/` 预留在线数据。目录骨架由 Git 管理，数据内容忽略。Web 自动预填相对输出路径，按采集配置调整；详见 [目录与命名说明](datasets/README.md)。

## 训练与评估产物

`outputs/` 按 `sft / rltoken / ac / handover / online / eval` 分层，建议使用 `<阶段>/<task_id>/run-<编号>/`。仅目录骨架和说明进 Git，实际权重、日志和报告忽略。用途、训练输出参数及与数据目录的边界见 [产物目录说明](outputs/README.md)。

## 配置、数据与 Git

设置 `VLA_ROOT` 指定可写工作目录；默认 Python 数据目录基于当前目录，源码脚本默认使用仓库根目录。安装包中的配置通过 `vla.paths.resource()` 定位，运行产物放在 `runtime/`、`datasets/`、`outputs/` 等忽略目录。

配置中的设备编号和网络地址是迁移时的示例值。归位默认关闭且不携带旧目标姿态，启用时必须提供通过校验的 10D 绝对动作目标。旧数据格式中的 `evo_rlt` 元数据字段保留，以维持格式兼容；额外字段同时保存到 `meta/vla_metadata.json`。

本地 `main` 使用独立历史，无 remote。本轮不发布。来源与哈希见 `docs/migration/`；0.5.1 基线保存在历史中，当前仅维护 0.6.1。

Python 项目沿用 Apache-2.0 声明；迁入 ROS 文件保留各自许可，其中 GELLO 包使用 MIT，个别驱动/launch 文件保留原 BSD 等许可。
