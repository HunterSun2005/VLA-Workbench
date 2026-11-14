# 官方 Franky 速度控制环境

采集、部署、GELLO 与保存点回位统一接收绝对逻辑 TCP 位姿，由官方 **Franky 2.0.0 Release + libfranka 0.21.2** 执行。ROS 负责唯一的 30 Hz 动作接受、执行队列及记录；独立 worker 在固定 100 Hz 时隙根据实测位姿生成速度，Franky 负责原生运动生成。模型环境不加载控制 binding。

## 固定依赖

| 项目 | 来源和身份 |
|---|---|
| Franky Release | [v2.0.0](https://github.com/TimSchneider42/franky/releases/tag/v2.0.0)，源码 `e7e63c7c7118f427ad33211500b33b81b4011bb4` |
| worker wheel | `franky_control-2.0.0+libfranka.0.21.2-cp310-cp310-manylinux_2_28_x86_64.whl` |
| wheel SHA-256 | `a057db2e1f7faeac491e110516a00affccaccaca8a08ced70799e7e2b8ffdd7e` |
| ROS libfranka | [0.21.2](https://github.com/frankarobotics/libfranka/releases/tag/0.21.2)，`9f9304ec0ac897eff3219a67f612b959948535e2` |
| common 子模块 | `2e090a65e51c94a98f9fc82b6ec3d1aa54d0f85f` |

Franky 与 libfranka 的版本号独立。使用官方带本地版本后缀的 wheel，不替换其中动态库，不回退到旧 fork。worker Python 3.10 独立 venv 固定 NumPy 1.26.4、SciPy 1.15.3、websockets 16.0；保留现有 ROS Python、模型 Python 和系统安装。

## 无硬件安装与构建

在仓库根目录执行以下命令；均不创建真实 Robot、不启动设备：

```bash
/usr/bin/python3 scripts/install_franky_worker.py
/usr/bin/python3 scripts/fetch_ros.py
export VLA_ROS_WORKSPACE="$(/usr/bin/python3 scripts/fetch_ros.py --print-workspace)"
bash scripts/build_ros.sh
unset VLA_ROS_WORKSPACE
PYTHONPATH="$PWD/src" runtime/franky/current/bin/python -m vla.control.franky.worker_main --check-only
bash scripts/test_ros.sh tests/ros -q
```

安装器已通过共享的干净环境函数执行同一 worker 预检。

安装器校验 wheel 哈希，在 `runtime/franky/envs/<lock hash>` 新建环境并预检，成功后切换 `runtime/franky/current`。源码下载器按清单在 `runtime/ros/workspaces/<identity>` 准备全新第三方归档；本仓库 ROS 包使用源码链接。构建不复用旧 build/install，完整重编译所有相关包，链接检查成功后才切换 `runtime/ros/current`。0.21.2 本身支持归档版本识别，不再应用旧 libfranka 打包补丁。

Web、CLI、ROS shell、构建和验证共用工作区解析器。`VLA_ROS_WORKSPACE` 可明确选择仓库内受管理的新工作区；旧 `ros_ws/install` 和外部 overlay 不自动继承。构建依赖仍使用已存在的 Eigen、Poco、Pinocchio、TinyXML2、fmt 等；缺失时报告失败，不修改已有环境。

worker 在创建 Robot 前验证官方包版本、实际加载唯一的 0.21.2 动态库、NumPy/Affine/Twist 转换与速度运动构造。ROS 检查硬件接口、semantic components、控制器与广播器的动态链接均指向所选工作区。两者分进程持有各自 0.21.2 产物，分别记录哈希，不要求二进制哈希相同。

## 控制参数与迁移

原 `control.compliance`、Web `compliance`、`tracking_*` 静态积分补偿和 Tracker 专属参数已移除，旧配置明确拒绝。改为：

```yaml
control:
  velocity_control:
    max_linear_speed: 0.05
    max_angular_speed: 0.20
    tracking_tau_s: 0.15
    dynamics_factor: 0.10
    target_smoothing_enabled: false
    velocity_smoothing_enabled: false
    target_smoothing_alpha: 0.25
    velocity_smoothing_alpha: 0.30
```

策略／GELLO 持续跟踪绝对目标。可选目标平滑使用位置及 SO(3) 插值；可选速度平滑包含角速度并按实际周期换算系数。两项默认关闭，独立于 SDK 的 `limit_rate=False`、`cutoff_frequency=100 Hz`。CLI 支持 `--target-smoothing`、`--no-target-smoothing`、`--velocity-smoothing`、`--no-velocity-smoothing`；Web 面板提供独立开关和 worker 确认状态。

手柄按实测位姿每接受时隙重新锚定，释放的通道不追赶累积参考。Normal XY/Z 为 0.02/0.01 m/s，倾斜/绕轴为 0.15/0.30 rad/s；Fast 只放大 XYZ 五倍，0.3 s 切换。X/B 是 TCP 工具轴旋转，不是关节 7 直接控制。保存点和快捷回位提交绝对最终目标，100 Hz 伺服以原反馈减速算法接近，保存点上限 0.03 m/s、0.15 rad/s，实测 0.5 mm/2° 内完成，30 s 超时。

统一逻辑 TCP 仍为 `flange_aligned_tcp_v1`：原点是物理 TCP，安装旋转只改变轴向表达。SDK Twist 使用 base 轴，`ee_frame` 为 identity，不重复施加 45° 或平移偏置。

目标先锁存，固定伺服时隙首次提交 SDK 成功后 ACK；ACK 不表示物理到位。每次新建运动实例，符合 Franky 2.0.0 单次使用约束。正常停止使用 `CartesianVelocityStopMotion`；停止失败锁存故障。reset 仅就绪，不启动运动；recover 不启用控制，必须再显式启用。health、内部伺服、重连、存储恢复均不能续期目标或恢复运动。

记录仍使用 schema 5，backend 为 `franky_cartesian_velocity`。动作保存被接受的绝对目标，`velocity_servo` 审计保存每次实测、派生速度、提交时间和代次/序号/slot。缓冲有界、消费确认，阻塞停止且保留未提交事件。历史 `diagnostic.tracker` 列保留 schema 5 字段名，其活动标志在新 backend 下表示速度伺服已就绪。控制身份改变后必须使用新数据目录，不续采旧 Tracker 数据；历史数据与姿态库不改写。

## 无硬件验收

```bash
export VLA_VALIDATION_ROOT=/tmp/vla-velocity-validation-unique
bash scripts/build_ros_validation.sh
bash scripts/validate_ros_control.sh --duration 600 --lingbot-rtc --franky-velocity
```

harness 使用独立 DDS domain、固定状态 mock ROS controller，以及官方 wheel 的真实运动构造与假 Robot 边界；记录 30 Hz 接受、100 Hz worker 提交和 1000 Hz mock controller 的独立频率。它不运行真实 FCI，不证明实机跟踪精度、手感、相机吞吐、GPU 时延或任务成功率。实际启动仍须使用 CLI 的显式 `--execute` 或 Web 的启动操作；本次开发不操作硬件。
