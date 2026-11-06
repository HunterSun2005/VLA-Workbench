# ROS 通信与控制器离线验收

本次验证使用真实 ROS 2 Humble DDS、`controller_manager`、FR3 URDF、PyKDL 和项目的 C++ 关节阻抗控制器。硬件接口仅允许 `vla_control_validation/TimingMockSystem`，没有连接 Franka、夹爪、GELLO 或相机。模型 Python 与系统 ROS Python 保持隔离。

## 复现

在仓库根目录执行，依赖为已构建的本仓库 ROS 工作空间与系统 Humble；不安装新依赖：

```bash
bash scripts/build_ros_validation.sh
bash scripts/validate_ros_control.sh --duration 600
```

专用构建输出为 `/tmp/vla-ros-validation-$UID/{build,install,log}`，不会替换 `ros_ws/install`。程序输出 `ARTIFACTS=...` 和 `REPORT=...`，报告、节点日志、临时配置和原始时序文件都保存在该独立 `/tmp/vla-ros-control-*` 目录，不进入 Git。短冒烟可用 `--duration 3`；默认测量 600 秒，最长 900 秒，故障注入和节点启动时间不计入测量窗口。

测试选取 170–199 中未被本验证工具占用且不同于调用环境的 ROS domain，启用 `ROS_LOCALHOST_ONLY`；发布前检查该 domain 没有其他节点，发现冲突立即失败。请勿在同一 domain 启动其他应用。退出时只清理本次创建的进程组，不查杀其他 ROS 进程。

C++ 行为测试独立运行；选择一个未用于硬件的 ROS domain：

```bash
source scripts/ros_env.sh
source /tmp/vla-ros-validation-$UID/install/local_setup.bash
ROS_DOMAIN_ID=169 ROS_LOCALHOST_ONLY=1 \
  /tmp/vla-ros-validation-$UID/build/franka_fr3_arm_controllers/test_joint_impedance_controller \
  --gtest_output=xml:/tmp/vla-controller-tests.xml
```

`BUILD_TESTING=ON` 仅强制行为测试所需依赖。原有源码 lint 改为显式 `CHECK_LINT=ON`，clang-tidy 仍通过 `CHECK_TIDY` 启用；本次未将 lint 当成已通过的验收项。

## 覆盖的链路

```text
真实 ControlCoordinator / CartesianServo
  → 与采集器共用的 JointState 命令封装
  → localhost DDS（独立左右臂命名空间）
  → controller_manager + 生产 JointImpedanceController
  → TimingMockSystem 的 effort 接口
  → 控制器诊断与 joint_state_broadcaster
  → coordinator 的观测、代次确认和故障处理
```

模拟器保持固定的有限关节状态，不模拟机械动力学。夹爪只保留动作中的标量，没有启动 Robotiq action server。没有运行采集器的相机、录制器或策略模型；这是实际 ROS 控制通信链路验收，不是完整采集会话或硬件验收。

本机测量使用 `5.15.0-1113-realtime` 内核，两个 manager 实际获得 FIFO 优先级 50；验证配置未指定生产启动配置中的优先级 98。控制器的增益、低通、死区、速度和力矩变化限制读取生产 `controllers.yaml`。结果不能直接外推到其他调度配置。

验证脚本的定时器异常转故障规则与采集器一致；消息编码／诊断解码直接复用 `vla.control.ros.transport`。C++ 控制算法未修改。

| 边界 | 验证内容 |
| --- | --- |
| TCP / joint | TCP 目标经过生产 CartesianServo；持续测量使用原生 joint 目标 |
| 双臂切换 | 单臂 reset 确认延迟时，两臂均不能进入 active；缺失确认超时后故障锁存 |
| 消息有效性 | 旧代次、过期／未来时间戳、倒序 ID、重复序列号不能覆盖当前目标 |
| 输入中断 | 过期观测撤销双臂控制；停止命令后 C++ 0.5 秒看门狗锁存，恢复须新代次 |
| 部分失败 | 第二臂发布异常撤销两臂控制；一侧 controller_manager 退出后另一侧进入故障处理 |
| C++ 边界 | 非有限／错误维度、未授权启动、重复包、看门狗、重置及力矩变化限幅；6 个无效参数组合 |

## 时序指标的含义

- 动作接受：coordinator 返回有效接受记录的单调时钟时间，目标 30 Hz。
- 跟踪：完成一次双臂提交的单调时钟时间，目标 100 Hz。
- 控制器循环：每臂 mock hardware `write()` 的 C++ steady clock，目标 1000 Hz。它包含 manager 的 read/update/write 周期，不等于 FCI 数据包频率。
- 首条诊断延迟：首次提交该代次／ID 后，Python 首次收到对应诊断的时间。包含 100 Hz 诊断定时器、DDS 和 Python 调度延迟。
- 双臂诊断观察偏差：同一代次／ID 的首条左右诊断接收时间差；不是机器人实际启动动作的偏差。

C++ `controller_state.position` 是命令低通后的关节目标，位于关节目标速度限制之前，不能当作最终控制器设定值或实测关节状态。单臂没有收到某个 ID 的诊断也不等于该目标未执行，诊断是最新值快照。

频率验收阈值固定为目标平均值的 ±3%，不因实测结果调整；脚本在故障断言失败、缺失时序文件或频率不合格时返回非零。周期分位数、最大值、诊断延迟、CPU 和 RSS 如实记录，没有用均值掩盖尾延迟。

时序缓存上限 1,300,000 个 int64／臂，启动预留容量，控制循环不做文件写入或扩容；退出后统一保存。Python 保存本次有限时长测量的命令与诊断样本，因此其 RSS 会随测量样本增加；这不是长期服务内存稳定性测试。报告同时记录缓存丢样数量。

## 结果与限制

2026-11-06：专用 overlay **2 packages finished**；C++ **8 passed**（2 项控制行为、6 项无效配置）；真实 ROS 集成脚本的 **9 项检查通过**。600 秒双臂测量退出码为 0，频率均通过 ±3% 验收。结构化结果、依赖版本和被测源码哈希见 [ROS_CONTROL_BENCHMARK.json](ROS_CONTROL_BENCHMARK.json)。

| 环节 | 平均频率 | 周期 p95 | 周期最大值 |
| --- | ---: | ---: | ---: |
| 动作接受 | 29.9999 Hz | 40.037 ms | 45.356 ms |
| 双臂跟踪提交 | 99.9954 Hz | 10.328 ms | 23.033 ms |
| 左臂 manager 循环 | 1000.0000 Hz | 1.008 ms | 1.074 ms |
| 右臂 manager 循环 | 1000.0000 Hz | 1.008 ms | 1.075 ms |

本脚本在 100 Hz 调度中尝试提交目标，30 Hz 接受间隔因此主要落在 30／40 ms；不是严格等间隔 33.33 ms。实际采集器与相机、模型共同运行时需重新测量。

首条诊断延迟 p50／p95／最大值为 **5.511／10.441／36.080 ms**；双臂诊断观察偏差为 **3.063／7.180／32.308 ms**。共收到 36,000 条首次命令诊断，对应 18,000 个双臂接受目标。两侧时序缓存均无丢样。

| 进程 | 平均 CPU（单核=100%） | RSS 开始 → 结束 | RSS 峰值 |
| --- | ---: | ---: | ---: |
| Python 验证桥接 | 16.45% | 97.96 → 113.27 MiB | 113.27 MiB |
| 左臂 manager | 2.45% | 397.84 → 396.91 MiB | 397.84 MiB |
| 右臂 manager | 2.48% | 398.51 → 397.59 MiB | 398.51 MiB |

10 分钟结束后补充了空 domain 检查和禁用 Python `-O` 的启动检查，并用最终启动脚本重新执行短冒烟；测量循环与被测生产控制器没有改变。构建仍有上游头文件弃用等警告，FR3 的扩展 URDF 属性也会产生解析器提示；未将这些警告过滤成通过证据。

完整 Python 回归结果见 [ROS_VALIDATION.md](ROS_VALIDATION.md)。原生动作的 Parquet 往返与策略重放单独见 [ROS_NATIVE_REPLAY_VALIDATION.md](ROS_NATIVE_REPLAY_VALIDATION.md)。

真实 FCI 网络、实时内核负载、机器人跟踪误差、接触稳定性、双臂物理同步、夹爪运动、相机录制与 GPU 推理并行负载仍需现场验收。本次不修改重叠 action chunk 的统计计算语义。
