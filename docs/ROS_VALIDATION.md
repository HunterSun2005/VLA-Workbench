# ROS 统一控制离线验收（2026-11-06）

2026-11-07 的协议 v2、schema 5、Replay 4 和十分钟 mock 测量更新见 [统一执行闭环](EXECUTION_RTC_RLT.md)。以下保留前一轮历史记录。

开发位置：`/data0/sht/VLA-Workbench`。初次验收使用本地 `main`，后续工作使用 `ros-dev`；未配置或发布远程。此报告只记录离线验证，不代表机器人现场验收通过。

## 已交付

- 动作契约与 NumPy 数学层：joint 8D／臂、TCP 10D／臂，absolute／chunk_relative，独立原始实测锚点、rot6d 矩阵组合、夹爪绝对开度。
- ROS 公共控制入口：30 Hz 目标接受、100 Hz 跟踪、现有 C++ 1000 Hz 关节阻抗控制；代际、命令 ID、重置确认、超时故障和控制器诊断。
- 绝对命令记录及派生导出：保持 50D／100D 实测状态；状态筛选保留锚点；不使用测量或任意 IK 解补造原始命令。
- SFT、RLToken、Actor-Critic、RTC 与在线 RLT 适配：请求锚点锁存、绝对执行队列、跨锚点前缀重编码、有效帧损失遮罩、稀疏目标的物理时间折扣与末尾 padding。
- Web／终端共用 ROS 生命周期；单／双臂、GELLO／gamepad、预览、姿态、接管与保存；旧 Franky 驱动、采集器、独占串口夹爪路线和依赖删除。旧 YAML 明确拒绝。

前四阶段提交：`ce89bc9`、`4d701aa`、`bfadc72`、`6e22d1b`；第五阶段提交包含本文、操作文档和现场验收清单。

## 验证结果

| 检查 | 结果 | 限制 |
| --- | --- | --- |
| 全量离线 pytest | **678 passed，6 skipped，0 failed**；34.11 s | 合成数据、模拟设备、小型网络；未加载真实机器人策略权重 |
| ROS 工作空间编译 | **12 packages finished**；2 min 11 s | 仅编译；未启动节点、机器人或相机 |
| ROS Python 导入 | 系统 Python 3.10；collection imports passed；`torch_loaded=false` | 验证解释器隔离，不验证真实 ROS 通信时序 |
| wheel 构建 | `vla-0.1.0-py3-none-any.whl` | 无依赖安装；未改变既有环境 |
| wheel 独立冒烟 | 临时解包、阻止 Franky 导入、8 种 CLI check-only 组合、Web demo 首页与 bootstrap 均通过 | 不创建真实硬件会话 |
| Web JavaScript | panel.js、processing.js 语法检查通过 | 交互行为由本地 HTTP／模拟 backend 测试覆盖；未做真实相机浏览器验收 |
| Git 与文件边界 | main，无 remotes，无数据／权重／第三方源码／构建日志入库 | 历史迁移清单保留旧路径作为来源记录 |

一次复跑曾在 `test_writer_process_exit_releases_lease` 的子进程启动握手处超过 30 秒；未调整断言或超时时间，单独复跑与随后完整回归均通过。dataset wrapper 序列化与训练集划分校验通过 13 项动作适配定向测试，随后纳入最终完整回归：spawn worker 可以反序列化；不允许计算统计量后再隐式划分验证集。

完整回归复现命令：

```bash
VLA_ROOT=$PWD VLA_PYTHON=/data0/miniconda3/envs/vla/bin/python \
HF_HOME=/tmp/vla-workbench-test-hf PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
bash scripts/test.sh -q -p no:cacheprovider -ra
```

本地 socket／子进程测试需要允许本地 IPC。测试脚本启用 Hugging Face／Transformers 离线模式。6 个跳过项为 5 项需要 `VLA_TEST_SFT_PI05_DIR` 的真实 processor 检查，以及 1 项需要本地 PI0.5 基础 checkpoint 的检查，均不计为通过。37 条警告来自 PyTorch nested tensor、Starlette 弃用提示和一个显式 legacy cache 兼容性测试。

ROS 编译使用 `ros_ws/dependencies.repos` 声明的精确上游版本。依赖仅放在忽略的 `ros_ws/src/third_party`；ROS Python 依赖只安装到本仓库新建的 `ros_ws/python_deps`。模型环境与系统环境没有安装／升级包，未修改迁移源仓库或正在采集的任务。ROS 构建存在上游编译／CMake 警告，构建退出码为 0。

本次临时日志：`/tmp/vla-final-offline-verified.log`、`/tmp/vla-ros-build.log`、`/tmp/vla-final-wheel.log`、`/tmp/vla-wheel-smoke.log`；它们未提交到 Git。

## 现场待验收

下列项目须另行授权、逐项记录，不能从离线结果推断：

1. 30／100／1000 Hz 实际时序、诊断延迟、双臂同步、丢帧与控制器重置确认。
2. 原生 joint／TCP 示教标签重放、机器人跟踪误差、接触稳定性和已有力矩／关节限速保护。派生空间数据不属于原生链路重放保证。
3. GELLO 跟随、gamepad Normal／Fast、恢复竖直／初始姿态／持久姿态、工作空间边界和夹爪松手保持。
4. 实际 RealSense 型号与序列号选择、side camera、多视角预览与录制帧率。
5. 实际 GPU 模型推理、RTC 延迟与截断、接管／归还期间旧请求失效、暂停及故障恢复。
6. 在线 RLT 保存后训练、训练中断与进程重启恢复、稀疏目标保持和相应奖励时序。
7. 四种动作空间／训练表示组合的真实训练收敛、模型效果与任务成功率。

操作命令见 [WORKFLOWS.md](WORKFLOWS.md)，数据与数学约定见 [ROS_ACTION_CONTRACT.md](ROS_ACTION_CONTRACT.md)。

## 后续补充：真实运动学与 ROS 模拟硬件通信

本节是上述初次基线的追加验收，不将离线模拟结果替代现场验收。

- 原生动作重放：系统 Python 3.10 下 **10 passed**，真实 FR3 URDF／PyKDL、生产 CartesianServo、单／双臂、joint／TCP、GELLO／gamepad、float32 Parquet 往返与策略侧 absolute／chunk_relative 恢复；详见 [原生重放报告](ROS_NATIVE_REPLAY_VALIDATION.md)。
- 模型 Python 完整回归：**678 passed，7 skipped，37 warnings**，28.88 秒。新增跳过项是 ROS 专用测试模块，已在系统 Python 单独验证；其余 6 项仍需真实模型资源。
- ROS 通信：专用 `/tmp` overlay 构建生产控制器及固定状态 mock plugin；使用真实 DDS、controller_manager 验证控制边界与故障。执行方式、指标定义和 10 分钟结果见 [通信验收报告](ROS_COMMUNICATION_VALIDATION.md)。

没有连接硬件或改变现有环境、原仓库和运行任务；未修改 action chunk 的统计计算。
