# 原生动作重放：真实 FR3 运动学

使用系统 Python 3.10、工作空间固定版本 franka_description 的 FR3 URDF，以及生产 PyKDL、CartesianServo、ControlCoordinator。没有启动 ROS 节点或连接硬件。

```bash
bash scripts/test_ros.sh -q tests/ros/test_native_replay.py
```

10 项测试通过（2026-11-06）：单／双臂 × joint／TCP × GELLO／Gamepad，分别将生产入口接受的动作写入 float32 Parquet，再读回并经 absolute／chunk_relative 变换交给独立策略控制器。每次对比 150 个 100 Hz 跟踪周期，包含非零滤波历史、连续观测、30 Hz 动作接受、旋转和中途代际重置。关节目标容差 1e-5 rad、夹爪 1e-6；另检验 TCP 工具偏置、工作空间裁剪与关节限位。

负向测试确认改变滤波历史或使用错误锚点会产生可检出的差异。生产 CartesianServo 仅从采集器机械抽取至控制模块，算法与参数没有调整。简单协调器测试保留为快速边界检查。

本结果证明相同测量和时间条件下的提交目标一致，不证明真实机器人轨迹、接触动力学或 ROS 网络时序一致。系统 Python 缺少 PyKDL/URDF 依赖应修复测试环境；模型 Python 下该 ROS 专用测试会跳过，不计作通过。
