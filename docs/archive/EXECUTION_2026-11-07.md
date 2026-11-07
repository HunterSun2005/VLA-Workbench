# 统一执行闭环首次验收（2026-11-07）

> 历史快照：仅代表文中日期、版本和范围；旧命令及依赖状态不作为当前操作指引。当前行为见[文档导航](../README.md)，当前结果由[自动化验收](../ACCEPTANCE.md)生成。

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

## 历史：首次完成记录（2026-11-07）

以下数量、产物路径及提交状态仅描述首次验收，不代表后续提交。当前结果由 `scripts/validate.py` 生成，绑定 commit、工作区状态和源码摘要，不以本文的旧数量作为通过标准。

- 模型环境全量回归：708 passed、7 skipped，42.67 秒。7 个跳过中，1 个 ROS 专用模块另在 ROS Python 下通过，6 个依赖未提供的真实 PI0.5 资源，未计通过。日志：`/tmp/vla-execution-completion-final.log`。
- 补充候选请求的任务／相机形状拒绝断言并清理旧测试名称后，服务／协议针对性复验：53 passed，3.19 秒。日志：`/tmp/vla-execution-last-targeted.log`。
- ROS Python 原生回放：10 passed，2.38 秒；无硬件启动。日志：`/tmp/vla-execution-completion-native.log`。
- 干净 wheel：176 个 Python 文件与当前源码清单一致；仓库外导入通过。ROS 侧不导入 Torch，模型侧不导入 ROS；未安装 wheel 或改动既有环境。产物：`/tmp/vla-wheel-validation-jzuxqrtz/vla-0.1.0-py3-none-any.whl`；日志：`/tmp/vla-execution-clean-wheel.log`。
- 十分钟 ROS mock 测量沿用上表及原始报告；后续改动为旧入口清理、停止状态校验及测试／文档收尾，没有重新测量真实相机或 GPU 负载。
- 初次验收形成五个本地提交，依次覆盖协议与执行器、候选块与 RTC、记录和恢复、Replay 与损失、集成验收；后经用户授权，与锁修复一并推送至 `origin/ros-dev`（截至 `bf77ed1`）。未合并 `main`；数据、权重、日志及构建产物均不入 Git。
