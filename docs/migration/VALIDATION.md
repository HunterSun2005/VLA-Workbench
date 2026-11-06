> 历史记录：其中 Franky 路线与旧测试结果不适用于当前 ROS 统一版本。当前行为和限制以 [ROS 动作契约](../ROS_ACTION_CONTRACT.md) 为准。

# 迁移验收记录

日期：2026-10-30。目标：/data0/sht/VLA；源：Evo-RLT c592f5047fd1ffb853cbc820606382e6d2822a9d。

## 已通过

| 项目 | 结果 |
| --- | --- |
| LeRobot 0.5.1 迁移基线 | tests/rlt：775 passed，6 skipped；基线提交 13d755c |
| 最终 LeRobot 0.6.1 全套离线测试 | 800 passed，6 skipped |
| 官方接口集成 | 真实工厂构建 rlt/rlt_token/rlt_ac，processor 发现，spawn 注册/加载、采样覆盖与恢复，原生 SFT 指标与 AC 优化步骤 |
| 图像与元数据 | HWC uint8 → BCHW float [0,1]；原始输入不被修改；typed DatasetInfo 往返后自定义 metadata 保留 |
| wheel | 构建、安装到独立 vla 环境，在 /tmp 导入 site-packages 包；21 个 CLI 入口可加载，YAML/HTML/launch/脚本资源可用 |
| ROS 构建 | 12 packages finished；仅 Humble、新工作空间和声明的原生依赖 |
| ROS 自有夹爪单测 | 22 passed |
| ROS 单/双臂启动器 | CHECK_ONLY 自检通过；未启动硬件 |
| ROS 采集依赖 | 单臂、双臂、rollout collector 与文件锁可用系统 Python 3.10 导入，无 Torch |
| ROS 隔离 | Dynamixel SDK/serial 来自专用目录；sys.path 不含其他工作空间；Franka 动态库解析到 VLA install，无未解析链接 |
| 安全默认值 | 所有部署 home 默认关闭，旧数据集推导的 target_action 已移除；启用但无有效目标时拒绝 |
| 来源保护 | 原 Evo-RLT 保持干净；外部 ROS 源文件哈希未变；未复制旧数据、模型或 checkpoint |

复现模型测试：激活 vla 后，`bash scripts/test.sh -q -rs`。
复现 ROS 构建：`/usr/bin/python3 scripts/fetch_ros.py`，`bash scripts/install_ros_python.sh`，`bash scripts/build_ros.sh`。
验证 ROS 采集环境：`source scripts/ros_env.sh` 后运行 `python3 scripts/check_ros.py`。

## 未验收项目

5 项 processor 测试需要 `VLA_TEST_SFT_PI05_DIR` 指定真实 SFT processor/tokenizer 文件；1 项基础 PI0.5 配置测试需要本地基础 checkpoint。这 6 项跳过不计作通过。

真实模型输出一致性、完整真实数据训练/恢复、GPU 模型性能、相机采流、机械臂运动、FCI 稳定性和真实 RTC/人工干预行为均未执行。合成样本与模拟策略通过不代表实机通过。

旧安装脚本没有记录 serial 上游提交，原源码也已被清理；新清单固定了实际获取并构建通过的提交，但不能证明其与旧二进制完全一致。正式实机接入时需要夹爪通信验收。

## 迁移调整

修复了旧测试中已经过时的 RTC 队列参数、机器人模拟接口及路径假设。移除只适用于 0.5.1 的 control_utils 引用；VLA 使用公开 observation preparation 和自有单步推理适配器。

独立 ROS 构建暴露的 franka_semantic_components 依赖声明已补齐。全局系统 Python 中的 editable SDK 与 NumPy/OpenCV ABI 冲突均通过项目内依赖隔离解决，未修改原环境。
