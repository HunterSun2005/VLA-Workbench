# 模块对照

这是当前模块导航；迁移时的文件快照与排除项见[本目录索引](README.md)，行为定义见[文档导航](../README.md)。

| 功能 | 当前模块 |
|---|---|
| 动作契约、数学、统计和导出 | `vla.actions` |
| ROS 控制权、关节／TCP 公共入口 | `vla.control.ros` |
| ROS 采集、原始命令与状态 | `vla.datacollection.franka_robotiq_rollout.collect_ros` |
| 模型动作 processor 与训练适配 | `vla.adapters.lerobot` |
| ROS 执行时隙、RTC 承诺与候选安装 | `vla.control.execution`、`vla.control.plan_service` |
| 候选块推理与模型适配 | `vla.adapters.lerobot.plan_inference` |
| RLT 损失、执行 Replay 与在线更新 | `vla.core.losses`、`vla.core.execution_replay`、`vla.core.online_training` |
| 分段持久化与回合恢复 | `vla.datacollection.episode_journal`、`vla.cli.recover_episode` |
| 终端公共入口 | `vla.cli.ros_session` |
| Web 采集／处理／部署 | `vla.web` |
| 独立相机缓存、几何、硬件锁 | `vla.sensors`、`vla.control.geometry`、`vla.core.hardware_lease` |

Franky 驱动、专用采集器、Modbus 独占夹爪与依赖已删除。当前行为及待验收范围见 [ROS 动作契约](../ROS_ACTION_CONTRACT.md)。
