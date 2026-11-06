# 模块对照

| 功能 | 当前模块 |
|---|---|
| 动作契约、数学、统计和导出 | `vla.actions` |
| ROS 控制权、关节／TCP 公共入口 | `vla.control.ros` |
| ROS 采集、原始命令与状态 | `vla.datacollection.franka_robotiq_rollout.collect_ros` |
| 模型动作 processor 与训练适配 | `vla.adapters.lerobot` |
| RTC、RLT、在线回放 | `vla.deployment`、`vla.core.online_training` |
| 终端公共入口 | `vla.cli.ros_session` |
| Web 采集／处理／部署 | `vla.web` |
| 独立相机缓存、几何、硬件锁 | `vla.sensors`、`vla.control.geometry`、`vla.core.hardware_lease` |

Franky 驱动、专用采集器、Modbus 独占夹爪与依赖已删除。当前行为及待验收范围见 [ROS 动作契约](../ROS_ACTION_CONTRACT.md)。
