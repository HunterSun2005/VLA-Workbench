# 功能迁移对照

| 功能 | 迁入位置 | 验证 |
| --- | --- | --- |
| Franky / Robotiq / RealSense | vla.robots.franka_robotiq | robot、recorder、gripper 模拟测试 |
| RTC / 归位 / 故障处理 | vla.deployment、deploy_franka_pi05_rtc | RTC 队列、停止、恢复、归位路由测试 |
| 单臂、双臂、31D 数据采集 | vla.datacollection | schema、writer、gamepad、dual collection 测试 |
| GELLO 与 ROS 控制器 | ros_ws/src 三个自有包及 vla_bringup | colcon 构建、单/双臂 CHECK_ONLY |
| 数据处理与 SFT | vla.cli、vla.diagnostics | 视频、特征、划分、原生训练指标测试 |
| RLToken / AC | vla.adapters.lerobot.policies | 原有算法/保存恢复测试、官方工厂与 spawn 测试 |
| 离线缓存及契约 | dataset_rlt_ac、build_transition_cache_v2 | manifest、SHA、采样覆盖与恢复测试 |
| 在线 RLT / replay / recovery | vla.core、serve_franka_rollout | online、replay、training safety、full episode 测试 |
| Handover / intervention | vla.core、datacollection | handover、intervention、phase、gamepad 测试 |
| 轻量通信 | vla_rollout_protocol | wire、collector、系统 Python 导入测试 |

删除项与理由见 CLEANUP.md、excluded.json。files.json 记录原仓库文件哈希与最终目标路径；ros-files.json 记录外部自有 ROS 工作树快照哈希。ROS 源快照中的未提交修改已纳入，不修改原工作树。
