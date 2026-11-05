# ROS 夹爪动作记录修复

本修复只在 VLA-Workbench 实施；不修改正在采集的 VLA、已安装包或 ROS 环境。

## 标签含义

ROS Gamepad 的 action 夹爪维度现在记录提交给 ActionClient 的目标开度（0～1），持续开／关时保持端点目标；松手时以当时实测值提交一次 hold，并锁定该目标。之后的测量漂移只影响 observation，不重写 action。同步提交失败不更新命令缓存；异步拒绝／执行异常仍报告错误。这里的“命令”表示已提交的请求，不保证服务器接受或物理到达。

FineTeleop 在 fine 和 returning 模式持续发布最后成功提交的目标，实测回调只更新测量。freshness 检查保留：缺少／过期测量、managed 未启用或故障时不发布有效夹爪记录心跳。普通 GELLO 模式仍使用其参考开度。

## 话题路由

| 输入链路 | 录制话题 | 驱动路径 |
| --- | --- | --- |
| 左臂 GELLO；Web 左／右臂 GELLO | `/<arm>/teleop/gripper_action_width_percent` | `/<arm>/teleop/gripper_drive_width_percent` |
| fine、returning、ROS Gamepad | 同上，持续发布已提交目标 | 直接 ActionClient；不重复发布 drive |
| ROS 策略控制 | 同上 | drive 话题 |
| 终端双臂右侧原始 GELLO | `/right/gello/gripper_width_percent` | 保持原始 GELLO 路由 |

启动器的 Robotiq remap 和订阅者预检同步更新：记录话题不再要求驱动订阅；独立 drive 至少一个消费者。原始 GELLO 共享话题仍要求录制器＋驱动两个订阅者。Web 双臂显式传入每臂协调器记录话题，不依赖模块导入后的环境变量修改。

GELLO／策略经话题驱动时，底层原有端点吸附和 deadband 仍存在；本修复不声称这些标签已等于驱动后处理值。observation 夹爪宽度仍是米，action 仍是开度比例。

## 数据兼容

新 ROS 数据写入 `gripper_recording_version=2`，GELLO 数据还记录实际订阅话题。双臂 Gamepad 使用 recording contract v2 和 `controller_joint_command_fk_absolute_tcp_rot6d_gripper_goal_v2`。离线 transition 转换同时识别旧 v1 与新 v2，不能把旧标签自动视为已修复。

禁止续写缺少／不匹配新契约的旧目录，需选择新目录。严格元数据检查发生在临时文件恢复之前，拒绝续采不会先清理旧临时数据。

## 验证与使用

离线测试用真实 collector、FineTeleop 和 recorder 方法配合模拟 ROS 传输，覆盖两臂目标记录、松手保持、测量漂移、持续心跳、驱动不重复发送、提交失败、订阅预检、Web 所用双臂话题覆盖和旧目录拒绝。

不在本轮启动任何设备。真实 DDS QoS、ActionClient 接受／抢占、夹爪到达和连续实机采集仍待验收。启用更新需后续部署 Workbench 版本并新建会话，驱动 remap、发布者和录制器须一起更新，不能混用新旧节点。

2026-11-06 离线验收：夹爪、双臂 Gamepad、单／双臂 recorder、入口、单臂录制、Web hardware/panel 共 **160 passed**；系统 Python 3.10 对采集 Python 源码语法检查及轻量契约导入通过。Web 测试使用本地模拟服务，无硬件连接；现有 Starlette/AnyIO 弃用告警未改依赖。
