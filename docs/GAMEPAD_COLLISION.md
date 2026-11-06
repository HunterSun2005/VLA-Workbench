> 历史记录：其中 Franky 路线与旧测试结果不适用于当前 ROS 统一版本。当前行为和限制以 [ROS 动作契约](ROS_ACTION_CONTRACT.md) 为准。

# Gamepad 与 GELLO 碰撞阈值

用户报告的现象是 gamepad 下稍加外力即触发碰撞／reflex。原单臂 Franky gamepad 入口没有显式设置碰撞阈值，而 GELLO 的 ROS 启动器默认调用 `set_full_collision_behavior`。此次将单臂 gamepad 遥操及采集配置对齐现有 GELLO 阈值；未读取或测量机器此前实际生效的阈值，不能据此断言具体提高了多少。

默认的 `single_left_gamepad.yaml`、`single_left_gamepad_record.yaml` 和兼容配置 `single_left_gamepad_31d.yaml` 均包含：

```yaml
robot:
  collision_behavior_profile: gello
```

Web 单臂采集与终端共用配置加载和机器人连接逻辑。自定义旧 YAML 如需采用该设置，在已有 `robot` 节下补充上述字段。省略字段或选择 `default` 时不调用项目新增的阈值设置，沿用 Franky 初始化行为；未知配置值会被拒绝。双臂 ROS gamepad 与 GELLO 原本共用启动器的阈值设置；本次没有改变部署模型配置中的默认 profile。

阈值来源：`src/vla/datacollection/start_teleop/lib/controllers.sh` 的 `apply_collision_behavior`。Franky 使用完整八参数接口，不能用两参数重载将上下阈值或加速／匀速配置混成一组。

| 类型／阶段 | 下阈值 | 上阈值 |
| --- | --- | --- |
| 关节力矩，加速（N·m，J1–J7） | 50, 50, 50, 40, 40, 40, 40 | 100, 100, 100, 80, 80, 40, 40 |
| 关节力矩，nominal（N·m，J1–J7） | 50, 50, 50, 30, 40, 40, 40 | 100, 100, 100, 80, 80, 40, 40 |
| 笛卡尔 wrench，加速（Fx/Fy/Fz 为 N，Tx/Ty/Tz 为 N·m） | 50, 50, 50, 30, 30, 30 | 100, 100, 100, 30, 30, 30 |
| 笛卡尔 wrench，nominal（同上） | 50, 50, 50, 30, 30, 30 | 100, 100, 100, 30, 30, 30 |

设置在 Franky 连接之后、外设连接及运动之前执行；失败直接中止启动并清理资源。成功后诊断日志记录 profile 和八组参数。既有碰撞报错、故障锁、状态／通信成功率检查继续有效。刚度、速度、夹爪力和负载质量补偿没有调整；此项改变碰撞保护灵敏度，不提高机器人额定载荷，也不等同于标定工具或抓取物的质量。

2026-11-04 离线验证：277 项相关测试通过，覆盖阈值与 GELLO 脚本完全一致、三个入口的配置选择、运动前调用顺序、设置失败清理、默认行为和非法配置；直接读取已安装 Franky 的接口说明确认八参数重载存在，未创建 Robot 对象。没有连接机器人或操作现有会话；实际接触任务的 reflex 行为仍待现场验收。

正常结束当前 gamepad 会话后，重新启动遥操／采集会话才会应用新设置；刷新网页不会更改正在运行机器人的阈值。
