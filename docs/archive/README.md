# 历史文档归档

这里保留已发生的离线验收、测量和变更记录，供追溯当时的版本、范围和限制。旧格式兼容说明、Franky 路线、安装命令及本机路径可能已失效，不应用来指导当前运行。当前入口见[文档导航](../README.md)，新验收见[自动化验收](../ACCEPTANCE.md)。

## 验收与测量

| 时间／范围 | 报告 | 配套数据 |
| --- | --- | --- |
| 2026-11-07：协议 v2、schema 5、Replay v4、首次统一执行验收 | [执行闭环](EXECUTION_2026-11-07.md) | 原始产物位置记录在报告中，未迁入 Git |
| 2026-11-06：ROS 统一控制基线及后续补充 | [ROS 验收](ROS_VALIDATION.md) | 报告内保留当时的构建／回归结果 |
| 2026-11-06：十分钟双臂 ROS mock 通信 | [通信与控制器](ROS_COMMUNICATION_VALIDATION.md) | [测量 JSON](ROS_CONTROL_BENCHMARK.json) |
| 2026-11-06：真实 FR3 运动学下的无硬件重放 | [原生动作重放](ROS_NATIVE_REPLAY_VALIDATION.md) | 测试与命令见报告 |
| 2026-11-06：逐列精确统计内存优化 | [统计基准](ACTION_STATISTICS_2026-11-06.md) | 合成数据规模与基准命令见报告 |
| 2026-11-02：离线数据处理 | [数据处理验收](DATA_PROCESSING_2026-11-02.md) | 合成数据、浏览器及导出检查 |
| 2026-10-30～11-05：Web 功能与 demo | [Web 验收与追加记录](WEB_VALIDATION.md) | [早期 demo](WEB_PERFORMANCE.json)、[30 FPS demo](WEB_PERFORMANCE_30FPS.json) |

不同日期的频率测量、通过数量和采样条件不能直接合并。特别是 11-06 的接受时隙实现与 11-07 固定截止时间实现不同；结果仅适用于各自记录的实现。

## 历史变更与来源

| 内容 | 历史记录 | 当前说明 |
| --- | --- | --- |
| 旧 Franky gamepad 碰撞配置调整 | [2026-11-04 记录](GAMEPAD_COLLISION_2026-11-04.md) | [ROS 碰撞阈值](../GAMEPAD_COLLISION.md) |
| 夹爪标签与话题修复 | [2026-11-06 记录](ROS_GRIPPER_RECORDING_2026-11-06.md) | [统一采集器夹爪记录](../ROS_GRIPPER_RECORDING.md) |
| Python／ROS 迁移、版本与来源清单 | [迁移资料索引](../migration/README.md) | [动作契约](../ROS_ACTION_CONTRACT.md) |

原有测量 JSON 原样保存。报告中的 `/tmp` 路径是当时的产物位置，不保证文件仍存在；它们不是可下载附件，也不替代当前源码的验证。
