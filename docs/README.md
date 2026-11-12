# 文档导航

本目录的顶层文档描述当前 ROS 统一链路；带日期的验收和变更记录位于 [archive/](archive/README.md)，来源清单与迁移补丁位于 [migration/](migration/README.md)。历史报告中的通过数量、环境状态和旧命令不代表当前版本。

首次使用先看仓库 [README](../README.md) 的环境与安装说明，再按下面的任务选择文档。开发约束见 [AGENTS.md](../AGENTS.md)；开发测试不启动硬件。

## 使用与维护

| 要做什么 | 文档 |
| --- | --- |
| 采集、SFT／RLT 训练、部署、暂停与保存恢复 | [工作流](WORKFLOWS.md) |
| Web 会话、接管、监控和姿态库 | [Web 面板](WEB_PANEL.md) |
| 独立 LingBot V2 服务、外部契约与串行推理 | [LingBot 部署](LINGBOT_DEPLOYMENT.md) |
| 离线预览、整回合筛选与字段／动作空间导出 | [数据处理](DATA_PROCESSING.md) |
| 诊断锁占用、处理废弃 Git index 锁 | [锁文件](LOCKS.md) |
| 运行当前提交的验收并读取报告 | [自动化验收](ACCEPTANCE.md) |
| 按代码改动选择专项测试 | [测试指南](../tests/README.md) |

## 契约与实现

| 要了解什么 | 文档 |
| --- | --- |
| action／anchor 的数学表示、接受与提交记录点 | [ROS 动作契约](ROS_ACTION_CONTRACT.md) |
| 记录 schema、字段维数、时间与数据文件说明 | [采集数据格式](DATA_COLLECTION_FORMAT.md) |
| 协议 v2、唯一执行器、RTC、Replay、梯度与奖励 | [执行与训练闭环](EXECUTION_RTC_RLT.md) |
| 夹爪目标、实测开度、命令提交和来源元数据 | [夹爪记录](ROS_GRIPPER_RECORDING.md) |
| ROS 碰撞参数来源与配置生效边界 | [碰撞阈值](GAMEPAD_COLLISION.md) |
| 精确 action 统计的内存／临时磁盘开销 | [统计内存](ACTION_STATISTICS_MEMORY.md) |
| 当前功能的模块位置 | [模块对照](migration/FEATURES.md) |
| 架构图与可编辑源文件 | [图表说明](diagrams/README.md) |

## 文档维护约定

- 操作步骤放在工作流／对应功能说明；契约定义放在对应参考文档，通过链接引用，避免多处复制版本表、恢复命令或损失公式。
- 当前验收以绑定 commit 和源码摘要的自动化报告为准。新报告、JUnit、日志和 wheel 放在忽略的 `artifacts/validation/` 或仓库外，不把测试数量复制到多份当前文档。
- 既有历史报告及测量 JSON 保留在归档，不回写其测量数字。来源清单、许可证依据及迁移补丁不是可随意删除的旧文档。
- 改名或移动时更新仓库内的相对链接。修改图表时同步 SVG 和可编辑 draw.io 页面，避免源文件与展示图表达不同的行为。
