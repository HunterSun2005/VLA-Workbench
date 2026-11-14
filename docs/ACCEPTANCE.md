# 自动化离线验收

当前源码的结果以 `scripts/validate.py` 生成的报告为准。旧 Markdown 中带日期的测试数量和性能表是历史证据，不代表后续提交已验收；不使用固定通过数量作为门槛。

## 执行

在仓库根目录使用已有模型解释器；脚本不安装依赖，也不修改任何环境：

```bash
export VLA_PYTHON=/data0/miniconda3/envs/vla/bin/python
# 默认：全量模型环境测试、ROS 原生回放、干净 wheel 构建与仓库外导入
python3 scripts/validate.py

# 按改动选择检查；未选择的项目明确记录为 not_run
python3 scripts/validate.py --checks model wheel

# 只有控制／时序改动需要运行十分钟固定状态 ROS mock 测量
python3 scripts/validate.py --checks timing

# LingBot RTC 与官方 Franky 速度 SDK 边界（fake Robot，不连接 FCI）
python3 scripts/validate.py --checks timing --timing-lingbot-rtc --timing-franky-velocity
```

默认产物位于忽略目录 `artifacts/validation/<UTC时间>-<commit>-<随机标识>/`。通过 `--output /path/to/new-report-directory` 指定新的归档位置；拒绝覆盖已有目录。指定目录应在 Git 忽略范围内或仓库外，避免报告自身被当成源码变化。报告、日志、JUnit、wheel 均不提交到 Git；需要长期保留时归档整个结果目录。

默认每项检查最多运行 900 秒，`--timeout` 可调整；超时先向本次检查的进程组发送中断，留出最多 30 秒进行清理，再终止该组。不会结束既有采集／训练／硬件会话。模型检查禁用 GPU 和在线模型下载；ROS 检查由独立 ROS Python 运行。`timing` 使用现有 mock overlay；缺失依赖时检查失败并保留错误，不自动构建或安装环境。

`--timing-franky-velocity` 使用隔离 worker 中官方 2.0.0 wheel 的真实运动构造器及固定状态 fake Robot，额外检查每臂 100 Hz 伺服；原 ROS mock 控制器仍独立检查 1000 Hz。该选项不执行原生 FCI 闭环，不证明实机连续性或精度。

## 报告

| 文件 | 内容 |
| --- | --- |
| `report.json` | schema 1；commit、分支、dirty 状态、源码 SHA256、前后源码一致性、命令、退出码、耗时、各项状态和验收边界 |
| `report.md` | 可直接阅读的状态表、跳过／失败原因及未验收项 |
| `model.xml`、`ros.xml` | pytest JUnit，分别统计通过、跳过、失败和测试错误 |
| `model.log`、`ros.log`、`wheel.log`、`timing.log` | 本次命令完整输出；只生成已选择检查的日志 |
| `wheel/` | 干净源码副本、wheel、外部导入目录、结构化 `result.json` |
| `timing.json` | 本次 mock 测量的频率、延迟、资源及故障结果；不会沿用以前的测量结果 |

源码摘要包括已跟踪文件的当前内容、删除状态和未忽略的新增文件。报告同时保存开始及结束时的源码身份；验收过程中更改源码或提交会将本次结果标为失败。工作区有修改时允许运行，但必须同时参考 `dirty` 和源码摘要，不能仅用 HEAD 宣称提交已验证。

检查状态：

- `passed`：执行成功且证据有效。
- `passed_with_skips`：测试成功，但存在跳过；报告逐项记录原因。
- `skipped`：报告中的测试全部跳过，不计通过。
- `not_run`：未选择，或因中断尚未开始。
- `failed`、`timed_out`、`interrupted`：未通过；缺失／损坏测试报告也不能算成功。

总结果在执行中为 `running`；检查失败或源码变化为 `failed`；成功完成所选离线检查为 `passed_with_gaps`，仍列出跳过、未运行及范围外项目。退出码 0 只表示所选检查没有失败，不能解释成所有现场项目通过。进程被强制杀死时，最后的 `running` 报告也不能当成完成结果。

## 当前验收范围

正常模型环境测试使用小网络、合成数据及模拟设备。真实 PI0.5 资源缺失的检查仍显式跳过。十分钟 timing 是调度与 ROS 通信测量，不包含真实相机、视频编码和 GPU 推理同时运行的负载。真实相机同步／吞吐、机器人跟踪与接触稳定性、轨迹连续性和任务成功率均不能从这些报告推断。

专项测试选择见 [测试指南](../tests/README.md)，存储恢复操作见 [工作流](WORKFLOWS.md#执行耗尽与保存恢复)，契约和首次验收快照见 [执行与数据闭环](EXECUTION_RTC_RLT.md)。
