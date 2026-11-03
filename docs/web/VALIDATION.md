# Web 面板离线验收

日期：2026-10-30。所有执行均为合成数据、缓存或模拟 IO，未连接机器人、未执行真实动作。模型环境 Python 3.12，ROS 导入使用系统 Python 3.10；没有提供旧数据或 checkpoint。

## 已执行检查

| 项目 | 结果与证据 |
| --- | --- |
| 全仓回归 | `HF_HOME=/tmp/vla-web-hf VLA_PYTHON=/data0/miniconda3/envs/vla/bin/python bash scripts/test.sh -q -rs -p no:cacheprovider`：826 passed，6 skipped。6 项均缺少真实 PI0.5 checkpoint，不能视为通过 |
| 最后局部修正 | `tests/web` + `tests/rlt/test_franka_sft_server.py`：22 passed；覆盖最后的状态来源修正，只有第三方 Starlette/AnyIO 弃用提示 |
| Web 集中测试 | 19 个参数化用例：4 种采集组合、6 种部署模式组合、重复命令／revision 冲突、保存失败、有限历史、令牌／Origin／Host、独立 runner 重关联、模式版本和迟到结果、reset 保持模式、轻量系统 Python 导入、启动取消及慢监控不阻塞命令 |
| ROS 轻量导入 | 在 `scripts/ros_env.sh` 后导入 ROSBackend、两类 recorder、ROS collector，无 Torch；FineTeleopNode 模拟 IO 验证待命、启用对齐、暂停和新鲜度门控 |
| 浏览器显示 | Chromium 无头浏览器：三路相机、默认曲线、维度开关、10/60/300 秒窗口、暂停与恢复跟随、刷新、离线重连、手机宽度无横向溢出；零 JavaScript 错误 |
| 浏览器操作 | 单臂采集启用→录制→Critical→保存→停止；同一页面再启动双臂部署，RLT→接管→接管中选择自动→归还→刷新→丢弃→停止；零 JavaScript 错误。发现并修复了旧 MJPEG 流占满连接导致命令排队的问题 |
| wheel | 构建并安装 wheel；在 `/tmp`、移除 PYTHONPATH 后验证 `vla-web --help`、site-packages 导入、静态资源和 vendor SHA256；`pip check` 无依赖冲突 |
| Web 真正重启 | 使用已安装 wheel 在独立端口启动 demo，启用后终止 Web；runner PID 不变，重启 Web 恢复同一会话和状态；随后浏览器完整采集／部署流程通过 |
| 策略服务断线 | 假模型服务中客户端提交后直接断开，服务继续接受下一次请求 |

保留的小规模回归测试位于 `tests/web/test_panel.py`，复用了既有模型／采集器 fixture；没有添加页面细节测试框架。Playwright 只用于本次临时浏览器验收，不成为项目依赖。

## 10 分钟双臂三相机 demo

测量结果见同目录 `PERFORMANCE.json`。使用 `scripts/web_demo_soak.py`，同时订阅 3 个 MJPEG 流和 1 个 WebSocket。合成图像为 320×240，不放大；合成遥测为两臂 56 个信号。CPU 百分比以 100% 为一个逻辑核，RSS 为进程驻留内存。

最终连续运行 600.033 秒；WebSocket 9.88 Hz，遥测 19.96 Hz，各相机约 9.54 FPS。遥测样本年龄 p50/p95 为 50.9/96.6 ms。最后约四分钟 runner RSS 73.09–73.86 MiB，Web RSS 49.98–50.02 MiB；CPU 中位数分别为 6.2% 和 2.3%（单核口径）。历史填满后内存稳定，没有随时间持续增长。

复现方式（仅允许 demo 服务，且必须没有活动 demo 会话）：

```bash
vla-web --demo --root /data0/sht/VLA
# 在另一终端传入上面 Web 进程的实际 PID；httpx/psutil 已在本机模型环境中。
python scripts/web_demo_soak.py --web-pid <PID> --seconds 600 --output runtime/web/soak.json
```

脚本拒绝真实硬件服务，正常结束自动丢弃合成回合并停止 demo runner。服务器侧时间差衡量的是已发送遥测样本年龄，不是相机到屏幕时延，也不是机器人控制或 GPU 推理时延。浏览器测试中显示的 35 ms 是明确的 demo 合成值，不用于性能结论。

## 待验收

- 真实 SFT / RLT / 分类头权重加载、输出一致性、实际推理速度和模式切换时的机器人运动连续性。
- Franky 及 ROS 后端在现场设备上的相机、夹爪、手柄、GELLO 交接与故障恢复；本轮只做导入、fixture 和模拟 IO 验证。
- 真实数据采集与续采、视频落盘、磁盘故障及崩溃后的数据恢复；mock 保存测试只验证面板不会误报成功。
- 硬件高负载下的 CPU、内存、图像新鲜度和控制抖动；demo 不能证明这些结果。

没有为 Web 重新构建或迁移 Humble；沿用已构建的专用工作空间。本轮不维护单右臂、不提供历史回放或在线训练管理、不新增双臂 RTC。实际硬件前须按操作文档填写本地配置，不能直接使用示例设备编号。


## 2026-11-02 · ROS 硬件预设与设备选择

- 新增随 wheel 打包的单左臂／双臂 JSON；测试与仓库对应 GELLO 校准、机械臂及夹爪 YAML 比对 IP、端口、手臂范围。
- `tests/web`：38 passed。覆盖设备被动枚举、断开的串口链接、USB 去重、重复 IP／相机／跨 GELLO 与夹爪的端口冲突、非法字段与 shell 字符、请求令牌、保存版本不相互覆盖、错误保存文件回退提示、相对路径与会话 JSON 快照。没有启动真实硬件子进程。
- Chromium demo：单双臂预设切换、各自草稿保留、设备选择、刷新保留输入、未检测设备提示、重复角色错误、自动生成 JSON、自定义文件不被覆盖、保存后刷新恢复、会话运行时禁改、Web 重连、浅色与移动视口检查通过，无页面 JavaScript 异常。
- 系统 Python 3.10 导入新模块不加载 Torch。只读系统元数据枚举到 4 个串口、3 台 RealSense；这不证明设备可控制、角色正确或没有被占用。仓库旧相机编号与当前检测结果不同，必须在 UI 确认 Base／左腕／右腕映射。
- wheel 构建成功；仅更新 `vla` 环境内本项目包，第三方依赖不变。在 `/tmp` 从已安装包验证两个 JSON、前端文件和带令牌的保存 API 成功。
- 未连接机械臂，未打开串口或相机流；真实 GELLO 遥操、标定和采集仍待现场验收。现有 Web 服务未重启，需正常结束会话后重启服务加载更新。


## 2026-11-02 · 配置检查错误排布

配置检查结果移到主区域独立卡片，显示采集／部署、通过状态或错误数量。错误逐条编号与分隔，长路径换行；检查按钮与启动前检查使用相同布局，避免重复输出整段错误。手动修改配置或切换页面后清除旧结果。

使用 Chromium 临时 demo 页面复现四条检查错误，确认独立条目、标题计数、无重复横幅、失败时不创建会话、旧结果清除、成功状态、深浅主题及 390px 移动布局；无 JavaScript 异常、无横向溢出。本次仅修改前端排布，未启动机器人会话。


## 2026-11-03 · Web 管理的 ROS 启动无需交互终端

现场会话 `386670f729704c99ad5d5fb153706e04` 在环境校验阶段退出，hardware.log 为 `An interactive terminal is required for keyboard shutdown`。Web 子进程使用 `/dev/null` stdin，原启动脚本却对所有控制路线要求终端；此时尚未启动硬件任务。

现在仅对 `CONTROL_OWNER=rollout` 且具有 Web 就绪文件的受管理启动豁免终端要求。后台监控每 0.1 秒检查子进程，不读取空 stdin，避免 EOF 空转；沿用 SIGTERM 和既有 EXIT 清理。独立 CLI 仍保留原终端要求与键盘退出。

`tests/rlt/test_teleop_entrypoints.py tests/web/test_panel.py`：54 passed。新增回归覆盖真实终端校验分支（模拟 ROS 依赖）、无终端的受管理启动、独立 CLI 拒绝无终端、check-only、后台故障监控、轮询等待及 SIGTERM 清理；Shell 语法检查通过。未启动硬件或自动重试现场会话；后续 ROS、设备通信与实机控制仍需现场验收。


## 2026-11-03 · Web／终端路线优化与 30 FPS 复验

Web 连接状态与业务故障分离，10 秒通信启动宽限；故障恢复按来源清除，命令失败可关闭且不重复弹出。`VLA_RUN_MODE` 明确区分 managed／terminal；终端按键与确认不进入 Web 业务调用。新增终端后台有界 reporter、持久日志和一次非阻塞丢弃确认；GELLO直接启动、暂停数据语义和既有安全检查保留。

离线全仓回归：918 passed，6 skipped；后续共享提示文案与状态行竞态修正的针对性回归251 passed（含新增竞态用例）；跳过项依赖未配置的真实模型资源，未计作通过。新增覆盖连接恢复／真实缺文件、抖动30 FPS源、单相机故障与恢复、无订阅停编码、慢输出／有界队列、按键确认取消／超时及暂停／EOF优先级。Chromium检查启动提示、真实错误显示、遥测／相机独立恢复、失败命令关闭不重放、跨会话重置、三视角双标签、明暗及移动布局通过，无JavaScript异常。

最终三视角采集demo连续600.023秒。完整测量见 [PERFORMANCE_30FPS.json](PERFORMANCE_30FPS.json)，使用独立帧时间戳和完整MJPEG帧计数；未用重复帧补足FPS：

| 视角 | 实际接收 FPS | 重复帧 | 接收帧年龄 p50 / p95 |
|---|---:|---:|---:|
| base | 29.998 | 0 | 24.64 / 25.67 ms |
| left_wrist | 30.000 | 0 | 24.27 / 25.44 ms |
| right_wrist | 30.000 | 0 | 24.41 / 25.55 ms |

WebSocket 9.93 Hz、遥测19.95 Hz。五分钟历史缓冲填满后，最后四分钟runner RSS为72.61–76.03 MiB、Web RSS为47.73–49.27 MiB；CPU中位数分别为17.5%和4.5%（100%代表一个逻辑核）。缓冲数量保持有界，RSS在上述范围波动；首五分钟的增长来自遥测历史填充。与先前10 FPS基线分开保留，未覆盖旧测量。

本次性能结果来自320×240合成图与本机HTTP接收时刻，不代表实际相机端到端屏幕延迟。真实RealSense、SSH隧道及硬件控制负载下的30 FPS仍待现场验收。未连接机器人，未重启或终止用户现有会话。

最终 wheel 构建并仅更新 vla 环境内本项目包。在源码目录之外验证 vla-web --help、已安装 HTML/JS/CSS 与启动脚本、连接状态 API、独立三相机 demo、30 FPS 元数据及退出清理通过。终端轻量模块可由系统 Python 3.10 导入，不加载 Torch/rclpy。现有 Web 服务和用户会话未重启；更新需要正常结束旧会话后重启 Web 并创建新会话。

## 2026-11-03 · 成功率边界与错误复制

- ROS Franka 健康监控与 gamepad 对齐阈值浮点表示边界；六个新增参数化用例覆盖阈值表示、连续计数复位、两臂独立、下一档较低 float32 值及真实低值拒绝、精确诊断。没有修改 90% 配置阈值或连续样本数要求。
- 采集、gamepad 与 Web 针对性回归：219 passed。同步更新一处既有英文提示断言以匹配此前已采用的中文故障提示；仅有第三方 Starlette 弃用警告。
- 独立临时 demo Web 的 Chromium 检查通过：完整多行复制、Clipboard API 拒绝时备用复制、全部复制方式失败提示、关闭单项后复制剩余错误、明暗主题移动布局、错误清除；无 JavaScript 异常。未创建硬件会话、未操作现有任务。
- 旧日志只有三位小数，无法追溯当时精确成功率或确认通信根因；此次修改不是现场通信性能验收。新健康监控逻辑需正常结束旧会话后创建新会话才生效。
