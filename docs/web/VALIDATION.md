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
