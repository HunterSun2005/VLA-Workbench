# VLA Web 采集、数据处理与部署面板

面板复用原有采集器、数据写入器、策略服务和人工接管逻辑。Web 服务只提供本地界面；设备由独立 runner 持有。没有在线训练、回放更新、浏览器虚拟手柄或跨后端热切换。

## 启动

模型环境使用 Python 3.12，ROS 仍为 Humble + 系统 Python 3.10。Web 不需要在 ROS 环境中启动。

```bash
cd /data0/sht/VLA
conda activate vla
source scripts/model_env.sh
python -m pip install -r requirements-web.lock
python -m pip install --no-deps '.[web]'
vla-web --demo --root /data0/sht/VLA
```

浏览器访问 `http://127.0.0.1:8765`。Demo 的采集／部署画面、遥测、档位与模型输出均为合成数据，不连接设备、不生成采集数据集。**数据处理页始终读取所选本地数据，点击转换会创建真实的新输出目录，即使启动了 `--demo`。**配置任务后点击“启动会话”，再启用控制、开始回合。取消 `--demo` 才会允许创建真实硬件会话；启动 Web 本身不会启动设备。

远程浏览器通过 SSH 隧道访问，服务固定监听 loopback，不提供公开监听选项：

```bash
ssh -N -L 8765:127.0.0.1:8765 用户@机器人主机
```

默认端口可用 `--port` 修改。`--root` / `VLA_ROOT` 指向工作目录；硬件运行必须指向完整 VLA 仓库（包含 `src` 和已构建的 `ros_ws`）。wheel 可独立提供界面和 demo，真实 ROS 后端还需要这个工作空间。

## 可选 Side 相机

启动数据采集前，可在「Side 相机（可选）」下拉列表中选择已检测到的 RealSense，点击「刷新摄像头列表」可重新枚举。留空使用 base + 左腕（双臂增加右腕）；选择后单臂为三视角、双臂为四视角，额外视角参与同步、实时预览及数据保存，字段名为 `observation.images.side`。支持 GELLO／gamepad；只在会话启动前设置，部署模式不增加此选项。

Side 不能与 base／wrist 指向同一设备。推荐采集目录随选择加上 `_side` 后缀，手动填写的路径仍保留。新增或移除 side 需要使用新的数据集目录，续采时相机视角集合必须完全一致。数据处理页会自动列出 side，字段提取转换保留其视频。采集和预览目标帧率沿用现有设置；四相机的实际 USB 带宽及控制负载需现场验证。

## 数据处理

导航顺序为「数据采集 → 数据处理 → 推理部署」。完整说明及 CLI 重现方式见 [离线数据处理](DATA_PROCESSING.md)。

1. 打开已完成的数据集：输入相对或绝对路径，或选择 `datasets/raw/`、`datasets/rollouts/` 下的候选项。
2. 点击回合「查看」，用帧号、时间轴、上一／下一帧或播放查看各视角图像、状态、动作和标注。
3. 勾选需要「剔除」的完整回合；默认全部保留。预览本身不修改选择，不支持删除回合内部的帧。
4. 全局选择需要导出的状态字段，或选 TCP 10D／双臂 20D 等预设；动作、相机与标注保留。「显示全部状态」仅影响数值预览。
5. 确认预填的 `datasets/sft/<源名称>_<维度>d` 输出路径，点击「开始转换」。已有目录不会覆盖，字段选择和剔除清单自动保存。
6. 完成后点击「打开结果」离线检查。转换不会自动启动训练、修改模型维度或执行机器人动作。

页面切换不会停止硬件会话，运行期间可返回当前采集／部署页监控，也可预览其他已完成的数据集。面板管理的真实硬件会话与转换任务互斥：必须结束会话才能转换，转换完成或取消后才能启动新会话。模拟会话可与转换并行；真实与 demo Web 使用同一工作目录时，转换也会检查真实会话。

## 功能矩阵

| 路线 | 数据／模型 | 回合和标注 | 控制与监控 |
| --- | --- | --- | --- |
| 单左臂 gamepad 采集 | 完整 50D；RecordingSession | 成功或失败保存／丢弃；Critical | Franky；Normal/Fast、实际倍率与过渡 |
| 单左臂 GELLO 采集 | 完整 50D；DirectLeRobotRecorder | 成功保存／丢弃 | ROS；GELLO / fine_gamepad / returning_to_gello |
| 双臂 gamepad 采集 | 完整 100D；ROS RolloutCollector demonstration | 成功保存／丢弃；Critical、配置中的阶段 | ROS；固定档位，不伪造 Fast |
| 双臂 GELLO 采集 | 完整 100D；DualLeRobotRecorder | 成功保存／丢弃 | ROS；每臂交接状态、固定档位 |
| 单左臂部署 | PI0.5 SFT 或兼容 RLT AC checkpoint | 原部署回合记录；成功／失败／丢弃 | 原 Franky RTC；gamepad 接管 |
| 双臂部署 | 一套联合 20D 策略 | 原部署回合记录；成功／失败／丢弃 | 原 ROS 时序；gamepad 接管，无新增 RTC |

所有路线都有待命、启用、软件暂停、保存／丢弃和停止。故障恢复仅在后端提供对应能力时开放；ROS gamepad 的 fault latch 需结束当前会话后排查并重启。GELLO 恢复只清除已排查的本地故障锁，不自动启用运动。没有单右臂模式。

按钮可见不代表当前状态可执行；命令在设备事件循环中校验，拒绝原因显示在操作区。双臂 gamepad 和 GELLO 采集不支持失败标签；面板禁用相应按钮。阶段标注使用双臂 gamepad YAML 的 `dataset.subtasks`，没有配置时不新增标签。

## 工作配置

会话选择、日志、socket、临时硬件配置放在 `runtime/web/<session_id>/`，active manifest 和独占锁放在 `runtime/web/`。这些文件不进 Git；日志不自动清理。离线处理草稿和任务记录位于 `runtime/web/processing/`。数据集输出目录自动预填相对路径，随单双臂和输入源切换，统一采用 `_full` 后缀；手动编辑后保留自定义值，“使用推荐路径”可恢复自动跟随。相对路径以 `--root` / `VLA_ROOT` / 启动当前目录为基准（页面会显示）；检查配置可查看最终绝对路径。已有非空目录必须选择续采。分层和命名见 [数据集目录说明](../datasets/README.md)。不要删除活动会话目录。

**单臂 gamepad 采集**：复制对应 YAML 到本地运行目录，修改 IP、夹爪串口、相机序列号、工作空间与运动限值。在面板的“采集／部署 YAML”中填入绝对路径。

```bash
mkdir -p runtime/web/config
cp src/vla/datacollection/start_teleop/configs/single_left_gamepad_record.yaml runtime/web/config/record.yaml
# 所有入口统一记录完整状态，不再选择 29D/31D
```

**ROS 路线**：先按 README 构建 `ros_ws` 并安装 `ros_ws/python_deps`。面板只接受 `VLA_ROOT/ros_ws`；不读取其他 overlay 或 Conda ROS 包。选择 GELLO 或双臂时，会显示 **ROS 硬件 · 预设与设备选择**：

1. 单左臂／双臂自动切换对应预设，预填仓库现有的机械臂 IP、GELLO 串口、夹爪串口和校准 YAML。各自记住上次保存的配置。
2. 从下拉列表选择当前检测到的串口和 RealSense 序列号，也可直接填写。点击「刷新设备」更新列表，保留已填值。IP 不扫描网络，需要按现场配置确认。
3. 确认 Base／左腕／右腕相机角色。旧序列号若未检测到会提示；列表顺序不能表示相机安装位置，不自动替换。检测只读取 `/dev/serial/by-id` 和 USB sysfs，不打开设备；检测到不保证有权限、未被占用或能正常工作。
4. 「保存并使用」生成 `runtime/web/hardware/` 下的本地 JSON；「检查配置」和「启动会话」也会自动保存。保存可离线进行，不表示硬件验收通过。检查会拒绝重复 IP、串口（含 GELLO 与夹爪混用同一端口）和相机角色。
5. 修改保存采用独立内容版本；每个硬件会话启动时再复制一份到会话目录。刷新页面不会重新启动设备，修改预设不会影响既有会话。

随 wheel 提供 `vla/web/presets/single_left.json`、`dual.json`。预设来自仓库现有 `franka_gello_left/duo.yaml`、`example_fr3_left/duo_config.yaml` 和相应 Robotiq YAML；串口选择只覆盖设备地址，不改写 GELLO 的关节符号、偏置或夹爪标定。更换 GELLO 本体仍需现场确认校准配置。

如需自行管理文件，选择「自定义 JSON 文件」，在下方「环境与 ROS 配置」填写路径。相对 JSON 路径以 `VLA_ROOT` 为基准；该模式不修改自定义文件。可提供本地 JSON，例如 `runtime/web/config/hardware.json`：

```json
{
  "LEFT_ROBOT_IP": "172.16.0.2",
  "RIGHT_ROBOT_IP": "172.16.1.2",
  "LEFT_ROBOTIQ_PORT": "/dev/serial/by-id/LEFT_GRIPPER",
  "RIGHT_ROBOTIQ_PORT": "/dev/serial/by-id/RIGHT_GRIPPER",
  "BASE_CAMERA_SERIAL": "BASE_SERIAL",
  "LEFT_WRIST_CAMERA_SERIAL": "LEFT_SERIAL",
  "RIGHT_WRIST_CAMERA_SERIAL": "RIGHT_SERIAL",
  "GELLO_CONFIG": "/data0/sht/VLA/runtime/web/config/gello.yaml"
}
```

这里的地址和编号仅为示例。单左臂可省略右臂项；gamepad 可省略 GELLO_CONFIG。还支持 `LEFT_GELLO_SERIAL_ID`、`RIGHT_GELLO_SERIAL_ID`、`ROBOT_CONFIG`、`GRIPPER_CONFIG`。只允许明确列出的字段及不含 shell 语法的路径／标识符。GELLO YAML 从 `ros_ws/src/franka_gello_state_publisher/config/` 的对应配置复制，保留所选臂的 `namespace: left/right` 并修改串口、校准；未提供所选臂时预检拒绝。运行时生成新的配置，不改写 ROS install。

GELLO 录制使用原 recorder 的数据 schema；面板中的“采集／部署 YAML”不用于这条路线。ROS gamepad 使用 `datacollection/franka_robotiq_rollout/configs/dual.yaml`，可复制后修改控制与阶段配置；JSON 中的相机序列号覆盖 YAML 中对应的序列号。更换输入后端或单双臂必须先停止会话。

**部署**：模型、tokenizer、分类头必须是已经准备好的本地目录。本轮不迁移权重或数据。预检检查 `config.json` 的模型类型、10D/20D 状态动作维度、相机字段，runner 在启动硬件前加载模型，缺失或不兼容资源导致启动失败。默认相机契约为 base、left_wrist，双臂增加 right_wrist。

- SFT 选择 `pi05` checkpoint，仅启用 VLA。
- RLT 选择 `rlt_ac` checkpoint（含兼容 Actor/RLToken），提供 VLA 和 RLT。
- 自动模式还需要兼容分类头目录及 `handover_config.json`；缺失时按钮禁用。所有资源预加载，运行中不加载新权重。
- 单左臂部署 YAML 需要 `control.rtc_deployment_config`；该文件继续使用既有 Franky 部署配置及限值。

## 操作语义

创建会话后进入待命；启用控制和开始回合是独立操作。单臂部署启用后可沿用 gamepad 定位，再开始策略回合。双臂部署启用只完成就绪检查，开始回合后才执行策略。机器人和手柄必须满足现有新鲜度、中立位置与健康检查。

保存结果以写入器完成为准，不将“命令已接受”当作“已保存”。保存错误保留 runner 并显示错误；存储故障需要检查会话日志与数据目录，不能通过重新点击启动绕过。停止活动回合会被拒绝，先保存或丢弃；成功处理后停止才清理本会话的子进程。

**软件暂停沿用原后端语义，可能丢弃当前未保存回合。** 页面在录制时要求确认。它不是物理急停；“停止会话”也不是可跳过回合处理的紧急停机按钮。故障后不自动恢复运动。

运行中选择 VLA / RLT / 自动，在动作边界清理旧队列和旧推理结果，再从当前观测推理。请求与响应携带模式版本；固定模式严格检查执行阶段，自动模式允许分类头选择实际阶段。分类头沿用原阈值与防抖逻辑，回合 reset 保留请求模式并清空投票状态。人工接管优先，接管期间选模式只更新待恢复模式，不隐式归还控制。界面分别显示选择、实际来源和等待状态。

## 监控和连接

右上角可切换浅色／深色模式，默认深色。选择保存在当前浏览器，多标签页同步；背景、控件和曲线同步换色，图表暂停和当前显示范围保持不变。

相机输入统一采用等比例缩放后居中补黑边：例如 640×480 → 224×168，再上下各补 28 像素到 224×224。该处理覆盖单／双臂 GELLO、gamepad（含旧 `_31d` 入口）采集以及 Franky／ROS 部署，预览与录制／推理使用同一处理后的缓存，保留完整视野。缩小仍会降低细节分辨率；已有裁剪数据不能恢复被裁掉的区域。新采集建议使用独立数据目录，使用旧裁剪数据训练的模型需重新验证输入变化后的效果。

相机只读设备拥有者缓存；无订阅者时不编码。每个相机只保留最新 JPEG，最大 640×480、不放大小图；采集预览目标 30 FPS，部署预览保持 10 FPS。单臂预览读取每视角最新缓存，不要求各相机同时到帧；策略和录制的同步接口保持不变。页面的“服务端预览 FPS”统计独立编码帧，过期时显示空值，不代表浏览器屏幕实测 FPS。遥测后台最高 20 Hz，WebSocket 约 10 Hz，缓冲最多 6000 个采样；慢客户端不积压旧显示数据。限制为最多 4 个 WebSocket 和 12 路相机订阅。

默认显示 TCP XYZ 和夹爪，其余按手臂、单位分组选择；目标值单独标注。缺失或超过 0.5 秒的数据不补零，画面过期有遮罩与时间标记。暂停图表仅冻结浏览器视图；采集和机器人仍运行。窗口与选择保存在浏览器本地。刷新后重新获取实时快照，不重放操作；不提供历史回合回放。

浏览器退出、断网、Web 服务重启均不停止 runner。使用相同 `--root` 和 `--demo` 设置重启 Web 可重新关联；会话 ID、PID 和进程启动时间共同校验身份。进程崩溃只记录最终故障，不自动重启硬件。每条命令携带唯一 ID 和状态 revision，重试不会重复执行，冲突操作拒绝后应等待新状态。

启动期间 runner 存活而通信端点尚未建立时，10 秒内显示“正在建立会话连接”；超过宽限显示延迟及日志位置。连接恢复自动清除连接提示。遥测和每路相机分别管理监控错误，恢复时仅清除对应项；真实配置缺失、设备故障和保存失败不会被过滤。命令失败可关闭，历史仍留在事件区域，不会由后续快照重复弹出。子进程退出显示名称、退出码、日志路径及有界诊断摘要。

报错栏右上角的“一键复制”复制点击时全部可见错误文本，保留换行与日志路径；显示“已复制”才表示成功。剪贴板接口不可用时尝试备用复制方式；均被浏览器拒绝时提示手动选择文本，不隐藏原错误。

`control_command_success_rate` 是机器人最近 100 条控制命令的成功接收比例；`for N samples` 是监控连续观察到低于阈值的新状态次数，不是秒数，也不是丢包数量。ROS 与 gamepad 监控均接受阈值本身的浮点表示误差，不降低配置阈值；真正较低的值仍触发故障。ROS 告警显示 17 位有效数字及实际比较边界，避免旧版三位小数产生 `0.900 below 0.900` 的歧义。单凭旧告警无法区分舍入误差与真实低值，也不能确定网络故障原因。

Web 显式使用 `VLA_RUN_MODE=managed`，由 runner 管理停止，不读取终端按键；独立 CLI 默认为 `terminal`。`VLA_WEB_READY_FILE` 只通知硬件就绪。修改 Python 服务逻辑后需重启 Web 服务加载；既有 runner 继续使用原代码，不自动重启或恢复运动。完整更新应在正常结束当前会话后，重启 Web 并创建新会话。

本地 HTTP 检查 Host、Origin、写请求令牌；Unix socket 使用版本化协议和 0600 权限。轻量协议、ROS 适配模块可用系统 Python 导入，不导入 Torch。前端 uPlot 1.6.32 本地打包，MIT 许可、下载来源及 SHA256 位于 `src/vla/web/static/vendor/`；浏览器无 CDN、Node 或数据库依赖。

实时监控的「显示维度」包含 TCP 外力 XYZ（N）与外力矩 XYZ（N·m），按手臂分别选择、分图展示，默认仍只显示 TCP 位置与夹爪。这些通道标为估计值：Franky 使用 `O_F_ext_hat_K`，ROS 使用 `o_f_ext_hat_k.wrench`，与原始数据集的 `base_tcp_force_*` / `base_tcp_torque_*` 一致；表示作用于刚度坐标系 K 原点、以基座坐标系 O 表达的外部 wrench，不是关节力矩或额外六维力传感器读数。没有读取到的维度和过期 ROS 状态留空，不补零。

桌面端三个页面的左右列之间均有可拖动分隔条，采集／数据处理／部署的比例分别保存在当前浏览器。双击分隔条恢复当前页默认宽度；聚焦后用左右方向键微调，Shift 加大步长，Home／End 调到边界。调整窗口时保留可用最小宽度；窄屏自动上下排列。

离线帧预览在新帧图像完成解码前保留上一帧图像和数值，加载状态单独提示；准备完成后统一切换。快速拖动时间轴时丢弃旧请求结果，单视角失败明确显示图像不可用，帧数值加载失败则保留上一帧并停止播放。该优化只涉及浏览器预览，不修改数据集或转换结果。

单臂 gamepad 默认遥操／采集配置在连接后、运动前应用与 GELLO 一致的完整碰撞阈值；配置字段为 `robot.collision_behavior_profile: gello`。已有会话需正常结束后重新启动才生效，自定义 YAML 需显式设置。阈值、适用范围及离线验证见 [Gamepad 碰撞配置](GAMEPAD_COLLISION.md)。

## 验收和限制

见 [Web 验收记录](web/VALIDATION.md)。真实模型输出、现场 ROS 控制、相机硬件、采集数据质量、存储故障恢复和机器人时延均待实机验收；demo 与 mock 通过不等于这些项目通过。首次接入应使用现场确认的配置和模型。
