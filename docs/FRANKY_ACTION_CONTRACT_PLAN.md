# Franky 单臂动作契约与控制状态统一方案

状态：**设计与现状诊断已完成，尚未实施新控制器**。本轮未修改 `robot.py`、`gamepad_recording.py` 或 Franky 运行配置。ROS 夹爪独立修复见 [ROS_GRIPPER_RECORDING.md](ROS_GRIPPER_RECORDING.md)。不连接硬件，不改变原 VLA 的运行任务。

## 1. 现状与证据

| 路线 | 实际控制入口 | 当前 action 来源 | 与策略入口的差异 |
| --- | --- | --- | --- |
| 单臂 Gamepad 示教 | `GamepadTeleop.tick` → `send_human_command`，直接速度控制 | `IntervalAssembler` 将成功下发速度沿相机区间积分；每区间从实测 TCP 重锚 | 标签位移约 `v × Δt_camera`；策略解释为目标误差并除以 τ |
| 普通部署人工接管 | `collect_franky.gamepad_tick` → `send_human_command` | 返回 `T_measured` 加本次命令的 `v × dt`／旋转增量 | dt 是实际控制间隔，不是 τ；未走策略的两级滤波 |
| `--online` RLT 接管 | `send_human_target` → `_servo_target_once`，后台约 100 Hz | 接受时构造 `T_origin + v × dt`；指定 `anchor_to_measured` 时重锚 | servo 用接受时 dt 将误差转速度，策略却用 τ；旧路径可能累计目标 |
| 策略执行 | `send_action` → `CartesianVelocityMotion` | 返回的是裁剪并平滑后的有效目标，失败时可能是当前位姿 | 若把返回值再次送入策略入口，会重复目标平滑；夹爪还有二值滞回 |

代码定位：`src/vla/robots/franka_robotiq/robot.py` 的上述方法；`src/vla/datacollection/start_teleop/gamepad_recording.py` 的 `command_from_debug`、`IntervalAssembler`；`src/vla/datacollection/franka_robotiq_rollout/collect_franky.py` 的策略提交、`gamepad_tick` 和接管边界。

当前配置也不同：示教最大线速度 0.16 m/s，两个部署 YAML 为 0.05 m/s；RTC 配置目标历史权重 0.25、速度历史权重 0.30，SFT takeover 配置二者为 0。Python 配置类默认速度历史权重还是 0.8，不能把某个 YAML 的数值当成所有路线的默认值。

新增 `tests/rlt/test_franky_action_contract_diagnostic.py` 直接运行上述生产方法，使用现有 FakeArm/Camera/Gripper，捕获传给 `CartesianVelocityMotion` 的真实参数。两独立机器人实例使用相同且固定的测量及初始历史，标签经过 float32 序列化，再送入第二个实例的 `send_action`。没有硬件误差或时间对齐误差，仍观察到：

- 关闭平滑、死区且不触及裁剪时，30 FPS 积分标签和 1/30 秒在线目标重放后的线／角速度为示教的 `(1/30)/0.15 ≈ 22.2%`；0.01 秒普通接管标签为 `6.7%`。这是选定 dt 的隔离实验，不是实机固定倍率。
- 相同非零历史下，目标位移 0.003 m、τ=0.15、目标历史权重 0.25、速度历史权重 0.30，原提交线速度 0.0135 m/s；返回的平滑目标再提交得到 0.010875 m/s。
- `stop()` 在没有活动 motion 时清除目标历史，但提前返回而保留速度历史。活动 motion 的 finally 分支会清零速度；异常路径不能假定与正常路径一样。

这些测试**以量化当前缺陷为通过条件**，不是统一契约的验收，不使用 xfail，也不证明真实 1 kHz 物理响应一致。

## 2. 待实施的共同契约

四条路线统一为 **30 Hz 接受动作、100 Hz 伺服、30 FPS 记录**。相机时间仍是观测时间，不把 30 FPS 当作动作的积分常数。保留现有观测字段与坐标定义。

`action = [p_base(3), R_base_tool_rot6d(6), gripper_open_fraction(1)]`：

- TCP 是当前 `O_T_EE` 使用的同一末端框架，不引入新的 TCP 偏置。
- 标签是在 `send_action` 公共入口完成验证和工作空间／相对步长裁剪后、目标平滑与速度转换前的目标。成功接受但尚未执行、提交失败和实际下发要在结果中区分；失败请求不能冒充有效动作帧。
- 最大线速度 0.16 m/s、角速度 0.30 rad/s；`τ=0.15s` 是位姿误差到速度的跟踪时间常数，**不是采样间隔，也不保证 0.15 秒到达**。工作空间和其他健康保护保持显式配置，统一上限并不代表直接提高旧模型运行速度已获验证。
- Gamepad 意图先转换工具角速度到基座坐标：`ω_base = R_measured × ω_tool`。目标 `p_ref = p_measured + τ v_base`、`R_ref = Exp(τ ω_base) R_measured`，随后送入与策略完全相同的入口。不同路线不能再次按 dt 积分或绕过滤波。
- 夹爪 action 为连续 0～1 目标；state 为米。共同夹爪适配器负责限速／deadband，松手时只锁存一次保持目标，后续测量不改变标签。移除 v2 路线的策略专用二值滞回，旧模型继续走 legacy 配置。

这会改变旧示教的响应，不承诺与直接速度示教逐周期相同；要求的是**新示教与新策略执行使用相同契约**。

## 3. 具体代码调整边界

### 3.1 共享执行器与结果对象

在 `robots/franka_robotiq` 增加轻量 `action_control.py`，只依赖 NumPy 和已有位姿数学，定义：

- `AcceptedAction`：不可变裁剪目标、夹爪目标、`action_id`、`generation`、source、接受观测时间及生效时间；数组拥有独立副本。
- `ControllerHistory`：上次平滑目标、线／角速度、上次 servo 时间、当前 action ID。
- `ActionReceipt`：接受／首次提交状态、裁剪前后目标及原因、命令 ID。另保留 diagnostics 的平滑目标和实际 twist，不能将 diagnostics 当训练 action。
- `accept_target(observation, target, source, generation)`：校验有限值、rot6d、工作空间和相对步长；一次接受一次目标平滑，更新 pending 候选。拒绝旧 generation。
- `servo_step(measured_pose, monotonic_time)`：重用已接受目标，计算并限幅 twist；不再次接受／平滑目标；由机器人对象提交 Franky motion。重复轮询、相机预览和取回 receipt 均无滤波副作用。

`robot.py.send_action()` 成为四条路线共同的公开动作入口；新 v2 返回／公开 prefilter receipt，不沿用旧返回值的 postfilter 含义。保留独立 legacy 入口供旧模型使用，禁止悄悄改变所有调用方。把 `send_human_command` 与 `send_human_target` 的 v2 分支缩为“意图 → 目标 → 共同 send_action”适配器；两者不能再直接构建不同 motion。

只有一条 100 Hz servo 线程拥有控制历史与 Franky motion 提交；接受线程仅投递有界最新目标。推理线程与采集线程不再竞争设备写入。运行中生成错误时不提交候选历史；motion 提交成功才提交速度历史和执行标记。当前 `send_action` 提交前更新 `_last_smoothed_target` 的行为要消除。

现有 `joint7_velocity != 0` 分支具有关节／冗余自由度语义，10D TCP action 不能无损表达。当前标准单臂 Gamepad 与接管调用传入 0、使用工具角速度；v2 保持此路线。直接调用旧 joint7 模式必须留在 legacy 并明确拒绝标成 v2，不通过 FK 假称已等价。

### 3.2 滤波频率与运动状态

目标历史系数初值 α=0.25，沿用当前函数定义：`new_smoothed = α × previous + (1-α) × accepted`，旋转用对应 SLERP。在新 action ID 的 30 Hz 边界更新一次；100 Hz 保持不变。

速度平滑从“每次调用固定系数”改为时间常数：初值对应 RTC 配置 β30=0.30（历史权重），`T_v = -(1/30)/ln(0.30) ≈ 0.02769s`，`β(dt_servo)=exp(-dt_servo/T_v)`；10 ms 时 β≈0.697。线／角速度统一执行该规则，长时间调度空洞触发 stale/stop，不以巨大 dt 补跑。这只是共同初始参数，并非实机最佳参数结论。

每次接受时先做 `limit_pose`，accepted target 在保持期间不随观测重写，保持 action 的含义固定。servo 每 tick 仍基于最新测量计算速度、做最终范数／工作空间保护；保护事件可见。位置／旋转到达死区若保留，必须所有适配器一致。Franky 自身还存在 C++ 插值／运动状态，不能把清空 Python EMA 当成完整控制器复位。

### 3.3 状态转换规则

| 边界 | 生命周期及队列 | Python 滤波历史 | Franky 与夹爪 |
| --- | --- | --- | --- |
| Episode 开始 | 新 generation；清 RTC／pending 命令；用新观测初始化 | target=实测姿态；速度=0；清旧时钟与 action ID | 完成旧 motion stop/join，随后保持待新动作；夹爪保持当前开度 |
| 策略 → 人工 Takeover | 立即失效旧推理请求；清动作队列；新 generation | 停止完成后按实测重锚、速度清零 | 先停／join，再允许人工目标；不把按键状态当运动重启许可 |
| 人工 → 策略 | 同样失效旧请求和队列；重新观测并 prime，等待新结果 | 同上；等待期间不能恢复旧目标 | 先停／join；等待时保持；首个新版本动作才恢复执行 |
| 普通接受／servo tick | generation 不变；只接受新 action ID | 保留历史；目标仅在新 action 更新，速度按 servo dt 更新 | 不逐帧 stop，不重复发送相同夹爪 goal |
| 暂停、故障、断连、超时 | 停止接受、作废代际和队列；禁止自动恢复运动 | stop/join 成功后清零；即使未启动 motion 也清零 | stop/join 失败保持 faulted，禁止宣称已复位；Ctrl-C 不等确认 |

统一 `reset_controller(reason)`，由 begin/end_human_control、Episode 启动／结束、暂停与恢复调用。停止 servo 并 join，再按既有安全接口结束并确认 Franky motion，最后重置历史；不能先清历史后让旧线程继续写回。超时只进入故障，不能生成新运动。

现有 `collect_franky` 接管已有 `policy_client.invalidate()`、清 `_command` 和 reprime，可复用，但须把 generation 贯穿命令、receipt、RTC 队列和迟到响应。人工优先级保持；恢复策略只能由明确归还操作触发。

### 3.4 录制与调用方

- `gamepad_teleop.py`：保留输入映射、Normal/Fast 和快捷姿态逻辑；所有意图最终经共同目标转换。只把成功执行 receipt 交给记录器。
- `gamepad_recording.py`：v2 用按生效边界匹配的 `AcceptedAction` 替代 `IntervalAssembler` 的速度积分；保存 prefilter 目标和 source/generation/action_id。旧 assembler 保留用于识别 legacy；不能用新定义解释旧文件。
- `collect_franky.py`：策略、普通接管、在线接管都从共同 receipt 获取 action；在线 `exec_chunk` 和帧动作含义一致。保留 reward／done／phase 算法，不能混存 raw policy、smoothed target 与 prefilter label。
- 相机与动作不必同一线程：以单调时钟记录 observed/accepted/applied，30 Hz 调度让每行对应一个明确生效动作；缺失、跨 generation 或失败的关联拒绝录制，不能用下一帧观测反推标签或以零填充。
- 不加新 ROS 控制栈；本方案只针对 Franky 单臂，ROS 双臂动作契约的整合是后续独立任务。

### 3.5 版本和迁移

新元数据应增加 `action_contract_version`、`controller_version`、`action_label_stage=clipped_pre_filter_target`、坐标框架、夹爪单位、30/100/30 频率、τ、限速与滤波参数／配置指纹。数据集、训练输出模型、部署预检都校验；缺失版本视为 legacy，不能默认启用 v2。

建议新编码 `franky_clipped_tcp_target_rot6d_gripper_fraction_v2`。不与 ROS 本轮 `gripper_recording_version=2` 混为同一版本，也不能仅依据 action shape=10 判断兼容。禁止原地续写或简单乘 τ/dt 批量修复旧数据：旧滤波、限幅、转动、夹爪滞回、丢帧和输入变化不具有统一可逆缩放。

## 4. 后续实施的最小验收集

1. **真实适配器等价**：参数化示教／普通接管／在线接管三入口与策略重放，用两独立共同控制器和各自状态；相同测量序列、初始化、非零历史、action 30 Hz 与 servo 100 Hz 的确定性时钟。示教侧真实生成和序列化标签，策略侧只读取标签；比较完整 `CartesianVelocityMotion` 提交序列与连续夹爪目标，而非让两个分支调用同一个假计算函数。
2. 同一测试序列覆盖三轴正反转、工具角速度／非单位姿态、工作空间裁剪、速度限幅、松手保持与多 servo tick。明确数值误差阈值（float32 后位姿约 1e-6、twist 约 1e-5，按误差传播验证），不放宽到掩盖 dt/τ 的程度。
3. 参数化 Episode、Takeover、归还、空闲 stop、stop/join 异常；断言旧 generation 永不提交，过滤历史只在边界重置，新目标只滤波一次，提交失败不提交有效 receipt。
4. 原有 recorder／RTC／在线 transition tests 验证字段、时间绑定和缓存契约；新增一项 legacy/v2 混用拒绝与保存再读一致性。无需新增大量页面测试。

完成上述离线验收才可称为“相同条件下提交命令等价”。Franky 内部插值和物理状态还受负载、网络和实时循环影响，不能据此声称机器人轨迹 100% 相同。真实首次启动、速度上限与滤波手感、接管边界、夹爪连续开度及模型效果仍需后续实机验收。

## 5. 实施顺序

后续按“共享执行器与状态机 → 四适配器及录制 receipt → 版本预检和配置 → 等价／切换验收”独立提交。先用上述可失败的等价测试驱动实现；本轮现状诊断在新版本落地时转为 legacy 回归或删除，不能要求新控制器保留已知缺陷。

## 6. 本轮离线验收记录

2026-11-06：新增 5 项现状诊断；连同已有 robot／gamepad recording 共 **102 passed**。完整离线回归 **987 passed, 6 skipped，31.50 秒**；6 项依赖未提供的本地模型资源，不能计作通过。存在既有 Torch/Starlette 和 legacy-cache 提示，未修改第三方依赖。

测试使用 `/data0/miniconda3/envs/vla/bin/python`，从 Workbench 源码运行，无 GPU、无网络模型下载、无真实设备连接。ROS 源码另用系统 Python 3.10 完成语法和轻量契约导入检查。原 `/data0/sht/VLA` 只做只读状态检查，工作树干净；未安装或替换其运行包。

复现（激活已有 vla 环境后，在 Workbench）：

```bash
bash scripts/test.sh -q tests/rlt/test_ros_gripper_commands.py tests/rlt/test_franky_action_contract_diagnostic.py
bash scripts/test.sh -q
```

本轮统一控制方案尚未实施，真实夹爪、ROS topic 心跳和新 Franky 控制器的硬件验收均未执行。
