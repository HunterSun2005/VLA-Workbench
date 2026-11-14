# Franky / libfranka 环境

统一机械臂执行链要求 **libfranka 0.19.0 + `CartesianImpedanceTracker`**。Franka 系统 5.9.2 的兼容矩阵支持 `libfranka >= 0.18.0`，本仓库固定到 0.19.0 以使用含 tracker 的 Brunch-Life fork wheel。ROS 负责输入、夹爪、相机和协调器；独立 Franky worker 独占 FCI，模型环境不加载机器人控制 binding。

## 固定源码

| 组件 | 来源与固定提交 |
| --- | --- |
| libfranka 0.19.0 | [frankarobotics/libfranka](https://github.com/frankarobotics/libfranka/tree/4df37dd2b527f41ef34b76d6d4d3805864b09993)，`4df37dd2b527f41ef34b76d6d4d3805864b09993` |
| libfranka common | [libfranka-common](https://github.com/frankarobotics/libfranka-common/tree/d96c221a32c343384cbe2f77c61116ed58b84231)，`d96c221a32c343384cbe2f77c61116ed58b84231` |
| 含 tracker 的 Franky | [Brunch-Life/franky](https://github.com/Brunch-Life/franky/tree/d6ba60f74010ddb03f517e40768d5753c0dbacb8)，`d6ba60f74010ddb03f517e40768d5753c0dbacb8`（`wheel-builds-enhance-impedance-control` 分支快照） |
| Franky 0.19 wheel | [`franky_control-1.1.3-cp310-cp310-manylinux_2_28_x86_64.whl`](https://github.com/Brunch-Life/franky/releases/download/wheels-libfranka-0.19.0/franky_control-1.1.3-cp310-cp310-manylinux_2_28_x86_64.whl) |
| Franky ruckig（源码构建时） | [pantor/ruckig](https://github.com/pantor/ruckig/tree/8a10e0129e0fc854215c0699da3ed42139bdc39b)，`8a10e0129e0fc854215c0699da3ed42139bdc39b` |

`wheels-libfranka-0.19.0` 发布提供 cp39/cp310/cp311/cp312/cp313/cp314 wheel；当前 ROS worker 使用 Python 3.10，因此选择上表的 `cp310 manylinux_2_28` wheel。该 wheel 的 native 扩展实际依赖 `libfranka-f9564df3.so.0.19.0`，并包含 `CartesianImpedanceTracker`。不能把其他版本 wheel 重命名、替换动态库或只依据 `franky-control==1.1.3` 判断兼容性。

## 构建与选择解释器

优先在独立 worker 环境安装上表 0.19 wheel；所有 Python 环境均使用新目录，不在现有 ROS/模型环境中升级。若从源码构建，libfranka 依赖 Eigen、Poco、Pinocchio、TinyXML2、fmt；Franky 还依赖 **pybind11 2.13.6**（该 fork 的构建版本）、Python 开发文件和 ruckig。系统 pybind11 2.9.1 与 NumPy 2.2.6 的组合曾在离线 `Affine(np.eye(4))` 检查中崩溃，不能只验证 import。上游 libfranka 0.19.0 的 CPack 配置需要 Git 标签元数据；本仓库的 `fetch_ros.py` 应用 `libfranka-archive-version.patch`，为没有 `.git` 的归档使用项目版本及 `.vla-revision` 中的固定提交生成打包标识，不修改控制实现。

如果从源码构建 Franky，CMake 输出必须是 `Found Franka Version: 0.19.0`。最终环境需包含 Franky Python 包及编译扩展、可解析的 0.19.0 动态库依赖，以及 NumPy、SciPy、websockets。wheel 自带 Franky 依赖库和 RPATH；Web 会清理 `LD_LIBRARY_PATH`，仍不能把 ROS shell 中的 `/usr/local/lib` 0.19 或旧 0.20.4 库混入 worker。不要混用模型 Python 与 ROS Python 的二进制扩展。

在仓库根目录运行可重复安装脚本：

```bash
/usr/bin/python3 scripts/install_franky_worker.py
```

脚本读取 `scripts/franky-worker.lock.json`，固定 Brunch-Life cp310 wheel、NumPy 1.26.4、SciPy 1.15.3、websockets 16.0 的 URL 和 SHA-256。在 `runtime/franky/envs/<锁文件哈希>` 创建 `venv --copies --without-pip`，通过已有系统安装工具向新环境安装已校验的 wheel，不更改 ROS、模型或用户环境。环境不继承系统/用户 site-packages。安装及离线验证成功后原子切换 `runtime/franky/current`；相同锁文件再次执行只检查，不重新安装。wheel、来源和安装指纹保留在忽略目录 `runtime/franky`，运行不依赖 `/tmp`。

解释器优先级为 `transport.worker_python` → `FRANKY_PYTHON` → 仓库 `runtime/franky/current/bin/python`，不再静默回退到系统 Python。保留解释器符号链接路径以维持虚拟环境身份。Web“检查配置”、终端 `--check-only` 和实际启动前调用同一个有超时的独立进程预检；预检清除 ROS、Conda、用户包、`LD_LIBRARY_PATH` 和 `LD_PRELOAD` 污染。冷磁盘导入可能较慢，依赖预检允许 120 秒，与运行时控制超时无关。

直接离线检查：

```bash
env -u PYTHONHOME -u LD_LIBRARY_PATH -u LD_PRELOAD \
  PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src" \
  runtime/franky/current/bin/python -m vla.control.franky.worker_main --check-only
```

`--check-only` 不创建真实 `Robot`、tracker、socket 或硬件租约。要求 tracker 的目标、增益和停止 API，读取 `/proc/self/maps` 确认实际只加载 libfranka 0.19.0，并执行 NumPy/Eigen `Affine` 转换。其他版本、混载、未知或删除的映射均拒绝。控制指纹使用实现版本、IPC 版本、native binding / Python tracker / libfranka 文件哈希及参数；安装路径仅作诊断，移动环境不会改变语义身份。

## 会话与故障

Franky 模式的运动学模型默认从当前 ROS 工作区安装的 `franka_description` 离线展开，按左右臂添加 `left_` / `right_` 前缀，关闭 `ros2_control` 和手部模型；此过程不启动节点或连接机械臂。可用 `FRANKA_URDF` 显式指定展开后的 URDF，文件须包含配置中的 base/tip 链接；指定文件缺失或模型不匹配时直接报错，不改用其他模型。有效模型写入运动学指纹。

启动顺序为离线依赖预检 → worker 全部机械臂 ready → ROS 外设 ready → collector。worker 独占机械臂租约，ROS 仅持有夹爪等外设资源；ready 必须携带全部有效状态、binding 指纹和阻抗参数，collector 使用显式会话 ID 和已验证连接生成 recording contract。关闭或启动失败时先撤销目标、停止 tracker、等待 worker 退出释放租约，再关闭外设。

Unix IPC 校验会话、generation、sequence 和 request ID。生命周期命令清空旧目标；重复或过期消息不能重新启动运动。独立 watchdog 分别检查 `control_discontinuity_timeout_s` 目标流和 `gamepad_state_timeout_s` 通信/状态期限，健康轮询不能掩盖目标中断。正常的一次请求一次连接不算失联；半包、超时、状态失效、异步 tracker 异常或任一臂失败都停止双臂并锁存故障。

IPC 收包使用配置中的通信期限（默认 0.5 秒），整包的头部与正文共享截止时间，不使用独立的 50 毫秒短超时。连接后短暂调度延迟在通信期限内可继续收包；等待收包不会延长目标流期限（默认 0.1 秒）。客户端报错包含连接/发送/接收阶段、请求 ID、generation、耗时和 socket 路径，worker 日志记录首次锁存故障及其来源。重试查询不会清除故障或恢复运动。

运行中调参按 request ID 确认，等待期间暂停新目标。刚度通过 `set_gains` 更新；只在结构参数实际变化时停止并重建 tracker，整表提交中未变化的字段不触发重建；任一臂失败则整体 fault，保留上次完整确认的参数。重连、健康查询、普通目标及重建不会隐式恢复；必须显式“恢复”，再启用控制。Gamepad 以实测 TCP 初始化参考、沿实测工具轴积分姿态，松杆保留最后 TCP 参考；GELLO FK 和策略更新同一 tracker。Web 的启动初值与有效参数、草稿和请求确认分离，详见 [Web 使用说明](WEB_PANEL.md)。真实接触前仍需核实工具、负载和安全设置。

## 绝对位姿的静态误差补偿

绝对逻辑 TCP 目标接口保持不变。worker 提供显式启用的静态误差积分补偿，
继续由现有 1000 Hz `CartesianImpedanceTracker` 执行，不使用另一个运动线程、
速度 SDK 模式或关节 IK。默认 `tracking_integral_gain=0`，不添加参考偏置。

| `control.compliance` 参数／Web 阻抗面板 | 默认值 | 含义 |
|---|---:|---|
| `tracking_integral_gain` | 0 | 静态误差积分增益，单位 1/s；0 关闭，允许 0–1 |
| `tracking_max_offset_m` | 0.01 | XYZ 补偿向量模长上限，m，最大 0.02 |
| `tracking_max_offset_rad` | 0.20 | 基座系旋转向量补偿模长上限，rad，最大 0.35 |
| `tracking_max_linear_rate` | 0.002 | 平移补偿变化速率上限，m/s，最大 0.01 |
| `tracking_max_angular_rate` | 0.03 | 旋转补偿变化速率上限，rad/s，最大 0.10 |

例如在新会话 Web 阻抗面板将“静态补偿增益”设为 `0.3`，保留其余补偿限制，
通过配置检查后启动；运行中应用需要等待 worker 的请求确认。终端配置同样可在
YAML 的 `control.compliance` 设置这些字段。配置文件或面板草稿不是运行时有效值。
补偿参数修改会重建 tracker、清空原补偿并从新鲜测量重新准备，不能隐式恢复故障运动。

补偿仅在连续接受相同目标至少 0.5 秒、实测误差不超过 30 mm／0.35 rad 时学习；
位置误差小于 0.5 mm、姿态误差小于 0.3°时保留已有补偿，不继续积分。
目标变化、关闭补偿或超出误差捕获范围时停止积分，并按速率逐步撤掉原偏置。
姿态补偿左乘于逻辑目标，安装旋转仍只在 TCP 转换层右乘一次。
补偿只在接受目标包时推进，每包最多一个 30 Hz 时隙；查询、重复包、旧代次包
不能推进积分，ACK 等待期间不补算时间。停止、故障、恢复和新 reset 都清空补偿。

补偿同时受向量模长、速率、工作空间和原单步限制约束；内层误差裁剪已饱和时，
不再向饱和方向积分。Franky 的旋转裁剪实际作用于基座系四元数误差向量部，
不能直接当作角度上限。补偿不会增加原有力矩／力矩变化率上限，无法克服超出
现有控制能力的扰动。它会增加持续接触力，接触任务应保持关闭，不能替代质量、
质心和 TCP 标定，也不保证实机达到某一误差。

接受动作及 RTC 承诺前缀保持原值；`nominal_tcp_pose` 是单步受限后的未补偿目标，
`reference_tcp_pose`／目标 ACK 的 `references` 是实际提交 tracker 的补偿后参考。
Web 输入诊断另显示 `nominal_tcp`、`tracker_reference_tcp`、`tracking_offset`
（XYZ m + 基座系旋转向量 rad）和 `tracking_status`。schema 5 不变，异步审计
`tracking_correction` 保存这两种参考、偏置和状态，原有硬件参考审计保持实际值。
worker 实现指纹版本更新为 7，参数参与控制身份；既有数据集不能混入新控制指纹。

带恒定负载偏差的一阶仿真验证收敛，真实 wheel + 假 Robot 验证传递、坐标、
边界和撤销语义。这些结果不代表真实机械臂精度或负载标定已经验收。

## ROS 与旧安装

`ros_ws/dependencies.repos` 同样固定到 libfranka 0.19.0。`scripts/build_ros.sh` 在构建前核对源码提交标记和版本，阻止悄悄沿用旧 0.20.4 源码。`scripts/fetch_ros.py` 遇到旧源码也会拒绝覆盖，需要按新清单准备新的工作区及构建/安装目录。

本仓库的 libfranka 源码已在用户授权后替换为上表固定的 0.19.0 提交，并完成 ROS 工作区重建。`ros_ws/src/third_party/libfranka`、`ros_ws/build`、`ros_ws/install` 中已无旧 0.20.x 库。未修改旧 `/data0/sht/VLA`、已有 Conda/用户 Python 或系统库。`build_ros.sh` 只扫描 `src`，避免将 `python_deps` 内的安装包误识别为待编译源码。

重建命令和无硬件验证：

```bash
/usr/bin/python3 scripts/fetch_ros.py
CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2 bash scripts/build_ros.sh
bash -c 'source scripts/ros_env.sh; /usr/bin/python3 scripts/check_ros.py'
```

`check_ros.py` 检查采集器和外设 Python 依赖能否导入，并使用 `ldd` 确认 `franka_hardware`、`franka_semantic_components`、机械臂控制器和状态广播器都解析到本工作区的 `ros_ws/install/libfranka/lib/libfranka.so.0.19.0`，拒绝误用系统库或未解析的依赖，不启动 ROS 节点。

ROS 重建和 Franky wheel 安装是两个独立步骤。当前默认 worker 来自仓库 `runtime/franky/current`；新检出仓库须先执行上面的安装脚本。

## 验证边界

离线单元测试覆盖错误/混合/未知版本拒绝、tracker 缺失、文件指纹、`--check-only` 无硬件副作用，以及旧 ROS 源码拒绝。临时源码编译与导入只能证明本机构建/API/动态链接兼容性；FCI/固件连通、实时调度、负载设置、真实接触和运动连续性仍需在部署阶段单独验收。

真实 wheel 测试使用持久环境与假 Robot，在独立 worker 子进程中运行真实 tracker，覆盖目标、调参、重建、停止、恢复及故障。ROS 600 秒固定状态 harness 验证协调器时序，不等同于 Franky FCI 的物理运动验收。测试报告保存在 `artifacts/validation`，缺失依赖或跳过项不计通过。

回归入口（模型、ROS 分开执行）：

```bash
VLA_PYTHON=/path/to/existing/model/python bash scripts/test.sh \
  tests/control/test_franky_binding.py tests/control/test_franky_worker.py \
  tests/control/test_franky_process.py tests/control/test_franky_runtime.py \
  tests/web/test_franky_startup.py -q
bash scripts/test_ros.sh \
  tests/control/test_franky_binding.py tests/control/test_franky_worker.py \
  tests/control/test_libfranka_source.py -q
```

GELLO 启动器将本检出的 `ros_ws/src/franka_gello_state_publisher` 放在子节点 Python 搜索路径首位，以免旧的复制安装遮蔽采样新鲜度修复；不修改已有 ROS 安装环境。驱动缓存保存采样时间和序号；JointState 的 stamp 为采样时间，最后一个 `gello_gripper` 位置为同帧夹爪开度，collector 不再从独立无时间戳的 Float32 消息刷新 GELLO 夹爪输入。

应用层使用一个 30 Hz 控制回调：读取 worker 缓存与外设输入、生成并接受目标、提交一次 IPC、关联记录。tracker 的 1000 Hz 阻抗闭环和 worker 独立 watchdog 不受上层调度控制。`max_step` / `max_step_rad` 仍限制每次参考变化，不因降频而放大；目标流超时保持 0.1 s，健康查询不能代替控制目标。控制指纹包含 `single_30hz`，旧控制指纹数据不能直接续写或混采。

## 单个应用控制循环

采集、部署和姿态恢复统一使用单个 30 Hz 回调，异步 IPC 最多一条在途目标，无待发目标队列。等待确认时冻结人工积分和记录，恢复后不补算；策略按当前真实时隙取动作。Franky 保持独立 1000 Hz 阻抗闭环，固定 wheel/libfranka 0.19.0。新会话只接受 Franky/TCP，旧独立 tracking rate 和 IK 执行参数明确报错。

启用先准备全部 tracker；停止、故障、恢复后须显式启用。目标请求／流默认 100 ms、通信／状态 500 ms，健康查询不能代替目标确认。运行调参暂停目标，匹配请求确认才结束等待，代际变化使迟到确认失效。worker 通信失败有界关闭，不自动恢复运动。

相机偶发缺帧跳过整条记录，连续过期超过 `sync_tolerance_ms` 才停控。提交时冻结数据，匹配 IPC v2 确认后记录；实际参考和应用时间写入审计。缺帧和 IPC 等待形成 slot／segment 缺口，训练和 Replay 不跨缺口。旧指纹的 checkpoint 需使用匹配的新数据训练，不做静默转换。详见 [动作契约](ROS_ACTION_CONTRACT.md)。
