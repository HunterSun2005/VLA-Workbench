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

启动顺序为离线依赖预检 → worker 全部机械臂 ready → ROS 外设 ready → collector。worker 独占机械臂租约，ROS 仅持有夹爪等外设资源；ready 必须携带全部有效状态、binding 指纹和阻抗参数，collector 使用显式会话 ID 和已验证连接生成 recording contract。关闭或启动失败时先撤销目标、停止 tracker、等待 worker 退出释放租约，再关闭外设。

Unix IPC 校验会话、generation、sequence 和 request ID。生命周期命令清空旧目标；重复或过期消息不能重新启动运动。独立 watchdog 分别检查 `control_discontinuity_timeout_s` 目标流和 `gamepad_state_timeout_s` 通信/状态期限，健康轮询不能掩盖目标中断。正常的一次请求一次连接不算失联；半包、超时、状态失效、异步 tracker 异常或任一臂失败都停止双臂并锁存故障。

运行中调参按 request ID 确认，等待期间暂停新目标。刚度通过 `set_gains` 更新，结构参数停止并重建 tracker；任一臂失败则整体 fault，保留上次完整确认的参数。重连、健康查询、普通目标及重建不会隐式恢复；必须显式“恢复”，再启用控制。Gamepad 松杆保留最后 TCP 参考，GELLO FK 和策略更新同一 tracker。真实接触前仍需核实工具、负载和安全设置。

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
