# Franky / libfranka 环境

统一机械臂执行链要求 **libfranka 0.18.0 + `CartesianImpedanceTracker`**。ROS 负责输入、夹爪、相机和协调器；独立 Franky worker 独占 FCI，模型环境不加载机器人控制 binding。

## 固定源码

| 组件 | 来源与固定提交 |
| --- | --- |
| libfranka 0.18.0 | [frankarobotics/libfranka](https://github.com/frankarobotics/libfranka/tree/c5c66bb25a987f122bb424549f5e3b283a41d2f2)，`c5c66bb25a987f122bb424549f5e3b283a41d2f2` |
| libfranka common | [libfranka-common](https://github.com/frankarobotics/libfranka-common/tree/d96c221a32c343384cbe2f77c61116ed58b84231)，`d96c221a32c343384cbe2f77c61116ed58b84231` |
| 含 tracker 的 Franky | [Brunch-Life/franky](https://github.com/Brunch-Life/franky/tree/d6ba60f74010ddb03f517e40768d5753c0dbacb8)，`d6ba60f74010ddb03f517e40768d5753c0dbacb8`（`wheel-builds-enhance-impedance-control` 分支快照） |
| Franky ruckig | [pantor/ruckig](https://github.com/pantor/ruckig/tree/8a10e0129e0fc854215c0699da3ed42139bdc39b)，`8a10e0129e0fc854215c0699da3ed42139bdc39b` |

2026-11-11 检查用户提供的 [wheels-libfranka-0.18.0 标签](https://github.com/Brunch-Life/franky/releases/tag/wheels-libfranka-0.18.0)时，页面仅列源码包，没有 wheel；该标签的源码也没有 `CartesianImpedanceTracker`。因此它不是可直接安装的新链路依赖。应使用上表含 tracker 的固定源码，针对 0.18.0 构建，不能把 0.19 wheel 重命名、替换动态库或只依据 `franky-control==1.1.3` 判断兼容性。

## 构建与选择解释器

在独立目录准备固定源码及对应子模块；所有构建、安装前缀和 Python 环境均使用新目录。libfranka 依赖 Eigen、Poco、Pinocchio、TinyXML2、fmt；Franky 还依赖 **pybind11 2.13.6**（该 fork 的构建版本）、Python 开发文件和 ruckig。缺失依赖应先单独准备，不在现有 ROS/模型环境中升级。系统 pybind11 2.9.1 与 NumPy 2.2.6 的组合曾在离线 `Affine(np.eye(4))` 检查中崩溃，不能只验证 import。上游 libfranka 0.18.0 的 CPack 配置需要 Git 标签元数据；直接解压源码归档会在 `SetVersionFromGit.cmake` 报错，不能将其误判为 C++/ABI 不兼容。

libfranka 构建指定独立 `CMAKE_INSTALL_PREFIX`，关闭 `BUILD_TESTS`、`BUILD_EXAMPLES`；Franky 构建明确指定该前缀中的 `Franka_DIR`、2.13.6 的 `pybind11_DIR` 和独立解释器的 `PYTHON_EXECUTABLE`，关闭 `BUILD_TESTS`、`BUILD_EXAMPLES`、`BUILD_PYTHON_STUBS`。检查 CMake 输出必须是 `Found Franka Version: 0.18.0`。最终环境需包含 Franky Python 包及编译扩展、可解析的 0.18.0 动态库依赖，以及 NumPy、SciPy、websockets。通过安装 RPATH 或打包依赖保证动态库可解析；例如使用 `/opt/openrobots` 的 Pinocchio 时，libfranka 的 `CMAKE_INSTALL_RPATH` 需要包含相应库目录。Web 会清理 `LD_LIBRARY_PATH`，不能依靠 ROS shell 的路径碰巧使导入成功。不要混用模型 Python 与 ROS Python 的二进制扩展。

会话配置 `transport.worker_python` 优先于环境变量 `FRANKY_PYTHON`；二者都未指定时当前使用 `/usr/bin/python3`。在启动 Web 的环境里设置已准备好的独立解释器，例如：

```bash
export FRANKY_PYTHON=/absolute/path/to/new/franky-env/bin/python
env -u PYTHONHOME -u LD_LIBRARY_PATH -u LD_PRELOAD \
  PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src" \
  "$FRANKY_PYTHON" -m vla.control.franky.worker_main --check-only
```

在仓库根目录执行。此命令只导入并检查依赖，不创建 `Robot`、tracker、socket 或硬件租约。失败时返回非零；成功时输出 JSON 指纹。不要用真实会话启动来代替依赖检查。

worker 启动时执行同一检查：要求 tracker 的目标、增益和停止 API，读取 `/proc/self/maps` 确认实际加载且仅加载一份 libfranka 0.18.0，并进行 NumPy/Eigen 位姿转换检查。支持普通库名和 auditwheel 哈希库名；其他版本、混载、无法确认版本、已删除的映射均拒绝。指纹包含实际 native binding、Python tracker 和 libfranka 路径及 SHA-256，以及 NumPy 版本、位姿检查结果，写入现有控制指纹；不能与此前不同指纹的数据直接续录。

## ROS 与旧安装

`ros_ws/dependencies.repos` 同样固定到 libfranka 0.18.0。`scripts/build_ros.sh` 在构建前核对源码提交标记和版本，阻止悄悄沿用旧 0.20.4 源码。`scripts/fetch_ros.py` 遇到旧源码也会拒绝覆盖，需要按新清单准备新的工作区及构建/安装目录。

修改源码锁定不等于更新本机已有二进制。本次不修改旧 `/data0/sht/VLA`、已有 Conda/用户 Python、`/usr/local`、ROS 第三方 checkout 或 `ros_ws/install`；这些位置仍可能含 0.18、0.19 或 0.20.4。不要把旧安装计为完成迁移。统一路线不启动 ROS 机械臂控制器，因此旧 ROS arm 库不应进入 Franky worker；worker 的实际加载检查是最后一道启动约束。

## 验证边界

离线单元测试覆盖错误/混合/未知版本拒绝、tracker 缺失、文件指纹、`--check-only` 无硬件副作用，以及旧 ROS 源码拒绝。临时源码编译与导入只能证明本机构建/API/动态链接兼容性；FCI/固件连通、实时调度、负载设置、真实接触和运动连续性仍需在部署阶段单独验收。

本次上表源码在临时目录完成构建：Python 3.10.12、Eigen 3.4.0、pybind11 2.13.6、Pinocchio 4.1.0、NumPy 2.2.6。清除用户包和动态库环境变量后，`--check-only` 确认加载 libfranka 0.18.0 并通过位姿转换；真实编译的 tracker 配合假机器人对象，通过创建、目标更新、增益更新、结构参数重建、停止和恢复检查。未构造真实 `Robot`，也未将临时产物安装到运行环境。

回归入口（模型、ROS 分开执行）：

```bash
VLA_PYTHON=/path/to/existing/model/python bash scripts/test.sh \
  tests/control/test_franky_binding.py tests/control/test_franky_worker.py \
  tests/control/test_libfranka_source.py -q
bash scripts/test_ros.sh \
  tests/control/test_franky_binding.py tests/control/test_franky_worker.py \
  tests/control/test_libfranka_source.py -q
```
