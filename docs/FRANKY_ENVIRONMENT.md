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

会话配置 `transport.worker_python` 优先于环境变量 `FRANKY_PYTHON`；二者都未指定时当前使用 `/usr/bin/python3`。在启动 Web 的环境里设置已准备好的独立解释器，例如：

```bash
export FRANKY_PYTHON=/absolute/path/to/new/franky-env/bin/python
"$FRANKY_PYTHON" -m pip install --no-deps \
  https://github.com/Brunch-Life/franky/releases/download/wheels-libfranka-0.19.0/franky_control-1.1.3-cp310-cp310-manylinux_2_28_x86_64.whl
env -u PYTHONHOME -u LD_LIBRARY_PATH -u LD_PRELOAD \
  PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src" \
  "$FRANKY_PYTHON" -m vla.control.franky.worker_main --check-only
```

在仓库根目录执行。安装命令仅针对已准备好的独立环境；`--no-deps` 要求该环境已有 NumPy、SciPy、websockets。随后 `--check-only` 只导入并检查依赖，不创建 `Robot`、tracker、socket 或硬件租约。失败时返回非零；成功时输出 JSON 指纹。不要用真实会话启动来代替依赖检查。

worker 启动时执行同一检查：要求 tracker 的目标、增益和停止 API，读取 `/proc/self/maps` 确认实际加载且仅加载一份 libfranka 0.19.0，并进行 NumPy/Eigen 位姿转换检查。支持普通库名和 auditwheel 哈希库名；其他版本、混载、无法确认版本、已删除的映射均拒绝。指纹包含实际 native binding、Python tracker 和 libfranka 路径及 SHA-256，以及 NumPy 版本、位姿检查结果，写入现有控制指纹；不能与此前不同指纹的数据直接续录。

## ROS 与旧安装

`ros_ws/dependencies.repos` 同样固定到 libfranka 0.19.0。`scripts/build_ros.sh` 在构建前核对源码提交标记和版本，阻止悄悄沿用旧 0.20.4 源码。`scripts/fetch_ros.py` 遇到旧源码也会拒绝覆盖，需要按新清单准备新的工作区及构建/安装目录。

本仓库的 libfranka 源码已在用户授权后替换为上表固定的 0.19.0 提交，并完成 ROS 工作区重建。旧源码备份在工作区之外，`ros_ws/src/third_party/libfranka`、`ros_ws/build`、`ros_ws/install` 中已无旧 0.20.x 库。未修改旧 `/data0/sht/VLA`、已有 Conda/用户 Python 或系统库。`build_ros.sh` 只扫描 `src`，避免将 `python_deps` 内的安装包误识别为待编译源码。

重建命令和无硬件验证：

```bash
/usr/bin/python3 scripts/fetch_ros.py
CMAKE_BUILD_PARALLEL_LEVEL=2 MAKEFLAGS=-j2 bash scripts/build_ros.sh
bash -c 'source scripts/ros_env.sh; /usr/bin/python3 scripts/check_ros.py'
```

`check_ros.py` 检查采集器和外设 Python 依赖能否导入，并使用 `ldd` 确认 `franka_hardware`、`franka_semantic_components`、机械臂控制器和状态广播器都解析到本工作区的 `ros_ws/install/libfranka/lib/libfranka.so.0.19.0`，拒绝误用系统库或未解析的依赖，不启动 ROS 节点。

ROS 重建不会安装 Franky Python wheel。当前默认 `/usr/bin/python3` 在禁用用户包后仍缺少 `franky`；此前 wheel 验证使用的是临时目录，不能等同于已部署 worker 环境。启用真实 Franky 会话前仍需配置通过 `--check-only` 的独立解释器。

## 验证边界

离线单元测试覆盖错误/混合/未知版本拒绝、tracker 缺失、文件指纹、`--check-only` 无硬件副作用，以及旧 ROS 源码拒绝。临时源码编译与导入只能证明本机构建/API/动态链接兼容性；FCI/固件连通、实时调度、负载设置、真实接触和运动连续性仍需在部署阶段单独验收。

本次 0.19 wheel 在临时目录完成离线检查：native 扩展实际依赖 `libfranka-f9564df3.so.0.19.0`，包含 tracker，清除用户包和动态库环境变量后通过 `--check-only` 及位姿转换检查。未构造真实 `Robot`，也未将 wheel 安装到现有运行环境。

回归入口（模型、ROS 分开执行）：

```bash
VLA_PYTHON=/path/to/existing/model/python bash scripts/test.sh \
  tests/control/test_franky_binding.py tests/control/test_franky_worker.py \
  tests/control/test_libfranka_source.py -q
bash scripts/test_ros.sh \
  tests/control/test_franky_binding.py tests/control/test_franky_worker.py \
  tests/control/test_libfranka_source.py -q
```
