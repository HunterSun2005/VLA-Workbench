# 离线测试与按需回归

测试服务于可观察的行为和故障边界。优先运行本次改动涉及的模块；跨模块接口、数据契约或依赖升级完成后，再运行全仓回归。不为普通文案、样式或可逆配置调整增加逐条断言，不通过跳过／隐藏必要测试缩短默认测试集。

## 独立开发副本

当前独立仓库为 `/data0/sht/VLA-Workbench`，在本地 main 维护，无 remotes。原采集仓库和已有环境不修改。

```bash
cd /data0/sht/VLA-Workbench
export VLA_ROOT="$PWD"
export VLA_PYTHON=/data0/miniconda3/envs/vla/bin/python
export HF_HOME=/tmp/vla-workbench-test-hf
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=''
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# scripts/test.sh 自行选择本副本源码、清理 ROS 环境并设置 OMP_NUM_THREADS=1。
nice -n 10 bash scripts/test.sh -q -p no:cacheprovider tests/web
```

上述环境仅影响本终端和子进程，不更改已安装的包。需要不同依赖时使用独立开发环境。测试只使用合成样本、模拟设备与临时目录；不要启动真实采集会话、相机或机械臂。不要复用活动 runtime 或给开发副本链接生产数据。合成数据写入、子进程和本地 socket 是离线集成测试的一部分。

## 如何选择测试

使用路径选择，避免仅用 `-k` 时仍导入整个测试目录。以下是起点；若同时修改公共协议或共享数据组件，应加入对应跨模块测试。

| 改动范围 | 优先运行（加在 `bash scripts/test.sh -q` 后） |
| --- | --- |
| Web 状态、接口、配置及监控 | `tests/web` |
| 单臂 gamepad、控制安全与采集 | `tests/control tests/rlt/test_cartesian_gamepad.py tests/rlt/test_dual_gamepad_record.py` |
| ROS／双臂启动与采集 | `tests/rlt/test_teleop_entrypoints.py tests/rlt/test_dual_gamepad_record.py tests/rlt/test_franka_dual_sft_recorder.py` |
| 数据字段、处理和视频 | `tests/rlt/test_franka_absolute_recorder.py tests/rlt/test_extract_dataset_features.py tests/rlt/test_dataset_processing.py tests/test_camera_image_padding.py tests/web/test_processing.py` |
| 部署协议、RTC 与人工接管 | `tests/rlt/test_franka_rollout_collection.py tests/rlt/test_franka_pi05_rtc_deployment.py tests/rlt/test_rlt_ac_rtc.py tests/rlt/test_franka_sft_server.py` |
| 核心训练、损失与模型保存 | `tests/rlt/test_losses.py tests/rlt/test_actor_critic.py tests/rlt/test_algorithm.py tests/rlt/test_training_step.py tests/rlt/test_pretrained_integration.py` |
| 在线训练、回放与恢复 | `tests/rlt/test_online_consistency.py tests/rlt/test_online_training.py tests/rlt/test_training_safety.py tests/rlt/test_training_upgrade.py` |
| LeRobot 依赖与工厂注册 | `tests/test_lerobot061.py tests/test_lerobot_registry_dataset.py tests/rlt/test_processor_rlt.py` |

运行单个故障用例可用 `文件.py::测试名`。完整离线回归：

```bash
nice -n 10 bash scripts/test.sh -q -p no:cacheprovider --durations=20
```

## 当前 ROS 迁移验收

完整离线回归为 678 passed、6 skipped、0 failed。ROS 编译、解释器隔离、无 Franky 的 wheel／CLI／Web demo 冒烟结果及现场待验收范围见 [ROS 验收报告](../docs/ROS_VALIDATION.md)。动作契约可从 `tests/actions` 开始；稀疏 RLT 时序见 `tests/rlt/test_sparse_action_timing.py`。

## 历史：迁移前的测试精简记录（2026-11-06）

以下历史轮次仅修改测试及文档，不修改控制、数据写入、模型实现或依赖。完整默认发现机制不变，没有新增排除目录、默认跳过标记或测试选择框架。

| 调整 | 仍保留的验证 |
| --- | --- |
| 删除仅检查 dataclass 存储输入形状和重复默认值的用例 | 配置 YAML 加载、派生值、打包默认配置；真实策略与 replay／训练集成 |
| 合并 DummyVLA 属性和策略中间输出形状检查 | 同一次真实前向检查输出和状态／参考块维度；保留监督损失与冻结语义 |
| 合并 actor／critic 的形状、标准差、采样、最小 Q、梯度检查 | 真实小网络执行并检查有限梯度；双 critic 仍各自参与计算 |
| 删除对单独生成的随机 mask 做统计的无效 dropout 测试 | 直接比较 actor 在全丢弃／保留参考输入下的输出，检验 training 参数实际生效 |
| 合并重复的基础训练步与软更新检查 | 公开 actor／critic update 验证参数确实变化和损失有限；保留精确 Polyak 公式、冻结参数、梯度隔离及端到端训练 |
| 合并 LeRobot policy 构造、权重键、保存加载和动作队列测试；将合成头 token 维度从 2048 缩为 32、encoder 从 3 层缩为 1 | 真实保存／加载往返、VLA 权重排除、RL 头存在、配置／特征校验；检查整块消费只推理一次、队列顺序及 reset 后重新推理 |
| 合并手动 phase 初始／切换／reset 检查，移除仅判断返回枚举类型的 learned 检查 | 保留独立 handover 测试中的真实阈值、防抖与冷却行为 |
| 删除一项重复启动器源码字符串检查和一项日志文案不存在检查 | 保留模拟启动器的单双臂 × teleop／rollout × check-only 运行矩阵；保留终端状态／诊断日志隔离及慢输出测试 |

控制安全、工作空间、碰撞配置、遥操作中立输入／断连、姿态恢复、资源独占、保存失败、协议版本与迟到响应、50D／100D 数据、缓存一致性、模型工厂 spawn、Web 重连与有界监控等用例均保留。没有因运行较慢而删除跨进程锁和 spawn 边界检查；也没有把所有读取源码的检查一概删除。

### 本机结果

同一现有 Python 3.12 环境，禁用 GPU、单线程数值库、降低进程优先级，在新副本执行：

| 状态 | 通过 | 跳过 | 失败 | pytest 报告时间 |
| --- | ---: | ---: | ---: | ---: |
| 修改前基线 `6fffc50` | 1005 | 6 | 0 | 55.87 s |
| 精简后 | 974 | 6 | 0 | 33.12 s |

减少 31 个独立用例。时间为各一次完整运行的观测值，包含导入／文件缓存和主机负载差异，不作为稳定加速比保证。第一次合并验证发现新测试误以为 reset 后仍是 RL 模式；按现有接口在 reset 后显式选择 RL，再次完整回归通过，未改变生产行为。

6 个跳过项与基线相同：5 项需要 `VLA_TEST_SFT_PI05_DIR` 指定的真实 SFT processor 文件，1 项需要本地 PI0.5 基础 checkpoint。它们未在本轮验证，不算通过；不为此复制旧权重。原采集目录工作树仍干净，未重启服务、连接机器人或更新安装环境。
