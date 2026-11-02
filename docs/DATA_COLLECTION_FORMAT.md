# 完整原始采集格式

采集保存 29D 与 31D 的字段并集，去除共有的前 10 维，得到每臂 **50D**。所有实时采集入口共用此定义：单左臂 GELLO／gamepad、双臂 GELLO／gamepad，以及 Franky／ROS 部署回合记录。双臂按左臂 50D、右臂 50D 拼接；字段加 `left_` / `right_` 前缀。

| 单臂索引（从 0 起） | 字段 | 维度 | 单位／来源 |
| --- | --- | --- | --- |
| 0–2 | `tcp_position_*` | 3 | m，实测 TCP 位置 |
| 3–8 | `tcp_rot6d_*` | 6 | 无量纲，姿态矩阵前两列 |
| 9 | `gripper_position` | 1 | m，实测夹爪开口 |
| 10–12 | `base_tcp_force_*` | 3 | N，基坐标系外力估计 |
| 13–15 | `base_tcp_torque_*` | 3 | N·m，基坐标系外力矩估计 |
| 16–18 | `base_tcp_linear_velocity_*` | 3 | m/s，相邻 TCP 位姿差分 |
| 19–21 | `base_tcp_angular_velocity_*` | 3 | rad/s，相邻 TCP 姿态差分 |
| 22–28 | `joint_position_1..7` | 7 | rad，实测关节位置 |
| 29–35 | `joint_torque_1..7` | 7 | N·m，`tau_J` 实测关节力矩 |
| 36–42 | `external_joint_torque_1..7` | 7 | N·m，`tau_ext_hat_filtered` 外部关节力矩估计 |
| 43–49 | `joint_velocity_1..7` | 7 | rad/s，`dq` 实测关节速度 |

原 29D 完整保留在前 29 维；原 31D 对应索引 0–9 与 29–49。动作仍是每臂 10D（目标 TCP XYZ + rot6d + 夹爪开度比例），夹爪动作单位仍为 0–1。实时策略不会直接接收 50D／100D；原策略提取每臂前 10 维。TCP 差分速度首帧延续原规则为零；新加的关节传感器字段缺失、维度不对或非有限值会拒绝样本，不补零伪造。

Franky 从同一次 RobotState 读取位置、力矩、速度；ROS 从 `measured_joint_state.position/effort/velocity` 和 `tau_ext_hat_filtered.effort` 读取，并按关节名称重排到 J1–J7。力和力矩是估计量，记录不等于经过标定的接触力传感器读数。

## 提取训练字段

先采集到 `datasets/raw/`，再用 `vla-extract-dataset-features` 按字段名提取到 `datasets/sft/`。源数据不修改；输出重算状态统计并保留动作和视频。

```bash
conda activate vla
source scripts/model_env.sh
vla-extract-dataset-features \
  --source-root datasets/raw/franka_single_left_gamepad_full \
  --config src/vla/configs/dataset_features/franka_single_left_tcp_rot6d_10d.yaml \
  --output-root datasets/sft/franka_single_left_gamepad_10d
```

随包提供以下 YAML（安装后也可通过 `vla.paths.resource("configs/dataset_features/<文件名>")` 获取）：

| 配置文件 | 输出状态 |
| --- | --- |
| `franka_single_left_tcp_rot6d_10d.yaml` | TCP + 夹爪，10D |
| `franka_single_left_tcp_rot6d_22d.yaml` | TCP + 夹爪 + 基坐标系力／力矩／TCP 速度，22D |
| `franka_single_left_state_29d.yaml` | 原标准 29D |
| `franka_single_left_state_31d.yaml` | 原力矩／关节速度 31D |
| `franka_dual_tcp_rot6d_20d.yaml` | 左右 TCP + 夹爪，20D |

修改 YAML 的 `components` 开关与 `output_order` 可选择其他字段组合。提取以数据集 `meta/info.json` 的字段名定位，不依赖固定列号；允许额外字段或未启用字段缺失，拒绝缺少所选字段、重复字段名和动作维度不符。已有 29D/31D/58D 数据也可按可用字段提取。训练配置仍需匹配提取后的状态维度；31D 不会自动启用新的 force-VLA 训练算法。`relative` 动作统计依赖状态与动作前部 TCP 块对齐，双臂推荐先提取到 20D。

## 续采和升级

旧数据无法补出过去未记录的传感器值，因此禁止向旧 29D/31D/58D 数据集中追加新格式。Web 预检与写入器均校验状态字段；写入器在清理恢复文件之前拒绝不兼容数据。Web 默认目录改为 `datasets/raw/franka_<arms>_<source>_full`，仍为相对路径，手工指定路径不会被选择变化覆盖。

旧 `_31d` 脚本／模块名仅作为转发入口，实际始终写 50D；历史 `schema_31d` 保留给离线读取。更新后先正常结束已有采集会话，再重启 Web 服务并刷新页面，新会话才使用新代码。

验收使用合成数据与模拟设备，不连接或移动机器人；真实 Franky／ROS 传感器输出及实机采集仍待硬件验收。

## 本次离线验收（2026-11-02）

- 完整仓库测试：`bash scripts/test.sh -q -p no:cacheprovider tests`，859 passed、6 skipped。五项依赖本地 SFT processor 文件、一项依赖 PI0.5 基础模型配置；跳过项不计作通过。
- 合成数据覆盖 50D／100D 采集写入、具名 ROS 关节重排、无效传感器值拒绝、旧格式续采拒绝且不改动恢复文件，以及 50D→10D/22D/29D/31D、100D→20D 字段映射。
- 系统 Python 3.10 + Humble 的真实 `FrankaRobotState` 消息类型：合成消息得到正确 50D／100D 数值，未导入 Torch、未启动硬件节点。
- Chromium demo：单／双臂 × GELLO／gamepad 的格式标签、相对默认路径、手工路径保留、会话创建和刷新重连通过，无页面异常。
- wheel 构建后在 `/tmp` 以已安装包验证 50D／100D 字段、五份提取 YAML、网页资源和两个 CLI 帮助命令。专用 `vla` 环境只更新本项目包，依赖版本未变。

真实模型训练和实机 Franka／RealSense／GELLO 采集性能尚未验收。
