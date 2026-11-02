# 数据集目录

沿用原仓库的阶段分层。Git 仅保存本文和各层 `.gitkeep`；实际数据、视频、缓存及 replay 数据库继续忽略。目录骨架不包含旧仓库的数据或软链接。

| 目录 | 用途 |
| --- | --- |
| `raw/` | 人工采集完整状态（单臂 50D／双臂 100D），动作 10D／20D |
| `sft/` | 清洗、特征转换后的 SFT 数据及 train/val 划分 |
| `transitions/` | RLToken / Actor-Critic 使用的 transition cache 和 shards |
| `rollouts/` | 部署阶段策略执行、人工接管的回合数据 |
| `online/` | 后续在线学习专用数据；当前 Web 不启动在线训练 |

每个实际数据集放在阶段目录的子目录中，不直接向阶段根目录写入。模型与训练产物放在 [outputs 分层目录](../outputs/README.md)，Web 会话配置／日志仍放在 `runtime/web/`。

## Web 默认命名

```text
datasets/raw/franka_single_left_gamepad_full
datasets/raw/franka_single_left_gello_full
datasets/raw/franka_dual_gamepad_full
datasets/raw/franka_dual_gello_full
datasets/rollouts/franka_single_left_sft
datasets/rollouts/franka_single_left_rlt
datasets/rollouts/franka_dual_sft
datasets/rollouts/franka_dual_rlt
```

默认名称用于区分采集配置。不同任务或独立批次请手动指定短任务标识和版本，例如 `datasets/raw/left_lid_v1/gamepad_full`、`datasets/rollouts/left_lid_v1/run-01`。完整任务描述保存在采集元数据中，不自动拼入文件路径。

前端预填相对路径；配置选择变化时，仅更新自动生成的路径。手动修改后保留自定义值，可点击“使用推荐路径”恢复自动跟随。部署目录按模型类型命名，运行中的 VLA/RLT/自动切换不会改变数据集目录。

相对路径基于 Web 的工作目录（`--root`，否则 `VLA_ROOT`，否则启动时当前目录）；页面显示这一基准，配置检查显示解析后的绝对路径。绝对路径和 `~` 路径也支持。runner 配置固定为解析后的绝对路径，避免子进程工作目录影响写入位置。

已有非空目录必须明确选择“续采”，并有 `meta/info.json`，且状态字段名称、顺序和维度与完整格式一致；新任务请另起目录。推荐路径不会自动覆盖、合并或清空旧数据。默认目录的最终数据集叶子由原有写入器创建，不预填 `.gitkeep` 或 README，以免被识别为非空数据集。
