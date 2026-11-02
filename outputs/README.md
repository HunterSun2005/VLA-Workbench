# 模型与实验产物目录

沿用原仓库的 SFT、RLToken、Actor-Critic、在线训练分层，增加当前 VLA 使用的 handover 分类头与独立评估目录。Git 仅跟踪本文和空目录标记；模型、日志、报告及其他运行产物均忽略。

| 目录 | 用途 |
| --- | --- |
| `sft/` | PI0.5 SFT 训练输出、checkpoint 和 processor 文件 |
| `rltoken/` | RLToken 训练输出与 checkpoint |
| `ac/` | 离线 Actor-Critic 训练输出与 checkpoint |
| `handover/` | VLA/RLT 自动切换分类头、配置与训练指标 |
| `online/` | 在线 RLT 训练 checkpoint、恢复状态及训练日志 |
| `eval/` | 跨模型评估、对比报告及汇总图表 |

## 命名约定

采用 `outputs/<阶段>/<task_id>/run-<编号>/`，例如：

```text
outputs/sft/left_lid_v1/run-01/
outputs/rltoken/left_lid_v1/run-01/
outputs/ac/left_lid_v1/run-01/
outputs/handover/left_lid_v1/run-01/
outputs/online/left_lid_v1/run-01/
outputs/eval/left_lid_v1/run-01/
```

`task_id` 与 [数据集目录](../datasets/README.md) 的任务标识一致，使用小写 snake_case 和版本号。不同超参数、随机种子、数据版本或独立训练使用新的 run 编号；详细配置留在 run 内，不拼成长目录名。编号由操作者指定，目前没有自动分配器。

只预建阶段目录。实际 run 由对应训练／评估工具创建，避免预填文件触发输出目录非空检查。各工具继续使用自身的 checkpoint 与恢复格式；例如 LeRobot 训练可在 run 下生成 `checkpoints/<step>/pretrained_model/`，handover 则使用自身的分类头配置和权重格式。不要统一改写这些内部结构。

可在每个 run 内保留 `logs/`、`diagnostics/` 和实验说明，记录输入数据路径、基模型／上游 checkpoint、代码提交、配置、随机种子及评估结果。已有工具生成的配置和 provenance 文件优先保留；额外实验说明由操作者记录，不会自动生成。

## 训练入口如何指定

以下是需要附加到现有完整训练命令中的输出参数；其余数据／模型参数按任务配置：

| 入口 | 推荐输出参数示例 |
| --- | --- |
| `vla-train-franka-pi05-sft` | `--output_dir=outputs/sft/left_lid_v1/run-01` |
| `vla-train-rl-token` | `--output_dir=outputs/rltoken/left_lid_v1/run-01` |
| `vla-train-actor-critic` | `--output_dir=outputs/ac/left_lid_v1/run-01` |
| `vla-train-handover` | `--output-dir outputs/handover/left_lid_v1/run-01` |

从 VLA 仓库根目录运行上述命令时，相对路径落在本仓库下；从其他目录运行请使用绝对路径或先切换目录。目录骨架只是组织约定，现有命令／脚本的默认输出参数不会自动重写。在线学习和评估涉及多个输出项，应将各自的 checkpoint、恢复状态、报告参数显式指向相应 run。

## 与数据和运行目录的边界

- 人工采集：`datasets/raw/<任务或配置>/`。
- SFT 输入与划分：`datasets/sft/<task_id>/`。
- Transition cache 和训练样本：`datasets/transitions/<task_id>/`，不混入模型输出目录。
- 部署回合：`datasets/rollouts/<任务或配置>/`；在线数据／回放：`datasets/online/<task_id>/`。
- Web 会话配置、socket、进程日志：`runtime/web/<session_id>/`。
- 预下载的基模型与 tokenizer：`models/`；任务训练结果放入本文的 `outputs/` 分层。

Web 的模型路径应指向实际可加载的 checkpoint 目录（例如某个 `pretrained_model/`），分类头路径应指向包含 `handover_config.json` 的目录。Web 仍只管理采集和部署，`outputs/online/` 是后续在线训练的存放约定。
