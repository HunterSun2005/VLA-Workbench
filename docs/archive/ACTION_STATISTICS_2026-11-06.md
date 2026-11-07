# Action 统计内存合成基准（2026-11-06）

> 历史快照：仅代表文中日期、版本和范围；旧命令及依赖状态不作为当前操作指引。当前行为见[文档导航](../README.md)，当前结果由[自动化验收](../ACCEPTANCE.md)生成。

两个独立 Python 3.12 进程，固定随机种子；双臂 joint 16D、`chunk_relative`、12,000 帧、chunk 长度 100、stride=1。有效展开样本 **1,195,050**。使用本机 ext4 `/tmp`，未读取真实数据或模型。

| 实现 | 峰值 RSS | 计算耗时 | 临时磁盘 |
| --- | ---: | ---: | ---: |
| 原算法：保留所有 chunk 后 concatenate | 488.42 MiB | 1.202 s | 0 |
| 新算法：临时列文件、逐维计算 | 58.89 MiB | 0.956 s | 145.88 MiB |

峰值 RSS 降低 **87.94%**。样本数、min/max、q01/q99 完全一致；mean 最大绝对差 `2.81e-13`，std 最大绝对差 `6.59e-14`。耗时是本机单次测量，不保证所有数据规模、临时盘或 Parquet 布局下都更快；该数字测量统计核心，不是完整训练内存。

复现（每个命令均启动全新进程，避免 peak RSS 被上次计算污染）：

```bash
export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=1
/data0/miniconda3/envs/vla/bin/python scripts/benchmark_action_statistics.py \
  --implementation expanded --output /tmp/vla-stats-expanded.json
/data0/miniconda3/envs/vla/bin/python scripts/benchmark_action_statistics.py \
  --implementation columns --output /tmp/vla-stats-columns.json
```

回归：**691 passed，7 skipped，37 warnings**，29.59 秒。新增 13 项覆盖单／双臂 × joint／TCP × absolute／chunk_relative、转换分批、掩码与步长、跨分片乱序回合／segment、列筛选和临时资源清理。7 个跳过项仍为独立 ROS 测试模块及缺少真实模型资源的检查，不计作通过。未连接硬件、修改环境或发布远程。
