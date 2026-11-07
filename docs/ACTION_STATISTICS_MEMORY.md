# Action chunk 统计内存优化

## 保留的统计含义

同一目标出现在多个训练 chunk 中，仍按出现次数参与统计；`chunk_relative` 下还必须分别使用每个 chunk 的起始实测锚点。没有去重、减少训练样本、加入 padding，也没有采用近似分位数。

`count`、最小／最大值、均值、总体标准差（ddof=0）、q01／q99 均保留。分位数采用 NumPy 的精确 linear 插值。当前统计契约为 v2，版本与来源约束见 [执行闭环](EXECUTION_RTC_RLT.md#协议与时间轴)；逐列计算不会放宽契约校验。

## 实现和资源边界

1. 数据读取先分批索引 episode ID，再按 episode 读取 action、独立 anchor、有效性、segment 和排序字段；不加载实测状态、图像等无关列。一个回合跨多个 Parquet 分片时仍按原顺序合并，segment 和 start/frame stride 语义不变。
2. 每个 chunk 的转换最多处理 8192 行。转换后的 float64 样本按维度写入带缓冲的临时文件，不在 RAM 中保留所有展开 chunk。
3. 每次读取一列并计算统计量；q01／q99 一次调用、原位 partition。临时文件由上下文管理，在正常返回、无有效样本、非法数据、输入中断和磁盘错误时关闭并删除；错误不会返回部分统计结果。

设有效展开样本数为 M、动作维数为 D：统计核心从多份 `M×D` 数组降为一列 `M` 数组及计算标准差时的一列临时数组，加上小批次转换缓冲。**不是恒定内存算法**，超大 M 时单列仍会增长。数据读取另需当前 episode 及 Parquet 解码／row-group 缓冲；数据指纹仍以 8 MiB 块读取完整文件内容。

临时磁盘容量约为 **`8 × M × D` 字节**。例如 M=100 万、D=20 时约需 153 MiB。临时文件遵循 Python `tempfile` 的目录规则，可在启动进程前指定磁盘目录：

```bash
mkdir -p runtime/statistics-tmp
export TMPDIR="$PWD/runtime/statistics-tmp"
# 随后照常运行 vla-action-contract 或启动 Web 数据处理服务。
```

应选择有足够空间的磁盘目录；若使用 tmpfs，临时文件本身仍消耗内存。Python API 的 `compute_statistics` / `dataset_statistics` 也接受可选 `temporary_directory`。没有新增依赖或持久统计缓存。

浮点归约顺序改变可能使均值／标准差出现末位差异。因此恢复已有 checkpoint／replay 时继续使用其中保存的契约，不能用新生成 JSON 替换旧契约后假定二进制指纹相同；现有完整契约一致性检查仍然有效。本次没有重写已有数据或契约。

历史测量与验证范围见 [合成基准（2026-11-06）](archive/ACTION_STATISTICS_2026-11-06.md)。当前结果请通过[自动化验收](ACCEPTANCE.md)生成。
