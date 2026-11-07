# 锁文件诊断与恢复

`.lock` 有三种含义，不能统一删除：

| 文件 | 含义 | 处理方式 |
| --- | --- | --- |
| `requirements.lock`、`conda-linux-64.lock`、ROS 依赖锁 | 版本固定清单 | 保留并纳入 Git |
| `.git/index.lock`、引用／配置事务锁 | Git 操作正在进行或被异常中断 | 先检查进程；只清理确认废弃的事务锁 |
| `.*.rlt-writer.lock`、Web／硬件／验证域锁 | `flock` 使用的稳定文件节点 | 允许文件长期存在，不删除；实际占用由内核维护 |

查看当前仓库的 Git 锁、Web 锁、数据／输出目录中的 writer 锁，以及本机硬件与 mock 验证域锁：

```bash
python3 scripts/check_locks.py
# 也可检查仓库外的数据集锁；传锁文件路径，不是数据集目录
python3 scripts/check_locks.py --runtime-lock /path/to/.dataset.rlt-writer.lock
```

`held` 表示采样时仍被占用，`idle_persistent_file` 表示文件存在但可取得锁，`unknown` 表示无法判断。状态是瞬时采样，不是启动运动或删除文件的授权。脚本不会启动 ROS、连接设备或结束进程。运行时探测使用非阻塞锁，并立即释放。

Writer／硬件独占租约正常释放时，会在仍持锁期间清空 PID 等诊断信息，再关闭描述符，保留文件节点。进程异常退出时内核自动释放锁，旧 PID 可能留下；下一位持有者会覆盖它。不要凭 PID 是否存活判断锁状态，也不要在释放时 `unlink`：其他进程可能已经打开同一节点，删除会造成两个独立的锁。

Git 状态检查使用 `git --no-optional-locks status --short`，避免后台轮询反复创建短暂的 `index.lock`。对同一仓库的 `add`、`commit`、checkout 等写操作必须串行；不要中断它们后立即重试或并行清锁。

遇到明确的 Git index 锁报错，停止发起其他 Git 操作，在能查看完整主机 `/proc` 的终端运行：

```bash
python3 scripts/check_locks.py --clear-stale-index
```

只处理当前仓库／worktree 的 `index.lock`。文件至少 60 秒未改变、属于当前用户、无符号／硬链接，且未发现当前用户的 Git 进程、打开此文件的描述符、检查权限缺失或检查中变化时，才会删除。仅凭“时间久”不会删除。工具不能协调不遵守约定的外部 IDE／Git 客户端，恢复期间应暂停它们的 Git 操作；检查不完整时保留锁并报告原因。其他 Git 事务锁只报告，不自动清理。

不要运行递归删除 `*.lock`，不要删除运行时租约以绕过正在进行的采集／训练／硬件会话，也不要为解决锁报错终止不属于本次测试的进程。
