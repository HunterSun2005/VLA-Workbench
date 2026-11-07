# 迁移来源与版本资料

本目录保留独立仓库建立时的来源、文件哈希、排除项和补丁，不是当前操作手册。新开发从[文档导航](../README.md)开始，当前验收使用[自动化入口](../ACCEPTANCE.md)。

| 资料 | 用途 |
| --- | --- |
| [SOURCE.md](SOURCE.md) | Evo-RLT 来源提交及许可说明 |
| [ROS.md](ROS.md) | ROS 自有包、上游修订、补丁取舍与 Python 隔离背景 |
| [LEROBOT061.md](LEROBOT061.md) | 官方 LeRobot 0.6.1 适配与依赖组合 |
| [FEATURES.md](FEATURES.md) | 迁入功能在当前仓库的模块位置 |
| [CLEANUP.md](CLEANUP.md) | 迁移时排除组件的历史记录 |
| [VALIDATION.md](VALIDATION.md) | 2026-10-30 的迁移验收快照 |
| [files.json](files.json)、[excluded.json](excluded.json) | Python 来源文件及排除清单 |
| [ros-files.json](ros-files.json)、[ros-source-status.txt](ros-source-status.txt) | ROS 来源文件和当时工作树状态 |
| [dynamixel-source.json](dynamixel-source.json) | Dynamixel SDK 来源比较 |
| [franka_ros2-local.patch](franka_ros2-local.patch)、[libfranka-local.patch](libfranka-local.patch) | 保存的 Franka 上游本地改动 |
| [libfranka-common-local.patch](libfranka-common-local.patch)、[ros-working-tree.patch](ros-working-tree.patch) | 保存的其他来源差异；保存不等于全部应用 |

当前依赖修订以 [dependencies.repos](../../ros_ws/dependencies.repos)、[模型依赖锁](../../requirements.lock) 和 [ROS Python 依赖锁](../../ros_ws/requirements-python.lock) 为准。不要因整理文档而重写来源哈希、重新应用补丁或修改第三方检出。历史报告提到但本目录没有收录的机器检查产物，不视为当前可用的附件。
