# LingBot V2 RTC / 串行部署

`lingbot-dev` 接入独立 LingBot 服务，CLI 与 Web 仍共用 ROS collector/coordinator。默认后端为 `pi05`，原 PI05 checkpoint 校验、RTC 和训练入口保持不变。LingBot 默认 RTC，保留显式串行模式。只支持单左臂 TCP 10D、base/left_wrist 两相机、30 Hz、stride=1、完整 50 步 chunk；不支持 Actor/RLT、在线训练或双臂。

## 进程与物理契约

LingBot 仓库的 `deploy/workbench_server.py` 在自己的 Python 环境运行，负责加载、归一化、图像处理、55D 槽位映射、反归一化。Workbench 不导入 LingBot、不管理外部模型进程。ROS 拥有唯一执行队列，执行投影、接管、记录与故障处理仍在 Workbench。

两端通过 TCP 上的 wire v2（`VLARLT02`、JSON header + NumPy payload）通信，不使用 pickle。默认地址 `tcp://127.0.0.1:8007`。推理与控制使用不同连接，服务模型 worker 同时只接受一次推理；健康查询和代次撤销不等待 GPU。此协议没有认证/加密，默认仅监听回环地址。

| 报文 | 内容 |
| --- | --- |
| `health` | ready、模型身份、external deployment v2、状态/动作字段、相机 HWC、50 步 horizon、30 Hz/stride=1、所选 rtc/serial 能力；Actor/online 均为 false |
| `plan` | session/episode/generation/request/observation/plan/model 身份、执行契约摘要、原始观测时间/期限、任务；`observation.state`、独立 `observation.anchor`、两幅 uint8 RGB HWC 图像、`committed_prefix` 为 absolute `[d,10]`，串行为 `[0,10]` |
| `plan_result` | 原样回显请求上下文与锚点/前缀；`actions` 和 `vla_reference` 为相同的 `[50,10]` absolute chunk，50 个 VLA 来源/phase=0；RTC 响应按原请求精确回填前缀，保留 float64 承诺目标 |
| `invalidate` | session/generation 撤销旧结果；不停止模型进程 |

远端不比较客户端单调时钟；安装期限由 Workbench 判断。请求保留观测快照的实际时间，不被网络入队或安装时间覆盖；提交前复制状态与图像，防止相机缓冲复用改变在途观测。跨连接延迟到达的旧控制消息被幂等确认，不降低服务端代次。重复、旧代次、过期、上下文/模型不匹配、形状或数值异常的结果不会安装。

外部契约必须显式声明字段顺序、单位、相机角色、采样率、基座 `left_fr3_link0` 和 TCP 变换。XYZ 单位米；rot6d 是旋转矩阵前两列；**观测状态的夹爪是米，锚点和动作的夹爪是 `[0,1]` 开度比例**。LingBot 把 TCP 放入 55D 的 `[14:23]`、夹爪放入索引 28，缺失的右腕相机 mask 为 false。

声明使用 `kind=external_deployment`、`model_backend=lingbot_v2`，允许 LingBot 原生统计，不要求冒充 Workbench SFT/RLT 的训练来源或分位数统计契约。服务计算 weights、training_config、processor、robot_mapping、norm_stats 的 SHA256，若声明已包含指纹则必须匹配；部署契约和推理设置参与模型身份。Workbench 保留物理动作校验与执行投影。此例外只适用于此次授权的外部 LingBot RTC/串行推理，不放宽 PI05 检查，也不把原生统计转换为 Workbench 训练统计。

## 启动 LingBot 服务

先将 LingBot 的 `configs/deployment/workbench_single_left.example.json` 复制到独立部署文件。填写真实的 `tool_definition.left`：`base`、`tip`、`position`（3 维米）、`rotation`（3×3 旋转矩阵）。它声明通信/控制端使用的 TCP，必须与 Workbench 的 URDF/工具配置一致。默认要求训练数据也使用相同定义；仅下述显式模型坐标适配允许姿态轴不同。模板故意为 null，未经确认不能直接运行；图像尺寸也须匹配部署 YAML。

### 当前旧硬件轴模型的显式适配

当前 `global_step_10000` RTC 模型的训练数据使用硬件 EE 轴：相对法兰 Z 偏移约 0.174 米，绕 Z 旋转 −45°。Workbench 使用 `flange_aligned_tcp_v1`：保留同一物理 TCP 位置，姿态轴与法兰平行。因此该模型的本机部署声明 `/data0/sht/VLA-Workbench/runtime/lingbot/deployment.json` 使用单位旋转的 `tool_definition`，另显式设置：

```json
{
  "model_frame_adapter": {
    "kind": "same_origin_tcp_rotation_v1",
    "model_tool_definition": {"left": {"base": "left_fr3_link0", "tip": "left_fr3_link8", "position": [0, 0, 0.17399999499320984], "rotation": [[0.7071067811865475, 0.7071067811865476, 0], [-0.7071067811865476, 0.7071067811865475, 0], [0, 0, 1]]}},
    "weights_sha256": "填入该 checkpoint 的 artifacts.weights SHA256"
  }
}
```

此字段要求完整、正确的工具定义、相同的 TCP 原点及明确绑定的权重 SHA256，不支持位置变换、隐式猜测或自动套用。服务加载时仍重算所有资产指纹；适配定义参与模型身份。服务在原生归一化前转换观测、锚点和 RTC 承诺前缀的姿态，在解码后把动作后缀转换回控制坐标，最后精确回填原通信前缀。旋转右乘 `R_wire_tool.T @ R_model_tool`，不是绕基座左乘；XYZ 与夹爪单位/数值保持不变。控制端、执行投影与记录始终使用逻辑 TCP。

后续使用 Workbench 逻辑 TCP 数据训练的模型**省略 `model_frame_adapter`**，走原始无转换路径；不要复制当前模型的适配声明到新 checkpoint。使用旧硬件轴数据重新训练的模型仍需独立核实坐标定义并显式绑定对应权重。

在 LingBot 环境中启动，例如：

```bash
cd /data0/sht/lingbot-vla-v2
/data0/miniconda3/envs/lingbotvla/bin/python -m deploy.workbench_server \
  --execution-mode rtc \
  --checkpoint /absolute/rtc_run/checkpoints/global_step_2000 \
  --checkpoint-format qlora-dcp \
  --training-config /absolute/rtc_run/lingbotvla_cli.yaml \
  --processor-path /data0/sht/lingbot-vla-v2/weights/Qwen3-VL-4B-Instruct \
  --robot-mapping /data0/sht/lingbot-vla-v2/configs/robot_configs/franka_single_left_10d.yaml \
  --norm-stats /data0/sht/lingbot-vla-v2/assets/norm_stats/franka_single_left_10d.json \
  --declaration /absolute/deployment.json \
  --tasks "Pick and place."
```

默认 RTC 声明包含 `execution_mode=rtc` 和：

```json
{"rtc": {"prefix_conditioning": "clean_tokens_time_zero_training_mask", "max_prefix_steps": 15}}
```

`max_prefix_steps` 必须等于训练保存的 `rtc_max_delay`。RTC 启动要求 `use_rtc=true`、`0 < rtc_max_delay < 50`、`0 < rtc_prefix_probability <= 1`，不符合即失败。模式、前缀语义和训练上限参与模型身份。服务使用训练期 attention mask：承诺 token 时间为 0，不能关注未来噪声；每个去噪步骤仅更新未承诺后缀。前缀先按完整 50 步原生统计归一化再裁取，支持逐时间步统计；缺失模型槽位继续按训练规则补零。

**旧串行部署迁移：** 服务显式加 `--execution-mode serial`，使用 `workbench_single_left.serial.example.json` 对应的声明（`execution_mode=serial`，无 `rtc` 字段）；客户端也须显式选择 `serial`。不自动降级，非 RTC 权重不能以默认 RTC 启动。

使用训练保存的完整配置，显式固定 checkpoint，不跟踪正在训练目录的 latest。DCP 路径下须有 `model/.metadata` 和模型 distcp shards；不加载 optimizer。普通完整权重目录使用 `--checkpoint-format safetensors`，其训练配置须关闭 QLoRA。两种格式都严格恢复模型键/形状；DCP 还检查 dtype，任何缺失都报错，不回退到基础权重或部分加载。

QLoRA 重建原 MoE/NF4/LoRA rank/alpha 布局，保留 FP32 LoRA 与量化缓冲，使用 BF16 autocast；不对量化模型整体做 BF16 转换。默认 eval、无梯度、关闭 compile。所有路径显式传入，相对路径以 LingBot 仓库根目录解析。模型加载及完整 chunk 预热成功后才监听并报告 ready。现有环境缺少依赖时明确失败，不自动安装。

## Workbench CLI 与 Web

CLI 无硬件预检：

```bash
cd /data0/sht/VLA-Workbench
source scripts/model_env.sh
python -m vla.cli.ros_session --workflow deploy --arms single_left \
  --model-backend lingbot_v2 --policy-kind sft --mode vla \
  --policy-endpoint tcp://127.0.0.1:8007 --policy-timeout-s 5 \
  --lingbot-execution-mode rtc --lingbot-inference-slots 14 --lingbot-guard-slots 1 \
  --task "Pick and place." --tasks "Pick and place." --check-only
```

任务列表及顺序须与服务声明一致。无需 PI05 checkpoint/tokenizer。`--check-only` 查询模型协议、任务和图像尺寸，并离线检查现有 worker binding，不启动机器人、夹爪或相机。完整 TCP 变换匹配还在 collector 初始化时校验。现场部署使用同一入口，配置独立 dataset/hardware 路径并显式传入 `--execute`；本次开发不执行该步骤。

Web 部署页选择 **LingBot V2**，填写服务地址、模式（默认 RTC）、推理时隙（14）、保护时隙（1）和网络超时（默认 5 秒），选择单左臂并配置相同任务。预检连接外部服务；开始会话不启动模型进程，关闭会话不关闭 LingBot。PI05 的模型路径草稿在切换后端时保留。后端参数由 CLI/Web SessionConfig 提供，控制/相机参数仍由原部署 YAML 提供。

## RTC 时序与记录

`lingbot_execution_mode=rtc`、`lingbot_inference_slots=14`、`lingbot_guard_slots=1` 只用于 LingBot。PI05 保持原预算和执行路径。30 Hz / stride=1 下，冻结前缀 `d=14+1=15`；预检要求 `d <= rtc_max_delay` 且 `50 >= 2d`。网络超时不延长 RTC 的 14 槽推理期限。

首块准备时连续提交启用目标，来源为 `system_hold`。剩余队列不超过 `2d` 时，以真实观测时间和已投影队列前缀异步请求下一块，最多一个在途请求。旧队列在推理期间继续执行。提前返回的候选也只安装到固定 `replace_slot` 之后，保护前缀的目标和来源。全块动作、VLA 来源、phase、精确前缀及所有后缀投影通过后整体安装。超时、迟到、缺槽、断连、接管、存储阻塞或投影失败撤销代次，要求手动恢复；旧结果及重连都不能启用运动。

沿用 recording schema 5，保存请求的实际观测时间、承诺前缀、期限、固定边界、预算、external deployment 和模型身份；`install` 事件另存实际安装时间和边界。初始 hold 不作为监督动作。外部原生统计不声明 Workbench SFT/RLT 训练合规性。

## 显式串行模式

正常循环为：新观测 → `waiting_inference`（保持目标）→ 完整 chunk 从索引 0 执行 → 最后目标提交 ACK → 新观测。推理等待与最终 ACK 等待均持续提交最后接受目标，首次使用启用目标；没有延迟裁头、提前预取或过期动作补发。网络工作线程不阻塞 30 Hz 回调。

只有正常 `waiting_inference` 可自动安装当前有效结果。候选动作及全部执行投影先校验再整体安装；任一校验或投影失败都撤销请求，禁止重试同一结果自动恢复。意外缺槽、推理超时、断连、存储阻塞、接管和故障撤销在途请求，要求人工处理/恢复。服务在 chunk 执行期间的退出在下一次网络交互被发现。手动故障恢复可重新建立客户端，但模型/部署契约变化要求新建会话；重连不启用运动。

沿用 recording schema 5。等待动作来源为 `system_hold`，source 编码 -1、supervision_valid=0；保存实际时间、原请求观测身份、独立锚点和 `install_serial` 的实际安装 slot/time。元数据保存 external deployment、模型身份和 serial 执行契约，不伪造训练来源。正常块间等待不改变控制 segment，故障/接管仍切段。

## 离线验证

```bash
VLA_PYTHON=/data0/miniconda3/envs/vla/bin/python bash scripts/test.sh \
  tests/control tests/actions tests/web \
  tests/rlt/test_franka_sft_server.py tests/rlt/test_franka_pi05_rtc_deployment.py -q
bash scripts/test_ros.sh tests/ros -q
# 固定状态 mock、独立 DDS domain；不得替换为硬件 plugin
bash scripts/validate_ros_control.sh --duration 600 --lingbot-rtc
# 串行最终 ACK 门控另由单元测试回归
VLA_PYTHON=/data0/miniconda3/envs/vla/bin/python python3 scripts/validate.py --checks model ros wheel
```

跨仓库测试默认读取同级 `lingbot-vla-v2`，也可设置 `VLA_LINGBOT_ROOT`；没有该仓库会明确 skip。双方独立 codec 使用同一 JSON 报文样例，无生产运行时跨仓库导入。LingBot 的 CPU/DCP 和可选 GPU 测试见其 `deploy/WORKBENCH.md`。CPU/mock 验证不能证明真实 QLoRA 延迟、物理连续性、相机吞吐或任务成功。

真实 RTC checkpoint 的完整服务/GPU 冒烟需显式指定 RTC 训练产物；无可用产物时列为未验证，不能用非 RTC 权重推理替代。
