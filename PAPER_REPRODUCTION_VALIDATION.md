# 论文 Can 实验训练协议验证（2026-10-31 UTC）

本文保留分阶段验证证据；当前训练和数据进度以 [整体实验安排与当前进度](EXPERIMENT_PLAN_AND_PROGRESS.md) 及后台日志为准。Paired seed 0 正式训练已于 23:00 UTC 启动，启动前最新回归检查为 22 项通过，日志见 [paired_first_launch_tests_20261031.log](logs/paired_first_launch_tests_20261031.log)。

本轮将此前 PH→MH 的顺序微调改为 REACH 表 2 的独立实验组织，保留 Pi0.5 策略骨干。规范文件为 [configs/paper_can.json](configs/paper_can.json)，操作说明为 [README](README.md)。

## 修改内容

- 策略任务：Can(Paired)/Can(MH) × baseline/conditional × 3 seeds，共 12 个；统一从同一个 Pi0.5 初始权重开始。
- 检测器任务：PH/MH/Paired × 3 seeds，共 9 个；独立训练 feedforward 图像 AE，优化像素 MSE并保存阈值标定结果。
- 评估清单：Paired 策略搭配 PH/MH/Paired 检测器、MH 策略搭配 MH/PH 检测器，扫描 S=0/0.2/2.5/4.5，并包含无检测器对照，共 72 组。带检测器的 S=0 仍标记启用 recovery，不与无检测器 baseline 混淆。
- 数据类型与策略模式解耦；不再默认“MH 就启用 AC”。跨样本 GT shuffle 最初为 0，现按用户要求改为条件策略默认 0.3，见文末训练变体说明。
- 策略训练支持 50 epochs，由各数据集 loader 长度计算 steps；默认 global batch=128、lr=1e-4、cosine。
- 官方 Paired low-dim v1.4.1 已下载。保留全部成功/失败轨迹，官方 mask 优先，初始场景分组校验防止配对泄漏。
- 新渲染器恢复原始 XML 和逐帧 simulator states，保留元数据和 mask，使用 Robomimic 图像方向约定，最终文件原子发布。
- AE 支持高斯 κ-sigma 阈值和有限样本 conformal 次序统计量标定，保存独立可加载的 detector artifact。
- 任务命令、参数和训练实现摘要参与 provenance；旧串行微调结果不会被新矩阵自动复用。

## 验证结果

| 检查 | 结果 |
| --- | --- |
| Python / Shell 语法 | 通过 |
| 回归测试 | 15 项通过；实验调度新增字段后另行重跑 3 项 pipeline 检查通过 |
| 官方 Paired 元数据 | 200 条轨迹，100 组相同初始场景，100 成功 / 100 失败 |
| 官方 Paired 切分 | train 180 / valid 20，配对场景不交叉 |
| 独立任务清单 | 12 policy + 9 AE，72 evaluation combinations；所有 policy 初始 checkpoint 相同 |
| CPU 双进程 DDP | 采样不交叉、参数同步、验证汇总通过 |
| 真实 Paired 图像渲染 | 原始 demo_0..3 的前两帧，共 8 帧；原 XML、双相机、states 和 pair_id 保留 |
| 真实 Pi0.5 Paired baseline | GPU 上 1 epoch / 2 steps，训练、验证和保存通过 |
| 真实 Pi0.5 条件策略 | 两张 GPU，每卡 batch=1，1 epoch / 2 steps，训练、验证和保存通过 |
| 实验公平性检查 | 两个真实 policy run 的初始 checkpoint、切分及归一化完全相同 |
| Epoch 模式恢复 | 从 baseline step 1 恢复后，813 个模型张量与不中断训练逐个完全一致 |
| 默认结构 AE | 双视图 84×84、hidden=512、latent=128、SGD，GPU 训练 / 标定 / 保存通过 |
| AE 重新加载 | 从 detector.pt 加载并计算真实图像的 score/is_ood 通过 |

上述 GPU policy 检查仅使用 8 帧 smoke 数据。baseline 的两步 loss 为 2.072695 / 0.410932；双卡条件策略为 1.248256 / 0.249034。AE 一轮 train MSE 为 0.084558、val MSE 为 0.084197，高斯标定 threshold 为 0.084492。所有数值仅用于证明执行路径和保存恢复正常，不代表任务成功率、收敛表现或检测质量。

日志保存在 [logs/paper_validation_20261031](logs/paper_validation_20261031/)。大体积临时 smoke policy/AE 权重在验证后清理；正式基础权重保留在 `checkpoints/pi05_base_pytorch/`。

## 数据及尚未执行的部分

PH 已转换为可用文件；Paired 新版图像已于 2026-10-31 22:40 UTC 完成，200 条轨迹 / 19,795 帧，官方切分和配对隔离检查通过；MH 新版图像已于 2026-11-01 01:34 UTC 完成，300 条轨迹 / 62,756 帧。三个文件均通过完整性检查。进度见 `logs/prepare_mh_paper.log`、[logs/prepare_paired_paper.log](logs/prepare_paired_paper.log)。旧版数据提取进程已按要求停止。新的 PH/MH/Paired 论文图像统一位于 `data/paper/`；已有完整文件会复用，活跃生成任务受文件锁保护。PH 复用了旧图像并转换方向和元数据，未重新逐场景渲染，详见 `data/paper/can_ph_image.conversion.json`。

最初协议验证阶段未启动完整的 21 个正式训练任务，evaluation 条目当时标为 `requires_rollout_runner`。后续已实现并验证真实 rollout、物理回退和基于失败历史的在线引导，见文末评测验证章节；当前清单标为 `pending_training_or_evaluation`。独立诊断或待执行的评测清单均不能当作已完成的论文成功率统计。

Pi0.5 的单帧输入、50 步 horizon、32 维 padding、flow matching、冻结 VLM以及线性 token 条件适配被保留。base 概率/值、AE 层宽、多相机融合、κ=3、held-out 标定、具体种子和评估回合数属于显式复现选择，详见规范中的 `adaptations_and_unspecified_details`；不声称这些值由作者公开。

## 固定输入监控与引导诊断验证

新增 `scripts/policy_diagnostics.py`、`scripts/diagnose_steering.py`。训练默认在固定留出面板记录 GT/base/foreign 分支；独立脚本使用同一观测和噪声检查采样一致性、提示跟随与正负向引导。所有指标只测 7 维物理动作，self 提示的其余通道补零；监控保留训练 RNG，不增加训练监督。实验任务实现摘要包含这两个新文件。

| 检查 | 结果 |
| --- | --- |
| 回归测试 | 19 项通过，含固定噪声/时间、RNG 恢复、padding 排除、引导符号和一致性失败退出 |
| CPU 双进程监控 | 留出样本仅 1 个、rank 1 无 validation batch；全局固定面板正常，开关监控时训练 loss、参数更新与 RNG 完全一致 |
| 真实 Pi0.5 条件训练 | Paired 8 帧 smoke 数据，batch=2，1 step，初始化与更新后 GT/base/foreign 监控、完整验证及保存通过 |
| 初始化零投影 | GT/base/foreign 的 flow MSE 均为 10.561532；GT/base 速度差为 0 |
| 一步更新后 | GT/base 速度 MSE 差异为 8.18185e-5，表明条件投影进入了训练路径 |
| 真实 Pi0.5 采样 | 2 个留出样本，固定 seed=2026，10 Euler steps；自定义与上游采样最大绝对误差为 0 |
| 零引导检查 | negative-self / negative-GT / positive-GT 的 S=0 均与 base 完全一致 |
| 引导扫描 | 上述 3 组各测 S=0/0.2/2.5/4.5，共 12 组，另测 GT/foreign/self 直接条件采样；全部有限值 |

日志与原始指标：

- [单元测试](logs/diagnostics_unit_tests_20261031.log)
- [双进程监控检查](logs/diagnostics_ddp_smoke_20261031.log)
- [真实模型训练日志](logs/diagnostics_train_smoke_20261031.log)
- [分支监控原始记录](logs/diagnostics_branch_monitor_smoke_20261031.jsonl)
- [采样诊断原始报告](logs/diagnostics_steering_smoke_20261031.json)

一步 smoke 训练仅用于验证接口和计算路径，尚未学出可靠的提示跟随能力；不据此要求距离随 S 单调，亦不报告任务成功率。临时 smoke 权重在验证后清理，基础权重、真实数据生成任务及上述日志保留。

## 用户要求的 30% GT shuffle 训练变体

条件策略现采用 30% base、30% 同一 GPU 本地 batch 中其他样本的 GT 提示、40% 自身 GT 提示。三类按样本互斥随机分配，概率相对于全部样本；shuffle 只改变动作条件，观测和监督目标均保持当前样本自己的数据，不改变动作序列内部的时间顺序。无自配置换保证提示的来源索引不同，每卡 batch 至少为 2。

直接 CLI、shell 入口和矩阵配置已统一；矩阵不再硬编码 shuffle=0，而读取 `configs/paper_can.json` 的 `gt_shuffle_probability=0.3`。协议更新为 `reach_cvpr2026_can_pi05_gtshuffle30_v2`，生成的任务清单与指纹记录该配比。baseline 与图像 AE 不启用这个条件分支。

此变体由用户明确要求，目标是引入错配动作条件训练；效果有待正式训练与 rollout 验证，不声称其概率来自论文。上文既有真实模型 smoke 结果使用 shuffle=0，不能当作本次新配比的效果结果。恢复旧任务时须显式保留原 `--gt-shuffle-prob`，严格恢复检查不允许静默更换概率。

该阶段验证：20 项回归测试通过，新增检查覆盖三类互斥提示选择、错配来源索引与训练目标不变；直接 CLI 和兼容 AC 开关的默认 shuffle 均为 0.3，显式设 0 仍可用。shell 入口参数及重生成的 21 任务 / 72 评估清单已核对，6 个条件策略任务均为 base=0.3 / shuffle=0.3。双进程监控检查再次通过；其中每卡 batch=1 的特殊用例显式关闭 shuffle。日志见 [gt_shuffle30_tests_20261031.log](logs/gt_shuffle30_tests_20261031.log) 与 [gt_shuffle30_ddp_check_20261031.log](logs/gt_shuffle30_ddp_check_20261031.log)。该阶段仅做验证；后续正式任务状态见本文开头的总览链接。

## JAX→PyTorch 权重核验与 AE 优化修正（2026-11-01）

原始 JAX 权重经映射、保存 dtype 转换后，与当前 safetensors 的 **811 个实际使用张量、3,353,433,872 个元素逐一完全一致**；无不匹配。共享 embedding 的别名和不参与动作预测的 expert LM head 已单独核验。报告见 [pi05_conversion_weight_audit_20261031.json](logs/pi05_conversion_weight_audit_20261031.json)。此项验证权重映射和保存内容，不是 JAX / PyTorch 相同输入的端到端前向对齐，也不能将既有 PyTorch 自定义 / 上游采样器一致性检查称为跨框架对齐。

AE_PH 原 SGD、lr=1e-4 训练 48 轮，train MSE 仅从 0.086348 降到 0.086341。固定训练面板 256 帧、独立官方 validation 面板 128 帧，所有候选使用相同初始权重、数据顺序和 120 次更新：

| 优化器 | 学习率 | momentum | 初始 validation MSE | 120 步后 validation MSE |
| --- | --- | --- | --- | --- |
| SGD | 1e-4 | 0 | 0.086475 | 0.086475 |
| SGD | 0.1 | 0 | 0.086475 | 0.086399 |
| SGD | 1 | 0 | 0.086475 | 0.086237 |
| SGD | 10 | 0 | 0.086475 | 0.045047 |
| SGD | 1 | 0.9 | 0.086475 | 0.057469 |
| Adam | 1e-4 | — | 0.086475 | **0.021313** |

SGD 原配置首步梯度 L2=0.004799，但相对参数更新仅为 4.09e-9，说明存在梯度，优化步幅却过小。训练集平均图像在 validation 面板上的 MSE 为 0.032471，Adam 已低于该基线。扩展 SGD lr=30/100/300 的 120 步结果分别为 0.033983 / 0.032526 / 0.244274；lr=300 明显退化。SGD lr=10 在完整 PH 数据上训练 3 轮的 validation MSE 为 0.036496 → 0.032540 → 0.032427，仍接近平均图像基线。

据此将检测器的默认优化器改为 **Adam、lr=1e-4**，其余网络、数据切分、50 epochs、batch=128、cosine 和标定方式保持不变。CLI 和矩阵配置同步修改；旧的 PH 结果归档到 `outputs/paper_can/detectors/can_ph/seed_0_legacy_sgd_1e-4_20261031/`，旧队列记录归档到 `outputs/paired_first/seed_0/detectors_legacy_sgd_1e-4_20261031/`。新的 seed 0 检测器队列于 2026-11-01 08:16:56 UTC 从初始权重重新启动。此优化器变更是显式的额外适配，不作为论文原始 SGD 配置报告。

每轮新增记录初始 / 当前 validation MSE 与相对下降率，并随 checkpoint 保存到 `optimization_diagnostics`，独立 JSON 为 `optimization_diagnostics.json`；恢复时沿用原基线。22 项回归测试通过，含诊断记录和恢复一致性检查。两组已运行的策略未重启、未修改配置，其启动 manifest 保留原快照。

原始证据：[优化器对照](logs/ae_optimizer_probe_20261031.json)、[SGD 扩展对照](logs/ae_sgd_lr_extension_20261031.json)、[完整 PH 的 SGD 对照](logs/ae_sgd10_full_data_probe_20261031.log)、[回归测试](logs/ae_optimizer_fix_tests_20261101.log)。重建误差下降仅说明优化开始生效，在线错误检测效果仍需后续验证。

Adam 修正后，完整 PH 正式重训前 5 轮的 validation MSE 为 0.018525 → 0.014899 → 0.013227 → 0.012000 → 0.011018；相对初始完整 validation 的 0.086574，下降约 87.27%。新队列已验证实际更新生效，仍会按计划完成 50 轮后训练 AE_Paired / AE_MH。实际记录见 [AE_PH 日志](outputs/paper_can/detectors/can_ph/seed_0/train.log) 和 [优化诊断](outputs/paper_can/detectors/can_ph/seed_0/optimization_diagnostics.json)。

## 真实仿真评测环境和流程验证（2026-11-01）

两组 Paired seed 0 策略均完成 6,900 步，baseline / conditional 的最终 checkpoint 分别于 09:42:47 / 09:41:45 UTC 保存。conditional 完整模型的离线 steering 诊断已自动完成。

新增 `eval_sim_worker.py`、`eval_sim_bridge.py`、`eval_policy.py`、`eval_recovery.py`、`eval_catalog.py`、`evaluate_can.py`、`evaluation_matrix.py`、`check_evaluation.py` 及 shell 入口 `evaluate_can.sh`。模型和仿真沿用各自现有环境；未修改共享环境安装。当前配置为 PickPlaceCan / Robosuite 1.4.1 / MuJoCo 3.1.0 / 20Hz；图形渲染固定 EGL 0，与模型 CUDA 卡选择隔离。

对齐表 2 的三种子、检测器组合、S 扫描，以及补充表 5 的 8 步执行。每 checkpoint 50 个测试初始场景、种子 100000–100049、Paired / MH 的 400 / 500 步预算来自论文引用的 [Diffusion Policy 官方配置](https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/diffusion_policy/config/task/can_image.yaml) 和 [Robomimic 原论文配置](https://raw.githubusercontent.com/ARISE-Initiative/robomimic/v0.3.0/robomimic/scripts/generate_paper_configs.py)，取代此前暂定的 100 回合。论文没有重新列出全部评测参数；未公开项、回退伺服选择、已有 Pi0.5 / shuffle / Adam 适配见 `configs/evaluation_can.json`，输出保留 `exact_paper_reproduction=false`。

| 检查 | 结果 |
| --- | --- |
| 回归检查 | 30 项通过；新增归一化、SE(3)、真实失败动作来源、S=0、物理预算、聚合、GPU 并行和恢复指纹检查 |
| 同一数据集状态重放 | EEF 位置误差 6.55e-16；agentview / wrist 图像 MAE 均为 0，逐像素一致 |
| 相同目录初始场景重置 | 两次状态重置得到相同双相机图像；50 个公共场景已持久化 |
| 物理回退 | OSC 真正执行回退，测试最终位置误差 4.24mm、旋转误差 0.0133rad；恢复过程未调用 reset 或 set_state |
| 最终 baseline，400 步预算单回合 | 第 80 步成功，正常动作适配、反归一化、成功判定和视频写出通过 |
| 最终 conditional + AE_Paired，400 步预算单回合 | 第 87 步成功，该回合无错误触发；不能据此证明恢复提升 |
| 强制错误，S=2.5，24 步诊断 | 5 次恢复、19 个真实回退步、4 个负向引导前向步 |
| 强制错误，S=0，24 步诊断 | 同样 5 次恢复、19 个真实回退步，零个引导前向步；证明 S=0 不是无检测器对照 |
| 同噪声引导对比 | 两个强制诊断的首动作完全相同；首次恢复后动作 L2 差异 0.176993 |
| 评测清单 | seed 0 每策略 13 个中间 checkpoint + final；共 196 个 checkpoint/configuration 单元 |

所有单回合、阈值覆盖和缩短预算结果均标为 smoke，汇总器排除这些结果。未启动正式 50 回合/单元的大规模评测；当前只有 seed 0，不能生成论文所需完整三种子聚合。汇总保留每 checkpoint 分数、final、各种子最大值、两种标准差和表注 / 正文两种聚合；checkpoint 扫描未完成时也不输出伪完整的最高成功率。

原始证据：

- [回归日志](logs/evaluation_tests_20261101.log)、[环境检查](outputs/evaluation/environment_report.json)
- [图像和位姿一致性](outputs/evaluation/environment_validation/report.json)
- [完整预算 baseline 单回合](outputs/evaluation/smoke_baseline_400/summary.json)、[视频](outputs/evaluation/smoke_baseline_400/episode_000.mp4)
- [完整预算 conditional 单回合](outputs/evaluation/smoke_recovery_400/summary.json)、[视频](outputs/evaluation/smoke_recovery_400/episode_000.mp4)
- [强制恢复和零引导对照](outputs/evaluation/recovery_path_validation.json)
- [公共初始场景](outputs/evaluation/can_initial_states_50.json)、[seed 0 全 checkpoint 清单](outputs/evaluation/paired_seed0_all_checkpoints/matrix.json)
- [本地缓存论文来源及 SHA256](references/source_manifest.json)

## 首轮正式评测启动（2026-11-01 11:13 UTC）

seed 0 的两份最终权重已开启 final-only 正式统计：baseline 单独 1 个设置 / 50 回合；conditional 单独 13 个设置 / 650 回合，共 14 设置 / 700 回合。共用同一份 50 个初始状态目录，每个设置在独立 GPU 任务中运行。AE_MH 尚未完成时，调度器先运行可用设置，之后自动加入 MH 检测器的 4 个 S。没有混入阈值覆盖或缩短预算的 smoke。当前尚无完整结果，不将 final-only 或单种子结果报告为三种子全 checkpoint 最高成功率。

[首轮启动记录](outputs/evaluation/paired_seed0_final/launch.json)、[队列状态](outputs/evaluation/paired_seed0_final/status.json)、[正式清单](outputs/evaluation/paired_seed0_final/matrix.json)。

## 作者公开源码对照（2026-11-01，372552d）

已将 [作者仓库](https://github.com/zundongke/Explicit-recovery-behavior-diffusion-policies) 克隆到 `/2025233210/lmx/Explicit-recovery-behavior-diffusion-policies`，提交固定为 `372552d6db59353cb3e1ee47126cbcbb12f34229`，上游工作区未修改。本节基于静态调用路径检查及提取原函数的 CPU 数值验证；没有安装上游依赖、执行其训练或评测，也没有修改正在运行的本地策略、检测器或评测代码。

原始审计：[audit.json](outputs/source_comparison/reach_372552d/audit.json)。下述“原仓库”均指该提交，不等同于已确认的论文最终运行配置。论文 PDF、上游当前默认配置、本地实际运行参数是三种不同证据，不能互相替代。

### 调用路径与发布完整性

- 训练入口 `train.py` 通过 Hydra 的 `cfg._target_` 创建 workspace；模型由 `cfg.policy._target_` 决定，workspace 的 Python 类型注解不决定实际类。
- `diffusion_policy/config/train_diffusion_unet_image_workspace.yaml` 默认 `task: real_task`，模型指向 `diffusion_unet_image_policy_yuan.py`，该文件是无动作提示的普通 DP。不能直接运行默认配置便称其为 Can Conditional DP。
- 动作提示实现位于 `diffusion_policy/policy/diffusion_unet_image_policy.py`，与 `_yuan.py` 为不同类实现。Can(Paired) 任务配置在 `diffusion_policy/config/task/can.yaml`；名为 `can_image.yaml` 的文件实际混有 PushT runner 和其他数据读取器。
- Can runner 的活动调用为 `RobomimicImageRunner.run -> error_aware_policy.predict_action -> detect_error/draw_back`。比较恢复时查看了这个活动路径，而不是文件中未调用的 `test_reverse` 辅助函数。
- 仓库保留 Git gitlink `diffusion_policy/ae` 和 `diffusion_policy/classifier-dp`，但缺少 `.gitmodules`，没有可用于标准初始化的子模块 URL。源码还引用缺失的 `diffusion_policy.env`、`diffusion_policy.robomimic`。
- AE YAML 中 `TrainVAEWorkspace` 与对应文件实际类名不一致，VAE 模型配置、policy 调用接口和 workspace 解包也有不一致。未找到能唯一确认论文 Can 检测器最终结构和训练参数的自洽配置；不能把某个备用 CNN/VAE 当成最终模型。
- README 仍主要为上游 Diffusion Policy 文档，未提供完整、明确的 REACH Can(Paired)/Can(MH) 复现命令。当前提交不能直接作为已验证可重现论文全部分数的发行版。

### 训练和数据：除模型家族外的差异

| 项目 | 作者公开源码 / 配置 | 本地实际运行 | 解释 |
| --- | --- | --- | --- |
| 条件提示配比 | Conditional `compute_loss` 写死 `base_prob=0.45`，45% 零提示、55% 自身 GT，没有 shuffle | 30% 零提示、30% 其他样本 GT、40% 自身 GT | 明确的方法变体，不是 Pi0.5 必然要求；本地 shuffle 来自用户指定的借鉴方案 |
| 提示编码 | `PromptProjector`：逐动作 MLP `action_dim→64→64→obs_feature_dim`，时间维 reshape 后再 Linear，与观测特征拼接，使全局条件维度翻倍 | 零初始化逐动作 Linear，直接加到 action suffix token | 模型接口需适配，但当前线性加法不是原始提示结构的等价实现，也不是 Pi0.5 的唯一选项 |
| 数据切分 | `can.yaml` 为 `val_ratio=0.02, seed=42`；ReplayImageDataset 按 episode 随机划分，不读官方 train/valid mask，不按 Paired 场景分组 | 官方 180/20 mask，验证成功/失败配对场景隔离 | 同为 200 条数据时，上游配置推得 196/4，本地 180/20；训练样本数不同 |
| 归一化 | `abs_action=False` 下，相对动作和 EEF quaternion 均为 identity；pos/qpos 用 range；统计读取整个 replay buffer | 七维动作和九维状态全部按训练集逐维 min/max 归一化；推理使用相同保存统计 | 训练推理自洽，但表示和噪声尺度不同；并非简单文件格式差异 |
| 采样窗口 | horizon16，pad_before1、pad_after7，当前动作从索引1开始取8步；不把每个末尾 timestep 都作为当前预测起点 | horizon50，以当前 timestep 为起点；不足50时重复最后动作 | 除窗口长度以外，末尾样本分布也不同 |
| 训练主干 | `freeze_encoder=False`，模型参数交给 optimizer | 冻结整个 PaliGemma，训练 action expert 及动作/时间投影 | 冻结是本地训练策略，不是 Pi0.5 必须条件 |
| 优化细节 | checked default YAML：AdamW β=(0.95,0.999)，weight_decay=1e-6，warmup500，有 EMA | AdamW 默认 β=(0.9,0.999)，weight_decay=1e-4，warmup0，无 EMA | 可控制的实验配方差异；需要消融验证，不应直接据此断言根因 |
| batch / epoch | checked default YAML 为256/500，但默认任务是 real_task；论文补充表5为128/50 |128/50，共6900步 | 当前与论文表5一致；没有证据说明原作者 Can 最终一定采用256/500，不建议机械改成500轮 |
| 验证和 checkpoint | 当前 workspace 的 rollout/validation/top-k 调用已注释；按 epoch 保存，保留 EMA；独立 eval 脚本使用 raw model，EMA选择被注释 | 每100步完整 validation，每500步 checkpoint；当前先测 seed0 final，后续才扫描 | 不能因为 YAML 写 use_ema=True 就声称上游独立评测一定使用 EMA；也无法从当前源码确认论文 best-checkpoint 选法 |

CPU 数据审计使用本地同一份 Paired 训练集进行公平计算：上游 horizon16/pad_before1/pad_after7 的目标填充约 **1.975%**，本地 horizon50/每 timestep 起点约 **24.803%**。后者增加了尾动作重复目标的占比；这不是已证明的性能主因。

将上游 seed42 的 2% episode 随机切分规则应用到本地官方 Paired 数据，会得到4条验证轨迹，其4个初始场景均有配对轨迹落在训练集。这个 CPU 实验只说明该规则不保证 Paired 场景隔离，不能反推作者未提供的 `image_v15.hdf5` 也有同样内容或实际运行时必然泄漏。本地场景隔离应保留。

动作归一化还改变旋转通道尺度：本地对应 scale 约2.083/2.632/1.433，而上游相对动作是1。这个映射的归一化零向量反变换后旋转通道为约 `[0.1046, 0.1677, -0.3020]`；这仅说明坐标表示不同，不能把作为无条件哨兵的零提示误解成一定要执行物理零动作。当前训练与推理均使用相同映射，未发现由此导致的直接反归一化不一致。

原作者数据路径是本地 `can_paird/relative_action/image_v15.hdf5`；未提供该文件或校验值。我们的数据来自官方低维数据并在 Robosuite1.4.1 下重新渲染，不能确认与作者图像文件的 XML、渲染、MuJoCo、数据转换逐项一致。仓库有多份不同环境 YAML，不能仅凭文件名 v15 就断言作者所有实验使用 Robosuite1.5。

### 推理和恢复：除 DP 与 flow matching 外的差异

| 项目 | 原仓库当前 Can runner 路径 | 本地实际运行 | 影响 |
| --- | --- | --- | --- |
| 检测频率 | 每次 policy 调用，通常已执行完整8步动作块后检测 | 每个物理前向动作后检测，即最多每块8次 | 本地触发机会更密，可能频繁中断动作块；与模型家族无关 |
| 阈值来源 | 在线积累历史误差，包含当前误差 | held-out ID 预标定，冻结 mean+3×std | 两种不同检测规则，本地不能再称与公开代码一致 |
| 负提示 | 保存整段 `action_pred`（horizon16，含未执行预测），恢复时传给策略归一化 | 保存真实失败执行动作，当前仅1步，重复到horizon50 | 提示信息内容和时间结构明显不同 |
| 回退动作 | `draw_back` 调用 `pushT_reverse`，仅将上一段8步动作按时间倒序，保留原符号和夹爪 | 根据实际前后 EEF 位姿，将上一步做闭环物理回退，最多10个控制 tick | 原活动路径没有对Can相对位姿求逆，本地闭环控制及容差均是额外设计 |
| 回退失败处理 | 活动逻辑中没有 pose tolerance 检查或对应的提前失败终止 | 5mm/0.05rad 未达标即 `rollback_failed` 结束该回合 | 本地新增终止条件，已影响 PH 各组17–18/50回合；不能归因于 Pi0.5 |
| 引导代数 | `(1+S)*epsilon_base-S*epsilon_failed` | `(1+S)*velocity_base-S*velocity_failed` | CFG 外推形式一致；epsilon→flow velocity 是骨干适配，不是相同采样过程 |
| S 参数 | Conditional 文件内部写死 `S=2.5` | 0/0.2/2.5/4.5 配置扫描 | 原仓库 YAML 的 guidance_scale 不控制这个硬编码，不能当成已完整实现论文扫描 |
| 回退后的引导 | pre_flag 状态下下一次 policy 调用使用负提示并清空相应历史统计 | 下一次重规划使用负提示，随后恢复基础提示；检测器本身不清空历史，因为阈值冻结 | 都有“回退后再引导”的组织，但检测/清空细节不同 |
| 步数预算 | MultiStepWrapper 每物理步计入预算 | 前向和闭环回退均逐物理步计入预算 | 计数原则一致；本地额外伺服 tick 仍可能消耗更多预算 |
| 评测配置 | Can YAML：50回合、seed100000、Paired400/MH500；`eval.py` 又覆盖为30回合、seed32、500步；`eval_vae_goal.py` 为10回合、seed42、500步 | 50回合、seed100000..100049、Paired400/MH500、公共初始状态目录 | 本地吻合 Can YAML，但不吻合两个手动覆盖脚本；无法确认哪个是论文最终协议 |

### 原仓库快照的已验证问题及不可照搬项

1. **错误标准差**：`get_error_statistics` 在均值和标准差两处均调用 `np.mean`，不是 `np.std`。提取原函数执行，历史 `[0.01,0.02,0.03]` 得 reported std=0.02，而真正 std≈0.008165。
2. **近乎禁用的错误触发**：`detect_error` 使用 `mean+3000000*std`，且先把当前误差加入历史。结合上述 bug，示例阈值为60000.02，当前0.03不会触发。更一般地，对非负误差且历史长度≤3000001，该阈值不小于历史最大误差，因此 strict greater-than 不会触发。这支持“该快照有关闭恢复的调试设置”的判断，不能据此推断作者最终实验也使用这个阈值。
3. **Can 活动回退不是求逆**：提取 `pushT_reverse`，输入连续8个 x=+0.25 的相对动作，输出仍为+0.25；只倒放顺序不会撤销相对平移。文件内虽存在矩阵求逆辅助函数，但 `draw_back` 没有调用它。
4. **无检测器路径需单独 runner**：当前 `robomimic_image_runner.py` 总是包装 `error_aware_policy`；None检测器返回布尔False，而后续 `draw_back` 对 flag 访问 `.shape`。仓库另有 `_base.py` runner 直接调用基础策略；不能假设默认 runner 同时支持两种路径。
5. **缺失子模块、包和失配配置**：见上文；没有通过直接执行该快照来验证论文分数。

这些问题应如实保留在对照记录中，不应复制进本地实现以追求源码表面一致。前述本地阈值/闭环回退虽然避免了一些原仓库问题，也属于不同算法选择，不能因此自动视为已严格复现。

### 结论与后续优先级

- 已实现负提示、基础/条件训练和恢复的大体框架，但目前既非作者源码的逐项移植，也不是“只将 DP 换成 Pi0.5”。普通 baseline 也存在切分、归一化、冻结策略、优化配方和窗口采样的差异。
- 将“论文实验复现”与“Pi0.5 + shuffle/闭环回退等变体”分别记录；30% shuffle 是用户指定的有意变体，暂不擅自删除或改变已训练模型。
- 先在相同本地协议下比较已保存的早期 baseline checkpoint。已知 baseline 验证损失最低在step1200为0.09311，final step6900为0.13252；尚未证明早期权重 rollout 更好。
- 若建立原方法对照，优先增设45%base/55%自身GT/0%shuffle条件训练；再比较非线性提示编码、动作块边界检测、负提示时间结构和回退失败处理。不能同时修改全部项目后把增益归于某一项。
- 不机械复制3000000倍阈值、错误std或仅倒序回退；先修复公开代码的可运行性、补齐其依赖来源并建立控制实验，才可能得到可信的原DP参考。
- 本轮首轮评测应继续保留原协议，源码对照没有改变正在运行的模型与评测进程。上述后续实验是建议，尚未启动新训练或更改阈值。

## 普通 Pi0.5 全参数 SFT 审计（2026-11-01）

用户明确要求 baseline 采用普通 Pi0.5 SFT。审计参照 Physical-Intelligence/OpenPI 提交 `215abfb217dbac7d5f1273282331b9b1866c0479`，不是将 DP 超参数作为 Pi0.5 必须标准。官方源码已缓存：[来源和SHA256](outputs/source_comparison/pi05_sft/upstream_manifest.json)。本次没有修改训练/评测源码、权重、数据或旧归一化统计，没有执行 optimizer step；单卡7上对完成的baseline checkpoint执行了前向和反向诊断，结束后显存已释放。

**结论：现有两组模型采用冻结VLM、仅训练动作专家的方案，不能代表用户要求的普通全参数SFT。** 另外存在输入token化顺序和损失范围两项明确的标准实现偏离；优化配方、归一化及checkpoint选择也需重新规范。不能据此声称每一项都已被证明是46%成功率的根因。

| 发现 | 证据 / 影响 | 分类 |
| --- | --- | --- |
| 无条件冻结VLM | `train_pi05.py:228`调用freeze函数；29.233亿个VLM参数全部requires_grad=False，实际仅4.301亿参数训练。审计复现训练参数开关，VLM梯度数为0 | 不符合用户要求的全参数SFT；不含LoRA替代 |
| 状态补零后才token化 | 真实state9维，先pad32，再将全部32维编码进文本；样本额外出现23个`128`，token数56→148。启用离散state时，OpenPI ModelTransformFactory的顺序为TokenizePrompt→PadStatesAndActions | 输入格式偏离；训练和推理同时存在，不能只改推理 |
| 仅监督前7维 | 本地`losses[..., :7].mean()`；普通OpenPI训练为`losses.mean()`，含补齐维度。实测25维输出head直接梯度为0，但其输入列梯度范数为0.01655 | 自定义损失目标，不是普通SFT等价替换；因果性能影响尚未测 |
| 优化配方未沿用Pi0.5参考 | 当前AdamW β=(0.9,0.999)、wd1e-4、lr1e-4无warmup→0；官方AdamW默认β=(0.9,0.95)、wd1e-10，默认cosine有warmup1000/peak2.5e-5/end2.5e-6，具体任务会覆盖 | 训练配方差异，不是确定代码bug；不能把较大学习率无warmup直接用于新全参训练而不检查 |
| min/max归一化 | 自定义adapter逐维min/max；OpenPI Pi0.5默认data factory启用quantile normalization。当前训练和推理使用同一保存统计，无直接错配 | 数据适配选择；建立普通SFT基线时应规范并保存新统计，不可改旧模型统计 |
| final明显偏离最低验证损失 | step1200为0.09311；step6900为0.13252，恶化42.32%，同时训练损失下降 | 过拟合线索及模型选择问题；尚不能断言早期checkpoint成功率必然更高 |

本次在真实checkpoint、真实训练样本、固定noise/time下验证：

- ExperimentalPI0Pytorch关闭动作提示后的loss逐元素与共享PI0Pytorch基类一致，最大差为0。共享PI0Pytorch的forward、sample_actions、embed_prefix、embed_suffix、denoise_step、sample_noise、sample_time、预处理调用等8个方法的AST与所缓存官方提交全部一致；预处理文件也逐字节一致。
- 仅启用前7维loss时，物理输出行梯度范数0.27464，其余25行严格为0；VLM无梯度。
- 解除VLM冻结并采用全32维loss后，只执行反向传播：vision patch梯度范数0.08778，语言层首层q_proj为0.04680，语言embedding为0.36554，其余25输出行梯度范数0.25232。说明现有图计算支持全参梯度，不需要将模型换回JAX或重写骨干。没有执行参数更新；仅证明梯度通路，不证明完整DDP训练的稳定性或批大小显存。
- 单样本诊断峰值显存15.29GiB；不能据此直接推断完整batch和optimizer的显存需求。

纠正几项不能当成bug的判断：

- **无EMA不是相对于官方PyTorch训练器的错误**：该训练器明确记录`EMA is not supported for PyTorch training`。JAX和部分task配置提供EMA，是另一种参考，不能混用标准。
- 单帧观测和horizon50是Pi0.5正常配置选择，不应强制变成DP的两帧/horizon16。末尾重复填充占比较高属于任务采样设计问题，尚未单独验证性能。
- 相对动作不应再额外套一次关节delta变换；当前没有证据表明相机布局、垂直翻转、动作反归一化或物理控制器约定出错。
- 权重转换逐张量核验、PyTorch内部前向一致性均已有证据；JAX/PyTorch同输入端到端数值一致性仍未验证，不能扩大上述结论。

修正顺序应为：建立可配置全参训练范围 → 真实维度state先token化 → 恢复标准32维flow loss和匹配验证指标 → 采用明确的Pi0.5优化/归一化配方 → 同配方baseline/conditional对照。旧冻结VLM结果保留为历史变体；新baseline应从原始pi05_base独立开始，避免把旧实验无缝resume成不同目标。用户本轮要求检查，以上修改和新训练尚未执行。

原始结果：[GPU/CPU探针](outputs/source_comparison/pi05_sft/probes.json)、[日志](outputs/source_comparison/pi05_sft/probes.log)、[结构化发现](outputs/source_comparison/pi05_sft/findings.json)、[方法AST对照](outputs/source_comparison/pi05_sft/core_ast_comparison.json)。

## 全参数 SFT v3 修复与验证（2026-11-01）

已按用户要求修复普通Pi0.5 SFT，并适配Can(Paired)。按后续明确指示继续使用现有`checkpoints/pi05_base_pytorch/model.safetensors`，没有下载或转换新base；来源、文件大小与SHA256见[权重记录](outputs/validation/full_sft_v3/base_checkpoint.json)。

修复包括：全量解冻VLM与动作专家，仅冻结未使用的expert LM head；32维flow loss及对应validation；真实9维state先token化再pad32；训练集q01/q99归一化与推理反归一化共用函数；保留旧统计和旧token化的读取兼容；旧训练协议不能resume成新协议。

训练使用OpenPI支持的FP32参数路径，AdamW β=(0.9,0.95)、wd1e-10、peak2.5e-5、warmup1000、cosine至2.5e-6。衰减长度适配Can的50epochs/6900更新步；global128和50epochs仍为任务实验选择，不冒充Pi0.5所有任务的固定值。梯度检查点及默认4次累积开启，4卡时micro8，8卡时micro4。DDP启用unused-parameter处理，因为最后的部分prefix输出不参与动作loss。

Can(Paired)保留180/20官方场景隔离划分、17780/2015帧，全部成功/失败轨迹；9维EEF/quaternion/gripper状态，7维相对OSC动作；两路RGB84→224，第三路mask；单帧/horizon50、末尾重复、不跨轨迹；已经是delta控制，不再施加joint delta变换。

新增best_validation.pt的完整checkpoint保存与恢复best loss；既有每500步checkpoint和final仍保留。全checkpoint评测会包含best_validation.pt，并将按验证loss选择与按rollout最高成功率选择区分。新默认输出/清单为outputs/pi05_full_sft_v3，不覆盖旧outputs/paper_can；评测可单独指定检测器根目录，默认复用已有独立AE。

验证结果：

- 36项单元及集成检查通过，含训练入口best/final保存、微批次位置恢复、梯度累积等价性、旧协议拒绝、真实状态token化、官方分位数公式一致性与旧归一化兼容。
- 双卡真实Can数据更新：FP32、每卡micro8、累积4次、等效global64，首个warmup学习率约2.4975e-8下视觉/语言/动作层均发生参数更新，25个补齐输出通道梯度非零；两rank动作输出权重校验和完全一致。
- 同一真实样本在训练与推理端的state、tokens及全部图像逐元素一致。
- 该配置实测峰值显存54.28GiB，GPU测试约59.7秒。验证相同micro8可用于4卡global128；8卡默认micro4。不把短测等同于长训练收敛或任务成功率。
- 旧base文件与旧策略/AE/评测结果未覆盖，诊断更新只存在测试进程内。没有启动新版长训练。

[完整验证记录](outputs/validation/full_sft_v3/validation.json)、[双卡更新原始数据](outputs/validation/full_sft_v3/gpu_smoke.json)、[测试日志](logs/full_sft_v3_unit_tests.log)、[GPU日志](logs/full_sft_v3_micro8_gpu_smoke.log)。

## 8卡配置与官方Pi0.5核心复核（2026-11-01）

已将train.sh及所有train_can_* shell默认GPU数统一为8；Paired-first lane也从4卡切换为8卡，策略队列共享文件锁按顺序使用8卡，CPU检测器保持独立。global128=8卡×micro4×累积4；不再按两组各4卡并行安排策略。生成了新版总清单和baseline/conditional两个8卡队列清单，未启动长训练。

32维loss依据是原始OpenPI目标：补齐干净动作`a[...,7:]=0`，但`x_t=t*noise+(1-t)*a`且`u_t=noise-a`，因此padding通道的速度目标是noise而非0。模型学习将这些通道还原为零动作；执行仍只有前7维。保留该目标是为了原始实现对齐，不把7维mask宣称为数学错误。validation新增full32/physical7/padding25三个指标；优化目标不变。

固定参照官方提交215abfb217dbac7d5f1273282331b9b1866c0479，检查12个模型/采样/tokenizer运行方法，计算AST全部一致；Gemma.forward仅多一个pytest.Cache类型注解，运行计算相同。预处理、优化器源码逐字节一致；另外下载并核对5个Transformers覆盖文件及Pi0Config源码，6个文件也逐字节一致。此处“下载”仅为源码核验，不涉及新权重。

8卡真实检查使用原有base、FP32、micro4、累积4、global128；视觉/语言/动作层均发生非零更新，8个rank这3层更新后的校验和相同。每个rank在同一真实Can样本、noise、time、随机图像增强下，扩展类无提示分支与官方等价PI0Pytorch基类的完整loss逐元素最大差均为0。实际耗时73.4秒，所有rank峰值显存最大54.27GiB，测试结束释放8卡。37项单元/集成检查通过，包含8卡队列清单及validation分解关系。

确认范围：模型与普通SFT的核心计算、损失、噪声/时间分布、tokenizer、官方覆盖代码、优化器数学配置一致。任务选择仍明确保留：Can9维state/7维relative action/两路84像素图像、官方180/20场景隔离切分，global128、50epochs/6900步（官方通用衰减默认30000步，实际task配置也会覆盖），FP32为官方支持选项但默认是BF16，梯度累积4和checkpoint/validation包装是本地工程选择。没有声称不同batching下随机优化轨迹逐位一致；没有新增JAX/PyTorch端到端对齐或长训收敛结论。Conditional提示与shuffle属于额外变体，普通baseline不含这些改动。

[完整对照报告](outputs/validation/full_sft_v3/eight_gpu_alignment.json)、[8卡原始更新数据](outputs/validation/full_sft_v3/gpu_smoke_world8_micro4_accum4.json)、[源码核心对照](outputs/validation/full_sft_v3/official_core_comparison.json)、[覆盖文件与模型配置SHA核验](outputs/validation/full_sft_v3/official_extended_comparison.json)、[8卡日志](logs/full_sft_v3_8gpu_smoke.log)。

## Kai0 padding mask核对与batch64切换（2026-11-01）

只读核对Kai0的pi05_baseline：config.py:3843定义FlowMatchingOutputSpec32，output_registry.py:93的flow_mse在完整动作维度mean，没有后补齐维度mask。AlohaInputs默认action_padding_mask_enabled=True，存在action_is_pad时取逻辑非，生成[B,H]时间步mask；OutputEngine在维度mean之后调用model.mask_action_loss，对无效时间步置零并按有效步数重新归一化。没有该数据标记时所有时间步有效。

提取原始函数CPU执行：有效步上仅第32维出现单位误差时最终平均loss为0.015625，仅第1维出现相同误差也为0.015625；误差仅落在被mask的时间步时为0。由此确认mask针对时间轴，不针对padding动作通道。[来源SHA及探针记录](outputs/validation/full_sft_v3/kai0_padding_mask_review.json)。没有改Kai0代码，也没有给本项目新增任何mask；用户先根据检查决定是否需要，当前保留32维loss和末尾动作重复。

按用户指定，8卡micro8、累积1、global64已写入Python/shell默认、canonical配置及两条策略队列清单。仍50epochs；实际每轮277步，共13850次optimizer更新。warmup1000、peak2.5e-5、end2.5e-6保持，cosine总长度随更新次数变化。旧global128配置及其8卡检查报告另存为historical，不与新配置混报。

37项回归/集成检查通过。新配置真实8卡更新：global64、FP32、所有rank的视觉/语言/动作层更新校验和一致，同输入/噪声/时间/增强下官方基类完整loss差均为0，峰值显存54.28GiB。训练/推理输入一致，仍用原转换base，数据切分180/20不变。没有启动长训练。

[8卡batch64原始报告](outputs/validation/full_sft_v3/gpu_smoke_world8_micro8_accum1.json)、[日志](logs/full_sft_v3_8gpu_batch64_smoke.log)、[当前对齐报告](outputs/validation/full_sft_v3/eight_gpu_alignment.json)。

## physical7 v4维度mask与8卡训练（2026-11-01）

按用户最新明确指示，在保持32维模型接口、噪声/时间采样及完整VLM训练的前提下，将优化目标改为`losses[..., :7].mean()`。后25维不参与监督；验证的loss及best_validation选择同步改为physical7，完整32维和padding25仅作诊断。未新增时间步mask。该目标是显式适配，不能把v3的官方完整32维loss对齐结论直接沿用为v4目标一致。

新recipe为pi05_can_full_sft_physical7_v4；配置和checkpoint记录supervised_action_dims=7；resume拒绝旧recipe和不同监督维度。默认输出独立为outputs/pi05_full_sft_physical7_v4，保留v2/v3文件。使用既有转换base，未重新下载或转换。

37项单元/集成检查通过，含仅物理7维参数更新、后25维无直接梯度、验证指标加权关系及best/final/resume流程。真实8卡micro8、累积1、global64检查通过：视觉/语言/动作层更新，8个rank更新校验和一致，后25维输出梯度范数严格0；最大峰值54.28GiB，检查耗时74.75秒。原始forward与官方等价基类在相同输入/noise/time/增强下完整loss最大差0；这里只证明原始计算一致，v4训练聚合主动改为7维。

[8卡原始报告](outputs/validation/full_sft_physical7_v4/gpu_smoke_world8_micro8_accum1.json)、[GPU日志](logs/full_sft_physical7_v4_8gpu_smoke.log)、[37项测试](logs/full_sft_physical7_v4_unit_tests.log)。

正式Paired baseline seed0已于2026-11-01 15:59:35 UTC以独立后台会话启动，supervisor PID3965756；GPU0–7，每卡8、累积1、global64，50epochs/13850步。仅该baseline已启动，conditional和MH未排队。checkpoint每500步保存，best按7维验证损失选择，最终保存final_model.pt。[启动](outputs/pi05_full_sft_physical7_v4/launch.json)、[状态](outputs/pi05_full_sft_physical7_v4/status.json)、[训练日志](outputs/pi05_full_sft_physical7_v4/policies/can_paired/baseline/seed_0/train.log)。

截至2026-11-01 16:04:04 UTC，已确认完成40次实际更新，稳定约3.29秒/步，单卡峰值55.72GiB。纯训练剩余约12.6小时；为每100步验证和checkpoint保存留出开销，当前估计baseline还需14–17小时。首轮验证/保存尚未计时，此为初步区间，不含conditional或评测。[估时记录](outputs/pi05_full_sft_physical7_v4/eta.json)。

## step8000中途评测启动（2026-11-02）

用户要求暂停训练并改测8000步。已终止原supervisor及8卡训练进程，确认8卡显存释放后，于01:13:57 UTC后台启动evaluate_can.py，PID4065492，GPU0。使用已有step_008000.pt、共同50场景目录及当前评测协议；Paired baseline、50回合、400步预算、无检测器，无短测覆盖项。不自动恢复训练。原训练主动中断记录为Interrupted，状态文件补注paused/user request。

[启动参数](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000/launch.json)、[评测日志](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000/driver.log)、[逐回合结果](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000/episodes.jsonl)、[训练暂停及恢复命令](outputs/pi05_full_sft_physical7_v4/pause_for_step8000.json)。本段记录启动事实，尚无最终成功率。

## 最新：5卡并行评测8000步

按用户追加指示，已停止原单卡评测，原部分结果保留；2026-11-02 01:18:59 UTC启动5卡队列（supervisor PID4068021）。GPU0/1/2/3/4分别评测目录索引0–9/10–19/20–29/30–39/40–49，每卡10回合，共50个原固定场景。每个场景继续使用原环境seed及策略noise seed，400步预算和baseline协议不变。训练仍暂停，不自动恢复。

分片入口保留完整rollouts=50，新增全局episode-start/end选择，不将每卡10回合误计为独立完整实验。每份分片summary明确complete_evaluation=False；全部完成后校验checkpoint、协议、归一化、实现、catalog及seed一致，拒绝重复/遗漏，输出完整50回合summary。11项评测及分片测试通过，日志logs/evaluation_shards_tests.log。原单卡的部分结果不混入新50回合。

[5卡状态](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/status.json)、[启动记录](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/launch.json)、[分片目录](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/shards)、[最终汇总](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/summary.json)。GPU5–7空闲；simulator使用既有EGL设备0，策略计算分别绑定GPU0–4。


## step8000最终统计与后续安排（2026-11-02）

8000步Paired baseline seed0的5卡评测已全部完成：**32/50成功，成功率64%，失败18回合均达到400步预算**。50个场景无重复或遗漏；成功回合平均99.59步。旧冻结VLM baseline为23/50=46%，相同场景下提升18个百分点：17场均成功、15场仅新版成功、6场仅旧版成功、12场均失败。多项训练配置同时改变，不能单独归因为VLM解冻、7维mask或某一超参数。单种子50回合的Wilson95%区间约50.1%–75.9%，尚不能作为三种子论文复现结果。

[完整汇总](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/summary.json)、[逐回合结果](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/episodes.jsonl)、[统计与旧版配对比较](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/analysis.json)。本轮从5卡启动到最慢分片完成约15.9分钟。当前训练仍暂停，评测进程均结束，GPU0–7空闲。

验证损失最低为step1000的0.13172；step3000/5000/8000分别为0.20640/0.27615/0.38115。后期训练损失仍低，存在过拟合信号，但尚不能推断哪个checkpoint的rollout成功率最高。不要仅因64%或验证损失就宣称已经收敛、需要延长至500epochs或确定某项修复有效。

| 优先顺序 | 实验 | 资源及预算 | 判断依据 |
| --- | --- | --- | --- |
| 1 | 补测baseline step1000、3000、5000 | 每个相同50场景，5卡×10；三组约45–60分钟 | 与已测step8000比较；step1000也是验证选中的checkpoint |
| 2 | 确定baseline续训顺序 | 如补齐原50轮，从8000恢复至13850，8卡约6–7小时 | 若早期明显更好，优先conditional；正式等预算比较仍需一致训练上限，不临时改LR schedule |
| 3 | 训练Paired conditional seed0 | 原base独立初始化，8卡micro8/accum1/global64，physical7；50轮13850步，暂估15–20小时 | 保持数据、split、全VLM训练、优化器一致；提示30%base/30%shuffle/40%自身GT，shuffle为额外变体 |
| 4 | 同步数评测baseline/conditional及恢复 | 先比较8000步及验证所选权重；conditional无ED+PH/MH/Paired各4个S，连baseline共14配置×50回合 | 分离conditional本身和恢复收益；保留final、validation选中、rollout最高的不同口径 |
| 5 | Paired seed1/2与消融 | 固定配方后补齐三种子；可增加0%shuffle对照 | 避免只用seed0结论；正式最高checkpoint口径补齐预设保存间隔的评测，早期筛查不是完整扫描 |
| 6 | MH策略实验 | Paired完成后，从同一base独立训练baseline/conditional×三种子 | 不从Paired权重继续；独立MH统计及split，复用适用AE |

同时回看若干失败轨迹，必要时生成视频，区分接近、抓取、滑落和放置问题；现有budget标签不能证明具体失败机制。现阶段不改控制器、mask、提示比例或评测预算，先完成checkpoint对照。

[后续实验及可执行命令](outputs/pi05_full_sft_physical7_v4/next_experiment_plan.json)已准备；**本轮只统计和规划，没有恢复训练或启动新评测**。下文保留已执行流程的历史记录。


## v4 conditional、检测器及引导与作者代码复核（2026-11-02）

本轮同时启动5000步baseline评测，GPU0–4各10回合，使用相同50场景和400步预算；训练保持暂停。[实时状态](outputs/evaluation/physical7_v4_paired_baseline_seed0_step005000_5gpu/status.json)、[启动记录](outputs/evaluation/physical7_v4_paired_baseline_seed0_step005000_5gpu/launch.json)。8000步64%仍是已经完成的结果，不能与进行中的5000步混合。

作者仓库仍为干净提交372552d6db59353cb3e1ee47126cbcbb12f34229。主条件分支参照diffusion_unet_image_policy.py，不能用默认YAML指向的普通yuan分支代替。此前audit.json里的冻结VLM、global128及旧归一化描述属于历史v2；本轮比较的是全参physical7 v4。新版conditional尚未训练，本轮只做源码、保存的AE配置及隔离CPU函数检查，没有执行作者端到端训练，也没有改动当前评测代码或训练配方。

| 项目 | 作者公开主路径 | 当前v4 | 判断 |
| --- | --- | --- | --- |
| 提示比例 | 45%零base、55%自身GT，无shuffle | 30%base、30%跨样本GT、40%自身GT | 用户指定额外变体；不是Pi0.5强制性质 |
| 提示编码与融合 | MLP action→64→64→视觉宽度，16步投影到2个观测位置，与视觉特征拼接 | 零初始化线性32维→动作token宽度，逐步加到50个suffix token | 明确结构适配，不是原编码器等价替换 |
| 检测器 | AE/VAE多分支配置不一致；可见BasicVAE为单图卷积重建，另有未填充AE gitlink | 双相机84像素拼接、MLP512/128、Sigmoid、mean MSE | 无法确认作者最终Can结构；当前是自主重建实现 |
| AE训练 | 公开train_ae默认PushT、AdamW、batch512、1000轮；不是已核实Can最终配方 | 已有PH/MH/Paired均Adam、batch128、lr1e-4、cosine50轮、官方场景隔离split | 独立训练差异，不能归因于策略骨干 |
| 检测阈值 | 在线历史含当前分数；std误写mean；乘数3000000 | 固定held-out ID均值+3×真实std | 原路径存在明确错误，不应照搬；固定标定仍属本地选择 |
| 检测时机 | 每次策略调用，通常8步动作块之后 | 每执行1个物理动作就检测，可中断当前块 | 恢复调度差异，非骨干必然属性 |
| 失败提示 | 缓存完整action_pred，含未执行预测 | 实际执行的最后1步失败动作重复至50步 | 失败提示时间结构不同 |
| 回退 | 活跃pushT_reverse只倒序8步，并非相对OSC的逆动作 | 实测pose闭环回退1步，最多10tick；5mm/.05rad；失败终止 | 恢复机制实质不同，不能机械移植倒序动作 |
| 引导公式 | (1+S)ε_base−Sε_failed | (1+S)v_base−Sv_failed，同一x/t，负dt Euler | 仿射负引导代数一致；ε/v属于骨干差异，无须额外翻转符号 |
| S的实际使用 | 主条件采样函数写死2.5，修改guidance_scale属性无效；其他分支另有0/.25 | 配置S真正传入采样，扫0/.2/2.5/4.5 | 当前可配置性已验证；相同S不等于相同物理引导幅度 |

CPU检查：10万样本提示分支约30%/30%/40%，shuffle自配0、监督目标不变；作者函数设置guidance_scale属性为0/.2/2.5/4.5，固定base=1/negative=3时结果均为-4（仍S2.5）；本地相应引导velocity为1/.6/-4/-8，参数确实生效。历史分数.01/.02/.03时作者std返回.02，真实std=.008165，作者阈值60000.02。本地同分数标定为mean+3std=.044495。40项回归检查全部通过。

本地已完成AE阈值PH=.01339062、MH=.01303989、Paired=.01403827，均在held-out ID上标定。AE重建已学习不代表错误分类质量已得到证明；下一步应检查正常rollout误报及真实失败的检出情况。

建议先固定conditional提示融合方案，再开新版长训；保留用户30%shuffle，额外添加45%base/55%自身GT/0%shuffle控制组以比较作者采样。检测时机和失败提示各单独做消融；保留正确std，别复制3000000或动作倒序。评测先比较conditional无ED、ED+S0，再看其他S，才能区分回退收益与负引导收益。S0带ED仍会检测和回退，不等于无ED。

[结构化差异及来源](outputs/source_comparison/reach_v4_recovery_audit/audit.json)、[隔离CPU数值检查与源码SHA](outputs/source_comparison/reach_v4_recovery_audit/probes.json)、[检查脚本](outputs/source_comparison/reach_v4_recovery_audit/probe.py)、[40项测试日志](logs/reach_v4_recovery_audit_tests.log)。本轮未修复或调整这些实验差异。


## 当前提示配比：作者主分支45%base/55%自身GT

按用户最新要求，conditional默认已改为45%零base、55%自身GT、0%shuffle。Python训练入口、train_step/train_update默认、shell、canonical配置与新manifest一致。通用共享build_action_condition助手未改动，训练调用显式传入45/0；不影响其他项目。旧30/30/40记录和执行manifest保留，新清单另存[author_prompt45_experiment_manifest.json](outputs/pi05_full_sft_physical7_v4/author_prompt45_experiment_manifest.json)。当前只有配方修改，没有启动conditional训练。

baseline从不使用动作提示，其历史base_prob=.3保留以通过严格resume检查；已用真实baseline config及原命令检查8卡从8000恢复，参数校验通过。监督7维、完整VLM训练、batch64、50轮及学习率计划均不变。40项检查通过；10万样本实测base约44.924%、shuffle0。旧source/config已存档。[修改与验证记录](outputs/source_comparison/author_prompt45_change/validation.json)、[测试日志](logs/author_prompt45_tests.log)。

作者检测器分支补充清单：

| 分支 | 实现与用途 | 当前可确认范围 |
| --- | --- | --- |
| model/vae/ae_nets.py的BasicVAE | 128像素RGB卷积AE，Conv/BatchNorm/ReLU编码、4×8×8 latent、反卷积/Tanh重建；MSE | 主YAML的error_detector明确指向它；PushT workspace有训练路径。虽然名含VAE，forward没有mu/logvar、采样或KL；不是变分模型 |
| model/vae/ae.py的BasicVAE | 84像素RGB卷积AE，latent调到22×22、输出调到84×84 | forward可运行，但loss有pdb断点；与vae_policy调用接口不匹配 |
| img_vae.py、image_vae.py、sequence_vae.py | 多个ImageSeqVAE原型，mu/logvar与重参数化；MLP或CNN/LSTM/时间解码 | img_vae有重建+KL；其他分支含缺失方法/固定形状问题，未确认最终Can路径 |
| AEImagePolicy + ae.model.ae_model.Network | train_ae指向外部Network，embedding64，BCE | ae是未填充gitlink、无.gitmodules，无法恢复完整结构；wrapper还切片batch14:16 |
| train_vae.yaml中的robomimic VAE | latent512、GMM prior等配置 | 目标模块缺失；与现有wrapper/workspace的参数和返回接口不一致，不能视为已核实最终Can实现 |

隔离CPU前向确认：ae.py的84输入输出84；ae_nets.py的128输入输出128，而84输入输出80。因此不能不调整图像尺寸就把128分支直接接当前84像素重建目标。检测器本身不依赖DP，可以单独适配到Pi0.5实验；需要决定输入/相机/重建评分接口并重训标定。[分支清单](outputs/source_comparison/author_prompt45_change/detector_inventory.json)、[尺寸检查](outputs/source_comparison/author_prompt45_change/detector_shape_probes.json)。

作者提示编码不能原样插入Pi0.5：它将DP16步提示投影到2个观测位置，并拼入U-Net的global_cond；Pi0.5当前单帧、50步动作token，通过prefix/suffix和动作专家工作，没有同一global_cond接口。但MLP编码本身并不与Pi0.5冲突，可把非线性输出宽度改为动作专家hidden，继续按当前suffix接口注入。当前零初始化线性注入是合法适配，不能称为原模块等价，也没有证据证明非线性一定更好。本轮按用户要求仅改提示比例，未改提示融合或检测器架构。

以下30%shuffle审计与规划属于变更前快照，当前配置以上述45/55/0为准。

