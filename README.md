# REACH Can 实验：Pi0.5 策略版本

本项目在 Robomimic **PickPlaceCan** 上实现 Pi0.5 baseline、conditional 策略、图像自编码器 error detector、物理回退和负向动作引导。Paired、PH、MH 策略分别从同一 Pi0.5 base 独立训练，不进行 PH→MH 串行微调。

更新至 **2026-11-04**：本文统一以 **2,000 optimizer steps 及以下**的策略 checkpoint 为训练示例和结果参考；主结果固定报告 2k，500/1k/1.5k 用于观察训练过程。当前结果为单训练 seed、每配置固定 50 个场景，尚未完成三训练 seed 的完整比较，也不声明与 REACH 原论文完全复现。

## 当前结果

以下均为 seed 0、无 detector；“MLP关”是对同一 conditional 权重的推理消融，不是独立训练的 baseline。

| 策略训练数据 | 独立 baseline 2k | Conditional 2k，MLP开 | 同一 conditional，MLP关 | Conditional开相对baseline |
| --- | --- | --- | --- | --- |
| Paired | 22/50（44%） | 35/50（70%） | 32/50（64%） | +26pp |
| PH | 49/50（98%） | 49/50（98%） | 50/50（100%） | 0pp |
| MH | 47/50（94%） | 49/50（98%） | 50/50（100%） | +4pp |

Paired 表中结果取自历史训练的 2k checkpoint；其学习率衰减计划不同于 PH/MH 的固定 2k 短训。下方命令为新实验的固定 2k 配置，不能把历史 Paired 数值当成这些新命令已复现的结果。PH/MH 的 baseline 与 conditional 均使用 2k 学习率时间尺度、1k warmup。

### 2k及以下训练曲线

| 策略训练数据 / 模型 | 500 steps | 1k steps | 1.5k steps | 2k steps |
| --- | --- | --- | --- | --- |
| Paired baseline | — | 16/50（32%） | — | 22/50（44%） |
| Paired conditional，MLP开 | — | 31/50（62%） | — | 35/50（70%） |
| PH baseline | 36/50（72%） | 46/50（92%） | — | 49/50（98%） |
| PH conditional，MLP开 | 49/50（98%） | 44/50（88%） | — | 49/50（98%） |
| MH baseline | — | — | — | 47/50（94%） |
| MH conditional，MLP开 | 45/50（90%） | 44/50（88%） | 46/50（92%） | 49/50（98%） |

成功率不随训练步数单调变化；validation loss 也不能代替环境成功率。“—”表示没有纳入本文的对应评测。

### Conditional 2k的恢复结果

固定同一 conditional 权重、MLP开启；detector 使用冻结 Gaussian κ=4 阈值，整块回退，完整预测计划作为 negative prompt。

| 策略训练数据 | Detector训练数据 | 无恢复 | S=0 | S=0.2 | S=2.5 | S=4.5 |
| --- | --- | --- | --- | --- | --- | --- |
| Paired | PH | 70% | 72% | 72% | 72% | 72% |
| Paired | Paired | 70% | 68% | — | 70% | 70% |
| Paired | MH | 70% | 68% | — | — | — |
| PH | PH | 98% | 96% | 96% | 96% | 96% |
| MH | MH | 98% | 98% | 98% | 98% | 98% |

Paired 配 PH detector 相对同一模型无恢复救回 1 局；强负向引导没有比 S=0 额外增加成功场景。PH 恢复额外丢失 1 局；MH 仅 episode 28 触发恢复，所有已测 S 均未救回。当前主要收益出现在 conditional 训练比较，尚未观察到稳定的额外负向引导收益。

完整指标、场景配对和历史记录见 [评测结果总览](docs/EVALUATION_RESULTS_SUMMARY.md)。本地逐局记录、trace 和权重存放在 `outputs/`，通过 `.gitignore` 排除，不随 Git 上传。

## 环境与文件布局

模型与仿真分别使用已有环境，通过独立进程通信：

| 用途 | 本机路径 / 版本 |
| --- | --- |
| 策略训练与推理 | `../RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python`，Python 3.11 / PyTorch |
| 仿真 | `../reach_simulation_env/venv/bin/python`，Python 3.10 / Robosuite 1.4.1 / MuJoCo 3.1.0 |
| 共同初始权重 | `checkpoints/pi05_base_pytorch/model.safetensors` |
| 图像数据 | `data/paper/can_{ph,mh,paired}_image.hdf5` |
| 固定评测场景 | `outputs/evaluation/can_initial_states_50.json` |
| 训练 / 评测脚本 | `scripts/` |
| 模型与评测配置 | `configs/` |
| 结果分析与实验记录 | `docs/` |

Git 仓库只保存代码、配置和文档。数据、权重、日志、输出、视频及论文 PDF 不上传；在新机器上需要另行准备这些文件和依赖环境。现有文件中的绝对路径反映本机目录，迁移时需要核对。

```bash
cd /2025233210/lmx/reach_can_training
export PYTHON=../RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python
export JAX_PLATFORMS=cpu
```

## 准备数据

```bash
# 下载并渲染三种数据
bash scripts/prepare_can_data.sh --dataset-types ph mh paired

# 仅下载 Paired low-dim 数据
bash scripts/prepare_can_data.sh --dataset-types paired --download-only

# 检查本地数据
bash scripts/prepare_data.sh
```

PH 包含 200 条熟练操作者成功轨迹；MH 包含 6 位不同熟练度操作者的 300 条成功轨迹；Paired 包含 100 个初始场景各一条成功和失败轨迹，共 200 条。Paired 是独立数据，不通过 PH+MH 拼接得到。

优先使用官方 `mask/train` 和 `mask/valid`。Paired 的同场景成功/失败演示始终位于同一集合；没有官方 mask 时按场景分组划分。策略与 Paired detector 保留成功和失败轨迹，不使用成功标签过滤。归一化统计只用训练集计算，推理从 checkpoint 恢复。

MH/Paired 图像恢复原始场景 XML 并逐帧设置保存的 simulator state，不通过执行动作推测下一状态。现有 PH 图像复用旧渲染像素，按 Robomimic 约定翻转并补齐元数据；它不是重新渲染的分支，转换来源保存在数据旁的报告中。未完成的 `.partial.hdf5` 不用于训练。

## 固定2k策略训练

新任务从共同 base 独立开始，使用新的 `OUTPUT` 目录，避免覆盖已完成的实验。以下命令依次运行，每个任务占用 8 张 GPU。

```bash
# Paired baseline：新实验，固定2k学习率计划
DATASET_TYPE=paired POLICY_MODE=baseline NUM_GPUS=8 NUM_STEPS=2000 \
  bash scripts/train.sh --lr-decay-steps 2000

# Paired conditional：MLP action-condition projection
DATASET_TYPE=paired POLICY_MODE=conditional NUM_GPUS=8 NUM_STEPS=2000 \
  bash scripts/train.sh --lr-decay-steps 2000 --action-condition-projection mlp

# PH baseline / conditional
DATASET_TYPE=ph POLICY_MODE=baseline NUM_GPUS=8 NUM_STEPS=2000 \
  bash scripts/train.sh --lr-decay-steps 2000
DATASET_TYPE=ph POLICY_MODE=conditional NUM_GPUS=8 NUM_STEPS=2000 \
  bash scripts/train.sh --lr-decay-steps 2000 --action-condition-projection mlp

# MH baseline / conditional
DATASET_TYPE=mh POLICY_MODE=baseline NUM_GPUS=8 NUM_STEPS=2000 \
  bash scripts/train.sh --lr-decay-steps 2000
DATASET_TYPE=mh POLICY_MODE=conditional NUM_GPUS=8 NUM_STEPS=2000 \
  bash scripts/train.sh --lr-decay-steps 2000 --action-condition-projection mlp
```

命令显式指定 `NUM_STEPS=2000` 和 `--lr-decay-steps 2000`。`train.sh` 未指定步数时仍有基于 epoch 的历史默认配置，因此应完整使用上述参数。conditional 必须显式传 `--action-condition-projection mlp` 才与本文 MLP 模型结构一致。

| 参数 | 本文新训练配置 |
| --- | --- |
| Optimizer更新 | 2,000 steps |
| GPUs / global batch | 8 / 64，每卡 micro batch 8，累积 1 次 |
| 优化器 | AdamW，β=(0.9, 0.95)，weight decay=1e-10 |
| 学习率 | peak 2.5e-5，warmup 1,000 steps，cosine 到 2.5e-6 |
| 精度 / 内存 | FP32，gradient checkpointing开启 |
| Checkpoint / validation | 每 500 / 100 steps |
| 策略训练 seed | 0 |
| Conditional投影 | MLP |
| Conditional提示混合 | 45%零base、55%自身GT、0%shuffle |
| 监督目标 | 当前样本自身动作，仅前7个物理动作维度 |

提示分支逐样本随机选择；base 是归一化动作空间中的零张量。使用 GT action prompt 只发生在训练阶段，评测时不提供未来真实动作。baseline 不使用动作提示分支。

### 观测、动作与损失

保留 Pi0.5 单帧观测、50步 action horizon；两路图像映射为 base/left_wrist，第三路屏蔽。真实状态先归一化并 token 化，再将连续 state 张量补零到32维。

Can 的7维相对 OSC 动作补零到32维，不额外做 joint delta 变换。模型保留32维 flow 输入输出，训练和 validation 使用 `losses[..., :7].mean()`；补齐的25个动作通道不计入目标，另记录完整和 padding loss 作诊断。轨迹末尾重复最后动作，不跨 episode；当前没有加入时间步 padding mask。

Conditional 使用 MLP 投影把动作提示接入去噪动作 token。该实现是对 Pi0.5 的适配，与作者 DP 的条件特征接入方式不同。

### Checkpoint与恢复

每500步保存 `checkpoints/step_*.pt`，验证损失改善时保存 `best_validation.pt`，结束保存 `final_model.pt`。checkpoint 包含模型、优化器、scheduler、各 rank RNG、采样位置、数据切分和归一化统计。`best_validation.pt` 按动作验证损失选择，不代表闭环成功率最高。

同一任务中断后，可设置 `RESUME=/path/to/checkpoint.pt` 恢复，必须保持原始模式、训练目标、数据、batch、学习率、seed、GPU数量和动作提示设置。旧的冻结VLM实验不能直接恢复成全参数训练。初始 JAX 权重转换入口为 `bash scripts/convert_checkpoint.sh`。

## Error detector训练与阈值

PH、MH、Paired detector 分别从各自数据独立训练，与策略权重无关。输入为两路原始 uint8 或 `[0,1]` 图像，不能直接输入 Pi0.5 的 `[-1,1]` 图像。

```bash
"$PYTHON" scripts/train_error_detector.py \
  --data data/paper/can_ph_image.hdf5 --dataset-type ph \
  --output outputs/ae_ph_manual_kappa4 \
  --epochs 50 --batch-size 128 --optimizer adam --threshold-method gaussian --kappa 4
```

这里的50 epochs是独立 AE 训练长度，不是策略更新步数。保存 `detector.pt`、`last.pt`、`calibration.json`、切分和优化诊断。Gaussian 阈值用留出的 ID 图像校准：`threshold = mean + 4 × std`，正式评测冻结，不按成功标签调整。后续 detector 训练、校准和正式评测统一 κ=4。

AE 重建误差下降不等于任务错误检测有效；不同 detector 的分数分布不同，不能仅凭原始阈值大小解释优劣。当前 AE 使用 Adam，是相对最初 SGD 配置的适配。

## 真实仿真评测

| 项目 | 当前协议 |
| --- | --- |
| 任务 | PickPlaceCan，从数据元信息创建 |
| 场景 | 固定50个初始状态，环境seed 100000–100049 |
| 策略checkpoint | 500/1000/1500/2000中已有候选；2k为主结果 |
| 预测 / 执行 / 去噪 | 50 / 8 / 10 steps |
| 物理步预算 | PH/Paired 400；MH 500，包含回退 |
| Detector时机 | 每个执行chunk之后、下一次策略调用之前 |
| 回退 | 反向访问最近8步记录的EEF目标，每个目标最多10步闭环控制 |
| 回退容差 | 位置5mm、旋转0.05rad；夹爪复用记录的前一指令 |
| Negative prompt | 最近完整50×7 `action_pred`，在环境动作clipping之前缓存 |
| 引导强度 | S=0、0.2、2.5、4.5 |

S=0 配 detector 时仍会检测、回退和重新采样，只关闭负向引导。普通推理使用零 base prompt；恢复后下一次重规划使用失败计划作 negative prompt。Gaussian 初始噪声为 `[1,50,32]` 的标准正态样本，每次 policy call 的噪声种子固定可复现。引导从 flow 时间1积分到0，使用 `v_base + S × (v_base - v_failed)`。

```bash
"$PYTHON" scripts/check_evaluation.py

# 仅在新部署且文件不存在时创建固定场景；已有catalog不覆盖
# "$PYTHON" scripts/eval_catalog.py \
#   --metadata data/robomimic/can/paired/low_dim_v141.hdf5 \
#   --output outputs/evaluation/can_initial_states_50.json --count 50 --seed-start 100000

# 示例：MH conditional的2k及以下checkpoint和推理MLP消融
"$PYTHON" scripts/evaluate_conditional_mlp_suite.py \
  --policy-dir outputs/pi05_full_sft_physical7_v4_mlp/policies/can_mh/conditional/seed_0 \
  --detector outputs/paper_can/detectors/can_mh/seed_0_kappa4/detector.pt \
  --detector-label mh --gpus 0 1 2 3 4 5 6 7 \
  --checkpoint-steps 500 1000 1500 2000 --baseline-only --ablate-steps 2000 \
  --output outputs/evaluation/mh_conditional_2k_policy_manual

# 固定2k checkpoint，比较同一个MH detector的各个S
"$PYTHON" scripts/evaluate_conditional_mlp_suite.py \
  --policy-dir outputs/pi05_full_sft_physical7_v4_mlp/policies/can_mh/conditional/seed_0 \
  --detector outputs/paper_can/detectors/can_mh/seed_0_kappa4/detector.pt \
  --detector-label mh --gpus 0 1 2 3 4 5 6 7 \
  --checkpoint-steps 2000 --recovery-only \
  --output outputs/evaluation/mh_conditional_2k_recovery_manual
```

评测输出目录必须是新目录。两轮依次运行：第一轮5个配置、250局，第二轮4个配置、200局。每配置拆成两个25局shard，调度器最多同时使用8张GPU。失败重试保留相同场景与噪声设置，不选择性丢弃失败episode。

输出包括 `run.json`、`episodes.jsonl`、`episode_*_trace.json` 和 `summary.json`，记录 checkpoint / normalization / catalog / implementation 哈希、实际环境、每局终止原因和物理步数。MH自动流程还生成总表、逐场景救回/丢失比较和审计；见 [MH训练与评测计划](docs/MH_CONDITIONAL_2K_EVAL_PLAN_2026-11-03.md)。

少于50局的指定场景视频仅作说明，不计为完整基准评测。改变阈值、缩短预算等诊断必须另存，不混入正式结果；平均步数下降也可能来自回退失败提前终止，不能直接解释为效率提高。

## 视频与检测曲线

`render_evaluation_traces.py` 可从保存的 action trace 重放双机位视频。`visualize_detector_comparison.py` 将同一场景的PH/Paired detector结果组合为左右同步画面，上方显示实际AE误差、冻结阈值、触发点和回退区间。两侧按物理步对齐，较早结束的一侧保留最后画面。

```bash
"$PYTHON" scripts/visualize_detector_comparison.py \
  --ph-run /path/to/ph_detector_run --paired-run /path/to/paired_detector_run \
  --episode-id 26 --expected-outcomes both-failed \
  --output outputs/visualizations/episode026_comparison_manual
```

输入目录需包含该场景的 `run.json`、`episodes.jsonl`、trace 和原始视频。默认对照模式为 `ph-success-paired-failed`；`both-failed` 要求两侧都实际触发物理回退且失败。曲线只使用已记录检测分数，连接线不表示额外测量。选例视频不能代替50场景统计。

## 诊断与验证

训练默认记录固定留出样本的 `branch_monitor.jsonl`，比较 base/GT/foreign 条件的物理7维 flow 指标；监控保留随机状态、不反向传播。validation 同样不更新模型参数。

```bash
"$PYTHON" scripts/diagnose_steering.py \
  --checkpoint outputs/pi05_full_sft_physical7_v4_mlp/policies/can_mh/conditional/seed_0/checkpoints/step_002000.pt \
  --output outputs/diagnostics/mh_conditional_2k.json \
  --samples 4 --seed 2026 --num-steps 10 --scales 0 0.2 2.5 4.5

JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 "$PYTHON" -m unittest discover -s tests -v
```

离线 steering 诊断检查固定观测与噪声下的提示作用、Euler采样一致性及S=0对照，不代表闭环成功率。最终结论以保存的实际rollout和逐场景比较为准。

训练和评测适配、作者源码对照及历史审计见 [实现核验记录](PAPER_REPRODUCTION_VALIDATION.md)。论文策略骨干为Diffusion Policy，本地为预训练Pi0.5全参数SFT；观测horizon、动作提示接入、AE优化器、冻结阈值和EEF伺服回退存在差异，所有本地结果均按适配实验解释。
