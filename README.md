# REACH Can 实验：Pi0.5 策略版本

当前状态（2026-11-02）：Paired conditional seed0已于04:11:48 UTC启动完整8卡训练，45%零base/55%自身GT/0%shuffle，global64、50轮/13850步；实际数据划分、归一化和训练超参数与baseline核对一致。baseline保持暂停，8000步评测为**64%（32/50）**，5000步评测已完成**26%（13/50）**。[当前训练状态](outputs/pi05_full_sft_physical7_v4/status.json)、[配置比较](outputs/pi05_full_sft_physical7_v4/conditional_seed0_configuration_check.json)、[baseline续训脚本](scripts/resume_can_paired_baseline_seed0_v4.sh)、[详细进度](EXPERIMENT_PLAN_AND_PROGRESS.md)。

训练流程按 REACH 论文表 2 组织。策略骨干保留 Pi0.5，**Can(Paired) 与 Can(MH) 是独立实验，不再进行 PH→MH 串行微调**。条件策略当前按作者公开主分支使用45%零base、55%自身GT，不使用shuffle；旧30%shuffle实验作为历史变体保留。

当前实现包括策略训练、自编码器错误检测器训练与标定、独立实验调度、离线 steering 诊断、真实 PickPlaceCan rollout、物理 SE(3) 回退和在线负向引导，以及多 GPU 评测与结果汇总。评测遵循公开流程并列明适配及未公开参数；尚未完成三种子的正式成功率统计，不声称与原论文数值逐项复现。

整体实验顺序、当前训练与数据进度、资源分配及时间估算见 [整体实验安排与当前进度](EXPERIMENT_PLAN_AND_PROGRESS.md)。

## 实验矩阵

| 策略训练集 | 策略 | 单独训练的检测器 |
| --- | --- | --- |
| Can(Paired) | Pi0.5 baseline、条件 Pi0.5 | PH、MH、Paired |
| Can(MH) | Pi0.5 baseline、条件 Pi0.5 | MH、PH |

三种子 `0/1/2`：12 个策略任务、9 个检测器任务。按 guidance scale `0/0.2/2.5/4.5` 生成 60 个带检测器组合，另加 12 个无检测器 baseline/conditional 对照，共 72 个待评估组合。

所有策略从同一 `checkpoints/pi05_base_pytorch/model.safetensors` 初始化。检测器从头训练，不加载策略权重。每个数据集的 baseline/conditional 使用相同官方切分及训练集归一化；不同数据集之间不传递权重或归一化统计。

PH 是 200 条熟练操作者成功轨迹；MH 是 6 个不同熟练度操作者的 300 条成功轨迹；Paired 是 100 个初始场景各一条成功、一条失败，共 200 条轨迹。Paired 使用官方独立数据，不能通过 PH+MH 拼接得到。

## 准备数据

```bash
cd /2025233210/lmx/reach_can_training

# 下载并渲染论文三种数据；耗时取决于渲染速度
bash scripts/prepare_can_data.sh --dataset-types ph mh paired

# 仅下载 Paired low-dim 数据
bash scripts/prepare_can_data.sh --dataset-types paired --download-only

# 查看数据是否完整
bash scripts/prepare_data.sh
```

Paired low-dim 已下载至 `data/robomimic/can/paired/low_dim_v141.hdf5`。PH/MH low-dim 优先复用原 `reach_simulation_env/data/robomimic/can/` 下的数据。

新图像写到 `data/paper/can_{ph,mh,paired}_image.hdf5`，与早期图像文件分开保存。渲染逐条恢复原场景 XML、逐帧设置已记录的 simulator state，遵循 Robomimic 图像翻转约定，不通过执行 action 推测下一状态。保留 states、rewards、dones、episode 属性及官方 mask。临时 `.partial.hdf5` 完成后才原子改名为最终文件；不会读取活跃 writer 的未完成文件。

优先使用官方 `mask/train` 与 `mask/valid`。Paired 按原始初始状态生成场景 ID，并验证每组两条演示始终位于同一集合。没有官方 mask 时按场景分组划分。Paired 的成功和失败轨迹全部保留，策略与 AE_Paired 都不使用成功标签过滤。

### 复用已完成的 PH 图像

PH 已转换为 `data/paper/can_ph_image.hdf5`：保留全部 200 条轨迹、23,207 帧，将旧图像上下翻转并从官方 low-dim 恢复 states、XML 属性、rewards、dones、next_obs 与全部 masks。完整图像 payload 已读回校验，报告为 `data/paper/can_ph_image.conversion.json`。

该分支复用旧渲染像素，**不是逐条恢复 XML 后重新渲染**；HDF5 中明确记录 `original_scene_xml_rerendered=False` 和转换来源。MH 与 Paired 继续使用新版原 XML/逐帧状态渲染。后续准备命令会复用这个已经完成的 PH 文件。

可重复使用的转换工具（输出必须是一个尚不存在的新文件）：

```bash
../reach_simulation_env/venv/bin/python scripts/convert_legacy_images.py \
  --legacy ../reach_simulation_env/data/robomimic_images/can_ph_image.hdf5 \
  --output /path/to/new_ph_image.hdf5
```

数据处理使用 `../reach_simulation_env/venv/bin/python`，策略及检测器使用 `../RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python`。分别可用 `SIM_PYTHON` 和 `PYTHON` 覆盖。

## 生成、执行实验

```bash
# 只生成可审查的任务清单，不训练
bash scripts/auto_pipeline.sh --plan-only

# 全部 21 个训练任务，策略使用 8 卡
NUM_GPUS=8 bash scripts/auto_pipeline.sh

# 只训练检测器
bash scripts/auto_pipeline.sh --stage detectors

# 先做一个 PH 检测器任务
bash scripts/auto_pipeline.sh --stage detectors --jobs detector_ph_seed_0

# 只做 seed 0 的策略实验
NUM_GPUS=8 bash scripts/auto_pipeline.sh --stage policies --seeds 0
```

清单保存在 `outputs/pi05_full_sft_physical7_v4/experiment_manifest.json`，每个任务包含完整命令、输入、初始 checkpoint、输出和指纹。输出布局：

```text
outputs/pi05_full_sft_physical7_v4/
  policies/can_paired/{baseline,conditional}/seed_{0,1,2}/
  policies/can_mh/{baseline,conditional}/seed_{0,1,2}/
  detectors/can_{ph,mh,paired}/seed_{0,1,2}/
  experiment_manifest.json
  status.json
```

调度器按任务独立执行，只有匹配任务指纹且存在完成 artifact 时才复用结果。失败即停止并保存具体任务状态和日志位置。`--wait-seconds` 可指定等待数据的时长。`monitor_pipeline.sh` 显示任务状态。

## 单独训练策略

```bash
# Paired baseline
DATASET_TYPE=paired POLICY_MODE=baseline NUM_GPUS=8 bash scripts/train.sh

# Paired 条件策略
DATASET_TYPE=paired POLICY_MODE=conditional NUM_GPUS=8 bash scripts/train.sh

# MH baseline / 条件策略，均从原始 Pi0.5 权重独立开始
DATASET_TYPE=mh POLICY_MODE=baseline NUM_GPUS=8 bash scripts/train.sh
DATASET_TYPE=mh POLICY_MODE=conditional NUM_GPUS=8 bash scripts/train.sh
```

默认所有训练shell入口和Paired-first策略队列均为8卡，每次一个policy独占8卡；baseline与conditional按顺序运行。Paired-first的CPU检测器仍可独立运行。当前采用 Pi0.5 全参数 SFT：global batch64、50 epochs，AdamW β=(0.9,0.95)、weight decay=1e-10，peak LR=2.5e-5、warmup1000更新步、cosine至2.5e-6。学习率数值与优化器采用 OpenPI 参考默认；将衰减长度压缩至实际 Can 更新次数是任务选择，并非照搬DP。使用 FP32 参数训练（已有BF16 base载入为FP32，不重新下载），默认累积1次、梯度检查点开启。8卡时每卡micro batch8，global batch64；Paired为每轮277次更新，共13850步。数据、观测和动作表示保持Can任务语义。仅rank0保存。

数据类型不再自动决定是否启用动作条件：`POLICY_MODE` 必须明确为 `baseline` 或 `conditional`。旧 `train_can_ph.sh`、`train_can_mh.sh` 等入口仍转发至同一训练器，默认均为 baseline；PH 策略可作为额外实验，但不属于表 2 的默认策略任务。

条件策略按作者公开主分支逐样本选择动作提示：

| 动作提示 | 概率 | 监督目标 |
| --- | --- | --- |
| base：归一化空间的零动作 | 45% | 当前样本自己的GT动作 |
| 当前样本自己的GT动作 | 55% | 当前样本自己的GT动作 |
| shuffled GT | 0%（关闭） | — |

训练器、train.sh及实验矩阵均采用45%base/55%自身GT/0%shuffle；canonical来源为configs/paper_can.json，协议reach_can_pi05_full_sft_physical7_author_mix45_v4。这是对齐作者diffusion_unet_image_policy.py的公开主分支，不声称唯一确认了论文最终Can运行配置。监督目标始终是自身动作。

历史30%base/30%shuffle/40%自身GT保留为额外变体；如单独复测需显式同时设置BASE_PROB=0.3和GT_SHUFFLE_PROB=0.3，不能在resume中改变概率。baseline不使用提示；为兼容已有checkpoint严格恢复，其未使用base_prob参数仍保留旧值0.3。旧执行manifest未覆盖，新计划另存outputs/pi05_full_sft_physical7_v4/author_prompt45_experiment_manifest.json。40项检查通过，并按真实baseline config验证恢复参数兼容。

保留 Pi0.5 单帧观测、50步action chunk，Can的7维相对OSC动作补零到32维；末尾重复最后动作，不跨episode，不再次做joint delta变换。训练全部参与前向的VLM/视觉/动作专家参数，只排除未使用的expert LM head；按用户选择仅对前7维flow loss取均值，屏蔽后25维；validation及best-validation选择也使用前7维。9维真实状态先分位数归一化并token化，再把连续state张量补零至32维；动作采用训练集q01/q99统计，推理复用对应checkpoint的统计。双相机映射为base/left_wrist，第三路屏蔽。Conditional才启用线性动作提示投影；baseline不含动作提示。论文补充材料描述将动作提示嵌入视觉特征空间并扩充条件特征，没有公开完整编码器实现；此前把 Fourier 编码写成论文明确要求的说法依据不足，现已更正。

### 32维模型接口与7维监督

Pi0.5基础模型的动作接口为32维。Can的干净目标是`a=[7维相对OSC动作, 25维零]`；训练对全部32维加入噪声，构造`x_t=t*noise+(1-t)*a`，速度目标为`u_t=noise-a`。所以后25维的干净动作虽为0，其速度目标仍为对应噪声；官方实现对全部32维的flow误差取平均。它们不是新关节，也不参与实际控制，执行时只反归一化并取前7维。

v3曾保留官方完整32维监督。按用户后续明确选择，当前v4改为`losses[..., :7].mean()`：训练、验证和best-validation均只看物理7维，另记录`full_action_loss`和`padded_action_loss`作为诊断。模型输入输出仍32维，噪声与flow计算不变；这是一项主动选择的目标适配，不能声称与官方完整32维loss相同。轨迹末尾仍重复动作，本轮未加入时间步mask。

## 错误检测器

`train_error_detector.py` 独立训练图像自编码器，输入双相机归一化到 `[0,1]` 的图像，最小化逐像素 MSE。不使用 OOD 或成功/失败标签作为监督。

默认网络为双视图拼接后展平 → 512 → 128 → 512 → 图像重建，输出 sigmoid；当前使用 Adam、lr=1e-4 与 cosine。**论文未公开 AE 的精确层宽及多相机融合方式，这些是明确记录的实现选择。**

检测器最初采用 SGD、lr=1e-4，但逐像素平均 MSE 下首步相对参数更新仅约 4e-9，训练 48 轮仍接近未训练的误差。固定训练 / 留出面板对照中，Adam 在 120 步后将 validation MSE 从 0.086475 降到 0.021313，低于训练集平均图像基线 0.032471；提高 SGD 学习率的对照仍接近该平均图像基线。因此检测器改为 Adam，并从初始权重重新训练。**这是相对原 SGD 复现配置的额外优化器变更，不声称与论文 AE 训练完全一致。** 当时未改策略的AdamW、学习率和数据；后续策略SFT v3修复及新配方见上文。旧检测器结果已归档。详细对照见验证记录。

训练结束后，在留出的同分布 validation 图像上标定 `threshold = mean + kappa * std`，默认 `kappa=3`；这些标定细节也不是作者公开的固定参数。支持补充材料提到的 conformal 标定，使用 `--threshold-method conformal --alpha 0.05`。

```bash
PYTHON=../RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python
"$PYTHON" scripts/train_error_detector.py \
  --data data/paper/can_ph_image.hdf5 --dataset-type ph \
  --output outputs/ae_ph_manual --epochs 50 --batch-size 128
```

保存 `last.pt`（按 epoch 恢复）、`detector.pt`、`calibration.json`、切分记录及 `optimization_diagnostics.json`。后者记录初始 / 当前 validation MSE 和相对下降率，用于检查是否学会重建；重建误差下降本身不证明错误检测有效。`error_detector.ErrorDetector` 提供重建误差及 `is_ood` 判定接口；输入必须为原始 uint8 或 `[0,1]` 图像，不能直接输入 Pi0.5 的 `[-1,1]` 图像。

## 权重与恢复

基础 JAX 权重可用 `bash scripts/convert_checkpoint.sh` 转换。训练器支持 safetensors 和本项目 `.pt`，完整 checkpoint 保存优化器、scheduler、数据切分、归一化、各 rank RNG 与采样位置。

当前策略及检测器都用 PyTorch 训练。基础权重的 811 个实际使用张量已逐一与原始 JAX 权重完成映射、保存 dtype 转换后的值核对，无不匹配，见 [核验报告](logs/pi05_conversion_weight_audit_20261031.json)。这验证了权重内容，尚不代表已完成 JAX / PyTorch 同输入的端到端数值对齐；已有采样一致性检查是在 PyTorch 内进行。

新版任务从现有共同base `checkpoints/pi05_base_pytorch/model.safetensors` 开始；中断后继续同一新版任务使用 `RESUME`，并保持原来的模式、epoch/step 目标、数据、batch、学习率、种子和 GPU 数量。旧的冻结VLM实验不能用`RESUME`改成全参SFT；恢复检查会拒绝旧训练协议。旧权重仍可评测，自动保留其min/max及32维状态token化约定。新版输出默认在`outputs/pi05_full_sft_physical7_v4/`，不覆盖`outputs/paper_can/`。

base与shuffle概率均是严格恢复检查的一部分。当前conditional默认45%base、0%shuffle；恢复已有同协议任务必须使用checkpoint保存的概率，不能在--resume时改动。已有v4 baseline的原参数保持可恢复。

## 动作条件监控与 steering 诊断

策略训练默认增加固定留出样本监控：初始化、每次 validation 和最后一步运行，写入输出目录的 `branch_monitor.jsonl` 及 TensorBoard 的 `monitor/*`。默认固定 4 个样本，使用独立种子的同一份噪声、分层时间点，关闭图像增强。支持 `--monitor-samples N`、`--monitor-seed 2026`；`--monitor-samples 0` 关闭。监控保留并恢复随机状态和模型模式，不参与反向传播，不改变原有训练 loss、条件概率或完整 validation。

条件策略分别测 base（通用动作）、GT（当前样本真实动作）与 foreign（循环移位后的另一条样本动作）；baseline 仅测自身。单样本时跳过 foreign。记录 flow MSE、速度方向余弦、相对速度误差、局部终点估计误差，以及 GT/base 预测速度差异和 GT loss 收益。所有指标只统计前 7 个物理动作通道。`endpoint_estimate_mse` 是单个 flow 时间点的局部估计，不能当作完整采样误差。初始化时条件投影为零，GT/base 指标相同是预期现象；训练后长期完全相同，才需要排查条件分支是否生效。

完成条件策略训练后，可运行独立的固定噪声采样诊断：

```bash
PYTHON=../RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python
"$PYTHON" scripts/diagnose_steering.py \
  --checkpoint outputs/pi05_full_sft_physical7_v4/policies/can_paired/conditional/seed_0/final_model.pt \
  --output outputs/diagnostics/paired_conditional_seed_0.json \
  --samples 4 --seed 2026 --num-steps 10 --scales 0 0.2 2.5 4.5
```

诊断恢复 checkpoint 的训练归一化、prompt 与留出划分，同一组观测和初始噪声贯穿所有比较：直接给 GT/foreign/self 提示能否改变输出、向 GT 正向引导及从 GT/self 负向引导时动作距离如何变化。报告记录样本 ID、噪声种子、时间点、checkpoint step 与归一化摘要，距离均为归一化空间的动作 MSE。`follow_gain > 0` 表示比 base 更靠近该提示；它不是任务成功率。foreign 样本索引不同，但动作数值可能相似。可更换 `--seed`、扩大 `--samples` 复核，默认小面板用于排查实现问题。

脚本强制检查自定义 Euler 与上游采样器一致，且 `S=0` 与 base 一致；误差超过 `--tolerance`（默认 `1e-4`）或产生非有限值则报错退出。正向使用 `v_base + S * (v_target - v_base)`，负向使用 `v_base + S * (v_base - v_target)`，从 flow 时间 1 积分到 0。非线性模型的动作距离不保证随 S 单调，不用单调性判定实现通过与否。

这两项用于确认条件分支和引导计算是否有效，帮助定位“loss 下降但提示被忽略”的问题。它们不增加训练监督，也不替代机器人 rollout、错误检测触发、失败动作历史、回退恢复与成功率评估；默认实验矩阵仍保持不变。

## 检查与使用边界

```bash
PYTHON=../RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python
JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 "$PYTHON" -m unittest discover -s tests -v
JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 "$PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=2 tests/ddp_smoke.py
JAX_PLATFORMS=cpu OMP_NUM_THREADS=2 "$PYTHON" -m torch.distributed.run \
  --standalone --nproc-per-node=2 tests/ddp_diagnostics_smoke.py
```

训练协议和未公开参数见 `configs/paper_can.json`；评测协议见 `configs/evaluation_can.json`。训练清单中的评估条目为 `pending_training_or_evaluation`，不代表已经得到成功率。论文表注与正文对“三种子最高成功率”的表述有差异，评测器保留逐 checkpoint、逐种子的原始结果，并同时报告各聚合方式。

## 真实仿真评测

评测直接复用现有两个环境，不改动共享依赖：模型使用 OpenPI 的 Python 3.11 / PyTorch 2.10，仿真使用 `reach_simulation_env/venv` 的 Python 3.10 / Robosuite 1.4.1 / MuJoCo 3.1.0，通过独立进程通信。任务必须从原数据元信息创建 **PickPlaceCan**，不能用 Lift 代替。两路图像、状态顺序、训练集归一化和动作反归一化均与训练一致；双相机与原始状态重放后的训练图像逐像素一致性检查已通过。

先检查环境；共同初始场景已生成，重复评测复用该文件，不覆盖它：

```bash
PYTHON=../RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python
JAX_PLATFORMS=cpu "$PYTHON" scripts/check_evaluation.py

# 首次部署到新目录才需要生成。文件存在时工具拒绝覆盖。
# JAX_PLATFORMS=cpu "$PYTHON" scripts/eval_catalog.py \
#   --metadata data/robomimic/can/paired/low_dim_v141.hdf5 \
#   --output outputs/evaluation/can_initial_states_50.json --count 50 --seed-start 100000
```

公开设定与本地配置的对应关系：

| 项目 | 当前评测 |
| --- | --- |
| 三种子、检测器、引导强度 | seed 0/1/2；Paired 搭配 PH/MH/Paired；S=0/0.2/2.5/4.5，按主文表 2 |
| 每次执行动作数 | 8，按补充表 5；Pi0.5 预测 horizon 仍为训练时的 50 |
| 测试回合与初始种子 | 50 回合，100000–100049；按论文引用的 Robomimic / Diffusion Policy 评测配置 |
| 物理步数预算 | Paired 400，MH 500；回退步数也计入统一预算 |
| 成功判定 | 原任务 `_check_success()`；首次成功结束，未成功则计至预算耗尽 |
| 错误检测 | 每个 8 步前向 chunk 后计算 AE 误差，使用 checkpoint 固定标定阈值 |
| 回退与重规划 | 回退最近执行 chunk 的 EEF 位姿；默认缓存该次调用完整、裁剪前的 50×7 `action_pred` 作为负向提示，另可切换为末次实际执行的 7D 动作重复 50 步；取消原计划未执行部分 |
| 相对动作回退 | SE(3) 逆变换转换为 OSC 的世界平移 / 左乘旋转控制，真实执行伺服回退；不恢复 simulator state，不瞬移物体 |
| S=0 对照 | 仍做错误触发、物理回退及新噪声重采样，只将引导强度置零；不同于无检测器对照 |
| checkpoint 与种子汇总 | 扫描所有已保存 checkpoint 和 final；保留 final、各 checkpoint、每种子最大值；同时输出表注和正文两种聚合 |

已有 rollout trace 可在不重跑策略的情况下重放成双相机视频：

```bash
PYTHON=../RoboTwin/XPolicyLab/policy/Pi_05/openpi/.venv/bin/python
"$PYTHON" scripts/render_evaluation_traces.py \
  --shard-dir outputs/evaluation/<run>/<condition>/shards/<shard> \
  --catalog outputs/evaluation/can_initial_states_50.json \
  --output-dir outputs/evaluation/<run>/<condition>/videos
```

50 回合、初始种子、400/500 步的参考来源是 [Diffusion Policy 官方 Can 配置](https://raw.githubusercontent.com/real-stanford/diffusion_policy/main/diffusion_policy/config/task/can_image.yaml) 和 [Robomimic 原论文配置生成器](https://raw.githubusercontent.com/ARISE-Initiative/robomimic/v0.3.0/robomimic/scripts/generate_paper_configs.py)。REACH 没有逐项重述回合数及完整 checkpoint 筛选过程；未将本地选择冒充为作者公开参数。

**精确复现边界：** `configs/evaluation_can.json` 逐项记录未公开选择，包括回退伺服最多 10 步、5mm / 0.05rad 容差、夹爪保持前一执行指令、固定 held-out 阈值以及短失败历史补齐到 Pi0.5 horizon。论文没有公开这些参数的可执行实现。现有Pi0.5提示融合、AE网络/Adam也属于适配；旧30%shuffle结果另作历史变体记录，因此结果明确标记 `exact_paper_reproduction=false`。论文表 2 的 best/average 及正文最大值描述不完全一致，工具保留所有原始分数，不自动将某一种 checkpoint 平均方式解释为作者唯一口径。

以下是新版模型训练完成后的评测命令；新版Paired baseline seed0已于2026-11-01 15:59:35 UTC后台启动，尚未进入评测。默认策略目录为`outputs/pi05_full_sft_physical7_v4`，检测器通过`--detector-artifacts`默认复用`outputs/paper_can`下已训练的独立AE。旧版seed0已经完成，但不能当成新版全参SFT结果：

```bash
# 每个checkpoint/configuration为50回合；all还包含best_validation.pt。
bash scripts/evaluate_can.sh plan --datasets paired --seeds 0 --checkpoints all \
  --output outputs/evaluation/full_sft_physical7_v4_paired_seed0_all_checkpoints

# 8 卡独立评测，每卡一个模型任务，无需 DDP；检测器未完成时最多等待两小时。
bash scripts/evaluate_can.sh run --datasets paired --seeds 0 --checkpoints all \
  --output outputs/evaluation/full_sft_physical7_v4_paired_seed0_all_checkpoints \
  --gpus 0 1 2 3 4 5 6 7 --wait-seconds 7200

# 汇总，不重新 rollout。
bash scripts/evaluate_can.sh summarize --output outputs/evaluation/full_sft_physical7_v4_paired_seed0_all_checkpoints
```

若先检查最终权重，可改为 `--checkpoints final` 并使用另一个输出目录，例如 `outputs/evaluation/full_sft_physical7_v4_paired_seed0_final`，共 14 个单元 / 700 回合；这是 final-only 对照，不能替代全 checkpoint 最高成功率统计。旧版14份权重的seed0全扫描为9800回合；新版batch64/50epochs若包含27份定期权重、best_validation与final，则为20300回合，三种子60900回合；实际耗时取决于成功提前结束、异常触发次数和渲染吞吐，不用单回合 smoke 外推准确总时间。

需要后台运行时，将上面的 `run` 命令通过 `nohup ... > outputs/evaluation/driver.log 2>&1 &` 启动。重启同一命令会检查清单和权重内容、恢复未完成 episode，并跳过有匹配指纹的已完成单元。更换协议、权重或实现后应使用新输出目录；不能混合旧结果。当前环境只暴露一个 EGL 图形设备，仿真子进程固定使用 EGL 0，各模型仍分配到不同 CUDA 卡；图形设备和计算设备的选择已隔离。

单次评测、视频与诊断示例：

```bash
# 一回合、完整 400 步预算的诊断；不是正式 50 回合统计。
JAX_PLATFORMS=cpu "$PYTHON" scripts/evaluate_can.py \
  --checkpoint outputs/pi05_full_sft_physical7_v4/policies/can_paired/conditional/seed_0/final_model.pt \
  --detector outputs/paper_can/detectors/can_paired/seed_0/detector.pt \
  --catalog outputs/evaluation/can_initial_states_50.json \
  --output outputs/evaluation/manual_one_episode --rollouts 1 --scale 2.5 --video
```

输出包括 `run.json`（权重 SHA256、归一化摘要、协议、代码摘要、执行设备）、`episodes.jsonl`（逐回合结果）、逐步 action/score/retreat trace、可选双视图 MP4 和 `summary.json`。矩阵还保存 `matrix.json`、`status.json`、单元 `job_record.json` 和 `aggregate.json`。汇总保留各 checkpoint 分数、final、每种子最大值、总体 / 样本标准差；种子不足或 checkpoint 未评完时，不输出伪完整的三种子聚合。`--threshold-override`、缩短预算或减少回合会标为 smoke，排除在正式汇总外。

已验证：30 项回归检查覆盖归一化、真实动作历史、回退步数预算、S=0、三种子 / checkpoint 聚合、并行分卡和断点复用；真实图像重放检查、400 步预算的 baseline / conditional 单回合以及强制恢复路径均通过。详细原始证据见验证记录。

参考：[REACH 主文](https://openaccess.thecvf.com/content/CVPR2026/papers/Ke_REACH_Explicit_Recovery_Behavior_for_Diffusion_Policies_CVPR_2026_paper.pdf)、[补充材料](https://openaccess.thecvf.com/content/CVPR2026/supplemental/Ke_REACH_Explicit_Recovery_CVPR_2026_supplemental.pdf)、[Robomimic v0.3 数据说明](https://robomimic.github.io/docs/v0.3/datasets/robomimic_v0.1.html)。

作者公开仓库已固定在提交 `372552d6` 并完成源码对照。当前实现存在提示配比、提示编码、数据切分/归一化、检测频率、负提示来源和物理回退等额外差异；公开快照也含失配配置及调试设置，不能把其所有默认值视为论文最终参数。详细证据见 [作者源码对照](PAPER_REPRODUCTION_VALIDATION.md#作者公开源码对照2026-11-01372552d) 和 [结构化审计](outputs/source_comparison/reach_372552d/audit.json)。

普通 Pi0.5 SFT 审计已确认：当前旧训练冻结整个 VLM，且存在state补零后token化、仅前7维loss等标准实现偏离。它属于冻结VLM的动作专家微调变体，不能当作全参数SFT基线。详见 [全参SFT审计](PAPER_REPRODUCTION_VALIDATION.md#普通-pi05-全参数-sft-审计2026-11-01)；该审计问题已在full-SFT v3修正，旧记录保留作为历史证据。

## 全参数 SFT physical7 v4 验证与训练

已于2026-11-01 15:59:35 UTC后台启动Paired baseline seed0，使用GPU0–7、每卡micro8、累积1、global64；50epochs共13850步。使用现有转换base，VLM/动作专家全参训练；仅前7维监督，后25维输出不计入loss。当前只启动baseline，conditional及MH未排队。

[启动记录](outputs/pi05_full_sft_physical7_v4/launch.json)、[后台状态](outputs/pi05_full_sft_physical7_v4/status.json)、[训练日志](outputs/pi05_full_sft_physical7_v4/policies/can_paired/baseline/seed_0/train.log)。不要重复执行启动命令。任务已脱离会话，聊天中断不会停止训练。

策略目录：`outputs/pi05_full_sft_physical7_v4/policies/can_paired/baseline/seed_0/`。每500步保存`checkpoints/step_*.pt`，7维验证损失改善时保存`best_validation.pt`，结束保存`final_model.pt`；均包含模型、optimizer、scheduler、RNG、训练位置和归一化统计。旧v2/v3协议不能resume成v4；历史文件保留。

37项单元/集成检查通过。[真实8卡更新报告](outputs/validation/full_sft_physical7_v4/gpu_smoke_world8_micro8_accum1.json)确认micro8/累积1/global64、完整VLM更新与8个rank同步；后25维输出梯度为0，峰值54.28GiB。未修改模型原始forward，同输入对照官方基类完整loss差为0；训练聚合仅取7维是本轮显式变化。[测试日志](logs/full_sft_physical7_v4_unit_tests.log)。

### Kai0 baseline的padding mask核对

已读`/2025233210/lmx/kai0/src/openpi/training/config.py`的pi05_baseline及后续损失链路。它配置32维flow输出，`_flow_mse`对最后一个维度直接mean，**没有屏蔽补齐动作维度**。另有`action_is_pad -> action_timestep_mask[B,H]`，默认开启，数据提供标记时屏蔽轨迹结束后补出的无效时间步；缺少标记时全有效。这是时间轴mask，与动作维度mask不同。提取其原函数CPU检查，首维和第32维的同量误差在有效步上贡献相同，只有无效时间步被屏蔽。

上轮只读检查Kai0；本轮按用户选择，**本项目已加入前7维监督的动作维度mask**，未加入时间步mask，轨迹末尾仍重复补齐。之前统计24.8%的末尾重复目标是时间轴填充，不是25个额外动作通道。证据：[Kai0 mask检查](outputs/validation/full_sft_v3/kai0_padding_mask_review.json)。
