# Can任务评测结果总览

## 范围与读表

更新至2026-11-04：本文按实验问题分组，97条完整评测记录及来源保留在文末折叠清单。记录包含独立重测，不等同于97个不同模型；每条记录完整覆盖50局，不把GPU分片或引用的对照组重复计为新增评测。未完成任务不计入成功率。本次新增MH baseline 2k的1条评测和MH conditional的9条评测，共500局；conditional的450局已于北京时间2026-11-04 08:53完成并通过审计。

`Paired/PH/MH策略`指策略训练数据；`PH/MH/Paired detector`指检测器训练数据，二者必须分别阅读。例如“Paired策略 + MH detector”仍然是Paired策略，不是MH策略。step是optimizer更新数，S是negative guidance scale。结果均为单训练seed，50局中1局对应2个百分点（下文记pp）；“—”表示没有对应评测。

当前协议为每次预测50步、执行8步再推理，10步去噪；Paired/PH每局预算400个物理步，MH为500步。平均步数包含回退。启用detector时，S=0仍然检测、回退并重规划，只关闭负向引导。当前恢复为整块EEF伺服回退，negative prompt使用完整预测计划；早期实现、25步执行和旧阈值结果单列归档。后续detector统一κ=4，历史κ=3结果不改写为κ=4。

## 1. Conditional与baseline：增益来自训练，还是推理时的MLP？

下表均不使用detector或恢复。MLP关闭是对同一个conditional checkpoint的推理消融，不是重新训练一个baseline。

| 策略训练数据，2k checkpoint | 独立训练baseline | Conditional，MLP开 | 同一conditional，MLP关 | Conditional开 vs baseline | MLP开 vs 关 |
| --- | --- | --- | --- | --- | --- |
| Paired | 22/50（44%） | 35/50（70%） | 32/50（64%） | +26pp | +6pp |
| PH | 49/50（98%） | 49/50（98%） | 50/50（100%） | 0pp | −2pp |
| MH | 47/50（94%） | 49/50（98%） | 50/50（100%） | +4pp | −2pp |

**结论：Paired上conditional的26pp增益不能全部归因于部署时的MLP。** 关掉MLP后仍比独立baseline高20pp；这部分包含条件训练对共享策略参数和优化路径的影响，当前实验不能再做因果拆分。PH上conditional没有提高2k成功率，MLP推理消融的方向也不同，因此不能概括为“加MLP一定提高成功率”。Paired conditional相对baseline逐场景救回17局、丢失4局；该比较包含训练差异。

MH conditional相对独立baseline救回episode 14、33、37，丢失28，净增2局（+4pp）；同一conditional关闭MLP后救回28、没有丢失，达到50/50。与PH方向一致，本次MH推理MLP未显示额外成功率收益；关MLP仍保留conditional训练后的共享参数，不等同于独立baseline。这些单seed、50场景结果不足以证明稳定优势。

Paired两模型在2k之前使用相同的13850步学习率时间尺度；PH与MH的对应baseline/conditional使用2000步时间尺度、1000步warmup。PH/MH和Paired之间同时改变了数据和训练计划，不能用98%对70%单独衡量数据质量。

来源：[Paired baseline](../outputs/evaluation/pi05_paired_baseline_seed0_step2000/summary.json)、[Paired MLP消融](../outputs/evaluation/paired_2k_paired_detector_mlp_ablation_sealab1_20261103_193538_utc/REPORT.md)、[PH baseline](../outputs/evaluation/pi05_full_sft_ph_baseline_seed0_2k_final/summary.json)、[PH conditional](../outputs/evaluation/ph_conditional_2k_8gpu_20261103_184946_utc/REPORT.md)。

MH来源：[独立baseline](../outputs/evaluation/mh_baseline_2k_8gpu_sealab1_20261103_204352_utc/summary.json)、[conditional及MLP消融](../outputs/evaluation/mh_conditional_2k_8gpu_20261103/REPORT.md)、[逐场景对照](../outputs/evaluation/mh_conditional_2k_8gpu_20261103/results.json)。

## 2. 训练更久是否更好？

### Paired：无detector，标准8步执行

| 训练步数 | Baseline | Conditional MLP | Conditional相对baseline |
| --- | --- | --- | --- |
| 1k | 16/50（32%） | 31/50（62%） | +30pp |
| 2k | 22/50（44%） | 35/50（70%） | +26pp |
| 4k | — | 32/50（64%） | — |
| 6k | — | 28/50（56%） | — |
| 8k | 32/50（64%） | 29/50（58%） | −6pp |
| 10k | 24/50（48%） | 30/50（60%） | +12pp |

**结论：成功率随训练步数不单调。** 已测Paired MLP checkpoint中2k最高；baseline已测候选中8k最高且独立重测复现。不能从“2k conditional优于2k baseline”推广到所有步数。这里是已测checkpoint的观察，不是独立测试集上重新验证的最优checkpoint结论。

来源：[1k/2k/4k/6k conditional](../outputs/evaluation/conditional_mlp_1k_2k_4k_6k_baseline_8gpu_20261103/)、[8k/10k conditional](../outputs/evaluation/conditional_mlp_8k_10k_ph_suite_20261103/)、[8k baseline](../outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/summary.json)；其余baseline链接见文末。

### PH：无detector，标准8步执行

| 训练步数 | Baseline | Conditional MLP开 | Conditional相对baseline |
| --- | --- | --- | --- |
| 500 | 36/50（72%） | 49/50（98%） | +26pp |
| 1k | 46/50（92%） | 44/50（88%） | −4pp |
| 2k | 49/50（98%） | 49/50（98%） | 0pp |

**结论：PH上的conditional优势只在当前500步checkpoint明显，2k最终成功数相同。** baseline的best-validation文件实际保存于500步，闭环成功率72%，低于最终2k的98%；validation loss不能直接代替环境成功率。2k关闭MLP为100%只是本次50局结果。

来源：[PH baseline 500步/best-validation](../outputs/evaluation/pi05_full_sft_ph_baseline_seed0_2k_best_validation/summary.json)、[PH baseline 1k](../outputs/evaluation/pi05_full_sft_ph_baseline_seed0_2k_step1000/summary.json)、[PH conditional报告](../outputs/evaluation/ph_conditional_2k_8gpu_20261103_184946_utc/REPORT.md)。

### MH：无detector，标准8步执行，500步物理预算

| 训练步数 | Baseline | Conditional MLP开 | Conditional平均物理步数 | Conditional相对baseline |
| --- | --- | --- | ---: | --- |
| 500 | — | 45/50（90%） | 214.34 | — |
| 1k | — | 44/50（88%） | 228.88 | — |
| 1.5k | — | 46/50（92%） | 210.58 | — |
| 2k | 47/50（94%） | 49/50（98%） | 182.92 | +4pp |

**结论：本轮MH conditional从500到2k总体改善，但也不单调。** 预先指定的2k主结果成功数最高，同时平均物理步数最低；2k唯一失败为episode 28，耗尽500步预算。独立baseline只评测了最终2k，不填充其较早checkpoint结果。MH预算500步，不能将其平均步数或成功率与400步预算的PH/Paired直接视为相同协议比较。

来源：[MH conditional训练曲线评测](../outputs/evaluation/mh_conditional_2k_8gpu_20261103/wave1_policy/results.json)、[MH baseline](../outputs/evaluation/mh_baseline_2k_8gpu_sealab1_20261103_204352_utc/summary.json)。

## 3. 同一Paired conditional 2k：MLP与S=0恢复的作用

固定同一模型、同一50场景；detector均κ=4。该表只改变推理MLP开关和是否启用S=0恢复。

| Detector / 恢复 | MLP开 | MLP关 | MLP开启差值 | 相对各自无恢复 |
| --- | --- | --- | --- | --- |
| 无 | 35/50（70%） | 32/50（64%） | +6pp | 参照 |
| Paired，S=0 | 34/50（68%） | 31/50（62%） | +6pp | 开/关均−2pp |
| MH，S=0 | 34/50（68%） | 31/50（62%） | +6pp | 开/关均−2pp |

**结论：这两种detector的S=0恢复均没有救回原本失败的场景，只额外丢失episode 14。** Paired与MH的逐局成功标签完全一致；MLP开关的净差值也一致。MLP关相对开救回episode 38，但丢失16、24、42、45，净少3局。该结果不代表“Paired策略 + MH detector”的非零引导效果，因为该组合尚未测对应S>0组；下文新增的是MH策略自身的非零引导评测。

来源：[Paired结果及逐局比较](../outputs/evaluation/paired_2k_paired_detector_mlp_ablation_sealab1_20261103_193538_utc/results.json)、[MH结果及逐局比较](../outputs/evaluation/paired_2k_mh_detector_mlp_ablation_sealab1_20261103_201552_utc/results.json)。

## 4. PH与MH conditional 2k：恢复和引导是否有额外收益？

### PH策略 + PH detector

固定PH conditional 2k、MLP开启；恢复组使用PH detector κ=4。

| 配置 | 成功率 | 平均物理步数 | 平均恢复次数 | 相对同一模型无恢复 |
| --- | --- | --- | --- | --- |
| 无detector | 49/50（98%） | 109.06 | 0 | 参照 |
| S=0 | 48/50（96%） | 111.32 | 0.20 | −2pp |
| S=0.2 | 48/50（96%） | 112.00 | 0.22 | −2pp |
| S=2.5 | 48/50（96%） | 110.68 | 0.18 | −2pp |
| S=4.5 | 48/50（96%） | 111.32 | 0.20 | −2pp |

**结论：四个S的成功标签完全相同，没有看到负向引导比S=0额外救回场景。** 同一conditional无恢复仅失败episode 38；恢复后失败14、38，因此是救回0局、额外丢失1局。

**更正旧摘要的参照组混淆：**“救回35、丢失14和38”是恢复组与另一个独立训练的PH baseline比较的结果。PH baseline无恢复失败35，PH conditional无恢复失败38；用两个不同模型作参照，不能将差异归因于恢复。

来源：[PH suite](../outputs/evaluation/ph_conditional_2k_8gpu_20261103_184946_utc/REPORT.md)、[同一conditional无恢复逐局结果](../outputs/evaluation/ph_conditional_2k_8gpu_20261103_184946_utc/wave1_policy_and_ablation/mlp_step002000_none_S0/episodes.jsonl)、[S=0逐局结果](../outputs/evaluation/ph_conditional_2k_8gpu_20261103_184946_utc/wave2_recovery/mlp_step002000_ph_S0/episodes.jsonl)。

### MH策略 + MH detector

固定MH conditional 2k、MLP开启，MH detector κ=4，完整50×7预测作为negative prompt；每局预算500个物理步，包含回退。

| 配置 | 成功率 | 平均物理步数 | 平均恢复次数 | 检测触发总数 | Guided预测总数 | 回退失败局数 | 相对同一模型无恢复 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 无detector | 49/50（98%） | 182.92 | 0 | 0 | 0 | 0 | 参照 |
| S=0 | 49/50（98%） | 180.90 | 0.06 | 3 | 0 | 1 | 0pp；救回0/丢失0 |
| S=0.2 | 49/50（98%） | 180.88 | 0.06 | 3 | 2 | 1 | 0pp；救回0/丢失0 |
| S=2.5 | 49/50（98%） | 180.88 | 0.06 | 3 | 2 | 1 | 0pp；救回0/丢失0 |
| S=4.5 | 49/50（98%） | 180.86 | 0.06 | 3 | 2 | 1 | 0pp；救回0/丢失0 |

**结论：MH上恢复和负向引导均没有增加成功场景。** 所有detector配置都只在episode 28触发3次：前两次完成整块回退，第3次回退失败；其余49局没有触发恢复。S=0在399步失败，S=0.2/2.5在398步失败，S=4.5在397步失败；无detector则在500步预算结束时失败。因此平均步数降低主要来自同一失败局提前终止，不能解读为任务完成效率提升。

非零S确实各执行了2次guided预测、16个guided物理动作，并非引导未运行；但引导只覆盖1个场景，当前结果不能区分各S的稳定优劣，也不能证明引导普遍无效。结合无恢复98%和旁路MLP100%的结果，本轮优先排查episode 28的action-condition分支及回退失败原因，而不是从相同成功率中选择一个“最佳S”。

来源：[MH汇总及逐局比较](../outputs/evaluation/mh_conditional_2k_8gpu_20261103/results.json)、[450局审计](../outputs/evaluation/mh_conditional_2k_8gpu_20261103/audit.json)、[MH恢复评测](../outputs/evaluation/mh_conditional_2k_8gpu_20261103/wave2_recovery/)。

## 5. Negative prompt和执行步数变体

Paired conditional 2k + Paired detector κ=4；以下A/B保持模型和整块回退相同，只改变negative prompt构造。

| S | Negative prompt | 成功率 | 平均物理步数 | 平均恢复次数 |
| --- | --- | --- | --- | --- |
| 2.5 | 完整预测action_pred | 35/50（70%） | 181.42 | 0.90 |
| 2.5 | 重复最后执行动作 | 35/50（70%） | 185.10 | 1.06 |
| 4.5 | 完整预测action_pred | 35/50（70%） | 180.36 | 0.94 |
| 4.5 | 重复最后执行动作 | 35/50（70%） | 184.04 | 1.04 |

**结论：完整预测prompt在这次A/B中略少用物理步和恢复，但没有成功率增益；每个S下两种prompt的逐局标签完全一致。** 当前默认保留完整预测，不据此宣称其成功率更高。

另外，同一Paired 2k权重无恢复时，8步执行为35/50（70%），25步执行为31/50（62%）。两轮checkpoint、归一化和catalog哈希一致，但实现快照不同；应作为执行频率变体记录，不能把它们当作相同协议的重复测量，也不据此宣称8步是普遍最优。

来源：[prompt A/B](../outputs/evaluation/conditional_mlp2k_paired_detector_prompt_ab_20261103/)、[25步执行](../outputs/evaluation/conditional_mlp2k_baseline_exec25_8gpu_20261103/summary.json)。

## 6. 与REACH原论文对比：是否复现了恢复收益？

原论文为[REACH主文](../references/reach_cvpr2026.pdf)第7页Table 2；训练参数见[补充材料](../references/reach_cvpr2026_supp.pdf)Table 5。以下论文数字均直接按该表抄录，百分比中的±保留原文形式，不擅自解释为本地单次评测的置信区间。

### Can(Paired)：固定策略数据，比较detector和S

论文无恢复Diffusion Policy为60.8±1.4%，Conditional DP为64.2±3.8%；本地Paired 2k分别为44%和70%。下表每格按“论文 / 本地”排列，本地均为Paired conditional MLP 2k、κ=4、完整chunk回退和完整预测prompt。

| S | PH detector：论文 / 本地 | MH detector：论文 / 本地 | Paired detector：论文 / 本地 |
| --- | --- | --- | --- |
| 0 | 74.1±2.8% / **72%** | 62.3±1.8% / **68%** | 64.1±2.8% / **68%** |
| 0.2 | 74.1±2.5% / **72%** | 65.0±2.5% / — | 67.5±2.5% / — |
| 2.5 | 84.1±1.4% / **72%** | 64.1±1.4% / — | 65.8±2.8% / **70%** |
| 4.5 | 85.0±0.0% / **72%** | 67.5±0.0% / — | 65.8±1.7% / **70%** |

**结论一：本地与论文都显示，已测Paired策略使用PH detector优于使用Paired/MH detector的S=0结果。** 本地为72%对68%；这是当前固定场景上的趋势，差2局，不是多seed稳定性证明。

**结论二：尚未复现论文中PH detector配合强负向引导的额外增益。** 论文PH detector从S=0的74.1%升至S=4.5的85.0%，增10.9pp；本地四个S均72%，增0pp。相对同一conditional无恢复，论文PH恢复S=0增9.9pp、S=4.5增20.8pp；本地均仅增2pp，逐局仅救回episode 27、没有丢失原成功场景。因此本地目前主要收益出现在conditional训练，尚不能声称复现了论文的负向引导收益。

本地Paired detector的S=2.5/4.5比S=0净多1局，但仍与无恢复的70%持平；不能仅凭相对S=0提高就说完整恢复系统优于无恢复。此处“Paired策略 + MH detector”的非零S仍为空缺，不作推断；新增的MH策略98%不填入Paired表。

来源：[本地PH detector四个S](../outputs/evaluation/conditional_mlp2k_ph_kappa4_chunk8_fullpred_8gpu_20261103/results.json)、[本地Paired detector非零S](../outputs/evaluation/conditional_mlp2k_paired_detector_prompt_ab_20261103/)、[Paired/MH的S=0](../outputs/evaluation/paired_2k_mh_detector_mlp_ablation_sealab1_20261103_201552_utc/REPORT.md)。

### Can(MH)：MH baseline与conditional已完成

| MH策略设置 | 论文Table 2原始记法 | 本地MH策略 |
| --- | --- | --- |
| Diffusion Policy / baseline | 1.00/0.96 | Pi0.5独立MH baseline 2k：47/50（94%） |
| Conditional DP，无恢复 | 1.00/0.95 | Pi0.5 MH conditional 2k、MLP开：49/50（98%） |
| Conditional，无恢复，推理MLP关 | — | 同一MH conditional 2k：50/50（100%），推理消融 |
| MH detector，S=0/0.2 | — | 两个S均49/50（98%） |
| PH或MH detector，S=2.5/4.5 | 1.00/0.98 | 仅完成MH detector：两个S均49/50（98%）；PH detector未测 |

论文Can(MH)使用斜杠双值，这里保留原始记法，不把左侧1.00换成本地“100%平均成功率”。Table 2表注写三种子各自最高成功率的平均，而§4.3正文另有三种子最大值的表述；本文保留这一统计口径不一致，不自行消解。**Paired策略 + MH detector的68%不能填入本表MH策略一栏；PH策略98%也不能拿来直接对标论文Can(MH)。**

**MH本地结果的收益发生在conditional训练比较，未显示恢复的额外收益。** 独立baseline到conditional由94%升至98%，再加MH detector和任一已测S仍为98%；同一失败场景始终未救回。不能因为恢复组98%与论文斜杠右值0.98相同，就认定恢复机制和统计结果已复现。旁路MLP100%是单seed、固定50场景的消融结果，也不是论文三seed统计量。

训练与自动评测设置见[MH baseline训练记录](MH_BASELINE_2K_SEALAB1_2026-11-04.md)、[MH conditional计划](MH_CONDITIONAL_2K_EVAL_PLAN_2026-11-03.md)；原始结果见[MH baseline报告](../outputs/evaluation/mh_baseline_2k_8gpu_sealab1_20261103_204352_utc/REPORT.md)、[MH conditional报告](../outputs/evaluation/mh_conditional_2k_8gpu_20261103/REPORT.md)。新增10条均已纳入97条完整记录。

### 数值比较的边界

| 项目 | 原论文/作者实现 | 当前本地 |
| --- | --- | --- |
| 策略骨干 | Diffusion Policy | 预训练Pi0.5，全参数SFT，仅物理7维loss |
| 观测 / 预测horizon / 执行步数 | 2帧 / 16 / 8 | 1帧 / 50 / 8 |
| 训练预算与batch | 50 epochs，batch128，lr=1e-4 | 以optimizer步数记录，global64，lr=2.5e-5；2k是checkpoint或固定短训 |
| 动作条件接入 | 动作prompt嵌入后与视觉条件特征拼接 | MLP投影加到去噪动作token |
| 检测与回退 | 策略调用处检测，缓存action_pred并使用反向动作块 | 策略调用处检测，缓存完整计划；记录EEF目标后伺服回退整个执行块 |
| 阈值与统计 | Gaussian阈值形式见主文；Can可复用的κ/精确评测seed未充分给出 | 冻结held-out阈值，κ=4；单训练seed、固定50场景 |

**比较目标是收益模式与机制，而不是把两种骨干的单次数值排成同口径排行榜。** 本地尚未完成三训练seed和统一checkpoint报告；本地回退也仍是适配实现。当前证据支持“Paired与MH的2k conditional优于各自当前baseline，PH两者持平；恢复增益有限，尚未观察到稳定的额外负向引导收益”，不支持“已完整复现REACH”或“本地70%已超越论文64.2%”这样的结论。

## 附录：完整评测记录与来源

为保留早期模型、旧κ、不同恢复版本和全部checkpoint结果，原始清单按原记录保留。PH恢复行的参照组说明已按逐episode记录更正；平均步数较小不自动代表更好，因为提前回退失败也会使回合更短。

<details>
<summary>展开97条完整配置评测记录（按模型及实现合并展示）</summary>


| 策略 / checkpoint | Detector / 推理配置 | 完成50局结果 | 来源与备注 |
| --- | --- | --- | --- |
| 早期Paired最终模型，冻结VLM baseline | 无detector，S=0 | 23/50（46%）；步数264.58；恢复0 | [首轮聚合](../outputs/evaluation/paired_seed0_final/aggregate.json)，早期实现 |
| 早期Paired最终模型，conditional | 无detector，S=0 | 13/50（26%）；步数323.44；恢复0 | [首轮聚合](../outputs/evaluation/paired_seed0_final/aggregate.json)，早期模型/训练配方 |
| 早期Paired conditional | PH detector，旧校准：S=0 13/50（26%），步数232.00，恢复29.46；S=0.2 12/50（24%），237.76，27.56；S=2.5 12/50（24%），235.70，27.38；S=4.5 13/50（26%），230.44，25.42 | 全部50局 | [首轮聚合](../outputs/evaluation/paired_seed0_final/aggregate.json) |
| 早期Paired conditional | MH detector，旧校准：S=0 13/50（26%），296.28，9.66；S=0.2 13/50（26%），296.26，8.98；S=2.5 13/50（26%），296.02，9.28；S=4.5 13/50（26%），296.14，7.08 | 全部50局 | [首轮聚合](../outputs/evaluation/paired_seed0_final/aggregate.json) |
| 早期Paired conditional | Paired detector，旧校准：S=0 13/50（26%），281.74，25.94；S=0.2 12/50（24%），287.40，22.76；S=2.5 13/50（26%），277.92，27.18；S=4.5 13/50（26%），280.98，22.88 | 全部50局 | [首轮聚合](../outputs/evaluation/paired_seed0_final/aggregate.json) |
| Paired baseline，v4训练checkpoint sweep | 无detector：1k 16/50（32%），308.02步；5k 13/50（26%），321.00；8k 32/50（64%），207.74；8k独立重测32/50（64%），207.74；8.5k 23/50（46%），261.50；10k 24/50（48%），255.22。恢复均为0 | 每个checkpoint 50局 | [1k](../outputs/evaluation/physical7_v4_paired_baseline_seed0_step001000_5gpu/summary.json)、[5k](../outputs/evaluation/physical7_v4_paired_baseline_seed0_step005000_5gpu/summary.json)、[8k](../outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/summary.json)、[8k重测](../outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu_retest_sealab3_20261102/summary.json)、[8.5k](../outputs/evaluation/physical7_v4_paired_baseline_seed0_step008500_5gpu_sealab3_20261102/summary.json)、[10k](../outputs/evaluation/physical7_v4_paired_baseline_seed0_step010000_5gpu/summary.json) |
| Paired conditional，v4线性条件模型8k | 无detector，S=0 | 25/50（50%）；步数249.34；恢复0 | [8k PH suite](../outputs/evaluation/physical7_v4_conditional8k_ph_suite_20261102/conditional_none_S0/summary.json) |
| Paired conditional，v4线性条件模型8k | PH detector，κ=3：S=0 25/50（50%），151.78步，17.06恢复；S=0.2 25/50（50%），154.46，18.02；S=2.5 26/50（52%），150.18，10.62；S=4.5 25/50（50%），150.44，9.12 | 每个S 50局 | [PH suite](../outputs/evaluation/physical7_v4_conditional8k_ph_suite_20261102/conditional_ph_S0/summary.json) |
| Paired conditional，v4线性条件模型8k | MH detector，旧校准：S=0 25/50（50%），194.88步，9.90恢复；S=0.2 25/50（50%），191.60，9.02；S=2.5 25/50（50%），192.74，5.20；S=4.5 25/50（50%），184.94，3.26 | 每个S 50局 | [MH suite](../outputs/evaluation/physical7_v4_conditional8k_mh_paired_suite_20261102/conditional_mh_S0/summary.json) |
| Paired conditional，v4线性条件模型8k | Paired detector，旧校准：S=0 25/50（50%），191.72步，16.62恢复；S=0.2 25/50（50%），193.78，16.28；S=2.5 24/50（48%），201.82，10.70；S=4.5 24/50（48%），167.80，5.86 | 每个S 50局 | [Paired suite](../outputs/evaluation/physical7_v4_conditional8k_mh_paired_suite_20261102/conditional_paired_S0/summary.json) |
| PH baseline checkpoint评测 | 无detector：step1k 46/50（92%），136.54步；best-validation 36/50（72%），202.18；final 49/50（98%），112.68；另一step1.5k结果46/50（92%），135.22。恢复均为0 | 每个checkpoint 50局 | [1k](../outputs/evaluation/pi05_full_sft_ph_baseline_seed0_2k_step1000/summary.json)、[best validation](../outputs/evaluation/pi05_full_sft_ph_baseline_seed0_2k_best_validation/summary.json)、[final](../outputs/evaluation/pi05_full_sft_ph_baseline_seed0_2k_final/summary.json)、[1.5k](../outputs/evaluation/pi05_ph_baseline_seed0_step1500/summary.json)。不同checkpoint/训练记录，不视为同一权重的重复测量 |
| Paired baseline step2k | 无detector，S=0 | 22/50（44%）；步数271.28；恢复0 | [summary](../outputs/evaluation/pi05_paired_baseline_seed0_step2000/summary.json) |
| Paired conditional MLP checkpoint sweep | 无detector：1k 31/50（62%），220.96步；2k 35/50（70%），191.82；4k 32/50（64%），208.70；6k 28/50（56%），233.98。恢复均为0 | 每个checkpoint 50局 | [checkpoint sweep](../outputs/evaluation/conditional_mlp_1k_2k_4k_6k_baseline_8gpu_20261103/step_002000/summary.json) |
| Paired conditional MLP step2k | 无detector，执行25步后再推理 | 31/50（62%）；步数221.62；恢复0 | [summary](../outputs/evaluation/conditional_mlp2k_baseline_exec25_8gpu_20261103/summary.json)。这是执行步数变体，不与标准执行8步的结果直接比较 |
| Paired conditional MLP step8k | 无detector：29/50（58%），226.66步，恢复0；PH detector κ=3：S=0 28/50（56%），178.38步，25.46恢复；S=0.2 28/50（56%），175.18，25.38；S=2.5 29/50（58%），164.98，11.54；S=4.5 28/50（56%），163.84，9.60 | 每个配置50局 | [8k/10k suite](../outputs/evaluation/conditional_mlp_8k_10k_ph_suite_20261103/mlp_step008000_none_S0/summary.json)，旧PH detector校准 |
| Paired conditional MLP step10k | 无detector：30/50（60%），220.56步，恢复0；PH detector κ=3：S=0 29/50（58%），182.82步，28.54恢复；S=0.2 29/50（58%），182.82，28.56；S=2.5 28/50（56%），172.28，12.64；S=4.5 29/50（58%），163.12，10.00 | 每个配置50局 | [8k/10k suite](../outputs/evaluation/conditional_mlp_8k_10k_ph_suite_20261103/mlp_step010000_none_S0/summary.json)，旧PH detector校准 |
| Paired conditional MLP step10k | PH detector，κ=4：S=0 29/50（58%），188.20步，19.88恢复；S=0.2 29/50（58%），194.68，21.76；S=2.5 29/50（58%），189.68，13.16；S=4.5 30/50（60%），179.04，9.56 | 每个S 50局 | [κ=4 suite](../outputs/evaluation/conditional_mlp10k_ph_kappa4_20261103/S0/summary.json) |
| Paired conditional MLP step2k，PH κ=4早期恢复实现 | 逐物理动作检测/回退，negative prompt为最近已执行动作重复：S=0 35/50（70%），168.80步，14.86恢复；S=0.2 35/50（70%），165.40，13.90；S=2.5 35/50（70%），168.12，10.16；S=4.5 35/50（70%），167.40，8.30 | 每个S 50局 | [早期恢复评测](../outputs/evaluation/conditional_mlp2k_ph_kappa4_20261103/S0/summary.json)，不等同后续8步chunk恢复 |
| Paired conditional MLP step2k，PH κ=4 boundary8 | 每8步执行块后检测，只回退最近一个动作并重复为prompt：S=0 35/50（70%），165.64步，3.00恢复；S=0.2 35/50（70%），166.26，2.74；S=2.5 35/50（70%），161.86，2.14；S=4.5 35/50（70%），166.30，2.60 | 每个S 50局 | [boundary8报告](../outputs/evaluation/conditional_mlp2k_ph_kappa4_boundary8_8gpu_20261103/REPORT.md)，与相同35/50无恢复场景结果一致 |
| Paired conditional MLP step2k，PH κ=4 full-chunk/full-plan | 每8步执行块后检测，整块反向EEF回退；negative prompt缓存完整50×7 action_pred：S=0 36/50（72%），167.50步，1.32恢复；S=0.2 36/50（72%），168.04，1.32；S=2.5 36/50（72%），169.98，1.20；S=4.5 36/50（72%），172.16，1.26 | 每个S 50局 | [full-plan报告数据](../outputs/evaluation/conditional_mlp2k_ph_kappa4_chunk8_fullpred_8gpu_20261103/results.json)。对无恢复35/50基线，每个S都救回episode 27、没有丢失原成功episode |
| Paired conditional MLP step2k，Paired detector κ=4，MLP推理消融 | 无detector：MLP on 35/50（70%），191.82步，0恢复；off 32/50（64%），210.82；detector S=0：on 34/50（68%），188.22，1.20；off 31/50（62%），200.34，1.34 | 每个配置50局 | [Paired MLP消融报告](../outputs/evaluation/paired_2k_paired_detector_mlp_ablation_sealab1_20261103_193538_utc/REPORT.md)，同一checkpoint，仅推理时关闭提示投影 |
| Paired conditional MLP step2k，MH detector κ=4，MLP推理消融 | S=0：MLP on 34/50（68%），187.04步，0.72恢复；off 31/50（62%），207.82，0.56 | 每个配置50局 | [MH MLP消融报告](../outputs/evaluation/paired_2k_mh_detector_mlp_ablation_sealab1_20261103_201552_utc/REPORT.md)，Paired策略不变，只换MH detector |
| Paired conditional MLP step2k，Paired detector κ=4，negative prompt A/B | S=2.5：完整action_pred 35/50（70%），181.42步，0.90恢复；重复最后执行动作35/50（70%），185.10，1.06。S=4.5：完整action_pred 35/50（70%），180.36，0.94；重复最后动作35/50（70%），184.04，1.04 | 每个条件50局 | [A/B评测目录](../outputs/evaluation/conditional_mlp2k_paired_detector_prompt_ab_20261103/)。每个S下两种prompt的逐episode成功/失败完全一致 |
| PH conditional MLP新训练step500/1k/2k | 无detector：500步49/50（98%），119.64步；1k 44/50（88%），148.94；2k 49/50（98%），109.06；2k关闭action-conditioning投影50/50（100%），102.82。恢复均为0 | 每个配置50局 | [PH conditional报告](../outputs/evaluation/ph_conditional_2k_8gpu_20261103_184946_utc/REPORT.md)，模型使用PH策略数据训练 |
| PH conditional MLP step2k，PH detector κ=4 | 整块恢复：S=0 48/50（96%），111.32步，0.20恢复；S=0.2 48/50（96%），112.00，0.22；S=2.5 48/50（96%），110.68，0.18；S=4.5 48/50（96%），111.32，0.20 | 每个S 50局 | [PH conditional报告](../outputs/evaluation/ph_conditional_2k_8gpu_20261103_184946_utc/REPORT.md)。四个S的episode结果相同：相比同一conditional 2k无恢复，救回0局、仅额外丢失episode 14；“救回35、丢失14/38”实际参照的是独立训练的PH baseline，不是该conditional无恢复组 |

| MH独立baseline step2k | 无detector，S=0 | 47/50（94%）；平均198.74物理步；恢复0 | [MH baseline汇总](../outputs/evaluation/mh_baseline_2k_8gpu_sealab1_20261103_204352_utc/summary.json)，8卡新训练2k、MH 500步预算；新增1条 |
| MH conditional MLP新训练step500/1k/1.5k/2k | 无detector：500步45/50（90%），214.34步；1k 44/50（88%），228.88；1.5k 46/50（92%），210.58；2k 49/50（98%），182.92；2k旁路MLP 50/50（100%），174.60。恢复均为0 | 每个配置50局 | [MH conditional报告](../outputs/evaluation/mh_conditional_2k_8gpu_20261103/REPORT.md)，新增5条；旁路仍为同一conditional权重 |
| MH conditional MLP step2k，MH detector κ=4 | 整块恢复、完整预测prompt：S=0 49/50（98%），180.90步，0.06恢复；S=0.2 49/50（98%），180.88，0.06；S=2.5 49/50（98%），180.88，0.06；S=4.5 49/50（98%），180.86，0.06 | 每个S 50局 | [MH完整结果](../outputs/evaluation/mh_conditional_2k_8gpu_20261103/results.json)，新增4条；仅episode 28触发，每组3次触发、2次完整回退，最终回退失败；相比同模型无恢复救回0/丢失0 |

</details>

## 排除项与复现入口

- 不纳入正式大表：1局smoke test；`conditional_mlp2k_ph_kappa4_chunk8_8gpu_20261103`中断且无完整50局汇总，已由full-plan实验替代；9k PH suite被取消；Paired step7k评测失败；Paired detector MLP消融19:29启动的中断版本被19:35:38完成版本替代。
- 完整轨迹、每局成功标签和检测器触发记录保留在各目录的`episodes.jsonl`与`episode_*_trace.json`。最新A/B视频在[conditional_mlp2k_paired_detector_prompt_ab_20261103](../outputs/evaluation/conditional_mlp2k_paired_detector_prompt_ab_20261103/)下，文件名包含`success`或`failed`。
- 当前报告基于聚合`summary.json`、suite的`results.json`/`REPORT.md`和场景配对分析；原始实验目录均保留，可继续按episode复核。
