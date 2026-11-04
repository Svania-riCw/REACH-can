# 整体实验安排与当前进度

更新时间：**2026-11-02 04:14:17 UTC**。Paired conditional seed0已启动完整8卡训练；baseline保持暂停。5000步评测已完成13/50=26%，8000步已完成32/50=64%。

## 当前运行：Paired conditional seed0完整8卡训练

按用户要求于2026-11-02 04:11:48 UTC启动，仅执行`policy_paired_conditional_seed_0`，从原始Pi0.5 base独立初始化，45%零base、55%自身GT、0%shuffle。8卡×micro8×accum1=global64，原50轮/13850步，不设中途自动暂停。监督前7维、全VLM与动作专家训练，FP32、AdamW及学习率计划均保持baseline配方。

实际保存的config逐项比较通过：仅policy_mode、base_prob和输出路径不同；baseline的base_prob=.3不参与其训练，保留以兼容严格恢复。两者episode_split.json和action_stats.json内容完全一致。[配置比较记录](outputs/pi05_full_sft_physical7_v4/conditional_seed0_configuration_check.json)。

- [启动记录](outputs/pi05_full_sft_physical7_v4/conditional_seed0_launch.json)，后台supervisor PID为4017152；[当前队列状态](outputs/pi05_full_sft_physical7_v4/status.json)只跟踪本次conditional任务。
- [conditional日志](outputs/pi05_full_sft_physical7_v4/policies/can_paired/conditional/seed_0/train.log)；输出为`outputs/pi05_full_sft_physical7_v4/policies/can_paired/conditional/seed_0/`。
- baseline的原队列状态与manifest保存在[启动前快照](outputs/pi05_full_sft_physical7_v4/history/before_conditional_seed0_20261102T041148Z/status.json)，原baseline启动和暂停记录保留。
- 已准备[baseline续训脚本](scripts/resume_can_paired_baseline_seed0_v4.sh)，从完整step8000恢复到原目标13850；CPU读取实际checkpoint元数据并执行严格resume检查通过，[检查记录](outputs/pi05_full_sft_physical7_v4/baseline_resume_command_check.json)。本机未启动baseline续训；可在另一台空闲8卡服务器、相同挂载路径及环境中运行该脚本。

以下包含先前阶段快照；当前运行状态以上方和实时日志为准。

## 当前提示配比：作者主分支45%base/55%自身GT

按用户最新要求，conditional默认已改为45%零base、55%自身GT、0%shuffle。Python训练入口、train_step/train_update默认、shell、canonical配置与新manifest一致。通用共享build_action_condition助手未改动，训练调用显式传入45/0；不影响其他项目。旧30/30/40记录和执行manifest保留，新清单另存[author_prompt45_experiment_manifest.json](outputs/pi05_full_sft_physical7_v4/author_prompt45_experiment_manifest.json)。conditional正式训练已启动，见上方当前运行记录。

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

## 历史审计：v4 conditional、检测器及引导与作者代码复核（2026-11-02）

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

## 已完成结果与下一步实验安排

8000步Paired baseline seed0的5卡评测已全部完成：**32/50成功，成功率64%，失败18回合均达到400步预算**。50个场景无重复或遗漏；成功回合平均99.59步。旧冻结VLM baseline为23/50=46%，相同场景下提升18个百分点：17场均成功、15场仅新版成功、6场仅旧版成功、12场均失败。多项训练配置同时改变，不能单独归因为VLM解冻、7维mask或某一超参数。单种子50回合的Wilson95%区间约50.1%–75.9%，尚不能作为三种子论文复现结果。

[完整汇总](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/summary.json)、[逐回合结果](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/episodes.jsonl)、[统计与旧版配对比较](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/analysis.json)。本轮从5卡启动到最慢分片完成约15.9分钟。当前训练仍暂停，评测进程均结束，GPU0–7空闲。

验证损失最低为step1000的0.13172；step3000/5000/8000分别为0.20640/0.27615/0.38115。后期训练损失仍低，存在过拟合信号，但尚不能推断哪个checkpoint的rollout成功率最高。不要仅因64%或验证损失就宣称已经收敛、需要延长至500epochs或确定某项修复有效。

| 优先顺序 | 实验 | 资源及预算 | 判断依据 |
| --- | --- | --- | --- |
| 1 | 补测baseline step1000、3000、5000 | 每个相同50场景，5卡×10；三组约45–60分钟 | 与已测step8000比较；step1000也是验证选中的checkpoint |
| 2 | 确定baseline续训顺序 | 如补齐原50轮，从8000恢复至13850，8卡约6–7小时 | 若早期明显更好，优先conditional；正式等预算比较仍需一致训练上限，不临时改LR schedule |
| 3 | 训练Paired conditional seed0 | 原base独立初始化，8卡micro8/accum1/global64，physical7；50轮13850步，暂估15–20小时 | 保持数据、split、全VLM训练、优化器一致；当前提示45%base/55%自身GT，shuffle关闭 |
| 4 | 同步数评测baseline/conditional及恢复 | 先比较8000步及验证所选权重；conditional无ED+PH/MH/Paired各4个S，连baseline共14配置×50回合 | 分离conditional本身和恢复收益；保留final、validation选中、rollout最高的不同口径 |
| 5 | Paired seed1/2与消融 | 固定配方后补齐三种子；可增加0%shuffle对照 | 避免只用seed0结论；正式最高checkpoint口径补齐预设保存间隔的评测，早期筛查不是完整扫描 |
| 6 | MH策略实验 | Paired完成后，从同一base独立训练baseline/conditional×三种子 | 不从Paired权重继续；独立MH统计及split，复用适用AE |

同时回看若干失败轨迹，必要时生成视频，区分接近、抓取、滑落和放置问题；现有budget标签不能证明具体失败机制。现阶段不改控制器、mask、提示比例或评测预算，先完成checkpoint对照。

[后续实验及可执行命令](outputs/pi05_full_sft_physical7_v4/next_experiment_plan.json)已准备；**本轮只统计和规划，没有恢复训练或启动新评测**。下文保留已执行流程的历史记录。

## 历史：5卡并行评测8000步

按用户追加指示，已停止原单卡评测，原部分结果保留；2026-11-02 01:18:59 UTC启动5卡队列（supervisor PID4068021）。GPU0/1/2/3/4分别评测目录索引0–9/10–19/20–29/30–39/40–49，每卡10回合，共50个原固定场景。每个场景继续使用原环境seed及策略noise seed，400步预算和baseline协议不变。训练仍暂停，不自动恢复。

分片入口保留完整rollouts=50，新增全局episode-start/end选择，不将每卡10回合误计为独立完整实验。每份分片summary明确complete_evaluation=False；全部完成后校验checkpoint、协议、归一化、实现、catalog及seed一致，拒绝重复/遗漏，输出完整50回合summary。11项评测及分片测试通过，日志logs/evaluation_shards_tests.log。原单卡的部分结果不混入新50回合。

[5卡状态](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/status.json)、[启动记录](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/launch.json)、[分片目录](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/shards)、[最终汇总](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000_5gpu/summary.json)。GPU5–7空闲；simulator使用既有EGL设备0，策略计算分别绑定GPU0–4。

## 历史：最初单卡评测step8000

2026-11-02 01:13 UTC按用户要求停止8卡训练；停止前最新训练日志约8410步，最新完整定期checkpoint为8000。原始训练器将主动中断记为Interrupted；状态文件已补注用户暂停原因，checkpoint未删除或覆盖。

已于01:13:57 UTC在GPU0独立后台启动step_008000.pt的baseline评测：50个共同初始场景100000–100049，Paired最长400步、10步flow采样、每次执行8步，无检测器，不启用恢复。使用既定评测协议及checkpoint自身归一化。当前只评测这一个baseline checkpoint。

[评测启动记录](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000/launch.json)、[评测日志](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000/driver.log)、[逐回合结果](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000/episodes.jsonl)、[最终汇总](outputs/evaluation/physical7_v4_paired_baseline_seed0_step008000/summary.json)。最终汇总完成后才生成。

训练保持暂停，不自动恢复。从8000继续到原定13850的完整命令已保存在[暂停及恢复记录](outputs/pi05_full_sft_physical7_v4/pause_for_step8000.json)，恢复应先确认评测结束和GPU释放，保持原8卡及学习率计划。下文5000步安排保留为历史计划，实际采用8000步。

## 当前版本：Pi0.5 全参数 SFT physical7 v4

使用现有转换base；VLM/动作专家全参训练；真实9维state先token化；训练集q01/q99。按用户要求只监督前7维物理动作，后25维不计入loss；验证及best-validation使用相同7维目标，模型接口仍32维。没有新增时间步mask。Can(Paired)保留官方180/20、全部成功及失败轨迹、17780/2015帧、双相机、7维相对OSC控制。

| 配置 | 当前值 |
| --- | --- |
| base | checkpoints/pi05_base_pytorch/model.safetensors，继续复用原转换权重 |
| 参数 | FP32，全VLM+动作专家，仅冻结未使用expert LM head |
| 优化器 | AdamW β=(0.9,0.95)，wd=1e-10 |
| 学习率 | peak2.5e-5，warmup1000，cosine至2.5e-6 |
| batch / 时长 | 8卡×每卡micro8×累积1=global64；50epochs=13850步 |
| loss / 模型选择 | physical7均值；full32与padding25仅诊断 |
| 输出 | outputs/pi05_full_sft_physical7_v4/policies/can_paired/baseline/seed_0 |
| 保存 | 每500步、best_validation.pt、final_model.pt |
| 训练状态 | 训练暂停；step8000评测已完成32/50=64%；8卡空闲，无自动续训 |
| 检测器 | 复用outputs/paper_can/detectors，不需因策略SFT变更重训 |
| 验证 | 37项检查通过，真实8卡更新同步，后25维输出梯度0，峰值54.28GiB |

[启动记录](outputs/pi05_full_sft_physical7_v4/launch.json)、[任务状态](outputs/pi05_full_sft_physical7_v4/status.json)、[训练日志](outputs/pi05_full_sft_physical7_v4/policies/can_paired/baseline/seed_0/train.log)、[性能事件](outputs/pi05_full_sft_physical7_v4/policies/can_paired/baseline/seed_0/performance.jsonl)、[清单](outputs/pi05_full_sft_physical7_v4/experiment_manifest.json)、[8卡验证](outputs/validation/full_sft_physical7_v4/gpu_smoke_world8_micro8_accum1.json)。训练以独立会话后台运行，聊天中断不影响；不要重复启动。

执行顺序：当前先完成新版Paired baseline，再以相同配方训练conditional（45%base/55%自身GT，无shuffle），完成seed0评测后补Paired seed1/2，最后MH。旧v2冻结VLM结果与v3完整32维审计文件保留；本次v4从base独立启动，不能resume旧目标。官方Pi0.5完整32维loss与本轮物理7维目标不同，这是用户明确选择的适配。

Kai0 baseline仍为完整32维loss，其action_is_pad只mask时间步；本项目本轮新增的是动作维度mask。历史审计：[Kai0检查](outputs/validation/full_sft_v3/kai0_padding_mask_review.json)。下文旧46%/26%结果属于v2，不能当成新版v4结果。

截至2026-11-01 16:04:04 UTC，已确认完成40次实际更新，稳定约3.29秒/步，单卡峰值55.72GiB。纯训练剩余约12.6小时；为每100步验证和checkpoint保存留出开销，当前估计baseline还需14–17小时。首轮验证/保存尚未计时，此为初步区间，不含conditional或评测。[估时记录](outputs/pi05_full_sft_physical7_v4/eta.json)。

## 5000步中途评测安排

用户计划在约5000步检查成功率；该步自然保存`checkpoints/step_005000.pt`，约第18.1轮。checkpoint包含optimizer/scheduler、训练位置、各rank RNG、归一化与split，支持中断后从该步恢复到原定13850步；保持8卡、global64、原50轮目标和LR schedule。若要超过13850步追加训练，需要另行定义延长阶段，不能直接改epoch后使用严格resume。

评测使用baseline checkpoint单独进行50个固定场景（100000–100049），Paired最长400步，无检测器。当前8卡全部训练；建议到完整checkpoint保存后暂停并释放GPU，评测后恢复。已准备[具体评测和恢复命令](outputs/pi05_full_sft_physical7_v4/step5000_evaluation_plan.json)。当前仍继续训练，**尚未设置5000步自动暂停或自动评测**；估计约5–6小时达到该步（依据当前约3.3秒/步并计入验证及保存开销）。

## 目标与执行顺序

使用 Pi0.5 骨干复现 REACH Can 实验组织，**先完成 Paired 的训练、恢复实现及评估，再开展 MH 策略实验**。Paired 与 MH 策略均从相同基础权重独立训练，不进行 PH→MH 或 Paired→MH 顺序微调。

| 阶段 | 工作内容 | 当前状态 / 完成条件 |
| --- | --- | --- |
| 1. 数据和启动检查 | PH 转换，Paired / MH 新版渲染，四卡短测 | PH、Paired、MH 均已完整可用；两组四卡短测通过 |
| 2. Paired seed 0 | baseline 与 conditional 并行训练，CPU 训练检测器 | 两组旧策略、AE_PH / AE_Paired / AE_MH 均完成 |
| 3. 打通 seed 0 恢复流程 | rollout、错误检测触发、失败动作历史、回退、在线负向引导 | 在线 rollout / 检测 / 物理回退 / 负向引导已实现，真实短测通过 |
| 4. 验证 seed 0 | 固定噪声诊断、正常执行与恢复的同条件评估 | 离线与真实短测通过；seed 0 final-only 14 个配置 / 700 回合均完成；全参physical7 v4 baseline已启动 |
| 5. 补齐 Paired | 训练 seed 1/2，并完成三个种子的 Paired 评估 | 未排队；待 seed 0 在线流程验证后启动 |
| 6. MH 策略实验 | MH baseline / conditional × 三个种子，复用已训练的相关检测器 | 未排队；Paired 实验完成后开展 |

MH 数据已完整生成，**Paired 实验也需要 AE_MH 检测器**，AE_MH 当前已完成。训练 AE_MH 不等于提前开展 MH 策略实验。

## 全部实验规模

当前配置见 [configs/paper_can.json](configs/paper_can.json)，协议为 `reach_can_pi05_full_sft_physical7_author_mix45_v4`。

| 实验部分 | 策略训练 | 检测器 | 待评估组合 |
| --- | --- | --- | --- |
| Paired | baseline / conditional × seed 0/1/2，共 6 个 | AE_PH / AE_MH / AE_Paired × 三个种子，共 9 个 | 42 组 |
| MH | baseline / conditional × seed 0/1/2，共 6 个 | 复用 AE_PH / AE_MH，无需再增加同配置训练 | 30 组 |
| 合计 | 12 个策略 | 9 个独立 AE | 72 组 |

Paired 每个种子有 2 个无检测器对照，以及 3 类检测器 × 4 个引导强度，共 14 组。MH 每个种子为 2 个对照加 2 类检测器 × 4 个强度，共 10 组。引导强度为 `S=0/0.2/2.5/4.5`；带检测器的 S=0 仍启用恢复流程，与无检测器对照分开。

评测改为每 checkpoint/configuration 50 个公共测试初始场景，种子 100000–100049；Paired 400 步，MH 500 步。此设置来自论文引用的基准。保留逐种子、逐 checkpoint 原始结果；当前 seed 0 final-only 已完成14配置/700回合，单回合 smoke 不计入正式结果。

## 数据准备进度

| 数据集 | 完成轨迹 | 完成帧数 | 状态 |
| --- | --- | --- | --- |
| PH | 200/200 | 23,207/23,207 | 可用，完整读回及元数据校验通过 |
| Paired | 200/200 | 19,795/19,795 | 可用，22:40 UTC 完成新版渲染 |
| MH | 300/300 | 62,756/62,756 | 已完成，2026-11-01 01:34 UTC 发布 |

Paired 官方切分为训练 180 条 / 验证 20 条，对应 17,780 / 2,015 帧；同初始场景的成功与失败演示始终在同一集合。Paired 策略与 AE_Paired 均保留全部成功、失败轨迹。

原始 low-dim 数据均已下载，三个正式图像文件现均已发布。正式图像目录为 `data/paper/`，MH 的 `.partial.hdf5` 仅在全部生成完成后原子发布为最终文件；不读取活跃 writer 的未完成文件。

PH 复用旧渲染像素，已转换图像方向并恢复官方元数据、全部 masks，**未重新逐轨迹恢复 XML 渲染**。MH 和 Paired 使用新版原始 XML / 记录状态渲染。PH 来源和读回校验详见 [转换报告](data/paper/can_ph_image.conversion.json)。

## 旧版 v2 资源分配和训练记录

正式任务于 **2026-10-31 23:00 UTC** 启动，目前只启动 seed 0。

| 资源 | 任务 | 最近已记录进度 | 最近训练 loss | 状态 |
| --- | --- | --- | --- | --- |
| GPU 0–3（训练时） | Paired baseline，冻结VLM变体，seed 0 | 6,900/6,900 步，100% | 0.047009 | 已完成 |
| GPU 4–7（训练时） | Paired conditional，seed 0 | 6,900/6,900 步，100% | 0.047135 | 已完成 |
| CPU，4 个计算线程 | AE_MH，seed 0，Adam | 50/50 epochs | 见训练日志 | 已完成 |
| 同一 CPU 队列 | AE_PH、AE_Paired，seed 0 | 均为 50/50 epochs | 见各自日志 | 已完成 |

旧检测器的 SGD lr=1e-4 更新过小，队列于 10 月 31 日 23:20 主动停止。经独立对照后，检测器已于 **11 月 1 日 08:16:56 UTC** 改为 Adam lr=1e-4，从初始权重重新训练 50 epochs，旧权重和日志已归档；不沿用旧任务的 epoch 计数。此优化器变更作为额外实现选择记录，策略进程和启动配置未改动。 当前完整 validation MSE 已从 0.086574 降至 0.009669，相对改善 88.83%。

8 张 H20 的策略训练已结束，首轮正式评测也已结束，当前8卡均空闲；本次审计仅短暂使用GPU7且已释放。CPU 检测器队列不额外占用策略 GPU。此处的 loss 是不同训练 batch 的记录，不据此判断哪条路线更优；后续需在相同留出数据及评估初始条件下比较。

### 策略配置

| 参数 | baseline | conditional |
| --- | --- | --- |
| 初始权重 | 同一 `checkpoints/pi05_base_pytorch/model.safetensors` | 同左 |
| 数据切分及归一化 | 同一官方切分，归一化仅拟合训练集 | 同左 |
| 训练时长 | 50 epochs / 6,900 更新步 | 同左 |
| global batch / GPU 数 | 128 / 4，每卡 32 | 同左 |
| 学习率 | 1e-4，cosine，warmup=0，min LR=0 | 同左 |
| 动作条件 | 不使用 | 30% base / 30% shuffled GT / 40% 自身 GT |
| 其他 | 冻结 VLM，训练 action expert，启用梯度检查点 | 额外训练动作条件投影 |

Pi0.5 使用单帧观测、50 步动作 chunk、7 维物理动作补零到 32 维；loss 只统计前 7 维。条件策略的 **30% GT shuffle 是用户要求加入的额外训练变体，不作为论文公开的原始设置报告**。它只替换动作提示，观测、监督目标和 chunk 内时间顺序均保持原样；来源是同一 GPU batch 的其他样本，三类概率相对于全部样本。

检测器默认使用双视图 84×84 图像，展平 → 512 → 128 → 512 → sigmoid 重建，Adam、batch=128、50 epochs、lr=1e-4；在留出 ID 数据标定 `mean + 3*std`。这些网络尺寸、标定参数与部分 Pi0.5 适配属于明确记录的实现选择，详见配置和 README。

## 训练框架与权重核验

策略和检测器均使用 PyTorch；JAX 用于原始基础权重读取及部分工具依赖，不负责当前策略反向传播。811 个实际使用的权重张量已与 JAX 原始值经映射和保存 dtype 转换后的结果逐一核对，无不匹配，见 [核验报告](logs/pi05_conversion_weight_audit_20261031.json)。尚未执行 JAX / PyTorch 相同输入的端到端前向对齐，已有采样一致性检查是在 PyTorch 内进行。

检测器优化对照和新训练的重建收敛指标见 [验证记录](PAPER_REPRODUCTION_VALIDATION.md) 及当前 AE 输出的 [optimization_diagnostics.json](outputs/paper_can/detectors/can_ph/seed_0/optimization_diagnostics.json)。重建改善不等于已验证在线错误检测质量。

## 预计用时

正式启动前，两组均在相同 4+4 卡、完整数据及 batch 下完成 6 步短测、完整验证和保存：

| 指标 | baseline | conditional |
| --- | --- | --- |
| 第 2–6 步平均耗时 | 4.968 秒/步 | 4.942 秒/步 |
| rank 0 GPU 张量峰值分配 | 20.87 GiB | 20.87 GiB |
| 完整验证及固定面板 | 45.08 秒 | 45.14 秒 |
| 保存 checkpoint | 6.06 秒 | 7.98 秒 |

两组策略实际分别于 2026-11-01 09:42:47 / 09:41:45 UTC 完成，符合启动时 10–12 小时的估计。纯更新约 9.5 小时，含验证、保存与换 epoch 后实际总耗时约 10 小时 42 分，两组并行，无需相加。

上述时间只包含 seed 0 策略训练。若后续种子保持相同速度，三种子的策略训练时段合计约 30–36 小时；在线恢复实现、评估及阶段间验证另计，不能据此承诺整个 Paired 实验的完成时间。

## 训练后检查与后续实现

1. 确认完整 checkpoint、切分、归一化和训练配置均已保存，检查完整 validation 与固定 GT/base/foreign 分支记录。
2. conditional 队列已自动完成固定噪声 steering 诊断，验证上游/自定义采样一致、S=0 与 base 一致，记录提示跟随和正负向引导距离；不要求非线性模型的距离随 S 单调。
3. 在线流程已实现并通过真实短测：观测/动作适配、实际执行动作历史、AE 错误触发、物理回退及负向引导。
4. 使用相同评估初始状态、随机种子和执行预算比较 baseline、conditional base 与恢复设置；验证正确后再补 seed 1/2 和剩余 Paired 组合。
5. Paired 完成后启动 MH 策略，对应检测器复用已有相同配置结果。

启动前已有 22 项回归检查通过，此外两组真实四卡短测均完成训练、全量验证和保存；恢复测试及采样一致性证据见 [验证记录](PAPER_REPRODUCTION_VALIDATION.md)。短测权重已清理，正式训练从基础权重重新开始。

## 已就绪的评测流程

完整操作命令见 [README 的真实仿真评测部分](README.md#真实仿真评测)，参数和未公开项清单见 [evaluation_can.json](configs/evaluation_can.json)。模型和仿真复用各自现有 Python 环境，不修改共享依赖；当前单 EGL 图形设备与模型 CUDA 分卡隔离。

- 数据集状态重放：两路相机逐像素一致，EEF 位置误差约 6.6e-16。
- 400 步预算的单回合测试：baseline 第 80 步成功，conditional 第 87 步成功。它们仅证明接口和流程正常，不是正式成功率。
- 强制异常的 S=2.5 / S=0 测试均触发 5 次恢复、真实执行 19 个回退步；前者有 4 个引导前向步，后者为 0。回退不使用 simulator state 恢复。
- 30 项检查通过，覆盖动作单位、SE(3)、失败历史、相同步数预算、S=0、聚合、并行和恢复指纹。
- seed 0 共有 14 个 checkpoint × 14 个配置 = **196 个评测单元 / 9,800 回合**；[评测清单](outputs/evaluation/paired_seed0_all_checkpoints/matrix.json)已生成。
- **首轮正式评测已于 2026-11-01 11:13:29 UTC 启动**，范围为两份 seed 0 最终权重、14 个配置 × 50 回合 = 700 回合。baseline 为 1 个无检测器对照，conditional 为 1 个无检测器对照 + 3 类检测器 × 4 个 S，共 13 个配置。两份权重独立出结果，共用相同 50 个初始状态。
- 当前 8 卡运行 final-only 队列；AE_MH 的四个配置在其最终 detector.pt 完成后自动接入，最多等待 7200 秒。全 checkpoint 的 196 个单元未启动，final-only 不能替代最高 checkpoint 成功率。
- 首轮状态：[status.json](outputs/evaluation/paired_seed0_final/status.json)；启动记录：[launch.json](outputs/evaluation/paired_seed0_final/launch.json)；清单：[matrix.json](outputs/evaluation/paired_seed0_final/matrix.json)。
- 只有 seed 0 已训练，三种子统计仍待 seed 1/2；聚合器在种子或 checkpoint 不完整时不会输出伪完整最高成功率。

当前 final-only 评测命令（已后台运行，不要重复启动）：

```bash
bash scripts/evaluate_can.sh run --datasets paired --seeds 0 --checkpoints final \
  --output outputs/evaluation/paired_seed0_final \
  --gpus 0 1 2 3 4 5 6 7 --wait-seconds 7200
```

后续全 checkpoint 补扫命令（尚未启动）：

```bash
bash scripts/evaluate_can.sh run --datasets paired --seeds 0 --checkpoints all \
  --output outputs/evaluation/paired_seed0_all_checkpoints \
  --gpus 0 1 2 3 4 5 6 7 --wait-seconds 7200
```

没有将本地选择称为原论文参数：回退伺服容差、最多控制步、夹爪处理、冻结阈值、短历史补齐以及 checkpoint 扫描节奏均有未公开项；Pi0.5、GT shuffle、AE Adam 是现有适配，输出 `exact_paper_reproduction=false`。完整严格数值复现仍受这些边界限制。论文表注和正文聚合口径有差异，结果同时保留两种聚合及所有原始分数。

## 日志、状态和产物

| 内容 | 入口 |
| --- | --- |
| baseline 训练日志 | [train.log](outputs/paper_can/policies/can_paired/baseline/seed_0/train.log) |
| conditional 训练日志 | [train.log](outputs/paper_can/policies/can_paired/conditional/seed_0/train.log) |
| baseline / conditional 队列 | [baseline 状态](outputs/paired_first/seed_0/baseline/status.json)、[conditional 状态](outputs/paired_first/seed_0/conditional/status.json) |
| 检测器队列 | [状态](outputs/paired_first/seed_0/detectors/status.json)、[当前 AE_MH 日志](outputs/paper_can/detectors/can_mh/seed_0/train.log) |
| MH 数据生成 | [渲染日志](logs/prepare_mh_paper.log) |
| 后台 PID、命令和资源 | [launches.json](outputs/paired_first/launches.json) |
| 短测原始性能 | [preflight_summary.json](outputs/paired_first/preflight_summary.json) |

策略输出为 `outputs/paper_can/policies/can_paired/{baseline,conditional}/seed_0/`：

- 每 500 步保存完整 checkpoint 至 `checkpoints/`，结束保存 `final_model.pt`。
- 每 100 步执行完整 validation 和固定分支监控；监控写入 `branch_monitor.jsonl` 及 TensorBoard。
- `performance.jsonl` 记录吞吐、验证和保存耗时；`config.json`、`action_stats.json`、`episode_split.json` 保存实验配置和数据约定。

AE 输出为 `outputs/paper_can/detectors/can_{ph,paired,mh}/seed_0/`，包含按 epoch 保存的 `last.pt`、最终 `detector.pt`、`calibration.json` 及优化诊断 `optimization_diagnostics.json`。条件策略训练后的诊断报告路径为 `outputs/paired_first/seed_0/conditional/steering.json`，当前已存在，诊断通过。

三个队列使用独立锁和状态文件，两组 DDP 端口分别为 29621 / 29622；各队列 manifest 和任务完成后的 `job_record.json` 记录命令、环境和任务指纹。任一队列失败会记录具体错误并停止该队列。严格恢复需要保留原数据、GPU 数、batch、种子、训练目标及条件概率，使用完整 checkpoint 的 `--resume`。

查看实时进度：

```bash
cd /2025233210/lmx/reach_can_training
tail -f outputs/paper_can/policies/can_paired/baseline/seed_0/train.log
tail -f outputs/paper_can/policies/can_paired/conditional/seed_0/train.log
cat outputs/paired_first/seed_0/detectors/status.json
tail -n 5 logs/prepare_mh_paper.log
```

## 文档入口

- 本文：整体安排、当前进度快照、资源、时间估算及下一步。
- [README.md](README.md)：数据准备、训练、诊断和恢复命令。
- [PAPER_REPRODUCTION_VALIDATION.md](PAPER_REPRODUCTION_VALIDATION.md)：分阶段验证证据及实现边界，当前进度以本文和实时日志为准。

## 首轮正式评测结果与SFT审计

快照时间：2026-11-01T14:19:39.891869+00:00；完成14/14配置、700/700回合。仅seed0最终权重，且属于冻结VLM变体，不能代表普通全参数Pi0.5 SFT。

| 策略 / 检测器 | S | 成功 / 已完成回合 | 当前成功率 | 状态 |
| --- | --- | --- | --- | --- |
| baseline（冻结VLM） | 0.0 | 23/50 | 46.0% | 完成 |
| conditional / 无检测器 | 0.0 | 13/50 | 26.0% | 完成 |
| conditional / ph | 0.0 | 13/50 | 26.0% | 完成 |
| conditional / ph | 0.2 | 12/50 | 24.0% | 完成 |
| conditional / ph | 2.5 | 12/50 | 24.0% | 完成 |
| conditional / ph | 4.5 | 13/50 | 26.0% | 完成 |
| conditional / mh | 0.0 | 13/50 | 26.0% | 完成 |
| conditional / mh | 0.2 | 13/50 | 26.0% | 完成 |
| conditional / mh | 2.5 | 13/50 | 26.0% | 完成 |
| conditional / mh | 4.5 | 13/50 | 26.0% | 完成 |
| conditional / paired | 0.0 | 13/50 | 26.0% | 完成 |
| conditional / paired | 0.2 | 12/50 | 24.0% | 完成 |
| conditional / paired | 2.5 | 13/50 | 26.0% | 完成 |
| conditional / paired | 4.5 | 13/50 | 26.0% | 完成 |

当前已完成的恢复配置尚未优于 baseline；conditional 无检测器本身只有 26%，不能把全部差距归因于恢复。PH 各组有 17–18/50 回合因回退失败提前终止。评测协议和运行代码保持原样。

baseline 审计发现：验证损失在 step 1200 最低 0.09311，step 6900 最终为 0.13252（约恶化 42%），同时训练损失仍下降，提示后期过拟合；已有 step 1000/1500 等权重可先行对照，尚不能凭验证损失推断其成功率。训练使用官方 180/20 掩码；horizon=50 的目标步约 24.8% 为末动作重复填充。视觉语言主干冻结、仅动作专家训练是本项目选择，非 Pi0.5 必须要求。当前无 warmup、无 EMA，需与任务适配一起验证；普通 baseline 不含 GT shuffle。

论文 Table 2 的 Can(Paired) DP baseline 为 60.8±1.4%，表注采用三个训练种子各自最高成功率的平均；当前单种子最终权重尚非同统计口径。DP 官方配置使用两帧观测、horizon16、execute8、可训练视觉编码器及 EMA；REACH 未逐项公开所有实现参数，不能直接宣称全部沿用。

优先诊断顺序：早期 baseline checkpoint 的同协议 rollout → 数据轨迹动作回放/失败过程检查 → 视觉适配与训练配方消融。保留现有首轮结果，不在中途改协议。

[中间结果原始快照](outputs/evaluation/paired_seed0_final/interim_snapshot.json)；[baseline 审计数据](outputs/evaluation/paired_seed0_final/baseline_audit.json)。

历史全参数SFT审计确认：硬编码冻结整个VLM、真实state补零后token化、仅监督前7维flow loss，均偏离目标；这些已在v3修正。另需规范Pi0.5优化/归一化配方和checkpoint选择。GPU实测无提示扩展类与官方基类前向一致，解除冻结后视觉/语言梯度可达。无EMA、单帧、horizon50不应自动当成错误。详见 [SFT审计](PAPER_REPRODUCTION_VALIDATION.md#普通-pi05-全参数-sft-审计2026-11-01) 和 [原始探针](outputs/source_comparison/pi05_sft/probes.json)。本轮已修复训练代码并完成短测，没有启动长训练。
