# PH conditional 2k评测计划

截至北京时间2026-11-04 02:32，训练已完成，进程退出码0。开始时间为11月4日00:09:52，结束时间为02:27:53，耗时约2小时18分钟。8张GPU均已释放。

策略目录：`outputs/pi05_full_sft_physical7_v4_mlp/policies/can_ph/conditional/seed_0/`。

已读取checkpoint元数据，确认`checkpoints/step_002000.pt`及`final_model.pt`均为step2000、PH、conditional、MLP；`best_validation.pt`为step500。启动时源码哈希保持一致，checkpoint数据划分和归一化统计仍与既有PH baseline完全一致。

两轮自动评测已于北京时间2026-11-04 02:49:46启动，并于03:26:55左右完成全部400局及审计。主进程PID为266090。输出目录：`outputs/evaluation/ph_conditional_2k_8gpu_20261103_184946_utc/`。结果与审计为该目录下的`REPORT.md`、`results.json`和`audit.json`。

最终结果：conditional step500为49/50，step1000为44/50，step2000为49/50；同一step2000权重跳过提示MLP为50/50。第二轮PH kappa4的S=0、0.2、2.5、4.5均为48/50。具体恢复覆盖及相同场景对比见完整结果。

## 对照与固定条件

主结果固定为conditional step2000的普通无恢复rollout，与相同PH数据、同学习率计划的baseline step2000比较。复用已完成的baseline评测：

| PH baseline checkpoint | 已完成成功率 | 结果目录（位于outputs/evaluation/） |
| --- | --- | --- |
| step500 | 36/50，72% | `pi05_full_sft_ph_baseline_seed0_2k_best_validation/` |
| step1000 | 46/50，92% | `pi05_full_sft_ph_baseline_seed0_2k_step1000/` |
| step2000 | 49/50，98% | `pi05_full_sft_ph_baseline_seed0_2k_final/` |

所有新配置使用同一份`outputs/evaluation/can_initial_states_50.json`，每组50局，环境种子100000–100049，目录哈希应为`9f3a555765e617499370f26e8c7b8a85b260274bf70b6fed10ffd1548b9f246b`。每次预测50步、执行前8步、10步去噪、400个物理控制步预算，回退计入预算。归一化使用本次PH checkpoint自身的统计。所有部署评测都不提供未来GT动作。

## 第一轮：普通策略表现与MLP消融，200局

| GPU | checkpoint与条件 | 检测器 | 局数 | 目的 |
| --- | --- | --- | --- | --- |
| 0–1 | conditional step500，正常零Base提示 | 无 | 50 | 与PH baseline step500比较 |
| 2–3 | conditional step1000，正常零Base提示 | 无 | 50 | 与PH baseline step1000比较 |
| 4–5 | conditional step2000，正常零Base提示 | 无 | 50 | 主结果，与PH baseline step2000比较 |
| 6–7 | 同一个conditional step2000，跳过提示投影 | 无 | 50 | 测量推理时MLP零Base偏置的影响 |

每组两张GPU各跑25局。step500对应本次best-validation；普通验证loss从step500的0.142522升至step2000的0.232992，因此补测500、1000用于诊断训练阶段变化。成功率仍须实测，不能根据loss直接判定2k模型更差。step2000始终作为主结果，不依据这轮成功率更换主checkpoint。

MLP消融保留完整checkpoint所有权重，只在推理时不传`action_condition`，使模型跳过提示投影贡献。它与当前“输入零Base”不同，因为训练后的MLP(0)可以非零。这项消融只衡量部署时Base偏置的影响，不能替代“有MLP但训练始终零提示”的重新训练对照。

## 第二轮：固定2k模型测试恢复，200局

| GPU | PH detector与引导 | 局数 | 目的 |
| --- | --- | --- | --- |
| 0–1 | PH κ=4，S=0 | 50 | 检测、整块回退与重规划的效果 |
| 2–3 | PH κ=4，S=0.2 | 50 | 弱负向引导 |
| 4–5 | PH κ=4，S=2.5 | 50 | 中等负向引导 |
| 6–7 | PH κ=4，S=4.5 | 50 | 较强负向引导 |

恢复使用当前已实现的协议：每执行8步，在下一次策略调用前检测；触发后按记录的前置EEF位姿倒序伺服回退整个最近执行chunk；完整环境裁剪前50×7预测作为negative prompt，按PH checkpoint统计归一化、补零并输入已训练MLP。每目标最多10个控制步，5mm/0.05rad容差，回退控制计入400步预算。失败目标可导致提前终止。

固定检测器为`outputs/paper_can/detectors/can_ph/seed_0_kappa4/detector.pt`，阈值0.01565774645848478，不重新训练或调阈值。先比较检测器S=0与第一轮2k无检测器结果，再比较各非零S与S=0，分别评估恢复流程和负向引导的增量。

## 执行准备与报告

准备工作已完成：`evaluate_conditional_mlp_suite.py`新增`--policy-dir`，本轮明确指向PH模型；`--disable-action-conditioning`保留完整checkpoint并在推理时跳过提示投影，该状态写入run身份，混合消融条件的分片会拒绝合并。新增`evaluate_ph_conditional_2k.py`负责两轮接续、失败停止、公共checkpoint哈希缓存、源码快照和最终报告。分片异常最多自动重试两次，沿用原episode索引续跑。相关40项测试及脚本语法检查均通过。

全部新测试共400局，分两轮使用8张GPU，不重跑已完成的150局PH baseline。每组报告成功率、平均物理步数、推理次数、恢复次数、完整chunk回退成功次数、回退失败/预算终止数、实际引导推理次数及控制步数。按相同episode_id报告相对于baseline或无恢复对照新增救回和丢失的场景，并检查重复、缺失、模型/目录哈希及回退步数记账。

PH baseline已达98%，当前50场景上的提升空间仅一局；恢复测试仍应重点检查触发覆盖、是否损害原本成功的场景，以及引导分支是否真正执行。如果2k普通策略也接近满分，恢复能力的后续验证可另建独立困难场景测试集，保留本轮固定场景结果，不用调过的场景替换主结果。
