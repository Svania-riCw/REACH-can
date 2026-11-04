# Paired conditional 2k与MH检测器的MLP消融

用户确认只补测MH检测器，PH不重跑。策略固定为Paired conditional step2000，不换成MH策略；检测器使用MH图像数据训练的AE，Gaussian kappa=4、S=0。

北京时间2026-11-04 04:15:52在sealab1启动，主进程PID为1302940。输出目录为`outputs/evaluation/paired_2k_mh_detector_mlp_ablation_sealab1_20261103_201552_utc/`，独立源码目录为`/2025233210/lmx/reach_can_training_mh_detector_ablation_20261103_201528_utc/`。

| GPU | 策略 | 检测器 | 提示投影 | 新增局数 |
| --- | --- | --- | --- | --- |
| 0–3 | Paired conditional 2k | MH，kappa4，S=0 | MLP开，零Base | 50 |
| 4–7 | Paired conditional 2k | MH，kappa4，S=0 | MLP关 | 50 |

共100局，每组4张GPU按13、13、12、12局分片。复用同一50场景、环境种子100000–100049、预测50步、执行8步、10步去噪和400物理控制步预算；整chunk回退计入预算，部署不提供未来GT。关闭MLP只改变推理提示投影，不删除或重新训练任何策略权重。

MH kappa4 detector由`outputs/paper_can/detectors/can_mh/seed_0/detector.pt`派生并另存`seed_0_kappa4/detector.pt`，不覆盖历史kappa3文件。使用已保存的6570个held-out ID样本统计：mean=0.005937463542038243、std=0.0023674755084418585，阈值mean+4*std=0.015407365575805677，AE权重不变。

对照复用已完成并通过审计的`paired_2k_paired_detector_mlp_ablation_sealab1_20261103_193538_utc`：无检测器MLP开/关为70%/64%，Paired kappa4、S=0时MLP开/关为68%/62%。程序启动前核对checkpoint、归一化和场景哈希；报告中将旧对照标为复用，进度只统计本次100局。

相关39项测试通过，覆盖MH独立选择、4卡分配、相同checkpoint/场景对照、最终报告的新旧局数区分及关闭MLP后的S=0整chunk恢复。完整结果完成后自动写入`REPORT.md`、`results.json`和`audit.json`。
