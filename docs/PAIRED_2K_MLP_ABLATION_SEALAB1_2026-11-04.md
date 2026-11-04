# Paired 2k MLP推理消融

用户要求后续detector统一kappa=4。训练CLI、Gaussian校准默认值及PH suite默认detector路径已更新；项目AGENTS.md也记录这一要求。历史kappa=3的文件和结果保留原值。

sealab1：root@10.15.171.204:30183。启动前两次确认8张H20显存均为1MiB、利用率为0%。与当前机器使用不同GPU UUID，但共享项目文件系统。

北京时间2026-11-04 03:35:38正式启动，主进程PID为1281546。输出目录：`outputs/evaluation/paired_2k_paired_detector_mlp_ablation_sealab1_20261103_193538_utc/`。使用独立源码目录`/2025233210/lmx/reach_can_training_paired_mlp_ablation_20261103_1920/`，该目录启动后保持不变。

03:29:04的首次启动因sealab1缺少libegl1/libegl-mesa0而未产生任何episode，已停止并归档。安装这两个运行库后，在sealab1完成真实仿真验证：两路初始图像与数据集样本的MAE均为0，EEF位置误差约6.6e-16，伺服回退在2步内达到既定容差。正式输出保留`render_preflight.json`和`preflight.json`；相关39项集成测试通过。

所有组使用同一份Paired conditional step2000完整权重；MLP off仅在推理时跳过提示投影，不删除权重、不重新训练。每组固定50局，两卡各25局，同一场景目录、环境种子、50步预测、8步执行、10步去噪和400物理步预算。部署评测不使用未来GT动作。

| GPU | 恢复条件 | 提示MLP | 局数 |
| --- | --- | --- | --- |
| 0–1 | 无detector | 开，输入零Base | 50 |
| 2–3 | 无detector | 关 | 50 |
| 4–5 | Paired detector，kappa=4，S=0 | 开，输入零Base | 50 |
| 6–7 | Paired detector，kappa=4，S=0 | 关 | 50 |

S=0仍执行检测、整chunk物理回退和重规划。关闭MLP时允许S=0恢复；S非零需要负向提示投影，不允许与MLP关闭组合。本轮不测非零引导，避免同时改变提示编码和引导机制。

Paired kappa4 detector由`outputs/paper_can/detectors/can_paired/seed_0/detector.pt`派生，保存到`seed_0_kappa4/detector.pt`。沿用同一2015个held-out ID样本的已保存mean=0.0063050449618991125、std=0.002577740218687983，阈值为mean+4*std=0.016616005836651043；AE参数不变，无需重新训练。出处及原始文件哈希见派生目录的`recalibration.json`。

本轮自动生成源码快照、checkpoint与detector哈希、分片续跑状态、完整轨迹、最终成功率、恢复计数和相同场景的救回/丢失对比。完成后结果为`REPORT.md`、`results.json`和`audit.json`。

## 最终结果与结论

北京时间2026-11-04 04:02:21完成全部200局，主进程状态completed，完整轨迹与模型/场景身份审计通过。

| 条件 | 成功 | 成功率 | 平均物理步数 |
| --- | --- | --- | --- |
| 无detector，MLP开 | 35/50 | 70% | 191.82 |
| 无detector，MLP关 | 32/50 | 64% | 210.82 |
| Paired kappa4，S=0，MLP开 | 34/50 | 68% | 188.22 |
| Paired kappa4，S=0，MLP关 | 31/50 | 62% | 200.34 |

本次无detector、MLP开组的50个场景成功/失败结果与之前Paired conditional 2k的35/50完全一致。关闭提示投影救回episode38，丢失16、24、42、45，净少3局；有无S=0恢复时，该差异均为6个百分点。推理时零Base经过MLP产生的固定偏置会影响动作及闭环成功，但单次50场景不能证明跨种子的稳定收益。

原Paired baseline 2k为22/50（44%）；关闭conditional提示投影后仍有32/50（64%）。因此原先70%相对44%的差异不能全部解释成部署时多了一个MLP。剩余差异包含训练时动作提示和共享权重优化路径等因素，本轮推理消融不能继续分离这些因素，不能将20个百分点直接视为GT提示的独立因果贡献。

S=0恢复在MLP开/关两种情况下均没有救回原本失败的场景，均额外丢失episode14，净降2个百分点。该场景无恢复时126步成功；恢复从48步首次触发，MLP开组触发11次，关组触发7次，最后均耗尽400步预算。因此当前检测、整chunk回退与S=0重规划的组合尚未带来成功率收益，并会干预原本能够成功的过程。恢复平均步数略短也受失败提前终止影响，不能据此认定更高效。

本轮四组全部S=0，负向引导推理次数为0；这些结果不能用于判断Paired detector下非零引导强度的效果。
