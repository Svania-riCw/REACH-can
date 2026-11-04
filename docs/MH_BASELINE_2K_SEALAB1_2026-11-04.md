# MH baseline 2k：sealab1训练与自动评测

2026-11-04 04:43:55北京时间启动，服务器为sealab1（10.15.171.204:30183），管理进程PID为1317577。训练使用全部8张H20共同训练一个MH baseline；训练正常退出并校验2000步权重后，管理进程自动释放训练阶段资源，调用8个独立GPU worker评测最终2000步checkpoint。

## 训练设置

- 从原始`checkpoints/pi05_base_pytorch`独立初始化，seed=0，不从Paired/PH策略或历史MH checkpoint续训。
- 完整MH数据`data/paper/can_mh_image.hdf5`，官方划分为270条训练、30条验证，共56186/6570个时间点。归一化仅使用训练集。
- baseline没有动作条件投影、GT动作prompt或negative guidance。任务文本为`Pick up the can and place it in the bin.`。
- 8卡DDP，global batch=64，每卡8，梯度累积1，全参数float32微调，gradient checkpointing。沿用physical7 v4损失和真实状态tokenization。
- 2000次optimizer更新，warmup=1000，cosine总长=2000，峰值学习率2.5e-5，最低2.5e-6；AdamW的betas=(0.9,0.95)、weight decay=1e-10。
- 每500步保存checkpoint，每100步验证，每10步记录训练性能。最终保存`checkpoints/step_002000.pt`和`final_model.pt`。

## 自动评测

- 仅评测最终2000步checkpoint；不使用error detector或动作恢复。
- 8卡分配50个固定场景，分片数量为7/7/6/6/6/6/6/6，环境seed为100000至100049。
- 共用`outputs/evaluation/can_initial_states_50.json`，SHA256为`9f3a555765e617499370f26e8c7b8a85b260274bf70b6fed10ffd1548b9f246b`。
- 预测50步，每次执行8步，10步去噪；遵循MH评测协议，每回合最多500个物理步。
- 每个worker故障后最多重试2次并复用已完成回合；汇总检查所有场景和seed覆盖、checkpoint和归一化哈希、baseline开关及逐步物理轨迹。
- 自动生成`summary.json`、`episodes.jsonl`、`audit.json`、`REPORT.md`。

## 文件与监控

训练目录：`outputs/pi05_full_sft_physical7_v4/can_mh_baseline_seed_0_sealab1_20261103_204352_utc/`。

评测目录：`outputs/evaluation/mh_baseline_2k_8gpu_sealab1_20261103_204352_utc/`。

冻结的运行代码：`/2025233210/lmx/reach_can_training_mh_baseline_2k_20261103_204352_utc/`。该目录的scripts/configs/tests为启动时副本，数据、初始化权重和outputs为指向主项目的软链接。

训练目录下`pipeline_status.json`记录阶段、最新训练步数和自动评测进度，每10秒更新；`train.log`记录训练日志，`evaluation.log`记录评测控制器日志，`pipeline.log`记录管理进程异常。`experiment.json`、`preflight.json`、`source_hashes.json`、`expected_episode_split.json`和`expected_action_stats.json`保留配置和预检记录。快捷索引为`outputs/mh_baseline_2k_sealab1_latest.json`。

可复用入口为`scripts/train_can_mh_baseline_2k_8gpu.sh`和`scripts/train_evaluate_mh_baseline_2k.py`。本次启动前，训练失败阻止评测、错误权重阻止评测、正确权重衔接8卡baseline评测、权重配方校验及既有分片合并/重试检查共11项测试通过；sealab1真实仿真重置、双相机渲染和物理动作执行预检通过。

04:46北京时间已实际完成第10步：loss=0.399824，首个10步窗口3.427秒/步，峰值已分配显存55.72GiB。实际训练配置、episode划分和归一化统计已与预检记录逐项核对一致。后续进度以`pipeline_status.json`及`performance.jsonl`为准。
