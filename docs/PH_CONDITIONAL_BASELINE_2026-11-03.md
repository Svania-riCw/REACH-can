# PH conditional 2k训练与baseline的区别

2026-11-03 16:09:52 UTC启动一版PH策略训练，8卡DDP共同训练一个MLP conditional模型，目标2000个optimizer updates。从原始`checkpoints/pi05_base_pytorch`独立初始化，没有接着Paired模型训练。实际启动配置与来源见输出目录的`experiment.json`、`source_snapshot/`，运行状态见`training_status.json`。

输出目录：`outputs/pi05_full_sft_physical7_v4_mlp/policies/can_ph/conditional/seed_0/`。

可复现入口：`bash scripts/train_can_ph_mlp_2k_8gpu.sh`。每500步保存一个checkpoint，每100步验证并记录固定样本上的base/GT/foreign分支诊断，最终保存`checkpoints/step_002000.pt`和`final_model.pt`。这是训练任务，尚未运行这版模型的环境成功率评测。

启动后已核验实际保存的训练配置、官方划分、归一化统计与源码哈希，见输出目录的`startup_verification.json`。8张GPU均参与训练，实际日志确认`world_size=8`、`global_batch=64`、`micro_batch=8`；首批更新loss有限。新启动脚本通过`bash -n`，已有动作条件投影测试5项通过，覆盖MLP零初始化、梯度、checkpoint加载及参数校验。运行包装器会在进程退出后将`training_status.json`更新为completed或failed；当前进度以`performance.jsonl`和`train.log`为准。

## 与已有PH baseline对齐的设置

已有PH baseline为`outputs/pi05_full_sft_physical7_v4/can_ph_baseline_seed_0_20261103_063328/`，已经完成2000步。启动前确认新训练使用相同PH图像文件，官方训练/验证划分及重新计算的动作、状态归一化统计均与该baseline保存内容完全一致。

| 设置 | PH baseline 2k | 本次PH conditional 2k |
| --- | --- | --- |
| 数据 | `data/paper/can_ph_image.hdf5` | 同一文件 |
| 划分 | 官方180条训练、20条验证 | 相同，已核验 |
| 可用时间点 | 训练20883，验证2324 | 相同 |
| 初始化 | 原始Pi0.5 base | 相同 |
| 随机种子 | 0 | 0 |
| 有效全局batch | 64 | 64=8卡×每卡8×累积1 |
| 优化器 | AdamW，β=(0.9,0.95)，weight decay=1e-10 | 相同 |
| 学习率 | 峰值2.5e-5，最低2.5e-6 | 相同 |
| 学习率计划 | warmup1000，cosine总长2000 | 相同，显式设置2000 |
| 精度与显存 | float32，全参数微调，gradient checkpointing | 相同 |
| 任务文本 | `Pick up the can and place it in the bin.` | 相同 |
| 动作与损失 | horizon50，32维接口，仅前7维物理动作计loss | 相同 |
| 动作提示模块 | 无 | MLP动作提示投影 |
| 训练提示 | 无额外动作提示 | 按样本45%零Base、55%自身GT、0%shuffle |

本次2k是固定2000次更新，约6.13个PH训练epoch，并非50个epoch。以前Paired MLP的2k checkpoint来自总长13850步的学习率计划；本次为匹配现成PH baseline，使用2000步计划。因此与此前Paired模型比较时，数据和学习率计划均有变化，不能只归因于PH/Paired数据。

## conditional具体增加什么

baseline同样接收图像、机器人状态和任务文本；这里的conditional特指额外的**动作序列条件**，不是说baseline没有视觉或语言条件。二者都使用两路真实相机输入和一个被mask的零占位相机输入，状态采用checkpoint记录的真实维度tokenization。

baseline训练为从观测及带噪动作预测flow velocity。conditional在同一训练目标上增加一个动作提示`c`，提示形状为`[B,50,32]`：前7维是按训练集统计归一化的动作，后25维补零。每个样本独立抽签，45%使用全零Base提示，55%使用该样本自己未来50步GT动作；本次不使用打乱GT提示。两种提示的监督目标始终是该样本自己的GT动作，不改变目标或额外引入分类标签。

投影为逐时间步共享的`Linear(32,64) → ReLU → Linear(64,64) → ReLU → Linear(64,action_token_width)`。投影结果直接加到Pi0.5去噪动作token的suffix embedding上；没有把50步展平成一个大向量，也没有将这些数值作为任务文本拼接。最后一层权重及bias初始化为0，使提示模块初始输出为0，随后与整个策略一起训练。

两者都使用flow matching：归一化GT动作记为`a`，噪声为`ε`，`a_t=tε+(1-t)a`，目标速度为`ε-a`。baseline拟合`v(o,a_t,t)`，conditional拟合`v(o,a_t,t,c)`；损失都只对50步中的前7个物理通道取均方误差。conditional没有新增“失败分类loss”或负样本对比loss。VLM和动作expert均参与全参数微调，不是只训练新增MLP；未使用的expert语言输出head按既有流程冻结。

额外提示能让同一个conditional模型学习Base与动作提示两种分支。它提供了引导接口，但不保证成功率自动高于baseline，仍需同场景rollout比较。55%GT提示分支的loss也不能直接当作部署能力：部署时没有未来GT。

代码开关由`policy_mode='conditional'`决定。配置里`enable_action_conditioning=false`可能仍然出现，该字段是旧命令行兼容开关；不能据此认定本次未启用动作条件。baseline保存的`base_prob=0.3`同样是未参与计算的历史兼容参数。

## 推理与恢复

| 阶段 | baseline | conditional |
| --- | --- | --- |
| 普通推理 | 不传动作提示 | 传全零Base提示 |
| 观测与任务文本 | 正常使用 | 正常使用，相同格式 |
| 采样输出 | 50步预测，取前7维反归一化 | 相同 |
| 当前执行协议 | 每次执行8步再推理 | 相同 |
| 当前负向引导接口 | 现有Policy拒绝向baseline传failed prompt | 可同时计算Base与negative分支 |

当前恢复过程在下一次策略调用前检测，触发后伺服回退最近实际执行的8步。完整环境裁剪前预测`[50,7]`作为negative prompt，归一化并补零后输入相同MLP；引导速度为`v_guided=(1+S) v_base-S v_negative`。S=0关闭负向引导，但保留回退与重规划。恢复后的下一次引导调用完成后，普通调用仍使用Base提示。

error detector、动作回退和negative guidance属于评测时的外部恢复逻辑，不参与此次2000步的策略训练。本次PH指策略示范数据，也没有重训PH error detector。训练没有把检测到的失败动作标为负样本；把失败候选计划作为negative prompt用于推理，是现有恢复算法的用法。

全零Base分支也不等同于baseline：它经过训练后的MLP且与GT分支共同更新策略权重，投影bias可以变为非零。因此不能把“conditional的S=0”当成独立训练baseline的同义词。

主要实现位置：`scripts/train_pi05.py`中的`train_update`和`main`、`scripts/openpi_robomimic_adapter.py`中的数据归一化及样本构造、`scripts/eval_policy.py`中的`Policy.predict`，以及`../RoboTwin/XPolicyLab/policy/Pi_05/openpi/src/openpi/models_pytorch/pi0_experimental.py`中的`enable_action_conditioning`、`embed_suffix`和`build_action_condition`。
