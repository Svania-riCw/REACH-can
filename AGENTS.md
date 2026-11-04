# Error detector设置

- 用户要求所有后续error detector训练、Gaussian阈值校准和正式评测统一使用kappa=4。
- 历史其他kappa的权重和结果保留原记录；需要复用时另存明确标记kappa4的派生detector，并记录原始文件及其哈希。
- 2026-11-04用户明确要求本轮将kappa=4改为kappa=2重测：允许另存明确标记kappa2的派生detector及独立评测结果，记录原始kappa4文件和哈希，不覆盖历史结果。此次专项对照不改变其他任务的默认kappa=4设置。
