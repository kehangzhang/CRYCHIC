# suggest_v5 缺口补齐与最新版单样本一致性报告

审计日期：2026-07-24

## 1. 结论

上一版审计列出的所有**可由已发布数据和现有统计合同估计**的缺口均已运行。结果不是“所有格子都有数值”：无法从公开资产估计、或与单样本设计不相容的端点继续明确记为 `NE`，数值未收敛的结果记为 `diagnostic_nonconverged`，没有补零或改写为成功。

核心结论如下。

1. 最新源码的单样本 `availability_state` 与已发布单样本结果逐行完全一致：CITE-seq 7 个数据集、IPF 56 位患者、HER2 CytoSig 1 个数据集，共 64 个数据集、4,757,604 行，键、数值和文本差异均为 0。因此同一 estimand 下既有 AUROC、AP 和排名不变。
2. 多组最佳分数头 RC12 依赖训练折产生的 sender assignment 和跨条件 receiver program；RC14 依赖多条件差异统计及空间 pair prior。二者对单个样本不可估，不能用 receptor expression 伪造。因此单样本一致性只检验当前源码中真正可估的 `availability_state`。
3. Tensor-cell2cell 扩展恢复指标、STACCato batch/校准/power、DCST 固定阈值二值投影、scACCorDiON Spearman/native/k-barycenter/Leiden/Friedman-Nemenyi 及 TCGA-PAAD survival 均已产生正式或明确标注为 diagnostic 的结果。
4. DCST 赛道的 CRYCHIC 固定二值投影是明确的负结果：平均 AUROC 约为 0.50，灵敏度为 0。它证明当前 `tau=0.25` projection 与 DCST 的二元链路真值不匹配，不应隐藏。
5. 患者图三队列 k-medoids 的平均名次中，CRYCHIC-compatible canonical Spearman 为 1/6，平均 rank 2.0；但只有 3 个队列，Friedman `p=0.1330`，不能声称统计显著优于 scACCorDiON。
6. PDAC 细胞下采样仍为有证据的 `NE`。作者发布 ZIP 只有患者级 LR 图和患者标签；作者 notebook 从其本机绝对路径读取未发布的细胞级 H5AD。对图表行做下采样不等价于对细胞做下采样，因而没有采用该替代。

正式结果根目录：

`/media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_v5_gap_completion_20260724`

## 2. 代码与 provenance

主要实现提交：

- `4d8427eec3b03f11ce97c995cb0fc1058a3f1449`：补齐 Tensor、STACCato、DCST、患者 native 距离、survival 和单样本一致性 runner。
- `9f758851c7af3ce98692bc8c5e6a7ed055fbbaf8`：复用已有 OT 距离、并行 k-barycenter，并记录 Sinkhorn 收敛状态。

两次正式运行时 worktree 均为 `dirty=false`。合同文件为 `benchmarks/configs/suggest_v5_gap_completion_v1.json`，SHA256：

`3171aa77ed96c91dce4a633e0acebb6a25956b013eda608d8508a1dd4375467a`

核心 runner：

- `benchmarks/comprehensive/complete_suggest_v5_posthoc.py`
- `benchmarks/comprehensive/run_staccato_gap_completion.py`
- `benchmarks/comprehensive/run_scaccordion_gap_completion.py`
- `benchmarks/comprehensive/run_native_patient_availability.py`
- `benchmarks/comprehensive/run_scaccordion_survival.py`
- `benchmarks/literature/compare_latest_single_sample.py`

## 3. 上一版缺口逐项关闭

| 上一版缺口 | 当前状态 | 结果或边界 |
|---|---|---|
| Tensor 扩展 C 指标 | complete | context scale-aligned RMSE/DTW/peak error、sender/receiver top-1 Jaccard、LR/event AUROC/MCC/P@K/R@K/F1/nDCG |
| Tensor CorrIndex | NE | 只有一个上游 score generator，没有可比较的成对 decomposition |
| Tensor 高阶/符号恢复 | NE | 冻结真值只有简单非负 LR event，不含多亚基超边或 signed truth |
| STACCato batch MSE/error SD | complete | 450/450 jobs，三种混杂设计、null 和 power grid |
| STACCato Type-I/FDR/CI/power | complete_diagnostic | 99 次 model-stage residual bootstrap；未重采样上游 score generation |
| STACCato formal full-pipeline p/q | NE | 上游生成未进入 bootstrap，不能把诊断性 p/q 称为正式校准 |
| DCST CRYCHIC fixed binary | complete | 475 realizations，预注册 `tau=0.25`；结果为负 |
| canonical Spearman | complete | PDAC、AKI、RCC fine/coarse |
| native CRYCHIC hyperedge distance | complete/NE | AKI、RCC fine/coarse complete；PDAC 因无细胞级输入为 NE |
| k-barycenter | complete/diagnostic | 主三队列收敛；RCC coarse correlation-OT 20/20 未达阈值，标记 diagnostic |
| Leiden resolution 0--1 | complete | 0.01 步长，所有已有距离 |
| Friedman-Nemenyi | complete_descriptive_low_n | 3 个 eligible 主队列，低功效，不作显著性胜负结论 |
| RCC annotation granularity | complete | fine 40 类与 coarse 7 类均含 canonical/native 和全部聚类后端 |
| PDAC cell downsampling | NE | 未发布作者细胞级 H5AD；禁止用 graph-row downsampling 代替 |
| TCGA-PAAD survival | complete | 作者冻结 scACCorDiON 数据及 top-10 Ductal2-to-Ductal1 LR，stage-adjusted Cox |

## 4. 最新源码单样本一致性

### 4.1 实际重跑

| track | 数据集 | 旧行数 | 新行数 | 未匹配 | 数值差异 | 文本差异 | 最大绝对差 |
|---|---:|---:|---:|---:|---:|---:|---:|
| CITE-seq | 7 | 653,647 | 653,647 | 0 | 0 | 0 | 0 |
| CytoSig HER2 | 1 | 2,692,500 | 2,692,500 | 0 | 0 | 0 | 0 |
| IPF | 56 | 1,411,457 | 1,411,457 | 0 | 0 | 0 | 0 |
| 合计 | 64 | 4,757,604 | 4,757,604 | 0 | 0 | 0 | 0 |

CITE-seq 与 IPF 的 63 个任务并行重跑 wall time 为 38.04 秒，63/63 成功；HER2 CytoSig 单独重跑约 22.6 秒。逐行比较过程另耗时 27.08 秒。结果 manifest 为 `single_sample_consistency/manifest.json`。

TNBC 的历史明细表仍存在，但生成它的 prepared H5AD 未保留，故当前源码重跑为 `NE/input_not_retained`，不进入 64 个 completed 数据集的分母。

### 4.2 与多组最佳版本的关系

| score head | 单样本状态 | 原因 |
|---|---|---|
| availability_state | complete | 只需要当前样本表达、细胞类型和冻结 LR/先验资源 |
| RC12 sender-response detection | NE | 需要 train-derived sender assignment 和跨条件 receiver program |
| RC14 differential DES | NE | 需要多条件 differential statistics 与 spatial pair prior |

因此“是否一致”的可证伪答案是：**可共同定义的 availability score 完全一致；RC12/RC14 在单样本问题上没有同一 estimand，不能比较数值一致性。** 因逐行 score 完全相同，既有单样本结果仍为：IPF AUROC 2/11、AP 1/11、balanced AUPRC 1/11；CITE-seq 和 CytoSig 各数据集的已发布 AUROC/AP 排名也保持不变。

## 5. Tensor-cell2cell 扩展指标

150 个冻结 factorization 全部后处理完成。举例：noise=0.01 时，context scale-aligned RMSE 为 0.01335，sender/receiver top-1 Jaccard 均为 1，LR 和完整 event 的 AUROC/MCC 均为 1。noise=1 时，LR AUROC/MCC 为 0.9996/0.9964，event AUROC/MCC 为 0.9999/0.9973。

noise=0 时 context RMSE 高于 noise=0.01，来自无噪声条件下等价 factor scaling/permutation 的非唯一性；报告的是匹配并 scale-aligned 后的诊断值，不据此选择模型。

## 6. STACCato 统计诊断

正式 campaign 为 450 jobs：3 个 condition/batch 设计、3 个 global-null 设计，以及 `n=20/40/60 x effect=0/0.25/0.5/1`，每格 25 replicates，每任务 99 次 bootstrap。48 workers 的 campaign wall time 为 117.81 秒，最大 worker RSS 约 282 MiB。

| design | disease MSE | batch MSE | active CI coverage | power q<0.05 | empirical FDR q<0.05 |
|---|---:|---:|---:|---:|---:|
| balanced | 0.001267 | 0.000944 | 0.7524 | 0.8544 | 0.1826 |
| moderate | 0.001447 | 0.000901 | 0.7583 | 0.7988 | 0.1892 |
| extreme | 0.002291 | 0.000898 | 0.6660 | 0.5694 | 0.2806 |

global null 下，名义 `p<0.05` 的 Type-I 分别为 0.0188、0.0349、0.0251，整体 CI coverage 为 0.9605、0.9350、0.9426。active CI coverage 和 empirical FDR 显示 model-stage bootstrap 并未达到可直接宣称 full-pipeline calibrated inference 的标准，故相应结果保留 diagnostic 标签。

## 7. DCST 固定二值投影

使用预注册阈值 `availability_state >= 0.25`，没有看结果后调参。

| sweep | 平均 AUROC | 平均 AUPRC | 方向准确率 | sensitivity q<0.05 | specificity q<0.05 |
|---|---:|---:|---:|---:|---:|
| receiver cells | 0.5000 | 0.2500 | 0.0020 | 0 | 1.0000 |
| subjects/group | 0.4893 | 0.2500 | 0 | 0 | 0.9874 |

该结果不支持 CRYCHIC 固定二值投影用于 DCST 的 binary-link estimand。此前 continuous arm 的低性能不能由简单阈值化修复。

## 8. scACCorDiON 患者图补齐

### 8.1 k-medoids 主队列结果

| cohort | CRYCHIC canonical Spearman ARI | CRYCHIC native ARI | scACCorDiON DW-OT ARI | 备注 |
|---|---:|---:|---:|---|
| PDAC | 1.0000 | NE | 1.0000 | native 输入未发布 |
| AKI | 0.2787 | 0.1188 | -0.0227 | 公开重建队列，不是作者受限对象 |
| RCC fine | 1.0000 | 1.0000 | 1.0000 | 40 cell types |
| RCC coarse | 0.7643 | 1.0000 | 0.7639 | 7 cell groups |

主三队列、六个全覆盖方法的平均 rank：

| rank | method | average rank |
|---:|---|---:|
| 1 | CRYCHIC-compatible canonical Spearman | 2.0000 |
| 2 | CRYCHIC-compatible canonical Pearson | 2.6667 |
| 3 | CORR-OT | 3.0000 |
| 4 | scACCorDiON DW-OT | 3.5000 |
| 5 | GOT | 4.3333 |
| 6 | Tabular PCA | 5.5000 |

Friedman statistic 为 8.4524，`p=0.1330`。所有 Nemenyi pairwise `p>0.19`。这只是 3 队列描述性排序；canonical projection 也不是 CRYCHIC 原生多组差异推断。

### 8.2 k-barycenter、Leiden 与数值状态

- PDAC：DW-OT k-barycenter ARI=1；correlation-OT ARI=-0.0332，40/40 method-start 组合均有收敛解。
- AKI：DW-OT/correlation-OT k-barycenter ARI=-0.0210/-0.0263；分别 19/20、20/20 starts 收敛。
- RCC fine：两者 ARI=1，均 20/20 starts 收敛。
- RCC coarse：DW-OT ARI=0.7639 且 20/20 收敛；correlation-OT ARI=0.5573，但 20/20 均未达到 Sinkhorn error 阈值，故为 diagnostic，不进入正式收敛结论。
- Leiden 已覆盖 resolution 0--1、步长 0.01。RCC fine native CRYCHIC Leiden ARI=1；AKI native CRYCHIC 在 protocol-selected true-k resolution 下 ARI=0.3319。

复用 checksum-bound 既有 OT 距离后，PDAC、AKI、RCC coarse/fine gap run 分别耗时 7.50、9.26、3.85、130.84 秒。RCC fine 较慢来自 1,600 维 cost 上 40 个并行 k-barycenter starts，不再重复计算患者两两 OT 距离。

## 9. TCGA-PAAD 生存验证

使用 scACCorDiON 作者冻结提交 `1850e472...` 的数据和 Ductal cell type 2 -> Ductal cell type 1 top-10 LR。177 位患者、93 个事件；有完整 stage 的 174 位进入 stage-adjusted Cox。

联合 LR geometric-mean 模型 concordance 为 0.6512，likelihood-ratio `p=0.000419`。两条 LR 在该模式内 BH `q<0.05`：

| LR | HR | p | q |
|---|---:|---:|---:|
| MMP7-SDC1 | 1.4073 | 0.005402 | 0.02701 |
| C3-CD81 | 1.9794 | 0.004537 | 0.02701 |

这些是作者候选 LR 的临床关联验证，不是 CRYCHIC 相对其他算法的独立 survival 排名，也不证明因果通信。

## 10. 最终边界

可表述：

- 可执行的上一版缺口已完成，且所有结果带 clean Git provenance 和 artifact checksum。
- 当前源码的单样本 availability 输出在 64 个可重跑数据集上逐行完全一致，原 AUROC/AP/排名不变。
- CRYCHIC-compatible canonical Spearman 在 3 个患者图队列的描述性平均 rank 为 1/6。

不可表述：

- “多组最佳 RC12/RC14 已在单样本上复现”，因为它们对单样本不可估。
- “所有 suggest_v5 端点都有数值”，因为 PDAC cell downsampling、full-pipeline STACCato calibration 和不具备真值的数据类型必须是 NE。
- “CRYCHIC 显著优于 scACCorDiON”，因为只有 3 个主队列，Friedman/Nemenyi 不显著，且 canonical projection 与 native differential inference 不同。
- “DCST 赛道得到改善”，因为固定阈值二值投影的正式结果接近随机且 sensitivity=0。

