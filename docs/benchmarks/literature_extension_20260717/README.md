# 文献 benchmark 扩展结果（2026-07-17）

> GitHub 发布副本: 仅包含汇总表、可审计的小型结果表和清理后的发布证明。原始 H5AD、Parquet、RDS、XLSX、日志、任务命令及机器绝对路径均未纳入。

## 结论摘要

- **CITE-seq 已覆盖文献 7/7 数据集**：原 4 个 10x 数据集，加上 CBMC、SLN111、SLN208。H-common 与 native 是不同资源和全集，不能混合排名。新增数据中，CBMC native 的 CRYCHIC 并非最佳（AUROC **8/11**、AP **4/11**）；mouse H-common/resource-fixed 下，SLN111 AUROC **1/9**，SLN208 **2/9**，是一胜一负。
- **CytoSig 相关数据已由 1 个扩展到文献实际使用的 2/2**：TNBC 与 HER2。HER2 用 19,275 cells、5 patients、25 target cell types 重建 43-signature CytoSig truth；受控 unique ligand-target 上 CRYCHIC AUROC/AP/balanced AP 为 **0.639/0.125/0.584**，AUROC 1/12。其 truth 状态是 `reconstructed_clean_activity_truth_not_historical_exact`，不是历史逐位复现。
- **IPF 已跑完整队列**：4 GEO studies、56 patients、138,248 cells；56 CRYCHIC score、56 LIANA score、56 evaluation，共 **168/168 tasks complete**。患者等权下 CRYCHIC AUROC **0.585**（**2/11**，不是第一），AP **0.178**（1/11），balanced AUPRC **0.652**（1/11）。
- 这些任务验证的是单条件静态通信排序、受体蛋白一致性或 CytoSig 下游响应一致性，**没有因此实现或验证多组别差异通讯分析**。CytoSig 是有价值的下游通路/转录响应旁证，但不等同于配体-受体结合或因果通信证明。

## CITE-seq：7/7

主结果见 [CITE-seq 汇总表](tables/citeseq_extension_summary.tsv)。7 个数据集均完成原始对象准备、ADT truth 构建和可用算法运行；新加入 CBMC 7,713 cells/12 clusters、SLN111 15,802/28、SLN208 15,049/28。truth 仍按作者协议：cluster ADT CLR mean 跨 cluster z-score，`z >= 1.645` 为阳性。

| 数据集 | 物种 | H-common/resource-fixed | native/independent |
| --- | --- | --- | --- |
| CBMC | human | **NE**：14 个 ADT truth receptor 与冻结 human H-common 重叠为 0 | CRYCHIC AUROC 0.591（8/11），AP 0.280（4/11） |
| SLN111 | mouse | AUROC 0.626（1/9），AP 0.199（4/9） | AUROC 0.735（1/9），AP 0.407（1/9） |
| SLN208 | mouse | AUROC 0.541（2/9），AP 0.122（1/9） | AUROC 0.723（1/9），AP 0.359（1/9） |

`NE` 是预先定义的不可评估状态，不应记为 0 分或运行失败。mouse H-common 是 LIANA/CellChat mouse 交集的 705 LR，不是 human 638-LR H-common；mouse native LIANA 为 4,015 LR，CellChat-native 为 3,379 LR。SLN mouse 的 CellPhoneDB 因无可用 mouse arm 跳过。表内 native 结果也显示资源选择会明显改变名次，因此不能用 native 结果声称跨算法资源受控的全面领先。

复核入口：[CBMC native 指标](citeseq/metrics_by_dataset.tsv)、[CBMC H-common NE manifest](citeseq/manifest.json)、[SLN111 H-common 指标](citeseq/metrics_by_dataset.tsv)、[SLN208 H-common 指标](citeseq/metrics_by_dataset.tsv)。

## CytoSig：TNBC + HER2（2/2）

文献发布的 Figure 6A source data 只有 Wu GSE176078 的 TNBC 与 HER2 两个 dataset arm；本轮补上 HER2，因此不再是“只用一个 CytoSig 数据集”。详情见 [CytoSig 汇总表](tables/cytosig_extension_summary.tsv)。

HER2 原始 19,311 cells/29 minor types，经作者 `>=25 cells/type` 规则保留 19,275 cells、5 patients、25 target types。使用 checksum 固定的 43-signature centroid，按 cell type raw-count pseudobulk、检测率/总量过滤、每个 signature 绝对权重 top-500、multivariate linear model，并在唯一 43 x 25 activity universe 上一次性 BH。所得 1,375 个 ligand-target truth rows 中 117 为阳性。

| HER2 终点 | CRYCHIC availability-state | 相对位置 |
| --- | ---: | ---: |
| controlled unique ligand-target AUROC | 0.638600 | 1/12 |
| controlled AP / balanced AP | 0.125084 / 0.584237 | AP 1/12 |
| current substitute top-250 OR（author top-vs-total） | 1.979049，primary-8 BH q=2.44e-4 | 1/8 |

top-250 union 有 168,750 rows，并由 CRYCHIC 的广覆盖主导；因此必须与 coverage-controlled endpoint 同看。TNBC 与 HER2 truth 也不同：TNBC 是 OpenProblems 冻结 binary proxy，HER2 是本轮 clean reconstruction。历史 LIANA 0.0.5/OmniPath、原 processed Seurat object、旧版本 multiplicity 未冻结；Crosstalk 当前 runner 不可用，且当前 LIANA 为 100 而非论文 1,000 permutations。故 [官方 Figure 6A reference](cytosig_her2/manifest.json) 仅用于参考校验，不能与当前方法数值直接排行。

复核入口：[HER2 报告](cytosig_her2/README.md)、[truth manifest](cytosig_her2/manifest.json)、[ranking metrics](cytosig_her2/ranking_metrics.tsv)、[Fisher curves](cytosig_her2/fisher_rank_curves.tsv)。

## IPF：完整 4-study 队列

[任务 ledger](ipf/manifest.json) 为 168/168 complete：每位患者各 1 个 CRYCHIC score、1 个 LIANA score 和 1 个 H-common-representable evaluation。队列组成如下。

| Study | patients | cells |
| --- | ---: | ---: |
| GSE122960 | 4 | 10,413 |
| GSE128033 | 8 | 20,627 |
| GSE135893 | 12 | 28,778 |
| GSE136831 | 32 | 78,430 |
| **总计** | **56** | **138,248** |

患者等权主结果见 [IPF primary metrics](tables/ipf_primary_metrics.tsv)：

| 方法 | AUROC | AP | balanced AUPRC |
| --- | ---: | ---: | ---: |
| CellPhoneDB composite | **0.588123** | 0.121009 | 0.583044 |
| CRYCHIC availability-state | **0.585464** | **0.177687** | **0.651724** |
| SingleCellSignalR LRscore | 0.581509 | 0.129119 | 0.591687 |

因此 CRYCHIC 的 IPF AUROC 是第二，不能写成全面第一；其优势主要在 AP 与 class-balanced AUPRC。按 study 拆分则存在明显异质性（[完整表](tables/ipf_crychic_by_study.tsv)）：CRYCHIC AUROC 在 GSE122960/GSE128033 为 **0.769/0.782**，在 GSE135893/GSE136831 为 **0.531/0.534**。56-patient、study-stratified 2,000 次 bootstrap 的 CRYCHIC 95% descriptive intervals 为 AUROC **0.564--0.609**、AP **0.160--0.196**、balanced AUPRC **0.635--0.670**；它们描述患者异质性，不是因果或生物学推断。

最重要的限制是 IPF gold 为 disease-level static curated presence/upregulation，不是 patient-specific truth，也不是 IPF-vs-control 差异 truth。其 250 个唯一 positive STLR 外的未标注 complement 是**开放世界未标注项**，不是实验验证 negatives；因此 specificity/MCC 使用的是 operational pseudo-negatives。当前 H-common representable universe 每患者为 960 rows、53 positives，亦不同于论文 union-all-method-predictions 的 100,352-row 历史全集。论文中的 CCInx、iTALK、scMLnet 等旧环境/输出本轮不可用而跳过；NicheNet 输出 ligand-to-target regulatory potential，不强行映射为 direct STLR 排名。

复核入口：[cohort summary manifest](ipf/manifest.json)、[patient-equal full table](ipf/cohort_patient_equal.tsv)、[study table](ipf/metrics_by_study.tsv)、[bootstrap table](ipf/patient_bootstrap.tsv)、[gold audit](ipf/manifest.json)。

## 跳过项、可比性与资源

- 可用的 CRYCHIC 与当前 LIANA/CellChat component arms 已运行；不可用算法按要求跳过。CITE-seq mouse CellPhoneDB、CytoSig Crosstalk，以及 IPF 旧独立 CCInx/iTALK/scMLnet 没有伪造替代结果。
- “CellChat composite”“CellPhoneDB composite”等是当前 LIANA component arms；除明确标注的 mouse CellChat-native 外，不应写成所有历史 standalone 软件的逐版本复现。
- IPF scheduler 使用 24 workers、每 task 1 thread，60% soft gate、70% hard gate；单任务最大 RSS 为 **2,324,107,264 bytes = 2.1645 GiB**。新增 CITE 成功任务中，`GNU time -v` 最大单进程 RSS 为 SLN111 native evaluation 的 **4,977,220 KiB = 4.747 GiB**。HER2 cgroup sampled peak 为 **8.77 GiB**。
- 本轮系统内存 spot-check 最高约 **60/251 GiB（约 24%）**，IPF ledger 记录的 task-start memory fraction 最高 **21.6002%**，均低于 70%。这些是离散观察，不是连续全局峰值，不能据此声称整个运行期间的精确峰值。
- IPF queue 最多 24-way 并行，wall time 1:54.94；ledger 中 165 tasks 实际执行一次成功，3 个任务复用已校验产物（attempts=0），最终 168/168 complete。摘要文件当前 checksum 已绑定 56 个 evaluation task manifests 与完整 ledger。

## 解释边界

本轮回答了“静态预测是否与 receptor protein、CytoSig response、IPF disease-level literature gold 一致”。它没有提供多 condition design matrix、subject-aware contrast、差异通信效应量或下游 pathway 的跨组变化检验，所以**不能把这些结果当作多组别细胞通讯和差异比较已实现的证据**。同样，CytoSig 只支持下游响应一致性；更强的可靠性论证仍需在独立队列中联合 ligand/receptor availability、receiver pathway activity、时间顺序和干预证据。

本目录的 [manifest](manifest.json) 固定所有报告表和关键上游证据的 SHA256；生成时未修改旧报告或共享 README。
