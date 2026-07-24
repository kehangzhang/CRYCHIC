# CRYCHIC 单组与多组 benchmark 简要汇总

更新日期：2026-07-24

版本边界：压缩包中的源码快照来自 `optimize/suggestions-next-m0-20260724`，精确 commit 记录在
`BUNDLE_METADATA.json`；最新核心算法变更为 `209e6a3`。两个已执行 notebook 及其独立图表是此前
冻结的正式 benchmark 结果；单组 notebook 最后更新于 `055c67d`。当前源码新增的 M0
absolute-activity 输出头仅通过开发集门禁，尚未进入跨算法冻结验证，因此不纳入下述正式名次。

## 1. 压缩包内容

本压缩包是便于传阅的紧凑快照，不包含原始单细胞数据、完整运行目录、软件环境或大体积中间矩阵。

- `notebooks/single_group/literature_benchmark_results.ipynb`：单组文献 benchmark，14/14 个代码 cell 已执行，内嵌 9 张图。
- `notebooks/multigroup/suggest_v5_under100k_benchmark_results.ipynb`：多组 / multi-context benchmark，10/10 个代码 cell 已执行，内嵌 7 张图。
- `figures/single_group/` 与 `figures/multigroup/`：上述 notebook 对应的 16 张独立 PNG；为控制体积，不重复保存 PDF/SVG。
- `source/src/crychic/`：当前分支完整核心源码，排除 `__pycache__` 和 `.pyc`。
- `source/benchmarks/`：benchmark Python/R runner、adapter、metric、report builder 和冻结配置；不含大结果目录。
- `reports/`：本汇总、最新版 bounded-evidence 报告、suggest_v5 缺口补齐报告及结构化 JSON 摘要。
- `results_compact/`：单样本一致性、scACCorDiON 跨队列平均排名等小型结果表。
- `BUNDLE_METADATA.json` 与 `BUNDLE_FILE_INDEX.tsv`：Git provenance、文件大小和 SHA256。

## 2. 单组 benchmark

当前单组 notebook 覆盖：

1. 7 个 CITE-seq receptor-protein benchmark。
2. TNBC 与 HER2 两个 CytoSig 下游响应 benchmark。
3. 4 个研究、56 位患者的完整 IPF 队列。
4. 4 类扰动下的 Top-250 稳健性。
5. 最新源码逐行一致性复跑。

截至单组 notebook 的冻结提交，`availability_state` 在可重跑的 7 个 CITE-seq、56 个 IPF 和 1 个 HER2 数据集上与旧结果完全一致：64 个数据集、4,757,604 行，未匹配、数值差异和文本差异均为 0。因此既有单组 AUROC/AP 排名不变。IPF 中 CRYCHIC 为 AUROC 2/11、AP 1/11、balanced AUPRC 1/11。

多组 RC12/RC14 不能直接套用到一个样本：RC12 需要训练折生成的 sender assignment 和跨条件 receiver program；RC14 需要多条件差异统计和空间 pair prior。这两项在单样本分析中是 NE，不用 receptor expression 伪造替代。

## 3. 多组 benchmark 范围

当前已运行的多组 / multi-context 赛道包括：

- 三组 implanted-truth event detection、contrast localization、方向和效应恢复。
- BRCA 2 x 2 semisynthetic canonical event benchmark、随机 global-null 校准和 native hypergraph truth。
- Kuppe 心肌梗死与 Lerma-Martin MS 空间 DES，包括 pair-rank、original-count、Top-K count-matched、continuous weighted 和机制分层端点。
- scACCorDiON PDAC、AKI、RCC 患者图表型恢复，包括 canonical Spearman、native CRYCHIC hyperedge distance、k-medoids、k-barycenter 和 Leiden。
- DCST subject/cell-count power sweep。
- STACCato condition/batch、null、置信区间和 power 诊断。
- Tensor-cell2cell 12-context program recovery。
- 10 个 score generator x 5 个 differential engine 的 component crossover。

不同赛道的 estimand 不同，不能把所有名次合成一个“总冠军”。下表只在各自合法比较面板内给名次。

## 4. CRYCHIC 当前名次

| benchmark 赛道 | CRYCHIC 版本 / 输出头 | 主指标 | CRYCHIC 名次 | 主要结论 |
|---|---|---|---:|---|
| 最新三组同 seed implanted truth | RC12 unsigned detection | prevalence-adjusted AP / AUPRC | **2/5** | scSeqCommDiff 1，RC12 2，CellChat 3，generic CRYCHIC 4，LIANA 5 |
| Kuppe pair-rank spatial DES | RC11 | median DES | **2/15** | 0.6287，低于 scSeqCommDiff 0.7008；Kuppe 是开发集 |
| MS pair-rank spatial DES | RC11 | median DES | **1/9** | 0.8330，高于 scSeqCommDiff 0.8250，8 strata 中胜 5 个；该 endpoint 不是原论文 event-count DES |
| Kuppe strict native original-count DES | 最新 directed LR ledger | median DES | **2/2** | 0.3869，低于 scSeqCommDiff 0.7008 |
| MS strict native original-count DES | 最新 directed LR ledger | median DES | **2/2** | 0.3000，低于 scSeqCommDiff 0.8250 |
| 三组 native end-to-end 旧 generic 面板 | generic CRYCHIC | adjusted AP / omnibus AUPRC | **3/4** | scSeqCommDiff 1，CellChat 2，CRYCHIC 3，LIANA 4；RC12 是该 generic 输出的后续改进 |
| 患者图三队列 | canonical event Spearman | true-k ARI 平均 rank | **1/6** | 平均 rank 2.0；Friedman `p=0.133`，不支持显著优于 scACCorDiON |
| DCST binary-link truth | fixed binary `tau=0.25` | AUROC/AUPRC | **2/2** | AUROC 约 0.50、sensitivity 0，明确负结果 |
| component crossover | native score + sample GLM | mean AUPRC | **10/27** | native + clustered CR2 为 11/27，native + STACCato 为 21/27 |
| synthetic native hypergraph truth | native hyperedge | exact hyperedge AP | **1/4** | exact AP 1.0；但 partial canonical-LR AP 为 3/4，优势来自更严格的高阶特异性 |

### 最新 RC12 的同 seed 五方法结果

| rank | 方法 | adjusted AP | AUPRC | AUROC | localization AP |
|---:|---|---:|---:|---:|---:|
| 1 | scSeqCommDiff | 0.2078 | 0.1451 | 0.7115 | 0.1329 |
| 2 | CRYCHIC RC12 | 0.1912 | 0.1392 | 0.5480 | 0.1214 |
| 3 | CellChat | 0.1301 | 0.0884 | 0.4956 | 0.0772 |
| 4 | CRYCHIC generic baseline | 0.0767 | 0.0509 | 0.1317 | 0.0502 |
| 5 | LIANA | 0.0757 | 0.0502 | 0.0972 | 0.0459 |

RC12 相对 scSeqCommDiff 的 AUPRC 差为 -0.0059，配对 bootstrap 95% 区间为 [-0.0356, 0.0346]，当前样本量下未分出差异；AUROC 差为 -0.1635，区间完全低于 0。因此可表述“RC12 超过 CellChat 和 LIANA，并在 AUPRC 上接近 scSeqCommDiff”，不能表述“全面超过 scSeqCommDiff”。

### RC14 strict Top-K DES

RC14 是在 Kuppe 上冻结的 benchmark-only Top-K head。它在 K=100 时于 Kuppe 和 MS 均高于 scSeqCommDiff；按 K=100/250/500/1000 的 mean DES 平均，Kuppe 为 0.7525 对 0.7208，MS 为 0.6716 对 0.6766。它没有在所有 K 上占优，native original-count 和 continuous-weighted DES 仍低于 scSeqCommDiff，且 MS 已在早期 RC11 工作中被查看过，不能作为 RC14 的全新独立验证。

## 5. 不能直接排名的赛道

- Tensor-cell2cell 扩展只有一个原生 factorization 方法，没有合法算法间排名。
- STACCato benchmark 的主要比较是 STACCato 与 eventwise OLS；CRYCHIC common-functional OOF 因独立 subject groups 不满足 paired contract 而为 NE。
- BRCA complete-replicate panel 中 CRYCHIC AUPRC 0.9292、scSeqCommDiff 0.9002，完成预注册 noninferiority；但 CellChat 在 19 个有限重复上更高且有 1 个 coverage failure，因此不报告统一全方法名次。
- scACCorDiON 的 CRYCHIC canonical projection 是患者图向量表示，不等于 CRYCHIC 原生多组差异推断。
- 方向保持空间 DES 因空间 truth 为 unordered pair 而 NE；严格 long-range DES 因 ConnectomeDB2020 没有传播距离注释而 NE。

## 6. 总体判断

目前最强、证据边界清晰的结论是：

1. RC12 在同 seed 三组检测面板中为 2/5，超过 CellChat、LIANA 和旧 generic CRYCHIC；AUPRC 与 scSeqCommDiff 尚未分出差异，但 AUROC 明显较低。
2. RC11 在独立 MS pair-rank spatial endpoint 为 1/9，但在开发集 Kuppe 为 2/15；这不是对所有多组任务的普遍胜出。
3. strict native event-count 与 continuous-weighted DES 仍落后于 scSeqCommDiff；RC14 只在部分 Top-K 预算上领先。
4. CRYCHIC 在 synthetic exact hypergraph truth 为 1/4，说明高阶表示有明确模拟优势，但尚不能替代真实数据验证。
5. 当前 RC12/RC14 均是 benchmark-only 输出头，未替换 public default；formal p/q 仍禁用。整体状态是“若干赛道竞争力明显提高，但尚未形成跨 estimand 的全面第一”。
