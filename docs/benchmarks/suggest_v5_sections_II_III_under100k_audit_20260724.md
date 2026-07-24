# suggest_v5 第二、三部分 `<100k` benchmark 落实审计

审计日期：2026-07-24

## 1. 结论

结论不是简单的“全部完成”或“未完成”，而应分两个层级表述。

1. **冻结主任务层面：已完成。** 严格按“观察到的单细胞数 `<100,000`”合同，共冻结 9 个任务；9/9 均有 `status=complete` 的正式结果、干净 Git provenance 和校验和。Section II 的 Tensor-cell2cell、STACCato、两个 DCST sweep，Section III 中符合规模条件的 PDAC、AKI、RCC，以及后续同名 Section II/III 的 signed-estimand contract 和 component crossover 均已运行。
2. **按 `suggest_v5.md` 字面逐条的完整协议层面：尚未全部落实。** 当前结果覆盖主模型、主要端点和所有符合合同的数据队列，但若把建议稿中的扩展 C 指标、全部聚类后端、鲁棒性分析和临床验证也计入“所有 benchmark”，则仍有明确缺口。主要缺口包括 Tensor 扩展指标、STACCato batch-effect/校准指标、DCST 的 CRYCHIC 固定阈值二值投影、scACCorDiON 的 k-barycenter/Leiden、原生超图距离、PDAC 细胞下采样和 TCGA 生存验证。
3. **后续 Section II 的语义合同已完整通过。** 15/15 个手算 case 和 11/11 个 invariance test 通过；但正式 P 值、置信区间覆盖和全流程重采样校准保持 NE，没有用诊断性 CR2 量替代正式推断。
4. **后续 Section III 的 10 x 5 交叉矩阵已建立，但不是 50/50 全部可估。** 50 个格中 27 个完整观察、2 个部分观察、21 个 NE。NE 来自设计或公开 API 约束，不是零分数。
5. **当前证据不支持“CRYCHIC 全面优于已有方法”。** native 三组端到端比较中 CRYCHIC 为 3/4；component crossover 中 `crychic_native + sample_glm` 为 10/27。RCC fine 和 PDAC 患者图上的 CRYCHIC-compatible projection 可达 ARI=1，但该投影不是 CRYCHIC 原生多组差异推断，且 RCC major annotation 下 ARI 只有 0.236615。

因此，对“第二、三部分是否全部落实”的最准确回答是：**冻结的 `<100k` 主任务全部落实；建议稿的完整字面协议只部分落实。**

## 2. 审计口径

`suggest_v5.md` 中存在两组重新编号的“第二、三部分”，本报告同时审计：

- 主 benchmark 设计的 Section II：Tensor-cell2cell、STACCato、DCST。
- 主 benchmark 设计的 Section III：scACCorDiON 七队列中的 `<100k` 队列。
- 后续诊断建议的 Section II：signed estimand contract 和三组分解指标。
- 后续诊断建议的 Section III：score generator x differential engine component crossover。

冻结合同为 [suggest_v5_under100k_v1.json](../../benchmarks/configs/suggest_v5_under100k_v1.json)，SHA256 为 `6a93f2427cd9057cc557568fb5d35e5325022333b5fe2e0264871d48cb07c26a`。

源建议文档为 `/media/subunit/bioinfo/crychic_dev/suggest_v5.md`，SHA256 为 `13bdfc24d4fcf2678567a56b22dd8767493d5e5998ebb389c35b8e1a7c0cfe84`。

纳入规则：

- 观察到的单细胞数必须严格小于 100,000。
- 100,000 个细胞也不纳入。
- 大队列不能通过临时下采样改变其原始 eligibility。
- 纯 score/tensor 合成任务记为 0 个观察单细胞。
- 缺失结构、不可比较接口和不满足设计合同的结果记为 NE，不补零。

Section III 因规模被排除的四个 scACCorDiON 队列为：COVID-19 647,366、MI 132,888、breast cancer 714,331、lung atlas 941,504 个细胞。

## 3. 冻结任务完成状态

| section | benchmark | 规模 | 正式重复/样本 | fidelity | 任务状态 |
|---|---|---:|---:|---|---|
| 主 II | Tensor-cell2cell 12-context | 合成张量 | 6 noise x 25 = 150 | 论文协议重建，非字节级复现 | complete |
| 主 II | STACCato condition/batch | 合成 score tensor | 3 designs x 100 | 论文设计及公开低秩生成器重建 | complete |
| 主 II | DCST subject sweep | 10,000-90,000 cells | 9 x 25 = 225 | 论文原生独立 sweep 重建 | complete |
| 主 II | DCST receiver sweep | 31,000-40,000 cells | 10 x 25 = 250 | 论文原生独立 sweep 重建 | complete |
| 主 III | scACCorDiON PDAC | 57,530 cells | 35 patients | 作者处理后的患者图 | complete |
| 主 III | scACCorDiON AKI | 76,020 cells | 36 patients | 公开数据设计匹配重建，非作者受限子集 | complete |
| 主 III | scACCorDiON RCC | 50,236 cells | 17 patients | 精确公开原始 cohort 重建；fine + major | complete |
| 后续 II | signed estimand contract | 确定性 fixture | 15 cases + 11 invariants | 原生公开设计/OOF API | complete, Gate A PASS |
| 后续 III | score-engine crossover | 合成 score universe | 20 active + 20 null seeds | 同 seed、同事件 universe | complete, 含 NE |

这里的 `complete` 表示冻结任务已完整产出，不表示建议稿中每一个可选扩展都已执行。

## 4. 主 Section II 结果

### 4.1 Tensor-cell2cell

正式运行：150/150 成功，24 workers，campaign wall time 526.006 秒。

| noise | context Pearson | context Spearman | LR Jaccard@100 | LR AUPRC | sender top-1 | receiver top-1 | event AUPRC | NRE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.672941 | 0.713270 | 1.000000 | 1.000000 | 0.900 | 0.930 | 0.894970 | 0.096909 |
| 0.01 | 0.998592 | 0.970178 | 1.000000 | 1.000000 | 1.000 | 1.000 | 1.000000 | 0.000136 |
| 0.10 | 0.998095 | 0.970178 | 1.000000 | 1.000000 | 1.000 | 1.000 | 1.000000 | 0.008232 |
| 0.25 | 0.976850 | 0.951356 | 0.990309 | 0.992429 | 0.990 | 0.990 | 0.986114 | 0.048820 |
| 0.50 | 0.972685 | 0.954613 | 0.989864 | 0.992323 | 0.990 | 0.990 | 0.987242 | 0.149018 |
| 1.00 | 0.962151 | 0.959137 | 0.995275 | 0.999510 | 1.000 | 1.000 | 0.999365 | 0.290819 |

`noise=0` 的 context 指标低于加入极小噪声后的结果，原因是精确零张量存在 CP 分解不可识别性；这不是数据加载失败。此 benchmark 只有 Tensor-cell2cell 一个原生分解方法，因此不能给出算法间排名。

已实现：Hungarian factor matching、context Pearson/Spearman、LR Jaccard/AUPRC、sender/receiver top-1、event AUPRC、NRE。

未实现：CorrIndex、多上游 score generator 间分解一致性、context RMSE/DTW/峰值误差、LR AUROC/MCC/Precision@K/Recall@K/nDCG、sender/receiver loading Jaccard、event MCC/F1、严格或部分超边恢复。后几项属于建议稿的扩展 C，而不是当前冻结合同端点。

### 4.2 STACCato condition/batch simulation

100 个随机重复全部完成；12 workers；campaign wall time 26.337 秒。下表比较协变量调整后的方法。

| design | method | disease MSE | RMSE | sign accuracy | opposite direction | top-100 recovery |
|---|---|---:|---:|---:|---:|---:|
| balanced | eventwise OLS | 0.001495 | 0.038660 | 0.849100 | 0.150900 | 0.600600 |
| balanced | STACCato | 0.001259 | 0.035471 | 0.994475 | 0.005525 | 0.361200 |
| moderate | eventwise OLS | 0.001686 | 0.041056 | 0.842525 | 0.157475 | 0.598200 |
| moderate | STACCato | 0.001410 | 0.037546 | 0.989050 | 0.010950 | 0.228600 |
| extreme | eventwise OLS | 0.005235 | 0.072340 | 0.759675 | 0.240325 | 0.502000 |
| extreme | STACCato | 0.002233 | 0.047194 | 0.890075 | 0.109925 | 0.031800 |

STACCato 的 disease-effect MSE 相对 adjusted OLS 分别下降约 15.8%、16.4% 和 57.3%，方向准确率在三个设计中均为第 1/2；但 top-100 recovery 均低于 OLS，为第 2/2，说明低秩收缩改善整体误差和符号，不等价于改善极端 top-k 排名。

CRYCHIC common-functional OOF 在该论文式独立 subject group 设计上为 NE，因为公开 OOF contrast 要求同一 subject 跨 context；没有伪造配对或把 NE 当作零效应。

已实现：disease MSE/RMSE、非零效应方向、反向率、top-100、失败率、运行时间和内存。

未实现：batch-effect MSE、逐事件估计误差 SD，以及扩展 C 的 null Type-I、经验 FDR、置信区间覆盖/宽度、power-effect-size 和 power-sample-size 曲线。冻结的 moderate/extreme 样本矩阵来自本次协议重建合同，也不是作者原随机 realization 的字节级重跑。

### 4.3 DCST 两个独立 sweep

两个 sweep 未被错误扩展成 Cartesian grid。共 475/475 个 realization 完成，24 workers，campaign wall time 122.573 秒。

| sweep | method | mean AUROC | mean AUPRC | direction accuracy | sensitivity@0.05 | specificity@0.05 |
|---|---|---:|---:|---:|---:|---:|
| receiver cells 50-500 | DCST | 0.999833 | 0.999333 | 1.000000 | 1.000000 | 0.988667 |
| receiver cells 50-500 | CRYCHIC generic continuous | 0.264333 | 0.250000 | 0.000000 | NE | NE |
| subjects/group 5-45 | DCST | 0.998704 | 0.997593 | 0.997778 | 0.924444 | 1.000000 |
| subjects/group 5-45 | CRYCHIC generic continuous | 0.311481 | 0.250000 | 0.000000 | NE | NE |

在 AUROC、AUPRC 和方向准确率上，DCST 均为第 1/2，CRYCHIC generic synthetic-prior baseline 为第 2/2。这个结果诊断的是当前合成 prior/score 与 DCST 二值链路真值之间的 estimand mismatch，不能外推为真实组织中的通用排名。

DCST 的平均原生 null Type-I error 分别为 0.011333（receiver sweep）和 0（subject sweep），平均经验 FDR 为 0.022667 和 0。

限制：

- 原论文 subject sweep 到 55/group；严格 `<100k` 仅保留 5-45/group，50/group 等于 100,000 cells，55/group 超过限制。
- 本次每点 25 个重复，高于原建议的 10 个初始化。
- 当前完成的是 DCST fixed-expression-threshold binary estimand 和 CRYCHIC generic continuous score；合同方法列表中的“CRYCHIC fixed-threshold binary projection”尚未形成预注册阈值后的正式结果。
- CRYCHIC continuous arm没有可比的正式 P/Q 值，因此 sensitivity、specificity、Type-I 和 FDR 为 NE，而不是零。
- 论文模拟中的“subject”来自 pooled cells bootstrap，是算法层 pseudo-subject，不是生物学供体。

## 5. 主 Section III：患者图 benchmark

### 5.1 数据和 fidelity

| cohort | cells | samples | labels | annotation | 数据 fidelity |
|---|---:|---:|---:|---|---|
| PDAC | 57,530 | 35 | 2 | 10 types | 作者仓库处理后的患者图，exact graph asset |
| AKI | 76,020 | 36 | 3 | 13 major types | 当前 CellxGene 公开对象重建；不是作者 restricted Zenodo subset |
| RCC | 50,236 | 17 | 2 | 40 fine types | GSE242299 完整公开 H5AD，exact cohort |
| RCC major | 同一 50,236 | 同一 17 | 2 | 7 `cell_group` | 同一 cohort 重新生成患者图 |

RCC 原始文件 SHA256 为 `8539723bf40f3d8647c9773c386db56256b305f69e0cdf80848cec5bae428f94`。17 个样本全部完成图构建；fine 和 major 分别耗时 37.811 秒和 12.714 秒，无失败。

### 5.2 ARI 结果

| cohort | method | ARI at true k | max ARI over k | phenotype silhouette | true-k rank |
|---|---|---:|---:|---:|---:|
| PDAC | CORR-OT | 1.000000 | 1.000000 | 0.318994 | 1-3 tie |
| PDAC | CRYCHIC projection | 1.000000 | 1.000000 | 0.320373 | 1-3 tie |
| PDAC | GOT | 0.883622 | 0.883622 | 0.410508 | 4 |
| PDAC | scACCorDiON DW-OT | 1.000000 | 1.000000 | 0.426728 | 1-3 tie |
| PDAC | Tabular PCA | 0.343289 | 0.525498 | 0.352738 | 5 |
| AKI | CORR-OT | -0.014199 | 0.058170 | -0.029527 | 3 |
| AKI | CRYCHIC projection | -0.008445 | 0.200379 | 0.047162 | 2 |
| AKI | GOT | 0.104826 | 0.104826 | -0.367656 | 1 |
| AKI | scACCorDiON DW-OT | -0.022716 | 0.084892 | 0.010905 | 4-5 tie |
| AKI | Tabular PCA | -0.022716 | -0.001046 | -0.022123 | 4-5 tie |
| RCC fine | CORR-OT | 1.000000 | 1.000000 | 0.663138 | 1-3 tie |
| RCC fine | CRYCHIC projection | 1.000000 | 1.000000 | 0.276290 | 1-3 tie |
| RCC fine | GOT | -0.013245 | 0.658515 | 0.253775 | 5 |
| RCC fine | scACCorDiON DW-OT | 1.000000 | 1.000000 | 0.335376 | 1-3 tie |
| RCC fine | Tabular PCA | 0.245734 | 0.295337 | 0.350350 | 4 |
| RCC major | CORR-OT | 0.557292 | 0.557292 | 0.532012 | 3 |
| RCC major | CRYCHIC projection | 0.236615 | 0.516874 | 0.223265 | 5 |
| RCC major | GOT | 1.000000 | 1.000000 | 0.603317 | 1 |
| RCC major | scACCorDiON DW-OT | 0.763889 | 0.775726 | 0.356739 | 2 |
| RCC major | Tabular PCA | 0.245734 | 0.245734 | 0.202716 | 4 |

RCC major 的 scACCorDiON DW-OT ARI/RI 为 0.763889/0.882353，与论文 major annotation 数值一致。RCC fine 的 CORR-OT 和 DW-OT ARI=1，也与论文表格一致。PDAC 中 DW-OT 为 1，与论文一致；其他基线因本次使用 100-start k-medoids，而论文参考表的单次结果较低，不能声称逐值复现。AKI 因公开重建子集与作者受限对象不同，所有方法的 phenotype recovery 均较弱。

RCC fine-major 患者距离矩阵 Spearman：

| method | Spearman |
|---|---:|
| CORR-OT | 0.687367 |
| CRYCHIC projection | 0.465927 |
| GOT | 0.630590 |
| scACCorDiON DW-OT | 0.665035 |
| Tabular PCA | 0.850971 |

### 5.3 不能作为 CRYCHIC 原生胜负结论的原因

`crychic_canonical_event_correlation` 是把同一批 CellPhoneDB/LIANA 患者图投影到 `sender-ligand-receptor-receiver` 向量后计算距离。它是 **CRYCHIC-compatible canonical projection**，不是每位患者从原始表达重新运行 CRYCHIC 后得到的 native hypergraph，也不是 CRYCHIC 多组 OOF differential inference。

此外，当前实现用的是 TMM 后向量的 Pearson correlation distance；建议稿文字指定的是 Spearman distance。这一偏差必须在将来同时报告 Pearson/Spearman sensitivity 后才能关闭。

### 5.4 Section III 未落实项

| 建议稿项目 | 当前状态 | 原因/影响 |
|---|---|---|
| 每患者独立图构建 | PDAC/AKI/RCC complete | PDAC 用作者图；AKI/RCC 用 LIANA 1.7.3 CellPhoneDB protocol |
| k-medoids, k=2..7 | complete | 本次使用 100 starts |
| k-barycenter | 未实现 | 只保存了论文参考数值，未运行当前数据 |
| Leiden, resolution 0..1 | 未实现 | 无本次运行结果 |
| ARI/RI/max ARI | complete | 三个 eligible cohort 均有结果 |
| Friedman-Nemenyi | 未实现 | 仅三个 eligible cohort，且后端不完整 |
| canonical Spearman distance | 偏差 | 当前为 Pearson correlation distance |
| native hypergraph distance | 未实现 | 当前患者图不含 CRYCHIC 原生高阶输出 |
| PDAC cell downsampling 5 x 5 | 未实现 | 当前只有作者处理图，未从细胞级重跑 |
| RCC annotation granularity | complete | fine 40 类与 major 7 类 |
| MI annotation granularity | 排除 | 132,888 cells，超过合同上限 |
| TCGA-PAAD survival validation | 未实现 | 不属于当前冻结单细胞主端点 |

所以 Section III 的准确表述是“3/3 eligible cohort 主患者图任务完成”，不是“论文全部聚类、鲁棒性和临床协议完整复现”。

## 6. 后续 Section II：signed estimand 和分解指标

### 6.1 手算 contract

Gate A 为 PASS，绝对误差容差 `1e-10`。

- 两组正/反 contrast：2/2。
- 三组 pairwise 和 balanced one-vs-rest：4/4。
- 三组 omnibus：1/1。
- 正/负 2 x 2 DID：2/2。
- context chain local edge、local neighbor、global one-vs-rest：6/6。
- 合计 15/15 cases 通过。

11/11 invariants 通过：reverse contrast、row reorder、context rename、factor order、positive score scaling、rank scaling、omnibus scaling、sender/receiver relabel、change-point localization、missing score 不补零、缺 context/cell type 返回 NE。

正式 P 值全部标记为 `NE_requires_full_pipeline_resampling`。这是推断边界，不是 benchmark 失败。

### 6.2 三组分解指标已实现范围

native end-to-end evaluator 已实现建议稿中以下端点：

- omnibus AUPRC、prevalence-adjusted AP、AUROC、MCC、Precision@K、Recall@K；
- localization macro/micro AUPRC、Hamming loss、exact contrast-set accuracy、每 contrast confusion；
- positive AP、negative AP、所有 active event 和 detected-active event 的方向准确率；
- native-scale RMSE/MAE、Pearson、Spearman、sign concordance、top-effect recovery；
- event/contrast coverage 和动态范围诊断。

置信区间 coverage 和统一正式 calibration 为 NE，因为四个方法没有共同的全流程重采样置信区间。native-scale RMSE 只作方法内诊断，不跨不同 score scale 排名。

### 6.3 native 三组端到端排名

20 个 active fixture 上的主要结果：

| rank | method | adjusted AP | omnibus AUPRC | AUROC | localization macro AP | direction accuracy | effect Spearman |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | scSeqCommDiff | 0.225873 | 0.163780 | 0.714286 | 0.153373 | 0.150000 | -0.197799 |
| 2 | CellChat | 0.133666 | 0.090872 | 0.504365 | 0.079741 | 0.828571 | 0.150281 |
| 3 | CRYCHIC generic baseline | 0.100398 | 0.067105 | 0.370040 | 0.054836 | 0.107143 | 0.013319 |
| 4 | LIANA | 0.075332 | 0.049954 | 0.090278 | 0.046515 | 0.814286 | 0.066817 |

CRYCHIC 在 omnibus detection 和 localization 上均为 3/4，未超过 scSeqCommDiff。CellChat/LIANA 的方向准确率较高但检出 AUPRC 低，说明“有没有差异”和“方向是否正确”必须分开。

## 7. 后续 Section III：component crossover

### 7.1 矩阵是否落实

矩阵包含 10 个 score generator 和 5 个 differential engine，共 50 个预声明格：

- 27 个 `observed`；
- 2 个 `partially_observed`；
- 21 个 `not_estimable`。

NE 原因：

| reason | cells |
|---|---:|
| independent subject groups violate paired OOF contract | 10 |
| scSeqCommDiff public engine has no external score injection API | 10 |
| CellChat score tensor not complete enough for STACCato | 1 |

两个 CellChat + GLM/CR2 格只有 43.93% event coverage，因此保留结果但不进入正式全覆盖排名。没有把缺失 event 置为最低分或零。

### 7.2 主要排名

27 个完整观察 arm 按 mean AUPRC 排名：

| rank | score generator | engine | AUPRC | AUROC | direction | empirical FDR |
|---:|---|---|---:|---:|---:|---:|
| 1 | simple LR product | clustered CR2 | 0.986192 | 0.999051 | 1.000000 | 0.558886 |
| 2 | simple LR product | sample GLM | 0.985299 | 0.998996 | 1.000000 | 0.574561 |
| 3 | simple LR mean | sample GLM | 0.867072 | 0.986496 | 1.000000 | 0.829529 |
| 4 | simple LR mean | clustered CR2 | 0.853976 | 0.985379 | 1.000000 | 0.828803 |
| 5 | LIANA magnitude | sample GLM | 0.713172 | 0.913337 | 0.821429 | 0.466378 |
| 6 | LIANA magnitude | clustered CR2 | 0.688258 | 0.904520 | 0.821429 | 0.440273 |
| 7 | simple LR product | STACCato | 0.396575 | 0.825307 | 0.892857 | 0.000000 |
| 8 | CellPhoneDB mean | clustered CR2 | 0.346191 | 0.782199 | 0.142857 | 0.511364 |
| 9 | CellPhoneDB mean | sample GLM | 0.342332 | 0.782087 | 0.142857 | 0.610000 |
| 10 | CRYCHIC native | sample GLM | 0.289663 | 0.628627 | 0.414286 | 0.200000 |
| 11 | CRYCHIC native | clustered CR2 | 0.280072 | 0.626563 | 0.414286 | 0.200000 |
| 21 | CRYCHIC native | STACCato | 0.065558 | 0.473577 | 0.357143 | 0.000000 |

高 AUPRC 不表示显著性校准良好。排名前两位的 simple product arm 经验 FDR 约 0.56；因此它们只说明 planted fixture 中的排序/检测信号强，不可报告为可靠的 q-value procedure。

### 7.3 根因判断

在同一个 GLM/CR2 engine 下，simple product、simple mean 和 LIANA magnitude 明显优于 CRYCHIC native；把 CRYCHIC native score 换到 STACCato 也没有改善，反而降到 21/27。当前最直接的诊断是：

- 该 fixture 上主要瓶颈更接近 score/representation，而不是 GLM 与 CR2 之间的 engine 选择。
- `crychic_availability` 和 `availability_prior` 完全一致，是因为本合成 universe 的 prior quality 固定为 1，不代表真实先验没有作用。
- CRYCHIC OOF engine 的 NE 来自独立 subject groups 不满足 paired contract，不能用这个矩阵评价其跨 context 配对设计性能。
- scSeqCommDiff 的 10 个 NE 来自公开 API 不接受任意外部 score tensor，不代表其 native 端到端方法失败；native 结果应看前一节的 1/4 排名。

## 8. 运行时间和资源边界

| stage | wall time | parallelism | peak memory | 计时边界 |
|---|---:|---:|---:|---|
| Tensor campaign | 526.006 s | 24 workers | 每 worker 未统一汇总 | 150 次 factorization 全 campaign |
| STACCato campaign | 26.337 s | 12 workers | 263.5 MiB max worker | 100 次 x 3 designs |
| DCST campaign | 122.573 s | 24 workers | manifest/task 记录 | 475 realization 全流程 |
| PDAC graph evaluation | 5.023 s | 单进程 | 425.5 MiB | 不含作者预计算图生成 |
| AKI preparation | 70.1 s | 单进程 | 约 8.18 GiB | 公开对象选择和样本拆分 |
| AKI graph generation | 27.564 s | 24 workers | 1.68 GiB max worker | 36 个患者图 |
| AKI graph evaluation | 8.315 s | 单进程 | 454.1 MiB | 五种距离、100-start k-medoids |
| RCC preparation, optimized | 23.96 s | 单进程 | 7.46 GiB | 50,236 cells 拆分为 17 H5AD |
| RCC fine graph generation | 37.811 s | 17 workers | 2.56 GiB max worker | 40 cell types |
| RCC major graph generation | 12.714 s | 17 workers | 842 MiB max worker | 7 cell groups；与 fine 并行 |
| RCC fine evaluation | 182.049 s | native BLAS | 435.6 MiB | OT 占 153.464 s |
| RCC major evaluation | 2.095 s | native BLAS | 232.5 MiB | 完整五距离评估 |
| component crossover | 531.047 s | STACCato 8 tasks x 4 cores | 未统一记录 campaign peak | 1.62M score rows，100 bootstraps |
| external CellChat/LIANA/scSeq panel | 943.476 s | 多进程 | 各 adapter manifest | 200/200 tasks |
| CellPhoneDB panel | 157.765 s | 多进程 | 各 adapter manifest | 40/40 tasks |

RCC preparation 的旧实现运行约 6 分钟仍只输出第 1/17 个样本。根因是每次 AnnData slice 都深复制 `uns` 中约 6.9 GB 的无关矩阵。提交 `83759c2` 改为只切片 `X/obs/var` 后，完整准备在 23.96 秒内结束；按细胞量外推相对旧流程约有 150 倍加速，且矩阵/样本计数测试通过。

不同时间不能直接合成算法总耗时：PDAC 不含原始细胞图生成，AKI/RCC 包含重建图；Tensor/STACCato/DCST 是并行 campaign wall time，不是 CPU-seconds。

## 9. 完整性审计表

| 审计对象 | 冻结主任务 | 建议稿字面协议 | 判断 |
|---|---|---|---|
| Tensor-cell2cell | complete | partial | 核心恢复指标完成，扩展 C 和 CorrIndex 未完成 |
| STACCato | complete | partial | condition 指标完成，batch/calibration/power 未完成 |
| DCST | complete | partial | 两个 eligible sweep 完成，CRYCHIC fixed-binary arm 未完成 |
| PDAC patient graphs | complete | partial | 主图聚类完成，下采样、其他后端和 survival 未完成 |
| AKI patient graphs | complete | partial fidelity | 公开设计重建完成，不是作者 restricted subset |
| RCC patient graphs | complete | partial protocol | fine/major 完成，native hypergraph/其他后端未完成 |
| signed estimand contract | complete | complete for Gate A | 15/15 + 11/11；正式重采样 P 值不在 Gate A |
| three-group decomposed metrics | complete | partial inference | 指标齐全，CI/calibration 为 NE |
| component crossover | complete | 27 full + 2 partial + 21 NE | 矩阵落实，但受设计/API 约束不能 50 格全观察 |

## 10. 发表表述边界

可以使用的表述：

- “All nine frozen under-100k primary tasks completed with checksum-bound outputs.”
- “The signed estimand Gate A passed 15/15 deterministic cases and 11/11 invariance checks.”
- “In the native three-group planted benchmark, CRYCHIC ranked third of four by omnibus AUPRC.”
- “In the component crossover, CRYCHIC native + sample GLM ranked 10th of 27 fully observed arms.”
- “RCC fine annotation yielded ARI=1 for CORR-OT, DW-OT, and the CRYCHIC-compatible canonical projection; the projection dropped to ARI=0.2366 at major annotation.”

不应使用的表述：

- “Section II and III were completely reproduced” without qualification。
- “CRYCHIC beat scACCorDiON on patient graphs”，因为 projection 不是 native CRYCHIC inference，且存在并列和 coarse 失败场景。
- “AUPRC 0.986 implies calibrated discoveries”，因为对应经验 FDR 约 0.56。
- 把 AKI 结果称为作者数据的精确复现。
- 把 NE 写成零效应、失败分数或最低排名。

## 11. 代码、结果和 notebook

核心代码：

- [Tensor runner](../../benchmarks/comprehensive/run_tensor_cell2cell_under100k.py)
- [STACCato runner](../../benchmarks/comprehensive/run_staccato_under100k.py)
- [DCST runner](../../benchmarks/comprehensive/run_dcst_under100k.py)
- [DCST campaign](../../benchmarks/comprehensive/run_dcst_under100k_campaign.py)
- [scACCorDiON evaluator](../../benchmarks/comprehensive/run_scaccordion_under100k.py)
- [public cohort preparation](../../benchmarks/comprehensive/prepare_scaccordion_public_cohorts.py)
- [LIANA/CellPhoneDB cohort runner](../../benchmarks/comprehensive/run_liana_cellphonedb_cohort.py)
- [signed estimand contract](../../benchmarks/comprehensive/signed_estimand_contract.py)
- [score-engine crossover](../../benchmarks/comprehensive/run_score_engine_crossover_under100k.py)
- [native three-group evaluator](../../benchmarks/comprehensive/evaluate_three_group.py)
- [notebook builder](../../benchmarks/report/generate_suggest_v5_under100k_notebook.py)

执行后的 notebook：[suggest_v5_under100k_benchmark_results.ipynb](../../tutorials/suggest_v5_under100k_benchmark_results.ipynb)。其 10/10 个 code cell 均已执行、均含 `%%time`、无 error output。Notebook 内含 7 组发表级图；外部结果目录同时保存 PNG/PDF/SVG。

正式结果根目录：

`/media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_v5_under100k_20260724`

原始/患者级数据和大体积结果未写入 Git。Git 只保存代码、冻结合同、执行后的 compact notebook 和本审计报告。

关键 clean provenance commits：

- Tensor/DCST：`04c30c12171ed6220379ca3d024ec8e497b175d2`
- STACCato：`d6b260a08d6734d964ccc090de8a00f66a267e82`
- PDAC：`6b116411639fec4f235c601b00af8e7bfb89be8c`
- AKI/signed contract：`f696a4a5c01b061c3ac7b48064970c172817ace6`
- RCC：`83759c23958805cc48a1e5c42d5757a3190b5afe`
- crossover：`8a4a3bdd6a56941edc147e9c59e13c9a4785ed33`
- native evaluator：`a4853d841369e3c5a661f4f75278687660b069a8`

## 12. 建议的补齐顺序

1. 先把 canonical patient distance 改为建议稿指定的 Spearman，并同时保留 Pearson sensitivity；随后重跑三队列和 RCC 两层注释。
2. 为 CRYCHIC DCST binary projection 预注册阈值，避免看过结果后选阈值，再完整重跑两个 sweep。
3. 补 STACCato batch-effect MSE/error SD 和 null calibration；Tensor 扩展指标可直接从已保存的 150 组 factors 后处理，不必重跑分解。
4. 增加 k-barycenter 和 Leiden 后端，再决定三个 eligible cohort 是否只做描述性 rank，避免用三个数据集过度解释 Friedman-Nemenyi。
5. 从原始患者表达运行 native CRYCHIC graph/hypergraph，严格区分 canonical projection 与 native hypergraph distance。
6. 最后补 PDAC cell-downsampling 和 TCGA survival；这两项数据链与计算边界最不同，不应阻塞当前主结果报告。

以上顺序优先修正 estimand/距离定义，再扩展更多后端和数据，能最大限度避免在错误语义上增加计算量。
