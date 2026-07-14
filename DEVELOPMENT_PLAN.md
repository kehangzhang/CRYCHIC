# CRYCHIC 详细开发计划

## 0. 文档控制

| 项目 | 内容 |
| --- | --- |
| 项目名 | CRYCHIC: Cellular Relational dYnamics for Contextual Hypergraph Inference of Communication |
| Python 包名 | `crychic` |
| 文档状态 | 架构与实施基线，尚未开始算法实现 |
| 编制日期 | 2026-07-12 |
| 主要需求来源 | 仓库上级目录 `../prompt.md`（1,787 行算法设想） |
| 规划分支 | `planning/development-blueprint` |
| 目标读者 | 计算生物学、统计学、算法、Python 工程、测试与文档维护者 |

本计划把 `prompt.md` 的研究设想转换为可执行的软件工作包、模块边界、
验收标准和发布门。`prompt.md` 中 `TopoCCC/topoccc` 是临时工作名，公开
命名统一为 `CRYCHIC/crychic`。原文部分 LaTeX 在 Markdown 转换中损坏，
实现公式前必须在 `docs/methods/` 中恢复、推导并经过统计审阅；不得直接按
损坏的公式片段编码。

本文的状态约定：

- **冻结**：实现不得绕过，变更需要 ADR、统计审阅和回归验证。
- **计划**：已经确定方向，但具体接口可在对应版本内通过 ADR 收敛。
- **实验**：必须有基线、消融和 go/no-go 决策门，未通过不得成为默认行为。

---

## 1. 项目目标与边界

### 1.1 产品目标

CRYCHIC 要解决的一般问题是：

> 在任意多个生物学上下文组成的图上，利用样本级单细胞转录组，联合估计
> 上下文特异的配体-受体可用性、receiver 下游响应、sender 来源及统计
> 不确定性，并输出可追溯的通信边和基因/TF/pathway signature。

首个稳定版本必须支持：

1. 严格验证的 AnnData 输入和多上下文字段。
2. `sample x context x cell_type` pseudobulk 与结构性缺失语义。
3. global、拓扑 local 和用户指定/factorial contrast。
4. 连续 receiver 响应，不依赖二元 DEG 列表作为模型输入。
5. context-gated LR-target basis 与多 LR 联合稀疏归因。
6. 任意上下文图上的 graph-fused 正则化和无拓扑退化路径。
7. receiver driver 与 sender 来源的分离建模。
8. state 与 ecosystem 两套通信量。
9. subject-level cross-fitting、正式差异模型与概率校准。
10. LR-attributed signature、未解释残差和通信超图输出。
11. 版本化结果目录、完整 provenance、可恢复执行和可复现实验。

### 1.2 科学差异化目标

项目的论文级核心不是“使用超图”本身，而是以下组合：

- 生物学样本感知的上下文特异 receiver 响应；
- 多个候选 LR 对 receiver 响应的联合稀疏分解；
- 任意上下文图上的局部自适应 graph-fused 正则化；
- receiver driver、sender assignment 与正式统计检验的职责分离；
- 强度、活跃概率、特异性概率和差异效应的不同统计语义；
- 可直接导出的 LR/TF/gene signature 和不可解释残差。

### 1.3 首版明确不做

- 不把细胞当作独立统计重复。
- 不以 `FindAllMarkers` 式细胞级 one-vs-rest 作为正式 receiver 模型。
- 不把 ligand expression 与 receptor expression 的乘积称为概率。
- 不从 graph-fused/elastic-net 系数直接计算普通 Wald p 值。
- 不强迫全部 receiver 差异表达归因于细胞通讯。
- 不在核心拟合中引入 GNN；超图是结果表达和后续分析结构。
- 不在 v0.1 同时实现完整 OmniPath 扩散、多视图、临床预后和生产级 ADMM。
- 不声称 sender assignment 是因果来源证明。
- 不在没有外部验证的情况下声称临床效用。

### 1.4 成功定义

项目成功必须同时满足四类标准，而不是只完成 API：

| 维度 | 成功标准 |
| --- | --- |
| 科学有效性 | 模拟中恢复真实 interaction/context/signature，并对模型错配稳健 |
| 统计可信度 | type-I error、FDR、区间覆盖和概率校准达到预注册阈值 |
| 软件质量 | 契约明确、可安装、可复现、可恢复、资源可追溯、失败不伪装成功 |
| 可用性 | 从合规 h5ad 到查询、导出和诊断报告的端到端工作流稳定 |

---

## 2. 冻结的方法学契约

以下规则从第一行实现代码起生效。

### 2.1 推断单位与重复测量

1. `subject_id` 是生物学独立重复和重采样单位。
2. `sample_id` 是文库/取样单位，必须与 `subject_id` 分开。
3. 同一 subject 的多个 region/time/treatment sample 在 split 和 bootstrap 中
   必须保持成块；permutation 不采用一条笼统的“整块置换”规则。
4. 每个设计必须生成 `ExchangeabilityMap`：between-subject 因子只在合法 strata
   内置换 subject block；paired within-subject 因子只执行零假设允许的受限 label
   swap 或 sign flip；固定协变量和不可交换因子永不置换。
5. 每次用于正式检验的 permutation 必须重跑过滤、gating、聚类、调参、归因、
   sender 和 scoring 全链，不能只置换已生成的 score label。
6. 细胞用于构成 pseudobulk、检测率和细胞丰度，不进入独立重复计数。
7. 重复测量设计只有在 cluster-robust/CR2、GEE 或经验证的加权模型可用时
   才进行正式推断；否则明确拒绝或降级。

### 2.2 缺失、零与不可估计

- 零捕获细胞先区分 `sampling_zero`、`qc_failure` 和有外部证据支持的
  `confirmed_absence`；scRNA 未捕获本身不能证明生物学缺失。
- 没有 eligible cell 时 state expression 必须为 `NA`，不能补零。
- 细胞数不足：可保留 abundance 证据，但 state-expression eligibility 由
  显式阈值和 QC 决定。
- cell type 只在一个 context 出现：可描述出现/消失，不能伪造普通组内 DE。
- ecosystem 使用 presence/abundance 与 conditional-state 的 two-part/hurdle
  estimand；只有 `confirmed_absence` 或经验证的 abundance observation model
  才允许确定性零。
- 未提供校正绝对丰度时，ecosystem 明确命名为 capture-weighted ecosystem
  proxy，不能解释成组织绝对通信总量。
- 完全混杂、秩亏、cluster 太少：`not_estimable`，不得静默换模型或给 p/q。
- 结果 schema 必须区分 `0`、`missing`、`not_estimable`、`filtered`、`failed`。

### 2.3 探索与推断模式

`exploratory` 输出可以包含 descriptive strength、ranking、网络和诊断；
`inferential` 才能包含经校准的 effect、SE、p、q 和 posterior probability。
模式不是一个全局布尔值，而要按 receiver、contrast、edge 记录可估计性。
v0.1 和 v0.2 的所有正式 p/q/probability 字段禁用；它们只输出 effect/SE
diagnostic、strength、ranking 和 stability。只有通过 G3-F 或 G3-P 的字段才能
分别启用。数据不足时返回 `NA + reason_code`，不得用别的分数补位。

### 2.4 训练折隔离

所有数据驱动步骤必须在 cross-fitting 的训练折中完成：

- gene/LR 过滤；
- 标准化和精度权重；
- baseline-expression/receptor/path gating；
- LR target profile 聚类；
- 超参数选择和稳定性选择；
- sender coupling 和 assignment 参数；
- outer-training `FrozenInteractionUniverse`、完整 receiver x interaction
  candidate-sender manifest、interaction-level ligand contrast support 和
  receiver-wise Holm family；
- activity/signature 权重学习。

测试折只使用冻结的训练产物生成 sample-level OOF score。任何 subject 不得
同时出现在同一折的训练和测试集合中。fold planner 必须在拟合前检查每个训练折
的 context/covariate/cell-type 支持、设计秩和 contrast estimability；可以按预先
声明的规则降低 `K`，但若不存在可估计的 `K >= 2` 必须拒绝 cross-fitting。

### 2.5 跨 context 的共同计分 estimand

context-specific `alpha`、gating 和 sender assignment 是机制发现结果，直接让
每个测试 context 使用自己的计分函数再比较，会把训练组差异写入 outcome 并造成
量尺不一致。因此每个正式 `ContrastSpec` 必须在训练折产生一个
`ScoringFunctional`，并满足：

- 对该 contrast 中所有被比较 context 使用相同 feature universe、target 权重、
  sender/LR/receiver 定义、变换和尺度；
- functional 只由训练数据产生，并有稳定 `scoring_function_id`；
- 测试样本 context 只能进入 frozen functional 的输入值，不能选择另一套权重；
- context-specific mechanistic score 可以另行输出，但标记 `not_comparable`，不得
  进入正式 context effect model；
- 单纯 fold scale calibration 不能修复不同 scoring functional 的 estimand 问题。

sender-side ligand contrast gate 是 contrast-common 的训练折产物，不属于 held-out
context-specific gate。它必须对 contrast 中所有测试 context 复用同一状态和 ID；
held-out ligand 值只进入 sample-local availability/assignment，不能改变 interaction
family、Holm rank、adjusted p、support status 或 gate ID。

正式 scoring functional 的精确定义、共同 eligible feature 规则和方向必须在
ADR-011 中冻结，并用 fold-specific-functional null 端到端验证。

### 2.6 nuisance 学习不确定性

普通回归把 OOF score 当固定观测，并不会自动计入同折 subject 共享的过滤、gating、
driver、tuning 和 sender 模型不确定性；当前 score 也未证明 Neyman-orthogonal。
因此 v0.3 的默认方案是：

1. repeated cross-fitting 产生 point estimate 和 repeat stability；
2. subject bootstrap/permutation 在外层重跑完整训练与计分链；
3. 每个 resample 内重新拟合未惩罚 effect model；
4. CI、empirical p 和校准从 full-pipeline resample distribution 得到；
5. 只对固定 OOF 表 bootstrap 的结果仅可作为 diagnostic。

未来若推导并验证了 influence-function/orthogonal score，可通过 ADR 替换默认方差
方案，但必须重新通过 coverage 和 FDR gate。

### 2.7 选择、归因、检验和 null 分离

- graph-fused/elastic-net：选择、去噪、mechanistic attribution。
- full-pipeline subject bootstrap：selection frequency 和 specificity support。
- common-functional OOF sample-score model：point effect；full-pipeline resampling
  提供其不确定性与检验校准。
- context-effect permutation：按 `ExchangeabilityMap` 检验 context contrast。
- sender-label null：只评估 sender assignment，不进入 active-edge posterior null。
- resource replacement/shuffle：除预注册 active null 外均作为资源敏感性分析。

`comm_probability` 的初始 estimand 冻结为：在给定表达、丰度和 context 的条件下，
sender-LR-receiver edge 相对“无机制匹配 LR-target link”零分量为 active 的后验概率。
主 null 由 degree/evidence-matched LR-target reassignment 在训练折内生成，并通过
workflow 重跑完整链。不得把 context permutation、sender shuffle 和多种 network
shuffle 混进同一个 local-FDR null。ADR-013 必须在实现前冻结唯一 null、strata、
最小候选数和诊断；失败时 `comm_probability=NA`，但合法 q-value 不受影响。

### 2.8 输出量的固定语义

| 字段 | 语义 | 允许来源 |
| --- | --- | --- |
| `comm_strength` | 0-1 连续综合生物学强度 | availability、downstream、sender、prior quality |
| `comm_probability` | context edge 属于预注册 non-null active component 的后验概率 | 单一机制匹配 null + local FDR，G3-P |
| `specificity_support` | edge-contrast 效应超过预注册最小效应的 bootstrap exceedance frequency；不是后验概率 | full-pipeline subject bootstrap |
| `selection_frequency` | receiver-driver/family 在完整重拟合中被选择的频率 | full-pipeline stability bootstrap |
| `effect_size` | common-functional OOF score 的组间点效应 | 未惩罚 subject-aware model |
| `standard_error` / `p_value` / `q_value` | 含 nuisance 重拟合的频率学不确定性与多重校正 | full-pipeline resampling + G3-F |

`specificity_support` 的最小效应 `delta`、尺度、方向、单/双侧和 contrast 必须在
看结果前写入 `ContrastSpec`。不得把 bootstrap exceedance frequency 解释为
“参数大于 delta 的后验概率”。

### 2.9 假设全集与多重性

v0.3 的初始 primary family 计划为：gene view、state mode 下的
`driver_family x receiver` context omnibus hypothesis；sender-resolved、ecosystem、
TF/pathway 和 post-hoc 属于明确标记的 secondary families。ADR-012 必须在实现前
唯一冻结：

- primary estimand 和 hypothesis key；
- omnibus 与 post-hoc 的层级和唯一默认 procedure；
- sender/receiver/LR/contrast/mode/view 的 multiplicity 边界；
- independent filtering 和缺失 hypothesis 规则；
- 所控制的 FDR/FWER 类型及 nominal level。

正式 hypothesis universe 由外部资源和不看 context label 的 pooled support 规则
预先冻结，所有 fold 使用稳定 ID。context-aware fold selection 不得改变 universe；
未选择项按预注册规则进入 full-pipeline null/多重性处理。列举多个可选 procedure
不能替代 ADR 和模拟验证。

---

## 3. 三类拓扑与架构边界

### 3.1 三类拓扑必须物理隔离

| 拓扑 | 符号 | 软件位置 | 含义 |
| --- | --- | --- | --- |
| 上下文拓扑 | `G_C` | `crychic.design` | treatment、region、time 等实验上下文节点与邻接 |
| 分子信号拓扑 | `G_S` | `crychic.resources` | ligand -> receptor -> signaling -> TF -> gene |
| 输出通信超图 | `H` | `crychic.network` | context、sender、LR、receiver、program 的结果超边 |

不得用一个未区分语义的通用 `Graph` 类表示三者。`H` 不参与核心模型拟合，
`G_S` 不承担实验 design contrast，`G_C` 不包含分子节点。

### 3.2 目标运行时目录

当前规划提交只创建目录和 `AGENTS.md`。Phase 0 才增加 `__init__.py`、实现
文件、构建配置和测试骨架。

```text
src/crychic/
├── api/             # 稳定 Python 门面
├── core/            # 类型、配置、协议、ID、错误、provenance、seed
├── data/            # h5ad、schema、gene mapping、输入 QC
├── pseudobulk/      # 聚合、检测率、丰度、结构性缺失
├── design/          # G_C、formula、estimability、contrasts
├── resources/       # LR/complex、G_S、静态 prior、manifest/cache
├── response/        # gene/TF/pathway receiver response
├── availability/    # ligand/receptor/complex、state/ecosystem
├── resampling/      # subject split/bootstrap/permutation
├── attribution/     # basis、equivalence、elastic-net、graph-fused
├── sender/          # evidence、coupling、soft assignment
├── scoring/         # sample-level OOF strength components
├── inference/       # omnibus/post-hoc、robust model、FDR/local FDR
├── signatures/      # observed/predicted/residual signatures
├── network/         # H、modules/hubs/topology export
├── results/         # schema、persistence、query、migration
├── workflow/        # orchestration、cache、resume、leakage barrier
├── visualization/   # 只消费 result
└── cli/             # 薄命令行封装
```

仓库级辅助目录：

```text
benchmarks/          # 模拟、外部方法 adapter、指标、隔离环境
docs/                # methods、ADR、API、tutorial、limitations
resources/           # manifest 和极小可再分发 fixture
schemas/             # 持久化 JSON Schema
tests/               # unit/integration/contract/statistical/regression/performance
```

### 3.3 依赖 DAG

```text
core ──> data ──> pseudobulk
data + pseudobulk ──> design
data ──> resources
design + subject metadata ──> resampling

pseudobulk + design + resources ──> response
pseudobulk + resources          ──> availability
response + availability + design + resources ──> attribution
pseudobulk + response + availability + attribution ──> sender
availability + attribution + sender ──> scoring
scoring + design + resampling ──> inference
response + attribution + sender ──> signatures
scoring + signatures (+ optional inference) ──> network
public upstream artifact contracts ──> results
all computational public contracts ──> workflow
workflow + results + design/core ──> api ──> cli
results + public api ──> visualization / top-level benchmarks
```

禁止的反向依赖：

- 数值模块不得导入 `workflow`、`api`、`cli` 或 `visualization`。
- 核心包不得导入顶层 `benchmarks/`。
- `plotting/visualization` 不得承载统计计算。
- `results` 读取/验证结果，不触发模型重算。
- 业务 artifact contract 由上游 producer 模块拥有，consumer 只导入其公开契约；
  `core` 仅保存通用配置、ID、provenance、seed 和协议基元，不导入私有实现。

### 3.4 模块交付矩阵

| 模块 | 第一职责 | 关键输出契约 | 首次启用 |
| --- | --- | --- | --- |
| `core` | 配置、协议、ID、provenance | `CrychicConfig`, metadata contracts | Phase 0 |
| `data` | 输入读取和验证 | `ValidatedInput`, `InputReport` | v0.1 |
| `pseudobulk` | 样本级聚合 | `PseudobulkDataset`, eligibility mask | v0.1 |
| `design` | 图、design、contrast | `ContextGraph`, `ContrastSpec`, diagnostics | v0.1 |
| `resources` | LR/complex/prior/G_S | `ResourceBundle`, `TargetPrior` | v0.1/v0.4 |
| `response` | 连续 receiver response | `ResponseEstimate` | v0.1 |
| `availability` | LR 可用性 | `AvailabilityEstimate` | v0.1 |
| `resampling` | subject fold/exchangeability/重采样计划 | `FoldManifest`, `ExchangeabilityMap`, `ResampleManifest` | v0.2 |
| `attribution` | baseline 与联合稀疏归因 | `AttributionResult`, `ResampledAttribution` | v0.1 experimental/v0.2 |
| `sender` | sender 证据分配 | `SenderAssignment` | v0.2 |
| `scoring` | contrast-level OOF 通信分数 | `ScoringFunctional`, `CommunicationScores`, `NullScoreDistribution` | v0.2 |
| `inference` | 正式检验/校准 | `DifferentialResult` | v0.3 |
| `signatures` | 归因和残差 signature | `SignatureTable` | v0.2 |
| `network` | 输出超图和拓扑特征 | `CommunicationHypergraph` | v0.2/v1.0 |
| `results` | 版本化持久化和查询 | `CrychicResult` | v0.1 |
| `workflow` | 阶段编排和恢复 | `RunManifest` | v0.1 |
| `api` | 稳定 Python 门面 | `Crychic` facade | v0.1 |
| `visualization` | 结果展示 | figure/data selection | v0.1+ |
| `cli` | 配置驱动命令 | exit status/run manifest | v0.5 |

### 3.5 从原 prompt 单文件建议到子包的映射

| 原建议 | 新边界 | 拆分理由 |
| --- | --- | --- |
| `io.py/schema.py/qc.py` | `data/` | 输入、映射和 QC 共享 AnnData 边界 |
| `aggregate.py` | `pseudobulk/` | 聚合、detection、abundance、missingness 独立演化 |
| `design.py/contrasts.py/context_graph.py` | `design/` | 共同拥有实验设计与 `G_C` |
| `resources.py/complexes.py` | `resources/` | 静态知识、复合物和 manifest |
| `activities.py` | `response/` | activity 是样本级响应，不是静态资源 |
| `target_prior.py` | `resources/` + `attribution/` | 静态 prior 在 resources，动态 gated basis 在 attribution |
| `attribution.py` | `attribution/` | basis、family、solver、diagnostics 拆文件 |
| `sender_assignment.py` | `sender/` | 与 receiver driver 明确分离 |
| `scoring.py` | `availability/` + `scoring/` | 原始可用性与共同计分 functional 分离 |
| `bootstrap.py` | `resampling/` + `workflow/` | 前者定义合法计划，后者重跑完整计算链 |
| `inference.py` | `inference/` | 只消费 OOF/null/resampled artifacts |
| `signatures.py` | `signatures/` | observed/predicted/residual 独立契约 |
| `hypergraph.py` | `network/` | 只表示输出 `H`，不参与拟合 |
| `result.py/plotting.py` | `results/` + `visualization/` | 持久化查询与展示分离 |
| `benchmark/` | 顶层 `benchmarks/` | 外部重型依赖不污染运行时包 |

### 3.6 契约和持久化 schema 的所有权

- 业务内存 artifact（如 `PseudobulkDataset`、`ResponseEstimate`）由 producer
  子包定义公开 typed contract。
- `core` 只拥有跨域基元、配置、ID、provenance、seed 和 schema-version ID。
- 顶层 `schemas/` 是持久化 JSON Schema 的唯一规范源。
- `results` 实现 schema validation、atomic persistence 和 migration，但不维护
  另一套竞争性定义。
- `workflow` 组合 producer/consumer；不会把 artifact 重新定义成 orchestration
  私有类型。

---

## 4. 输入、内部数据和结果契约

### 4.1 AnnData 必需字段

| 位置 | 默认字段 | 契约 |
| --- | --- | --- |
| `adata.layers` | `counts` | inferential mode 使用有限、非负整数原始 counts |
| `adata.obs` | `sample_id` | 文库/取样唯一 ID；映射到单一 subject/context |
| `adata.obs` | `subject_id` | 生物学独立重复 ID；API 可由 `subject_key` 覆盖 |
| `adata.obs` | `cell_type` | sender/receiver cell type 或 state |
| `adata.obs` | 用户 context keys | 一个或多个离散/有序上下文列 |
| `adata.var_names` | gene ID | 明确 species 与 HGNC/Ensembl 等 namespace |
| `adata.X` 或显式 layer | exploratory expression | counts 缺失时必须声明 source 和 transform |

可选协变量包括 `batch`、`sex`、`age`、`stage`、`response`、`timepoint`、
`library`、`chemistry` 和 `cell_state`。输入报告至少给出：

- subject/sample/context 交叉表；
- 每个 cell type 的 subject 和 cell 支持度；
- 设计矩阵秩、别名列和完全混杂；
- counts 类型、稀疏格式、基因重复与映射损失；
- inferential/exploratory eligibility 及 reason code。

仅有 normalized expression 时，用户必须显式配置 `expression_source` 和
`expression_transform`（首版允许 `linear_normalized` 或 `log1p_normalized`），
不得由数值范围猜测。该路径生成 sample-cell-type descriptive mean，而不是把
log-normalized 值求和冒充 counts。只有零值仍明确表示未检测时才计算 detection，
否则 detection 为 `NA`。normalized-only run 强制 exploratory，所有正式 p/q 和
posterior probability 字段为 `NA` 并带 `normalized_only` reason code。

### 4.2 Pseudobulk 契约

定义样本单元 `u`、cell type `c`、gene `j`：

```math
Y_{u,c,j} = \sum_{i \in (u,c)} Y_{i,j}.
```

每个 unit 同时保存：

- raw aggregated counts 和 library size；
- cell number、cell proportion、median UMI；
- raw detection fraction 及 shrinkage 后 detection；
- `subject_id`、`sample_id` 和规范化 context tuple；
- `state_eligible`、`abundance_eligible` 和 missingness reason；
- 输入 gene namespace、映射和聚合 provenance。

counts 路径输出 `PseudobulkDataset`；normalized-only 路径输出语义不同的
`ExploratoryAggregate`。二者不能通过同一个无标记矩阵类型互换。零捕获 cell
type 的 missingness 分类和 two-part abundance 状态必须随 artifact 一起保存。

大型矩阵首选 AnnData/Zarr，表格首选 Arrow/Parquet；不得依赖 DataFrame 行顺序
形成 ID 或关联。

### 4.3 Context 和 ID 规范

- 多因素 context 采用带字段名的规范 tuple，例如
  `((region, core), (treatment, treated))` 的稳定序列化。
- context graph 节点必须与输入 observed context 一一校验；允许显式声明未观察
  节点，但不得默认补数据。
- interaction ID 由资源版本、ligand complex、receptor complex 和 species 的
  canonical representation 生成。
- result row key 不依赖输入排序、线程数或 hash 随机化。
- fold、bootstrap、permutation 和 solver run 都有独立稳定 ID。

### 4.4 内部类型契约

Phase 0 先定义契约归属；具体 contract 位于其 producer 子包，再并行开发：

| 契约 | 最少字段 |
| --- | --- |
| `InputSchema` | layer/key/namespace/context/covariate 声明 |
| `PseudobulkDataset` | counts、metadata、QC、eligibility、feature IDs |
| `ExploratoryAggregate` | normalized mean、transform、detection eligibility、强制 mode |
| `ContextGraph` | typed nodes、weighted edges、components、provenance |
| `ContrastSpec` | name、vector/rule、family、mode、estimability |
| `ExchangeabilityMap` | factors、strata、immutable covariates、合法 permutation ops |
| `ResponseEstimate` | effect、SE、z/precision、sample support、status |
| `TargetPrior` | gene x interaction sparse matrix、direction、evidence、version |
| `AvailabilityEstimate` | ligand/receptor/complex、state/ecosystem、missingness |
| `AttributionResult` | directional coefficient、family、objective、solver diagnostics、fold |
| `ResampledAttribution` | resample、driver selection events、完整重拟合 provenance |
| `SenderAssignment` | evidence components、weight、entropy、support、fold |
| `ScoringFunctional` | contrast、共同 feature/weights/scale、training fold、stable ID |
| `CommunicationScores` | subject/sample/context、functional、components、strength、fold、QC |
| `NullScoreDistribution` | 唯一 null ID、resample scores、full-pipeline provenance |
| `DriverEstimate` | context/receiver/family、coefficient、selection frequency |
| `ActiveProbabilityResult` | context edge、active-null ID、posterior、diagnostics |
| `DifferentialResult` | edge-contrast、effect、SE、p/q、specificity support、status |
| `SignatureTable` | observed、predicted、residual/contribution、stability |
| `RunManifest` | config/input/resource/code/schema digest、seed tree、stages |

### 4.5 目标公共 API

首个 API 先保持窄门面；名称在 Phase 0 ADR 中最终确认：

```python
import anndata as ad
import crychic

adata = ad.read_h5ad("cohort.h5ad")

graph = crychic.ContextGraph.product(
    graphs={
        "region": crychic.ContextGraph.chain(
            ["adjacent", "border", "core"]
        ),
        "treatment": crychic.ContextGraph.complete(
            ["control", "treated"]
        ),
    },
    edge_weights={"region": 1.0, "treatment": 0.5},
)

config = crychic.CrychicConfig(
    counts_layer="counts",
    sample_key="sample_id",
    subject_key="subject_id",
    cell_type_key="cell_type",
    context_keys=["treatment", "region"],
    covariates=["batch", "sex", "stage"],
    design="~ batch + treatment * region",
    communication_modes=["state", "ecosystem"],
    random_seed=20260712,
)

model = crychic.Crychic(config)
report = model.validate(adata, context_graph=graph)
result = model.fit(adata, context_graph=graph)
```

结果查询面：

```python
result.rank_interactions(
    context={"treatment": "treated", "region": "core"},
    receiver="Tumor",
    contrast="global_one_vs_rest",
)

result.get_signature(
    context={"treatment": "treated", "region": "core"},
    sender="Macrophage",
    receiver="Tumor",
    interaction="TGFB1_TGFBR2",
)
```

以下 patient feature API 只属于 v1.0 计划，不在 v0.x 首个稳定查询面中：

```python
result.get_patient_features(level="communication_module")
```

### 4.6 结果目录和表

```text
crychic_result/
├── config.yaml
├── provenance.json
├── run_manifest.json
├── pseudobulk.h5ad
├── drivers.parquet
├── interactions.parquet
├── differential.parquet
├── signatures.parquet
├── sample_scores.zarr
├── context_graph.graphml
├── communication_hypergraph.parquet
└── diagnostics/
```

核心表约束：

| 表 | 主键建议 | 关键内容 |
| --- | --- | --- |
| `drivers` | context + receiver + driver family + response direction | coefficient、equivalence、selection frequency、solver status |
| `interactions` | context + sender + receiver + interaction + mode | availability、assignment、strength、active probability、support |
| `differential` | hypothesis level + stable hypothesis ID + contrast + mode/view | effect、SE、statistic、p/q、specificity support、change class、status |
| `signatures` | context + sender? + receiver + interaction? + gene/view | observed、predicted、residual、weight、stability |
| `sample_scores` | subject + sample + normalized context + design-row ID + edge + scoring-functional ID + repeat/fold + mode | OOF components、strength、eligibility、immutable covariate linkage |
| `hypergraph` | context + hyperedge ID | typed nodes、direction、weight、uncertainty |

`comm_probability` 只属于 context interaction edge；`specificity_support` 属于
edge-contrast；`selection_frequency` 属于 receiver-driver/family，不复制到每个
sender edge。`state_driven`、`abundance_driven`、`mixed`、`unstable` 和
`not_classifiable` 由 state/ecosystem differential effect、方向一致性和各自
estimability 按版本化规则产生，不能只比较原始 point score。

写入必须是原子/事务式；中断目录标记 `incomplete`，不得作为成功结果加载。

---

## 5. 核心计算主链

### Step 0：验证和运行模式判定

读取 AnnData、资源和 context graph，生成设计可估计性报告。任何正式模型开始
前明确每个 receiver/contrast 的 mode、样本支持、expression source 和失败原因。
normalized-only 输入直接锁定 exploratory。

### Step 1：样本级 pseudobulk

counts 路径按 sample/context/cell type 聚合，同时保存 detection、abundance 和 QC；
normalized-only 路径生成明确标记的 descriptive mean。零捕获 cell type 分类为
sampling zero、QC failure 或 confirmed absence，state 不补零，并为 ecosystem
two-part estimand 保存 presence/abundance 信息。

### Step 2：design、exchangeability 和 fold 计划

构造 `G_C`、formula、EMM 和预注册 contrast，检查整体设计和每个潜在训练折的秩、
context/covariate/cell-type 支持与 estimability。v0.2 起，按 subject 和设计结构
生成 `FoldManifest`；v0.3 再生成 repeated-cross-fit、bootstrap 和 permutation 的
`ExchangeabilityMap`。不存在可估计 `K >= 2` 时拒绝交叉拟合。

### Step 3：receiver 连续响应与静态资源

对每个 receiver/gene 进行样本级 count 或 validated weighted model，得到协变量
校正后的 context marginal mean 与 contrast。模型输入为完整连续向量：

```math
Z_{g,r} = (z_{g,r,1}, \ldots, z_{g,r,J}),
```

其中 `z` 可为 moderated Wald z、shrunken logFC/SE 或带方向 rank-normalized
统计量。v0.1 先实现 gene view；TF/pathway 在 v0.4 扩展。同时加载版本化
LR/complex 和预计算 ligand-target prior，完成 namespace、许可、checksum 和
evidence 检查，但此时不做动态 gating。

### Step 4：availability 与 receiver gating

先计算 ligand、receptor、multi-subunit complex 的 training-fold availability，
保留 state 与 capture-weighted ecosystem proxy。按 receiver receptor eligibility
和 baseline signaling 可用性形成 gating artifact。禁止使用待解释的差异响应或
测试折信息门控，避免循环论证。

### Step 5：directional gated basis 与 LR 可辨识性

将静态 prior 与 Step 4 gating 合成 gene x LR sparse basis，统一 target profile
L2 norm，计算 cosine similarity 并聚类高度相似 basis。先输出 `driver_family`，
再用 ligand/receptor 表达和证据质量分配具体 LR；不可辨识时提高 uncertainty。

v0.1/v0.2 的非负 ligand-target prior 不能闭合解释 signed `Z`。首选基线方案是按
预注册方向拟合 direction-compatible response：增加方向拟合 `Z` 的正向通道，
减弱方向通过显式 reverse contrast 单独拟合；负向/不匹配 gene 保留在完整 signed
residual 中。不得把“减弱的激活”称为“主动抑制”。ADR-009 在 G2 前比较并冻结
该方案与 signed-coefficient/双通道备选；v0.4 有符号 `G_S` 另行定义 repression。

### Step 6：联合稀疏分解

对 receiver `r` 联合上下文拟合：

```math
Z_{g,r} \approx B_{g,r}\alpha_{g,r}, \qquad \alpha_{g,r} \ge 0.
```

目标函数包含加权残差、L1、ridge 和 graph-fused 项：

```math
\min_{\alpha \ge 0}
\sum_g \left\|Q_{g,r}^{1/2}(Z_{g,r}-B_{g,r}\alpha_{g,r})\right\|_2^2
+ \lambda_1\|\alpha_r\|_1
+ \lambda_2\|\alpha_r\|_2^2
+ \lambda_F\sum_{(g,h)\in E(G_C)}w_{gh}
  \|\alpha_{g,r}-\alpha_{h,r}\|_1.
```

这里的 `Z` 表示 Step 5 冻结的 direction-compatible channel，而不是未经处理的
任意 signed 向量。MVP 用 positive elastic net 和 CVXPY 验证；生产版在 golden
parity 建立后实现稀疏 incidence-matrix ADMM/proximal solver。

### Step 7：sender 独立归因

对 context/LR/receiver 的候选 sender 组合 ligand availability、cell-type
specificity、跨 subject prevalence 和调整 context/batch 后的 coupling。soft
assignment 输出权重、entropy 和支持度。receiver attribution 不允许偷带 sender
身份先验以伪造可辨识性。

当前 common-sender v3 先接受 producer-owned `FrozenInteractionUniverse`，再冻结
每个 receiver x interaction 的非空 candidate-sender tuple；每个 receiver 必须覆盖
完整 interaction universe，训练行若落在 manifest 之外则拒绝。ligand contrast
support 在 interaction 层计算，而不是逐 sender 检验：先在
sender x subject x context 内平均重复 sample/technical rows，再在冻结候选 sender
间取最大值。只有 contrast 全部 context 有限的 subject 进入估计，保持原 contrast
权重且 subject 等权，不因缺失 context 重归一化。

对每个 interaction，以至少 `max(2, min_subjects)` 个 complete subject 做单侧
Student-t 检验，原假设边界由预先冻结的
`ligand_contrast_minimum_effect` 给出。每个 outer-training fold x receiver x
contrast 的全部冻结 interactions 组成一个 Holm step-down family；`not_estimable`
interaction 以 effective p=1 留在 family size `m` 中，不得在看见结果后缩小全集。
只有 `holm_adjusted_p < 1 - ligand_contrast_confidence_level` 才为 `supported`。
这些 p 值只服务于训练 gate，不是结果层 inferential p/q，也不声称跨 receiver、
contrast、fold 或 repeat 的全局错误率控制。

### Step 8：共同计分 functional 与 OOF score

每个训练折针对注册 contrast，从 attribution/sender 结果生成一个共同
`ScoringFunctional`。同一 functional、feature set、transform 和 scale 必须用于
所有被比较 context。在测试 subject 中用冻结 functional 计算 target activity、
availability 和 sample communication score；合并折和 repeat 形成 OOF table。

- `state`：不乘 abundance，无 eligible cell 时为 `NA`。
- `ecosystem`：two-part abundance x conditional state 的 capture-weighted proxy。

context-specific mechanistic weights 可保存为 attribution 结果，但不能直接作为
不同 context 的可比较 outcome。v0.2 提供最小一次 subject cross-fitting 和
exploratory OOF strength；v0.3 使用 repeated cross-fitting 进入正式校准。

family-common v2 的同一 interaction row 必须同时通过 hard receptor eligibility 和
冻结 ligand-contrast gate，才可进入 family availability 与 member evidence。
precedence 固定为：无 receptor-eligible interaction ->
`receptor_family_ineligible` 结构零；无 supported 且存在 receptor-eligible
`not_estimable` gate -> `ligand_contrast_not_estimable`；全部 eligible gate 已观察但
unsupported -> `ligand_contrast_not_supported` 结构零；只有其后才进入 family
selection、availability、incremental gain 和 soft-min。

若同一 family 同时含 supported 与 receptor-eligible `not_estimable` member，且
family core 在此前 precedence 下为正，则 core 可仅基于 supported rows 保持
observed，但 within-family entropy、member weights、所有需分配的 member score 和
sender-resolved descendants 必须 fail closed 为 `not_estimable`，不得在不完整
denominator 上归一化。更早成立的 family structural zero 继续为精确零；
receptor-ineligible 或已观察为 unsupported 的 member 也为精确零，不能借用 family
score。

该链的兼容边界为 common-sender parameter/functional schema `3.0.0`、interaction
support/gate identity schema `3`、family-common producer/binding/edge-evidence `v2`
和 `score_version=family_first_mechanistic_ligand_contrast_gated_softmin_v2`。旧版
in-memory artifact、cache 和持久化 family rows 必须从 raw fold scope 重拟合，不能
与新 ID/score 混合。

### Step 9：full-pipeline resampling、概率和正式检验

`comm_strength` 使用 availability、downstream、sender 和 prior-quality 的加权几何
组合并保留全部分量。workflow 按唯一预注册 null 重跑完整 pipeline，生成
`NullScoreDistribution`；inference 只消费该 artifact。active-edge posterior 的
主 null 是 degree/evidence-matched LR-target reassignment；context-effect permutation
和 sender-label null 分别服务于另外的 estimand，不得混合。

正式 point effect 对 common-functional OOF score 拟合未惩罚 subject-aware model。
repeated cross-fitting 和 full-pipeline subject bootstrap/permutation 重新执行过滤、
gating、聚类、tuning、attribution、sender、scoring 和 effect model，以纳入 nuisance
学习不确定性。只 bootstrap 固定 OOF 表不足。

先进行 primary family omnibus，再使用 ADR-012 唯一冻结且经模拟验证的层级 procedure
做 post-hoc；不能把 stage-wise、gatekeeping 等当可随意互换的插件，也不能假定
“omnibus 筛选后普通 BH”天然控制整体 FDR。频率学 q-value gate 与 local-FDR
posterior gate 独立。

### Step 10：signature 和残差

输出 receiver context、LR-attributed、完整 sender-LR-receiver 三层 signature。
预测贡献与观察方向一致规则显式版本化，同时保存：

```math
R_{g,r} = Z_{g,r} - \widehat{Z}_{g,r}.
```

残差代表 cell-autonomous effect、未知信号、资源缺失、signed prior 不匹配或模型
误差，是一级结果。state/ecosystem differential result 进一步按版本化规则分为
`state_driven`、`abundance_driven`、`mixed`、`unstable` 或
`not_classifiable`；不可估计结果不能硬分类。

### Step 11：结果超图和患者特征

将 context、sender、ligand/complex、receptor/complex、receiver 和 gene/TF/program
组织为有向异质超边。超图在拟合完成后构造，用于 hub/module/gradient/jump 和患者
特征提取，不参与 v1 核心统计拟合。

---

## 6. 分阶段实施路线

估时假设：至少 1 名统计方法负责人、2 名 Python/算法工程师和 1 名测试/benchmark
工程师可并行投入；以下为相对工作量，不是发布日期承诺。任何决策门未通过时，
版本不得因日期压力自动升级。

### Phase 0：工程和方法规范基线（1-2 周）

**目标**：让多人可以在稳定契约上并行开发。

| ID | 工作包 | 负责模块 | 依赖 | 交付物 |
| --- | --- | --- | --- | --- |
| P0-01 | 确认项目/类/API 命名 | `api`, `core`, docs | 无 | ADR-001 |
| P0-02 | 恢复核心数学公式和符号表 | docs/methods | `prompt.md` | 审阅后的 method spec |
| P0-03 | 建立 `pyproject.toml`、src layout、版本来源 | build | P0-01 | 可安装空包 |
| P0-04 | 定义 core dataclass/Protocol/错误/枚举 | `core` | P0-01/02 | 类型契约 |
| P0-05 | 定义 config/provenance/result JSON Schema v0 | `schemas`, `results` | P0-04 | schema fixtures |
| P0-06 | 建立 Ruff、mypy、pytest、pre-commit、nox | tooling | P0-03 | 本地质量命令 |
| P0-07 | 建立 PR/nightly/release CI 骨架 | CI | P0-03/06 | workflow |
| P0-08 | 建立 synthetic fixture generator 骨架 | tests/benchmarks | P0-04 | 固定 seed tiny fixture |

**G0 验收门**：

- wheel/sdist 可以构建并在干净环境导入；
- 公共占位 API、配置和 schema contract tests 通过；
- 核心公式已由统计负责人审阅，不再依赖损坏 Markdown；
- CI 运行 format/lint/type/unit/build/doc smoke；
- 没有患者数据、下载资源或绝对路径进入 Git。

### v0.1：统计 baseline（6-8 周）

**目标**：冻结输入、设计、pseudobulk、baseline response/availability 和结果格式。
此版本全部输出为 exploratory/diagnostic；正式 p/q/posterior 字段保持 `NA`，
直到 v0.3 的独立发布门通过。

| ID | 工作包 | 负责模块 | 依赖 | 验收要点 |
| --- | --- | --- | --- | --- |
| V01-01 | counts/normalized AnnData schema 和 gene mapping | `data` | P0 | sparse/backed、不原地修改、强制 mode |
| V01-02 | subject/sample/context 设计审计 | `data`, `design` | V01-01 | rank/confounding/support report |
| V01-03 | pseudobulk counts/detection/abundance | `pseudobulk` | V01-01 | 手算和 sparse/dense 一致 |
| V01-04 | sampling zero/QC/absence 与 eligibility 状态机 | `pseudobulk` | V01-03 | state 不补零、two-part 字段 |
| V01-05 | ContextGraph constructors/product | `design` | P0 | chain/complete/custom/disconnected |
| V01-06 | formula、EMM 和 contrast matrix | `design` | V01-02/05 | global/local/factorial 可估计性 |
| V01-07 | 独立样本 receiver gene diagnostic backend | `response` | V01-03/06 | effect/SE/continuous Z/无正式 p/q |
| V01-08 | LR/complex/precomputed prior adapter | `resources` | P0 | manifest/checksum/namespace |
| V01-09 | soft-min 与 two-part availability | `availability` | V01-03/08 | limiting subunit/missingness |
| V01-10 | state/ecosystem availability estimate | `availability` | V01-09 | proxy 标签、abundance-only null |
| V01-11 | 单 LR target enrichment baseline | `attribution` experimental | V01-07/08 | 与手算/参考实现一致 |
| V01-12 | result schema、atomic writer、query | `results` | P0/V01 outputs | keys/null/provenance/round-trip |
| V01-13 | validate/dry-run/fit baseline workflow | `workflow`, `api` | V01-01..12 | tiny h5ad E2E |
| V01-14 | 基础 QC/context/result visualization | `visualization` | V01-12 | 不重算统计量 |

**G1 验收门**：

- 输入非法、sample 映射非单值、design 秩亏、contrast 不可估计时有明确失败；
- global/local/factorial contrast 与手算 EMM 一致，且不受 cell count 权重支配；
- pseudobulk counts、detection、proportion 在 dense/sparse 路径逐元素一致；
- abundance-only null 中 state 基本不变而 ecosystem 改变；
- 输出主键唯一，所有值可追溯到输入/config/resource/schema/code digest；
- normalized-only 不求和 log data，强制 exploratory；
- 未支持的重复测量设计不得伪装成正式 inferential result；
- 所有 p/q/posterior 字段保持 `NA`，G1 不授权任何正式统计声明。

### v0.2：核心联合归因（8-12 周）

**目标**：在最小 subject cross-fitting 基础上证明 joint sparse + graph fused +
sender separation 的方法学增益；所有真实数据结果仍为 exploratory。

| ID | 工作包 | 负责模块 | 依赖 | 验收要点 |
| --- | --- | --- | --- | --- |
| V02-01 | directional-response 与 signed-prior ADR | docs/attribution | v0.1 | 非负 basis 语义闭合 |
| V02-02 | estimability-aware subject fold planner | `resampling`, `design` | v0.1 | 每折秩/support、合法 K |
| V02-03 | fold-scoped leakage barrier | `workflow` | V02-02 | train-only transforms |
| V02-04 | gated basis 标准化与 sparse contract | `attribution` | V02-01/03 | fold-safe gating |
| V02-05 | LR cosine clustering/equivalence class | `attribution` | V02-04 | family 与 uncertainty |
| V02-06 | positive elastic-net 无图基线 | `attribution` | V02-04 | sklearn/reference parity |
| V02-07 | CVXPY graph-fused 原型 | `attribution` | V02-06/design | convex/KKT diagnostics |
| V02-08 | graph-aware tuning/stability selection | `attribution` | V02-07 | deterministic、无泄漏 |
| V02-09 | explained/predicted/residual response | `signatures` | V02-06/07 | 完整 signed residual |
| V02-10 | sender specificity/prevalence/coupling | `sender` | v0.1/V02-03 | frozen candidate manifest、context/batch null |
| V02-11 | common-sender v3 gate 与 soft assignment | `sender` | V02-10/attribution | subject-equal complete case、receiver-wise Holm、不可辨识 sender |
| V02-12 | family-common v2 contrast-level scoring functional | `scoring` | V02-08/11 | receptor x ligand gate、跨 context 同量尺、member fail-closed |
| V02-13 | minimal OOF exploratory strength | `workflow`, `scoring` | V02-02/12 | fold provenance、无 p/q |
| V02-14 | LR/full signatures 与 exploratory hypergraph | signatures/network | V02-09/11/13 | stable directed hyperedges |

**数值验收**：

- 小型凸问题对 CVXPY 参考解：目标函数相对误差 `< 1e-6`、系数最大误差
  `< 1e-4`；solver residual 默认目标 `<= 1e-5`；
- `lambda_F=0` 与无拓扑模型一致；`lambda_F` 极大时同一连通分量趋同；
- 断连分量独立，context 重命名/重排不改变映射后的结果；
- topology jump 不被过度抹平；高共线 LR 主要在 family 层正确恢复；
- predicted contribution + residual 在容差内重构 observed response；
- 每个训练折可估计且 train/test subject 零交叉；
- 同一 contrast 的所有测试 context 使用相同 `scoring_function_id`；
- 每个 receiver 的 Holm family 与 `FrozenInteractionUniverse` 完全一致，缺失/NE
  interaction 不缩小 multiplicity denominator；
- supported 与 eligible-NE member 混合时 family core/member allocation 的状态分离
  符合冻结 precedence，且不破坏 observed sender conservation；
- signed negative response 不会被非负 prior 错误宣称为解释完成。

**G2 方法学决策门**：

模拟前锁定 primary manifest：chain 与 product graph、smooth-gradient 与
single-local-jump、两个 SNR 和两个 LR-collinearity 水平，按 scenario cell 等权。
唯一 primary metric 是 `driver_family x context` recovery 的 macro-AUPRC。相同 seed
配对比较 fused 与 unfused，平均 improvement 的 95% bootstrap 下界必须 `>= 0.02`；
no-topology 和 wrong-topology 集合的非劣下界必须 `>= -0.02`。context assignment、
AUROC 和 jump localization 仅为 secondary，不得替代 primary gate。scenario、
metric、聚合、margin 和 bootstrap 方法在看结果前写入 manifest；未通过时 graph
fusion 保留 opt-in experimental，不得宣传为默认核心优势。

### v0.3：重复交叉拟合和统计校准（8-12 周）

**目标**：把 exploratory attribution 转换为可校准的 subject-level inference。

| ID | 工作包 | 负责模块 | 依赖 | 验收要点 |
| --- | --- | --- | --- | --- |
| V03-01 | repeated estimability-aware cross-fitting | `resampling`, workflow | v0.2 | repeat/fold support 与 seed |
| V03-02 | frozen common-functional OOF scoring | `scoring` | V02-12/V03-01 | context 同量尺，不靠 scale 补救 |
| V03-03 | repeated-measure receiver backend | `response` | design | clustered/GEE；subject-FE 吸收效应时拒绝 |
| V03-04 | OOF effect backend（CR2/GEE/WLS） | `inference` | V03-02/design | small-cluster diagnostics |
| V03-05 | 固定 hypothesis universe/family/procedure | inference/docs | ADR-012 | primary/secondary 和 filtering |
| V03-06 | omnibus 与唯一 hierarchical post-hoc | `inference` | V03-04/05 | 未惩罚 point effect |
| V03-07 | `ExchangeabilityMap` 和合法 permutation plans | `resampling` | design | factor-specific operations |
| V03-08 | full-pipeline subject bootstrap | `workflow` | V03-01 | 重跑 nuisance + effect model |
| V03-09 | full-pipeline context permutation | `workflow` | V03-07 | context-effect null scores |
| V03-10 | active-edge primary null rerun | workflow/scoring | ADR-013 | `NullScoreDistribution` |
| V03-11 | resampled attribution/selection events | workflow/attribution | V03-08 | `ResampledAttribution` |
| V03-12 | effect/SE/empirical p/hierarchical q | `inference` | V03-06/08/09 | nuisance-aware calibration |
| V03-13 | specificity support/selection frequency | inference/results | V03-08/11 | delta 预注册、正确 grain |
| V03-14 | local FDR active posterior/fallback | `inference` | V03-10 | 独立 G3-P gate |
| V03-15 | change class 与 inferential report | results/visualization | all | state/ecosystem/uncertainty |

**G3-F 频率学 effect/p/q 发布门**：

- 每个预注册完全 null 场景至少 1,000 次重复；`alpha=0.05` type-I error 的
  one-sided 95% Monte Carlo 上置信界 `<= 0.06`，保守性通过独立 power/interval-
  width 非劣测试评估，而不是要求 CI 必须包含 0.05；
- FDR 在含真阳性、0.5%/1%/5%/10% 稀疏度及预注册相关结构的 mixed scenarios
  评估；目标 `q=0.05` 时 empirical FDR 的 one-sided 95% Monte Carlo 上界
  `<= 0.07`。global null 只作为 FWER-like 边界检查，不能代替 mixed FDR；
- 95% CI coverage 的 one-sided 95% Monte Carlo 下界 `>= 0.92`，且 interval
  width 相对预注册 baseline 不劣于 margin；
- 正确 exchangeability full-pipeline permutation 的 QQ、分位数和尾部错误率达标；
- rank-deficient、完全混杂、cluster 过少、fold 不可估计和 leakage case 全部拒绝。

通过 G3-F 后可启用 effect/SE/p/q；未通过时这些字段为 `NA`，但不影响已经通过
独立校准的 active posterior。

**G3-P `comm_probability` 发布门**：

- 唯一 active-null、strata 和 candidate universe 已冻结；运行时每个 calibration
  stratum 默认至少 200 个 eligible statistics，不足则返回 `NA`；
- release simulation 覆盖上述 non-null prevalence 和相关结构，且至少 1,000 次
  replicated datasets；
- 相对 prevalence-only predictor 的 Brier score 至少改善 5%；
- ECE point estimate `<= 0.05` 且其 one-sided 95% upper bound `<= 0.07`；
- calibration-in-the-large 绝对值 `<= 0.02`，slope 位于 `[0.8, 1.2]`；
- 这些阈值和 probability strata 在看模拟结果前由 ADR-013 最终锁定。

通过 G3-P 后才启用 `comm_probability`。G3-P 失败时仍可发布通过 G3-F 的合法
q-value，并回退到 empirical active-null score/p；G3-F 失败也不自动否定独立通过
G3-P 的 active posterior，但 UI 必须清楚区分 active 与 differential inference。

### v0.4：分子网络与多视图（6-10 周）

**目标**：用有向带符号 `G_S` 扩展 receptor -> TF -> gene 影响，并验证多视图
确有增益而非复杂度堆叠。

| ID | 工作包 | 负责模块 | 依赖 | 验收要点 |
| --- | --- | --- | --- | --- |
| V04-01 | OmniPath manifest/adapter/license | `resources` | v0.1 | version/checksum/offline |
| V04-02 | 有向带符号 signaling graph | `resources` | V04-01 | sign/direction preserved |
| V04-03 | PageRank/RWR/diffusion backend | `resources` | V04-02 | toy network analytical cases |
| V04-04 | training-fold intracellular gating | resources/availability | V04-03 | 无 outcome circularity |
| V04-05 | CollecTRI/TF activity | `response` | resource adapter | direction/uncertainty |
| V04-06 | pathway activity | `response` | resource adapter | gene-set versioning |
| V04-07 | gene/TF/pathway multi-view loss | `attribution` | V04-04..06 | view weights/tuning |
| V04-08 | resource replacement 与 ablation | benchmarks | all | sensitivity report |

**G4 决策门**：

- toy network 上激活/抑制、不可达节点、路径衰减与解析预期一致；
- outcome label permutation 后无 circular gating 残余信号；
- 模拟前锁定 primary mismatch manifest：缺失 direct gene edges 的 TF-mediated
  activation 与 signed repression，各含两个 SNR 和两个 prior-missingness 水平；
- 唯一 primary metric 是 signed target-program recovery macro-AUPRC，scenario cell
  等权；multi-view 相对 gene-only 的 paired improvement 95% simultaneous-bootstrap
  下界 `>= 0.02`；
- G3-F type-I/FDR 上界相对 gene-only 的恶化不超过 `0.005`；
- LOSO stability、cosine similarity 和 pathway recovery 为预注册 secondary，不能
  事后替换 primary gate。

未通过时 multi-view 保持 opt-in，默认无损退化到预计算 gene-view prior。

### v0.5：生产化与发布候选（6-8 周）

**目标**：在冻结 v1 API 前完成性能、恢复、CLI、文档和竞品 benchmark。

| ID | 工作包 | 负责模块 | 依赖 | 验收要点 |
| --- | --- | --- | --- | --- |
| V05-01 | sparse incidence ADMM/proximal solver | attribution | CVXPY golden | parity/convergence |
| V05-02 | receiver-level parallelism | workflow | V05-01 | serial parity |
| V05-03 | checkpoint/resume/cache invalidation | workflow/results | prior versions | output digest parity |
| V05-04 | CLI validate/dry-run/fit/export | cli/api | stable workflow | exit codes/log safety |
| V05-05 | Parquet/Zarr/GraphML schema migration | results/schemas | v0.3 | compatibility suite |
| V05-06 | isolated competitor environments | benchmarks | manifests | reproducible adapters |
| V05-07 | documentation/tutorial/limitations | docs | stable APIs | strict docs build |
| V05-08 | medium real-cohort release rehearsal | all | approved data access | end-to-end report |

**G5 发布候选门**：

- ADMM 与 CVXPY golden cases 在声明容差内一致，未收敛不输出 success；
- 固定 seed、线程和依赖时可复现；serial/parallel 在容差内一致；
- checkpoint 恢复结果与完整运行的语义摘要和关键数组 hash 一致；
- 记录目标规模的 wall time、peak RSS、磁盘量和扩展曲线；
- 一个中等规模真实队列从 h5ad 到全部输出成功演练；
- 所有负对照、资源替换和竞品比较形成版本化报告。

### v1.0：稳定接口与临床衔接（8-12 周）

**目标**：稳定公共 API/schema，输出患者通信特征和可外部投射 signature。

| ID | 工作包 | 负责模块 | 依赖 | 验收要点 |
| --- | --- | --- | --- | --- |
| V10-01 | 冻结公共 Python/CLI API | api/cli | G5 | SemVer contract |
| V10-02 | edge/module/hub/gradient 患者特征 | network/results | v0.5 | OOF/frozen features |
| V10-03 | stable LR/TF/gene signature export | signatures | v0.4/5 | namespace/missing genes |
| V10-04 | bulk external projection interface | api/signatures | V10-03 | frozen weights |
| V10-05 | survival adapter | optional package/API | V10-02/03 | nested CV contract |
| V10-06 | external cohort validation workflow | benchmarks/docs | frozen model | no retuning on outcome |
| V10-07 | method/API/reproducibility release docs | docs | all | review + archived report |

**G6 v1.0 门**：

- 患者特征来自 OOF 或完全冻结模型；
- survival 评估采用 nested CV，标准化、选择和 tuning 全在内层；
- 报告 time-dependent C-index、integrated Brier、calibration 和 censoring
  diagnostics；
- 外部投射冻结 gene set、方向和权重，处理缺失基因，不在外部 outcome 上重选
  signature 后仍称独立验证；
- 原始 h5ad 到发布结果可在锁定环境中复现，包含 input/resource hash 和运行日志。

---

## 7. 测试与质量策略

### 7.1 测试分层

| 测试层 | 目的 | PR / nightly / release |
| --- | --- | --- |
| unit | 纯函数、公式、边界、错误 | PR |
| contract | 跨模块类型、schema、API、持久化 | PR |
| property | 排序/重命名不变性、稀疏/稠密等价 | PR + nightly |
| integration | tiny h5ad 到结果目录 | PR smoke + nightly full |
| statistical | type-I/FDR/coverage/probability | nightly + release |
| negative control | 生物学/技术 null 行为 | nightly + release |
| regression | golden numeric summary、schema migration | PR |
| performance | time/RSS/storage/scaling | nightly + release |
| docs | 示例和 API 文档可执行 | PR |

### 7.2 必须手算/属性测试的对象

- pseudobulk counts、library size、detection、cell proportion；
- global/local/factorial contrast weights 和 EMM；
- ContextGraph chain/product/disconnected components；
- multi-subunit soft-min limiting behavior；
- LR target basis normalization 和 equivalence clustering；
- graph-fused objective、KKT、primal/dual residual；
- sender soft assignment normalization 和 entropy；
- frozen receiver x interaction candidate-sender manifest 完整性、row-order
  invariance、interaction-level sender max、subject-equal complete-case 和 Holm
  family 联合重算；
- `not_estimable` interaction 以 effective p=1 保留在 Holm denominator，伪造
  support/gate lineage 必须拒绝；
- communication geometric score 和 missingness propagation；
- family-common v2 receptor/ligand-gate precedence，以及 mixed supported+NE
  member allocation fail-closed 与 sender conservation；
- common `ScoringFunctional` 跨 context 一致性；
- full-pipeline resample 确实重跑 filtering/tuning/attribution；
- contribution + residual 重构；
- multiple-testing family、schema primary key 和 migration。

### 7.3 覆盖率目标

- 全包 line coverage：`>= 85%`。
- `data/design/attribution/inference/resampling` 分支覆盖：`>= 90%`。
- 关键拒绝路径（leakage、rank deficiency、checksum、solver failure）：100% 场景覆盖。
- 覆盖率不能替代 statistical calibration 和 negative control。

### 7.4 随机测试规范

- 每次运行记录根 seed 和派生 seed tree。
- 统计测试保存 repetitions、Monte Carlo interval 和预期操作区间。
- release null/calibration 至少 1,000 重复；power/robustness 场景至少 200
  重复，除非预注册 power analysis 要求更多。
- 不使用单次 KS p 值作为概率校准的唯一证据，同时检查 QQ、分位数、尾部、
  Brier、ECE 和 calibration slope。

---

## 8. 负对照与模拟 benchmark

### 8.1 强制负对照

| 场景 | 数据变化 | 预期结果 |
| --- | --- | --- |
| global null | 无真实 context/communication effect | p 校准、FDR 受控、无系统 edge |
| end-to-end context permutation | 按合法 exchangeability 置换 context 并重跑全链 | G3-F type-I/FDR 达标 |
| donor-only null | 固定 subject 数，只增加每个 subject 的 cell 数 | 不产生虚假的有效样本量和显著性 |
| abundance-only | 只变 cell proportion | ecosystem 变，state 基本不变 |
| receiver-autonomous | 直接改变 receiver genes | response 检出，通信归因弱，residual 上升 |
| ligand-only | sender ligand 上升 | availability 可升，functional probability 不应升 |
| target-only | receiver target 上升但无 sender ligand contrast support | downstream signal 可见；无 support 时 integrated family 为结构零，support 不可估时传播 NE，不能从同 family 其他 member 借分通过 |
| receptor knockout | receptor/必要亚基缺失 | 对应 integrated score 被门控 |
| prior shuffle | degree-matched target prior 置换 | recovery 近随机，probability 不虚高 |
| composition imbalance | cell 数/比例强不平衡 | 不产生大规模 state 假阳性 |
| topology jump | `A ≈ B`、`C` 突升 | 保留 B-C jump，不过度平滑 |
| wrong topology | 用户图错误 | 给敏感性警告，无融合基线可比较 |
| disconnected graph | 分量独立 | 信号不跨分量泄漏 |
| confounded null | context 与 batch 部分/完全混杂 | 可校正时控制错误；完全混杂拒绝估计 |
| context-correlated missingness | 捕获失败概率随 context 变化 | two-part model 标记偏差，不把缺失当 ecosystem zero |
| fold-functional null | 各 context 错用不同训练权重会造假差异 | 正确共同 functional 控制 type-I，错误路径被 contract 禁止 |
| paired exchangeability | paired context 受限 swap/sign flip | 正确 full-pipeline procedure 控制 type-I/FDR；cell-level permutation 永久禁用 |
| annotation perturbation | cell label 噪声 | 性能退化曲线和不稳定 edge 标记 |

### 8.2 分层模拟生成器

不要只模拟最终 score。生成链应包括：

1. subject covariate、随机效应、配对 context 和 batch；
2. logistic-normal 或 Dirichlet-multinomial cell abundance、真实 presence 和
   context-correlated capture/sampling-zero process；
3. 经验 library size、NB dispersion 和 pseudobulk counts；
4. ligand/receptor complex availability；
5. 可控稀疏度、方向和共线性的 LR-target basis；
6. shared、context-specific、smooth、jump、factorial interaction pattern；
7. sender-driven、receiver-autonomous 和 abundance-driven components；
8. dropout、低表达、annotation error、batch、固定 donor 下 cell 数增加、样本/
   细胞数不平衡；
9. prior omission、错误 topology 和非线性/饱和模型错配。

维护两套生成器：

- **well-specified**：与拟合模型一致，定位实现和统计错误；
- **misspecified**：加入非线性、未知 pathway、错误 prior 等，评估鲁棒性。

### 8.3 指标

| 类别 | 指标 |
| --- | --- |
| interaction recovery | AUROC、AUPRC、top-k precision、family recovery |
| context recovery | context assignment accuracy、gradient/jump localization |
| calibration | type-I、empirical FDR、coverage、Brier、ECE、slope |
| signature | gene/TF AUROC、cosine similarity、direction accuracy、explained variance |
| stability | LOSO、bootstrap selection、downsampling、annotation/resource replacement |
| performance | wall time、peak RSS、storage、LR x receiver x context scaling |

### 8.4 外部方法比较

至少比较：CellChat、CellPhoneDB、NicheNet、MultiNicheNet、LIANA+、
Tensor-cell2cell、scHyper 和简单 pseudobulk ligand x receptor product。

公平性规则：

- 每个方法在隔离 Conda/container 环境运行；
- 同时做“统一 LR 资源”和“方法默认资源”两套比较；
- 固定方法/资源版本、seed、数据 hash、硬件、线程和失败状态；
- 只比较方法真实输出的量，不能把 rank 强行当 calibrated p；
- Git 跟踪 adapter、config、manifest 和 compact metrics，不跟踪 raw output。

---

## 9. 性能与优化计划

### 9.1 正确性优先的 MVP

- 无拓扑用 positive elastic net；
- graph fused 用 CVXPY 建立数值 golden；
- 稀疏矩阵保存 pseudobulk/prior/basis；
- 以 receiver 为隔离执行单元；
- 先生成可审计中间结果，再优化内存和并行。

### 9.2 生产 solver

在 G2 通过后实现 sparse incidence matrix `D_G alpha = v` 的 ADMM/proximal
求解，包含 quadratic、soft-threshold、graph total variation 和 non-negative
projection update。每次求解保存：

- objective 分项和单调/收敛轨迹；
- primal/dual/KKT residual；
- iteration、tolerance、初始化、wall time、peak memory；
- convergence/failure reason；
- CVXPY parity case identifier。

### 9.3 优化顺序

1. 表达、receptor eligibility 和最低 subject support 预筛；
2. 高相关 basis 合并为 equivalence class；
3. sparse storage 和 block computation；
4. receiver-level parallelism；
5. fold/bootstrap cache 和 checkpoint；
6. solver 热启动；
7. profiling 后再考虑底层编译优化。

性能基线建立前只报告回归；建立后 wall time 或 peak RSS 退化超过 20% 时阻断
合并，除非 PR 给出经审阅的正确性/功能理由和新基线。

---

## 10. 工程、依赖和 CI

### 10.1 Python 与构建

计划采用：

- Python 3.11-3.13；
- `src` layout；
- Hatchling + `hatch-vcs`，版本单一来源为 Git tag；
- Ruff format/check、mypy、pytest、nox、pre-commit；
- MkDocs strict build；
- wheel/sdist clean-environment smoke test。

Phase 0 ADR 应比较 `uv` 与标准 lock/constraints 工作流，选择后提交 lockfile。
核心依赖保持小，重型能力按 extras 分组：`de`、`optimization`、`activities`、
`plotting`、`docs`、`dev`、`benchmark`。可选 backend 延迟导入。

### 10.2 CI 分层

**Pull request CI**：

- Ruff format/check；
- mypy（数值内部严格，AnnData/DataFrame 边界局部豁免）；
- unit/contract/property tests；
- tiny E2E；
- wheel/sdist build + installed-wheel import；
- `mkdocs build --strict`；
- schema compatibility 和无大文件/凭据检查。

**Nightly CI**：

- 慢速 statistical/negative-control suite；
- resource URL/checksum smoke；
- benchmark smoke；
- ASV/core microbenchmark 与 RSS regression；
- serial/parallel determinism；
- optional backend matrix。

**Release CI**：

- 完整校准和 benchmark report；
- supported Python clean install；
- `twine check`；
- 文档/version/schema 一致；
- tag 只从通过所有 gate 的受保护提交触发；
- PyPI Trusted Publishing，不保存长期 token。

---

## 11. 资源、provenance 与数据治理

### 11.1 Resource manifest

每个 LR、ligand-target、OmniPath、CollecTRI/pathway 资源记录：

```text
resource_id
version/release
species
gene_namespace
source_url
sha256
license
citation
retrieved_at
adapter_version
transformation_log
```

计划用 `pooch` 或等价 checksum-aware cache，支持离线、镜像和 hard failure。
资源 payload 存在平台 cache，不入 Git。

### 11.2 Run provenance

每次 run 至少记录：

- package/Git commit 和 dirty flag；
- config canonical JSON/hash；
- input file/hash、AnnData shape 和 schema report；
- resource manifest/hash；
- result schema version；
- context graph/contrast definitions；
- root seed 和 seed tree；
- folds/resamples/permutations；
- backend/solver versions、thread/hardware 摘要；
- stage timing、cache hit、convergence 和 warning/error status。

日志不得记录 patient-level 表达值、可识别元数据或凭据。

### 11.3 数据策略

- Git 只接受极小、合成、固定 seed、可再分发 fixture；
- 真实队列必须由外部受控路径提供并写入 `.gitignore`；
- benchmark 中间结果、大矩阵、完整数据库、`.venv`、cache、build artifact 不入库；
- 任何真实示例在进入仓库前完成去标识化和再分发许可审查。

---

## 12. 文档计划

`docs/` 最终分为：

```text
docs/
├── adr/
├── concepts/
├── methods/
├── api/
├── tutorials/
├── reference/
├── development/
└── limitations/
```

优先文档：

1. 符号表、核心目标函数和 statistical estimand；
2. inferential/exploratory 决策树；
3. AnnData schema、缺失语义和 design estimability；
4. ContextGraph 和 contrast 指南；
5. cross-fitting/no-leakage 规范；
6. strength/probability/effect/q-value 字段解释；
7. 资源版本、许可和替换敏感性；
8. solver diagnostics 和失败排查；
9. synthetic end-to-end tutorial；
10. 统计限制、非因果性和临床声明边界。

文档中的实现状态必须标记 `implemented`、`experimental` 或 `planned`；所有示例
固定 seed 并在 CI 中执行。

---

## 13. 团队并行与工作流

推荐四条工作流并行，但通过 core contract 和 gate 汇合：

| 工作流 | 主要职责 | 不可自行决定 |
| --- | --- | --- |
| A 数据/设计 | schema、pseudobulk、G_C、contrast、availability | inferential 语义变更 |
| B 算法/资源 | prior、joint attribution、solver、sender、G_S | 普通 p-value 构造 |
| C 统计/验证 | crossfit、resampling、inference、simulation、benchmark | 绕过 leakage/estimability |
| D 平台/产品 | core、workflow、results、API/CLI、CI、docs | 在 facade 重写算法 |

协作规则：

- 先合并 contract/ADR，再并行实现 producer 和 consumer；
- 一个 PR 尽量只改变一个模块契约或一条行为链；
- 跨模块变更必须列出输入、输出、迁移、测试和 benchmark 影响；
- 统计方法 PR 至少一名统计审阅者，schema/API PR 至少一名 consumer 审阅者；
- gate report 由独立于主要实现者的人复核。

---

## 14. 风险登记与缓解

| 风险 | 影响 | 早期信号 | 缓解/决策 |
| --- | --- | --- | --- |
| 生物学重复不足 | 无法正式推断 | 每组 subject/edge support 低 | 自动 exploratory，逐 hypothesis reason |
| 完全混杂/秩亏 | effect 不可辨识 | design rank/alias fail | hard fail，不换 backend 掩盖 |
| 小 cluster 方差偏差 | p/CI 失真 | cluster 数低、CR diagnostic | CR2/模拟校准；不足不报 p |
| LR prior 共线 | driver 不稳定 | cosine 高、bootstrap 互换 | family 结论 + assignment uncertainty |
| 数据库偏差/缺边 | signature 错误 | resource replacement 敏感 | 多资源敏感性 + residual 一级输出 |
| topology 误设/过平滑 | context jump 丢失 | fused/unfused 差异异常 | lambda_F=0 基线 + wrong topology test |
| crossfit 泄漏 | 性能虚高/FDR 失控 | test 标签影响 preprocessing | typed fold scope + automated leakage test |
| context 使用不同计分量尺 | 组差异被写入 score | scoring function ID 随 context 变 | contrast-level common functional + hard contract |
| nuisance 不确定性遗漏 | CI 过窄/p 反保守 | 固定 OOF bootstrap 与 full rerun 差异 | repeated CF + full-pipeline subject resampling |
| outcome-derived gating | 循环论证 | permutation 仍有信号 | baseline/train-only gating |
| interaction support 多重性遗漏 | target-only/噪声 ligand contrast 通过 | candidate 数增加时 gate 阳性增加 | 完整冻结 universe + receiver-wise Holm；NE 仍计入 `m` |
| 不完整 family denominator | supported member 在 NE member 缺失时吸收全部分数 | member 权重和为 1 但 family evidence 不完整 | mixed supported+eligible-NE allocation fail closed |
| 层级检验不控 FDR | 假阳性 | post-hoc family 模拟超标 | 唯一 family/procedure 预注册并校准 |
| local FDR 不稳定 | 假概率 | mixture 不收敛/边数少 | empirical p fallback，probability=NA |
| 多种 null 混合 | posterior 无明确 estimand | null 来源异质 | active/context/sender null 分开 |
| sender 被误解为因果 | 过度解读 | 单 sender 权重过于确定 | evidence/entropy/support 明示 |
| sampling zero 当 absence | ecosystem 偏差 | 缺失随 context 变化 | two-part estimand + capture proxy 标签 |
| signed Z 与非负 prior 不闭合 | 负向响应误归因 | residual 方向性异常 | directional contrast/双通道 ADR + signed tests |
| 计算爆炸 | 无法运行 | basis/bootstrap 内存时间陡升 | 预筛/family/sparse/parallel/cache |
| solver 未收敛 | 错误系数进入结果 | KKT/residual fail | failure status，不输出成功 |
| 资源许可/漂移 | 不可发布/不可复现 | URL/version 未固定 | manifest/license/checksum/cache |
| 临床过拟合 | 无法外部泛化 | 同队列选特征评估 | nested CV、冻结 signature、外部验证 |
| API 过早冻结 | 技术债 | 内部对象频繁泄漏 | v0.x experimental，G5 后冻结 |

---

## 15. 需要 ADR 冻结的决策

| ADR | 决策 | 截止版本 |
| --- | --- | --- |
| ADR-001 | 主类名 `Crychic`、公开 import 面和命名规则 | Phase 0 |
| ADR-002 | core dataclass/array/table 技术选择 | Phase 0 |
| ADR-003 | build backend、版本来源、lock 策略 | Phase 0 |
| ADR-004 | context tuple、interaction ID、result key 编码 | v0.1 |
| ADR-005 | gene namespace 和 many-to-one mapping policy | v0.1 |
| ADR-006 | low-cell、sampling zero、confirmed absence 与 ecosystem two-part estimand | v0.1 |
| ADR-007 | independent-count response backend 与 shrinkage statistic | v0.1 |
| ADR-008 | receiver 与 OOF effect 的 repeated-measure backend、最小 cluster 支持 | v0.3 |
| ADR-009 | signed response、非负 prior、reverse contrast/双通道语义 | v0.2 |
| ADR-010 | LR equivalence、graph tuning、G2 primary metric/scenario/margins | v0.2 |
| ADR-011 | common scoring functional、fold estimability 和所有 train-only transforms | v0.2 |
| ADR-012 | primary hypothesis universe、filtering、唯一 hierarchical FDR procedure | v0.3 |
| ADR-013 | active-edge 唯一 null、local-FDR strata/阈值/fallback 与 G3-P | v0.3 |
| ADR-014 | ExchangeabilityMap、repeated CF 和 full-pipeline nuisance uncertainty | v0.3 |
| ADR-015 | result schema versioning/migration policy | v0.1/v0.5 |
| ADR-016 | parallel backend、determinism 和 cache key | v0.5 |
| ADR-017 | patient feature/survival boundary | v1.0 |

每份 ADR 必须包含问题、选项、证据、统计/兼容性影响、迁移和回滚路径。

---

## 16. Git、版本和发布策略

### 16.1 分支与提交

- `main` 受保护，只通过 PR 合并；
- 功能分支使用 `feature/`、`fix/`、`docs/`、`benchmark/`、`planning/`；
- Conventional Commits，例如：
  `feat(design): add topology-local contrasts`；
- contract、implementation、test、benchmark 尽量形成可独立审阅的原子提交；
- 不改写或回退他人无关改动；
- PR 描述列出 schema/API/statistics/resource/performance 影响。

### 16.2 版本

- SemVer 管包版本；规划骨架不打 release tag；
- v0.1.0 在 G1 通过后发布，v0.2.0-v0.5.0 依次对应上述 gate；
- v1.0.0 只在 G6 和 API/schema compatibility suite 通过后发布；
- result schema 使用独立版本，资源版本也不与包版本绑定；
- tag 必须指向 CI 和 release report 均通过的提交。

### 16.3 Changelog 分类

`Added`、`Changed`、`Deprecated`、`Removed`、`Fixed`、`Security`，另在统计方法
变更时增加 `Statistical behavior` 小节，说明 estimand、calibration 或默认值影响。

---

## 17. 首批迭代顺序

### Sprint 1：契约和构建

1. 完成 ADR-001 至 ADR-003。
2. 创建 `pyproject.toml`、`__init__.py`、版本和最小 wheel。
3. 建立 core config/ID/provenance/seed contracts。
4. 建立 config/result schema v0 和 contract tests。
5. 建立 Ruff/mypy/pytest/nox/PR CI。

### Sprint 2：输入与 pseudobulk

1. 实现不修改输入的 AnnData schema validator。
2. 实现 sample-subject-context 审计和 mode report。
3. 实现 dense/sparse pseudobulk、detection、abundance。
4. 实现 missingness/eligibility 状态机。
5. 建立 tiny synthetic h5ad 和手算 golden tests。

### Sprint 3：design 与 baseline response

1. 实现 ContextGraph constructors/product/serialization。
2. 实现 formula、rank/alias diagnostics、EMM。
3. 实现 global/local/factorial contrasts。
4. 接入首个 independent-sample response backend。
5. 完成输入到 continuous receiver response 的 integration test。

### Sprint 4：资源、availability 和结果

1. 定义 resource manifest/adapter 和 gene namespace mapping。
2. 实现 LR complex 与 precomputed target prior。
3. 实现 soft-min、state/ecosystem availability。
4. 实现 result schema、atomic writer 和 basic query。
5. 跑通 v0.1 tiny E2E 和 G1 negative-control smoke。

关键路径为：

```text
method spec/core contracts
  -> data/pseudobulk/design
  -> estimability-aware fold scope
  -> response/resources/availability
  -> attribution/sender/signatures
  -> common-functional OOF scoring
  -> repeated full-pipeline resampling/inference
  -> calibration gate
  -> production solver/API freeze
```

---

## 18. Definition of Done

任何工作包只有同时满足以下条件才可标记完成：

1. 行为和非目标有文档，公开接口有类型和示例。
2. 输入、输出、缺失、失败和 provenance 契约明确。
3. 单元/contract/property 测试通过，风险相称的统计测试已加入。
4. 不产生 subject leakage，不把结构性缺失当零。
5. 可选依赖缺失时给可操作错误，不破坏 base import。
6. solver/backend 失败显式传播，不产生伪成功结果。
7. schema/API 变更有迁移、changelog 和 consumer test。
8. 性能影响已测量；新增大对象说明内存/持久化策略。
9. resource/data 许可、版本、checksum 和 citation 完整。
10. Ruff、mypy、pytest、build 和适用 docs/benchmark gate 通过。
11. PR 经对应模块和统计/契约审阅者批准。
12. Git 提交不包含患者数据、cache、绝对路径或无关修改。

版本只有在对应 `G0-G6` 决策门有可复现报告、未解决 blocker 为零且失败项明确
降级后，才可发布。路线总原则是：

> 数据与 contrast 正确性 -> 核心归因有增益 -> 推断完成校准 -> 分子网络确有
> 增益 -> 性能与复现达标 -> 冻结 v1.0 API。
