# Component-swap benchmark (2026-07-22)

本目录对应 `suggestions_v4.md` 的 RC0 分层诊断。所有输入、
ConnectomeDB2020 LR resource、Figure 3 spatial truth 和 CRYCHIC backbone 均保持
冻结；本轮没有修改或优化算法主体。A 臂与此前正式 CRYCHIC 结果逐行、逐值完全
一致：Kuppe 132 行，MS 90 行。

## Arm 定义与可用性

| Arm | 实际运行定义 | 状态 |
|---|---|---|
| A | CRYCHIC score + sample-level HC2 + one-SE + hard count | exact |
| B | CRYCHIC score + Mann-Whitney + raw p < 0.05 + hard count | exact head swap |
| C | CRYCHIC score + nominal Gaussian HC2 Wald p < 0.05 + hard count | declared proxy |
| D | native per-sample scSeqComm `S_inter` + HC2 + one-SE + hard count | exact head swap |
| E | LIANA statistic + hypergraph reweighting + calibrated posterior count | not estimable |
| F | CRYCHIC effect/SE + no hard selection + continuous signed evidence count | uncalibrated diagnostic |

C 不是 PyDESeq2 的精确复现。CRYCHIC 的 bounded continuous score 不能合法进入
PyDESeq2 negative-binomial raw-count likelihood，因此这里只回答“Wald 式阈值 head”
的局部问题。E 所需的 full-pipeline null 与通过 G3P calibration gate 的 soft
posterior 在当前冻结 run 中不存在；本轮没有用 `1-p` 等任意量冒充 posterior。
F 也不是可发布的后验概率。

## DES 结果

DES 使用与正式 Figure 3 相同的 multi-sample、`scoreType=pos`、
`gseaParam=1`、fgsea-native ties、exclude-self 和 raw-cardinality 口径。

| Dataset | A | B | C | D | F | native scSeq | native LIANA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kuppe | 0.428 (8/8) | 0.478 (8/8) | 0.418 (8/8) | 0.195 (8/8) | 0.419 (8/8) | 0.701 (8/8) | 0.490 (8/8) |
| MS | 0.050 (8/8) | 0.232 (6/8) | 0.393 (8/8) | 0.321 (6/8) | 0.254 (8/8) | 0.825 (8/8) | 0.500 (6/8) |

括号为 observed strata / 8。MS 的 B、D 和 native LIANA 不完整，其 median 只描述
已观察的 6 个 strata，不能进入正式排名。

## 主要判断

1. **MS 的 one-SE head 是主要问题之一，但不是唯一问题。** C 相对 A 从
   0.050 升到 0.393，F 升到 0.254。相同 CRYCHIC edge effect 的 rank Spearman
   为 1，改善来自选择/聚合 head，而不是 edge score 被替换。
2. **不能据此认定 CRYCHIC score 已经足够。** Kuppe 的 B 只提高 0.050，仍低于
   LIANA 和 scSeqCommDiff；MS 的 B 又只有 6/8 strata。native scSeq 的完整结果仍为
   0.825。
3. **把 CRYCHIC head 放到 scSeq score 上会明显损失性能。** D 在 Kuppe 为 0.195，
   远低于 native scSeq 的 0.701。A 与 D 的 common-edge sign concordance / rank
   Spearman 为 Kuppe 0.730 / 0.497、MS 0.739 / 0.391。因此 base score 不同，且
   HC2/one-SE/estimability head 会进一步破坏 scSeq 的 native 优势。
4. **MS 低分可定位到两个被系统性排低的空间命中 pair。** 在 CRYCHIC 的 21 个
   eligible unordered pairs 中，CA 的 `OL | OPC` spatial rank 为 1，但 A rank 为
   19；Ctrl 的 `MG | TC` spatial rank 为 3，但 A rank 为 21。C(p<0.05) 将它们移到
   14 和 4，解释了大部分 DES 改善。
5. **`n_estimable_lr` 不是 A/C/F 内部排序的直接原因。** Kuppe 的每个 A pair 均为
   3,704 条 estimable directed LR，MS 的 21 个 eligible pair 均为 3,930 条，因而
   pair score 与 opportunity 数量的相关性不可定义。B/D 的主要问题反而是 coverage：
   Kuppe 分别只覆盖 54/55 和 34/55 pairs，MS 只覆盖 15/21 和 14/21。

## Threshold path

Threshold path 仅用于定位问题，**属于 post hoc diagnostic，不能在 Kuppe/MS 上选出
阈值后再声称性能提升**。MS 中 A 的 z 阈值从 1 提到 1.96/2.576 时，median DES 从
0.050 变为 0.393/0.649；对应空间命中 pair 的 rank 从 19/21 改为 14/4，再改为
12/3。这说明当前 one-SE cutoff 在 MS 中引入了大量不利于 cardinality ranking 的
边，但任何新 cutoff 必须在独立数据或模拟中预注册和冻结。

## 后续优化方向（本轮未实施）

1. 在独立 development datasets / simulations 上校准 selection head，重点验证
   signed expected cardinality，而不是在 Kuppe/MS 上选择阈值。
2. 完成 LIANA/PyDESeq2 raw-count baseline 的逐行 parity，再评估 hypergraph residual
   gain；这才是精确 C/E 的前置条件。
3. 运行 full-pipeline null、G3P calibration 和 signed posterior，之后再执行 E，避免
   把未校准 evidence 当成 probability。
4. 单独审计 `OL | OPC` 与 `MG | TC` 的 edge composition、sample coverage 和
   condition polarity，并解决 B/D 的 pair coverage 损失。

## 文件

- `spatial_des_summary.tsv`: primary DES、完整性、相对 A 的差值和完整方法排名。
- `coverage_summary.tsv`: 8 strata 完整性、expected-pair coverage 和 ranked pairs。
- `arm_diagnostics.tsv`: sign counts、selected edges、ties、zero pairs 和 opportunity。
- `pairwise_diagnostics.tsv`: edge sign/rank、pair rank 和 top-10 overlap。
- `pair_rank_comparison.tsv`: Kuppe 55 pairs/condition 和 MS 21 pairs/condition 的逐对审计。
- `threshold_path_summary.tsv`: post hoc threshold sensitivity。
- `manifest.json`: 两个完整外部 run、实现和紧凑输出的 SHA256 绑定。

百万级 directed-edge 表、逐 strata DES 和原生逐样本 scSeqComm score 保留在 Git 外；
本目录只包含可审计的紧凑结果。
