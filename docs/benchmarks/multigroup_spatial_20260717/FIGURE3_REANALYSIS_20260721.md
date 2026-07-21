# Figure 3 DES reanalysis (2026-07-21)

## Conclusion

The previous benchmark was not a faithful Figure 3 reconstruction. Its DES
walk used an unweighted running sum (`gseaParam=0`), whereas the paper reports
`fgsea` without an override and the installed fgsea default is
`gseaParam=1`. This was the main reason that the previous scSeqCommDiff DES
values were too low.

The paper does not publish its Figure 3 script. Four details therefore cannot
be recovered uniquely: `scoreType`, whether raw cardinalities or cardinality
ranks were passed to fgsea, self-pair handling, and fraction rounding. The
Kuppe authors' MISTy importance table is also unavailable. Results below keep
these choices in separate, checksum-bound arms instead of choosing a setting
post hoc for each dataset.

## What Figure 3 requires

Each method has eight DES values: two conditions by top 10%, 20%, 30%, and
40% expected spatial sets. Figure 3 ranks methods by the median of those eight
values.

| Dataset | Scenario | Paper methods |
|---|---|---|
| Kuppe MI | conditions-aware | scSeqCommDiff, CellChat, scDiffCom |
| Kuppe MI | multi-sample conditions-aware | LIANA+, scSeqCommDiff, MultiNicheNet |
| Lerma-Martin MS | conditions-aware | scSeqCommDiff, scDiffCom, CellChat |
| Lerma-Martin MS | multi-sample conditions-aware | scSeqCommDiff, LIANA+, MultiNicheNet |

All methods use the frozen 2,293-pair human ConnectomeDB2020 resource.
MultiNicheNet 2.0.1 is a declared skip because a validated native adapter is
not available. It is never assigned a zero score.

## Paper protocol audit

The scSeqCommDiff adapter matches Supplementary Section S4:

- version 2.0.0 and scSeqComm intercellular scores;
- 1,000 permutations, Wilcoxon, and BH q-value below 0.05 for the
  conditions-aware scenario;
- pseudo-Wilcoxon and raw p-value below 0.05 for the multi-sample scenario;
- maximum intracellular score above 0.5, or all intracellular scores missing;
- condition assignment from the intercellular log-fold-change sign.

The corrected MS input contains 69,168 nuclei from the paper's 5 control and
6 chronic-active tissues. The prior 75,004-cell input contained the unmatched
control sample CO45. The Kuppe spatial truth uses 4 control and 9 ischemic
sections. The corresponding snRNA-seq input has 4 control and 11 ischemic
libraries; this cross-modality sample-count difference is present in the
public data and is not silently collapsed.

LIANA+ was rerun with sample/tissue pseudobulks using LIANA+ 1.5.0,
decoupler 1.8.0, and pydeseq2 0.5.0. Earlier subject-level LIANA+ output is a
separate sensitivity analysis and is not reused as Figure 3 output.

## DES arms

The Figure-compatible report uses a non-negative fgsea walk
(`scoreType="pos"`), `gseaParam=1`, raw interaction cardinality, stable native
fgsea tie ordering, sample-level spatial truth, floor top-set sizes, and no
self-pairs. The paper plots only non-negative DES values, making `pos` more
compatible with Figure 3 than the literal fgsea default `std`.

The following arms remain separate:

- literal fgsea defaults: `scoreType="std"`, `gseaParam=1`;
- cardinality average ranks instead of raw cardinalities;
- the former unweighted `gseaParam=0` walk;
- include-self plus ceil top-set construction;
- strict native pair estimability instead of paper-style zero completion.

For scSeqCommDiff, paper-style rankings set tested or unreported
cell-type-eligible pairs with no significant LR interaction to cardinality
zero. The strict arm requires at least one finite native intercellular p-value
for a pair. Both files are emitted from the same native result.

## Figure-compatible result

These values use the public-data spatial truth and are not claimed to be the
authors' unreleased source values.

| Dataset | Scenario | Method | Median DES | Rank/status |
|---|---|---|---:|---|
| Kuppe | conditions-aware | scSeqCommDiff | 0.403 | 1 |
| Kuppe | conditions-aware | scDiffCom | 0.328 | 2 |
| Kuppe | conditions-aware | CellChat | 0.285 | 3 |
| Kuppe | multi-sample | scSeqCommDiff | 0.701 | 1 |
| Kuppe | multi-sample | LIANA+ | 0.490 | 2 |
| MS | conditions-aware | scDiffCom | 0.453 | 1 |
| MS | conditions-aware | scSeqCommDiff | 0.413 | 2 |
| MS | conditions-aware | CellChat | 0.395 | 3 |
| MS | multi-sample | scSeqCommDiff | 0.825 | 1 |
| MS | multi-sample | LIANA+ | 0.500 | not ranked (6/8 DES observed) |

Approximate medians read from the published plot for scSeqCommDiff are 0.53,
0.44, 0.51, and 0.82 for Kuppe conditions-aware, Kuppe multi-sample, MS
conditions-aware, and MS multi-sample, respectively. The corrected MS
multi-sample value (0.825) reproduces the plot closely.

Kuppe cannot be reproduced exactly because the paper used a MISTy importance
table supplied by the original authors. On the public-data MISTy
recomputation, changing only the fgsea input from raw cardinality to average
cardinality rank changes Kuppe multi-sample scSeqCommDiff from 0.701 to 0.425,
which is close to the plotted value. The same transformation reduces MS
multi-sample scSeqCommDiff from 0.825 to 0.623, so it is not a defensible
global correction.

## scSeqCommDiff sensitivity

| Panel | Unweighted p=0 | Weighted pos p=1 | Average-rank pos | Include-self/ceil pos | Strict native estimability |
|---|---:|---:|---:|---:|---:|
| Kuppe conditions-aware | 0.224 | 0.403 | 0.424 | 0.499 | 0.306 |
| Kuppe multi-sample | 0.243 | 0.701 | 0.425 | 0.695 | 0.216 |
| MS conditions-aware (5+6) | 0.280 | 0.413 | 0.354 | 0.467 | 0.260 |
| MS multi-sample (5+6) | 0.575 | 0.825 | 0.623 | 0.923 | 0.600 (6/8) |

The old six-control MS arm gave 0.522 and 0.791 for conditions-aware and
multi-sample, respectively. Its apparently close conditions-aware value was
cohort-sensitive and cannot be retained as the paper-matched result.

## Reproducibility limits

The exact Figure 3 script, Kuppe author MISTy table, and exact fgsea argument
list are not publicly available. Therefore this work can establish protocol
compatibility and quantify each ambiguity, but cannot claim byte-identical
reproduction. The primary publication bundle is in
`benchmarks/results/spatial_des_figure3_reanalysis_20260721`; all alternative
arms are indexed in
`benchmarks/results/spatial_des_figure3_sensitivity_20260721`.
