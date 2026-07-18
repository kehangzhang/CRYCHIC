# Multi-group spatial DES benchmark protocol

## Reference

This benchmark reproduces Method 1 from Cesaro et al., "Differential
cellular communication inference framework for large-scale single-cell
RNA-sequencing data", NAR Genomics and Bioinformatics (2025),
doi:10.1093/nargab/lqaf084. Method parameters were recovered from Section S4
of the supplementary material.

Both paper datasets define a two-condition contrast. This reproduction tests
condition-aware and biological-replicate-aware differential CCC workflows; it
does not constitute an omnibus benchmark for an arbitrary number of groups.

## Frozen inputs

| Dataset | Molecular input | Primary contrast | Primary unit | Spatial evidence |
|---|---:|---|---|---|
| Kuppe myocardial infarction | 76,141 nuclei, 29,126 genes | IZ vs CTRL | subject | MISTy 1.3.5 recomputation over 13 public Visium sections |
| Lerma-Martin multiple sclerosis | 75,004 nuclei, 32,115 genes | chronic active vs control | subject | Pearson correlation of public Visium cell-type proportions |

Both molecular inputs retain raw integer counts. Repeated sections from one
person are averaged within subject and condition before subject-level spatial
testing. Sample-level outputs are retained only as sensitivity analyses and are
never ranked against subject-level outputs.

Frozen molecular-input SHA256 values are
`c47112ce01a192bb157af1ba5feb09c1616601e280102fd11a658f570698c926`
for Kuppe and
`612fe9c4cdaf88694a47e13ba4458c828f206cd945eba9e9c72f46e9bd0196c7`
for MS. The corresponding primary expected-set SHA256 values are
`151fadcb1a805a43c1ffb673e6b43ed2019b1c5a6d61c704ef2e1c6a635ed57e`
and `ab7e12a62aeedb343d8abc961741786f2a451549e0828c8e85cc30c7d5c67bd0`.

The Kuppe authors' exact MISTy importance table was not publicly available.
The primary Kuppe truth is therefore a protocol-level recomputation from the
public CELLxGENE spatial inputs, not a byte-identical copy of the authors'
table. The primary view is `spatial_neighbor_max`, the maximum finite
importance across the juxtaview and paraview after the paper-compatible
multi-R2 filter. This limitation is part of the comparison-panel identity.

## Interaction resource

All methods use the same frozen human ConnectomeDB2020 resource: 2,293
directed simple ligand-receptor pairs. The resource table, method-specific ID
maps, and manifest are checksum-bound. A method can mark an interaction or
cell pair `not_estimable`; absent evidence is never converted to a numerical
zero. The shared table SHA256 is
`e781363288a26c15e03246500111bfecb818eef997f5ebe1b936aaa465151c3a`.

## Spatial truth

Cell-pair direction is collapsed to an unordered canonical pair, including
self-pairs. For each condition, spatial cell-pair effects are ranked and the
top 10%, 20%, 30%, and 40% define expected sets. Exact zero spatial effects are
tied and are not expected members.

The condition-aware truth uses the absolute difference between condition
means. The multi-sample truth uses a two-sided Mann-Whitney U test on subjects,
ordered by raw p-value and then absolute effect. No p-value correction is
applied to the multi-sample spatial truth, matching the paper's small-cohort
policy.

## Metric and ranking

The Distance Enrichment Score (DES) is an unweighted positive-running-sum
GSEA analogue over each method's condition-specific unordered cell-pair
ranking. Each method has eight required strata: two conditions by four top
fractions.

The primary summary and rank use median DES across all eight strata. Mean DES
and its separate within-panel rank are secondary summaries. Higher DES is
better. A method is ranked only when all eight strata are observed and finite
and all expected sets are available.

Rankings are isolated by dataset, scenario, expected-set SHA256, truth
variant, analysis unit, condition/fraction strata, cell-pair direction, and DES
semantics. No ranking is allowed across comparison panels.

## Method arms

| Scenario | Method | Version | Paper S4 policy used here |
|---|---|---:|---|
| condition-aware | scDiffCom | 1.1.1 | 1,000 permutations; BH p<0.05; abs logFC>log(1.5); UP/DOWN |
| condition-aware | CellChat | 2.1.2 | separate `computeCommunProb(type="triMean")`; official-vignette ligand-logFC primary plus ligand/receptor-concordant sensitivity |
| condition-aware | scSeqCommDiff | 2.0.0 | 1,000 permutations; Wilcoxon; BH p<0.05; max intracellular score >0.5 or unavailable |
| multi-sample | scSeqCommDiff | 2.0.0 | pseudo-Wilcoxon; raw p<0.05; max intracellular score >0.5 or unavailable |
| multi-sample | LIANA+ | 1.5.0 | decoupler 1.8.0 and pydeseq2 0.5.0 Wald workflow; raw interaction p<0.05 |
| multi-sample sensitivity | LIANA rank aggregate | 1.7.3 | descriptive common-resource sensitivity arm; never ranked as the exact S4 method |
| multi-sample | CRYCHIC | source tree | subject-blocked three-fold descriptive cross-fit; subject-equal condition effect |

CRYCHIC emits descriptive held-out strengths, not formal p-values, in this
benchmark. Its condition-specific ranking is the sum of positive subject-equal
LR strength changes after collapsing communication directions. Independent
outer folds may be scheduled in parallel. The fold-worker count affects only
execution scheduling: fold membership, held-out predictions, aggregation, and
the scientific comparison identity are unchanged.

CellChat's Supplementary S4 prose describes sender-ligand and
receiver-receptor fold-change filtering, while the official 2.1.2 comparison
vignette uses `ligand.logFC=+/-0.05` and `receptor.logFC=NULL`. The benchmark
therefore labels the ligand-only result as the official-vignette primary and
emits a concordant ligand-plus-receptor result as an S4-literal sensitivity.
The two arms are never merged.

The primary LIANA+ arm pins LIANA+ 1.5.0, decoupler 1.8.0, and pydeseq2 0.5.0
and follows the paper's pseudobulk Wald workflow. The legacy LIANA 1.7.3 rank
aggregate over the same resource is retained only as a descriptive sensitivity
arm and must not be presented or ranked as the exact paper reproduction.

MultiNicheNet 2.0.1 is not included in the primary leaderboard because no
validated frozen native adapter was available in this run and its ligand-target
prioritization estimand is not directly interchangeable with the LR-only arms.
This is a declared skip, not a zero score.

## Estimability exceptions

The MS cell type `BC` is absent from controls. Native scSeqCommDiff permutation
code cannot subtract per-condition result tables with unequal cell-type axes.
The adapter applies a symmetric preflight rule requiring at least two cells in
both conditions, retains the full 45-pair universe, and marks all affected
pairs `not_estimable`. CellChat runs each condition on its supported cell-type
composition and uses the official `liftCellChat` alignment workflow before
comparison; condition-specific unsupported pairs remain `not_estimable` rather
than forcing a common cell-type intersection. scDiffCom likewise retains its
declared full pair axis and reports unsupported pairs explicitly.

## Reproducibility contract

Raw H5AD, RDS, and spatial model objects remain outside Git because of size.
The publication bundle contains checksummed manifests, source-data tables,
figures, software versions, method deviations, and the exact code needed to
regenerate the reported summaries from the frozen local outputs.
