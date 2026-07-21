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
| Kuppe myocardial infarction | 76,141 nuclei, 29,126 genes | IZ vs CTRL | sample/library (4 CTRL + 11 IZ) | MISTy 1.3.5 recomputation over 4 CTRL + 9 IZ public Visium sections |
| Lerma-Martin multiple sclerosis | 69,168 nuclei, 32,115 genes | chronic active vs control | tissue/sample (5 control + 6 chronic active) | Pearson correlation of public Visium cell-type proportions |

Both molecular inputs retain raw integer counts. Figure 3 uses the declared
sample/library or tissue as the multi-sample unit. Kuppe has 4 CTRL + 11 IZ
snRNA-seq libraries but 4 CTRL + 9 IZ spatial sections; this public-data
cross-modality difference is retained rather than silently collapsing repeated
sections. Subject-collapsed outputs are separate sensitivity analyses and are
never ranked against the sample-level Figure 3 outputs.

Frozen molecular-input SHA256 values are
`c47112ce01a192bb157af1ba5feb09c1616601e280102fd11a658f570698c926`
for Kuppe and
`433717d9fd98e57e15a444a338a6e1002f7ca3224022321d28a386c0a8498e1c`
for MS. The corresponding primary expected-set SHA256 values are
`1a9ad459a7c2cf7eb1e52d47ec7f6d77815b28b604b63315fe8524a95fdf7655`
and `402f6e8255cf032a40f456c441d6881d07131d05d21608406475fdf0de0a44a2`.

The Kuppe authors' exact MISTy importance table was not publicly available.
The primary Kuppe truth is therefore a protocol-level recomputation from the
public CELLxGENE spatial inputs, not a byte-identical copy of the authors'
table. The primary view is `spatial_neighbor_max`, the maximum finite
importance across the juxtaview and paraview after the paper-compatible
multi-R2 filter. This limitation is part of the comparison-panel identity.

## Interaction resource

All methods use the same frozen human ConnectomeDB2020 resource: 2,293
directed simple ligand-receptor pairs. The resource table, method-specific ID
maps, and manifest are checksum-bound. Structural ineligibility remains
`not_estimable`. In the strict arm it is never converted to a numerical zero;
the paper-compatible scSeqCommDiff arm separately zero-completes only
cell-type-eligible pairs with no significant interaction. The shared table SHA256 is
`e781363288a26c15e03246500111bfecb818eef997f5ebe1b936aaa465151c3a`.

## Spatial truth

Cell-pair direction is collapsed to an unordered canonical pair. The Figure 3
primary arm excludes self-pairs, matching the Liu et al. DES implementation;
an include-self arm is reported separately because the paper does not state
the background policy. For each condition, spatial cell-pair effects are
ranked and the top 10%, 20%, 30%, and 40% define expected sets. The primary
top-set size uses floor, as in the published Liu code, while ceil is retained
as a sensitivity arm. Exact zero spatial effects are tied and are not expected
members.

The condition-aware truth uses the absolute difference between condition
means. The multi-sample truth uses a two-sided Mann-Whitney U test on the
declared sample/tissue units, ordered by raw p-value and then absolute effect.
No p-value correction is applied to the multi-sample spatial truth, matching
the paper's small-cohort policy.

## Metric and ranking

The paper-compatible Distance Enrichment Score (DES) uses the weighted fgsea
running sum over each method's condition-specific unordered cell-pair ranking.
The paper reports `fgsea` without parameter overrides, so the strict-default
arm uses `gseaParam=1` and `scoreType="std"`. Figure 3 displays non-negative
scores, but the unpublished benchmark script does not reveal whether the
authors explicitly used `scoreType="pos"`; a separate Figure-compatible arm
therefore uses `gseaParam=1` and `scoreType="pos"`. The implementation
reproduces `fgsea::calcGseaStat`, including its uniform-hit fallback when all
expected hit weights are zero and its stable input-order treatment of tied
cardinalities. Each method has eight required strata: two conditions by four
top fractions.

Additional sensitivity arms remain separate: `gseaParam=0` tests the former
unweighted definition, simultaneous equal-strength tie blocks remove
input-order dependence, average cardinality rank tests the paper's ambiguous
"sorted" wording, and include-self/ceil variants test universe construction.
None is merged silently into the Figure 3 comparison.

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
| multi-sample | MultiNicheNet | 2.0.1 | declared skip until a validated native adapter is available; never scored as zero |
| multi-sample sensitivity | LIANA rank aggregate | 1.7.3 | descriptive common-resource sensitivity arm; never ranked as the exact S4 method |
| multi-sample | CRYCHIC | source tree | subject-blocked two-fold descriptive cross-fit; subject-level condition effect |

CRYCHIC emits descriptive held-out strengths, not formal p-values, in this
benchmark. Its exploratory V3 ranking counts sender-specific LR effects whose
absolute subject-level, covariate-adjusted contrast is at least one
contrast-specific HC2 standard error, after collapsing communication
directions. Absolute condition means, signed effects, opportunity-normalized
magnitudes, stable-edge breadth, and the correlation between breadth and the
number of estimable LR edges are reported separately. Independent outer folds
may be scheduled in parallel. The fold-worker count affects only execution
scheduling: fold membership, held-out predictions, aggregation, and the
scientific comparison identity are unchanged.

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

scSeqCommDiff emits two explicitly named ranking files from the same native
result. The Figure-compatible file zero-completes every cell-type-eligible
pair, matching the likely paper/CClens counting universe. The strict
estimability file marks a pair observed only when at least one native
intercellular p-value is finite; a tested pair with no significant interactions
has cardinality zero, while an entirely untested pair is `not_estimable`.
Pair-level finite-test counts are emitted with every run, and the two policies
are never mixed in one panel.

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
