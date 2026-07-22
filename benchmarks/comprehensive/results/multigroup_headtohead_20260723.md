# Corrected multi-group head-to-head pilot, 2026-07-23

## Scope

This corrected pilot compares the CRYCHIC generic multigroup baseline,
scSeqCommDiff 2.0.0, CellChat 2.1.2, and LIANA 1.7.3. It supersedes the earlier
local run that incorrectly supplied raw counts to CellChat/LIANA, used the
2,293-pair native scSeqCommDiff resource against a five-pair H-common arm, and
simulated repeated rather than independent groups.

The planted-truth panel has five active seeds and five matched global-null
data sets. Every data set contains three independent groups with 8/10/12
subjects, approximately 5,400 cells, three cell types, five exact four-method
H-common LR pairs, and 45 frozen sender-receiver-LR events per contrast. The
three prespecified contrasts are B-A, C-A, and C-B. CRYCHIC, CellChat, and LIANA
run once on each full A/B/C data set; scSeqCommDiff runs each pairwise contrast.

CRYCHIC reads raw `layers["counts"]`. CellChat, LIANA, and scSeqCommDiff read
CP10K-log1p `X`. All method outputs are bound to clean commit `d5705f5`; the
read-only evaluator correction is bound separately to clean commit `63ac428`.

## Planted Truth

The primary metric is event-level prevalence-adjusted AP at 10% target
prevalence, averaged equally over 15 active seed-contrast combinations. The
observed positive prevalence is 5.19%. Missing events are not zero-imputed,
and all four methods pass the 80% coverage gate.

| Rank | Method | Adj. AP | AUPRC | AUROC | Top-k recall | Direction | Spearman |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | scSeqCommDiff | 0.2212 | 0.1259 | 0.6751 | 0.0222 | 0.2444 | -0.1029 |
| 2 | CellChat | 0.1429 | 0.0762 | 0.5262 | 0.0000 | 0.8333 | 0.1603 |
| 3 | CRYCHIC generic baseline | 0.1025 | 0.0535 | 0.3494 | 0.0000 | 0.0444 | -0.0102 |
| 4 | LIANA | 0.0883 | 0.0456 | 0.1890 | 0.0000 | 0.8000 | 0.0569 |

scSeqCommDiff has the best discrimination but poor signed-effect agreement.
CellChat is second by discrimination and first by direction accuracy and
effect Spearman. CRYCHIC is third and essentially at the 0.10 adjusted-AP
random baseline on this pilot; it does not outperform scSeqCommDiff or
CellChat. LIANA ranks fourth by discrimination but second on direction and
effect correlation.

Only scSeqCommDiff emits native between-condition p-values. Its mean null
`p < 0.05` fraction is 0.040 across 15 null contrasts, but one 45-event
contrast reaches 0.178. Five null seeds are not sufficient for a calibration
claim. The other methods receive NE for type-I error. No method is assigned an
omnibus rank: the current CRYCHIC generic head has no validated omnibus p/q,
CellChat/LIANA are compared through external subject effects, and
scSeqCommDiff is pairwise-only.

## MIS-C And Olink

The real panel contains 13,343 cells, 16 subjects, 11 family blocks, three
conditions, three broad cell types, and an exact 455-LR four-method H-common
axis. CRYCHIC, CellChat, and LIANA consume the full three-group input;
scSeqCommDiff runs M-S, C-S, and M-C pairwise comparisons.

The frozen M-vs-S prediction universe contains 251 ligands; 123 are measured
by the external Olink endpoint. Only LIANA and CellChat score at least 80% of
those measured ligands and enter the primary ranking.

| Primary status | Method | AP | Spearman | NDCG | Measured coverage |
|---|---|---:|---:|---:|---:|
| Rank 1 | LIANA | 0.6504 | 0.1539 | 0.6921 | 1.000 |
| Rank 2 | CellChat | 0.5742 | 0.0651 | 0.6478 | 1.000 |
| Diagnostic | scSeqCommDiff | 0.7227 | 0.1590 | 0.6636 | 0.577 |
| Diagnostic | CRYCHIC generic baseline | 0.6566 | 0.0902 | 0.6868 | 0.577 |

The diagnostic AP values are not ranks because they are computed on only
71/123 measured ligands. In the explicitly post-hoc 71-ligand complete-case
sensitivity, AP orders LIANA (0.7315), scSeqCommDiff (0.7227), CRYCHIC (0.6566),
and CellChat (0.6318). CRYCHIC has the highest sensitivity NDCG (0.6868), but
is third by AP and Spearman. This sensitivity does not replace the primary
coverage-gated result.

## Runtime

Full-command wall times include adapter startup and were measured during the
parallel run. Simulation means are 34.8 s for one full CRYCHIC A/B/C data set,
49.5 s for CellChat A/B/C, 22.5 s for LIANA A/B/C, and 24.0 s for one
scSeqCommDiff pairwise contrast. On MIS-C, the corresponding wall times are
201.1 s, 88.2 s, 30.4 s, and 33.6-37.2 s per scSeqCommDiff contrast. Logged
maximum RSS values are command-level diagnostics, not process-tree sums.

## Interpretation And Limits

- The runnable general CRYCHIC entrypoint is a generic baseline. It is not the
  Kuppe/MS-specific RC9 held-out replay and must not be reported as RC9.
- This is a five-seed pilot, not the registered 200 active/500 null campaign.
- The result evaluates prespecified pairwise effects in a three-group design;
  validated omnibus power and calibrated FDR remain NE.
- MIS-C condition, batch, and family structure are not fully separable.
  scSeqCommDiff does not model family blocks; external point effects have no
  formal population-level p/q values.
- Olink is orthogonal ligand support, not complete LR-event truth, and external
  healthy controls proxy the healthy-sibling reference. The endpoint existed
  locally before this rerun, so no strict prospective blinding claim is made.
- There is no defensible grand winner across planted-event discrimination,
  direction, external-protein support, coverage, and runtime.

## Checksum Bindings

Large outputs remain outside Git under
`benchmark_work/comprehensive_multicontext_20260722/runs/multigroup_headtohead_corrected_20260723`.

- five-seed fixture manifest: `f8590c22bce61d313e811c0eb5d4b4efff76189f021971d619ba11a27e6584ad`
- five-seed evaluation manifest: `f2fe7fcd0de88920844cb2d80837e949922379e53fb7f1f178b1f2a60f627278`
- MIS-C preparation manifest: `a9e7060af2e7bed3a4005b9ac737e31045ef72abed29d924727102f6f00ec869`
- frozen prediction manifest: `dbf619da50170ea5e1ec69a2cd3506779ef0f99f9e08a3760eda9e7f179764a5`
- frozen prediction table: `abc06329eaa470a03f57650d813fbae9f5818ade180b0d88f801f323207bb6e2`
- primary Olink evaluation: `c2b782cbdd78a35eb849a5cf001048ea8c30477d346790ce433c9ecdc092a2db`
- complete-case sensitivity: `607645b511e06dc05a6a43b76063d71ce01cbcdbedc75b318fe2ba86387442e9`
- complete-case Olink evaluation: `8eae9b459230670dd5b49b026cf25ce7f8db9fdc341e2f56912a08f6bd775e05`
