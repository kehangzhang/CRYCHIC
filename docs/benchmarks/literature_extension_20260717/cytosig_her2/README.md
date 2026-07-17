# HER2 CytoSig validation

## Scope and reconstruction status

This run adds the HER2+ arm used by Dimitrov et al. 2022. The Wu GSE176078
input contains 19,311 cells from five patients and 29 minor cell types. Applying
the released author rule of at least 25 cells per type retains 19,275 cells,
29,733 genes and 25 target cell types.

CytoSig activity was reconstructed from the 43-signature centroid using
cell-type pseudobulk counts, the released expression filters, top-500 absolute
signature weights and multivariate linear modelling. BH correction was applied
once over the unique 43-signature by 25-target activity universe, before method
scores were joined. The resulting truth contains 1,375 rows (55 expanded
ligands by 25 targets), including 117 positives.

Truth status is explicitly:
`reconstructed_clean_activity_truth_not_historical_exact`.

## Current-method results

The top-250 endpoint uses the current union/max-imputed protocol extension. Its
168,750-row universe is dominated by CRYCHIC's broader returned STLR coverage,
so it is reported together with the coverage-controlled ligand-target endpoint.

| CRYCHIC arm | Top-250 OR | Fisher p | BH q | Primary rank |
|---|---:|---:|---:|---:|
| availability-state | 1.981613 | 3.023e-05 | 2.418e-04 (primary 8 arms) | 1/8 |
| availability-ecosystem | 1.889963 | 1.084e-04 | 3.251e-04 (all 12 arms) | descriptive |

| CRYCHIC arm | AUROC | AP | Balanced AP mean (SD) | Returned truth rows |
|---|---:|---:|---:|---:|
| availability-state | 0.638600 | 0.125084 | 0.584237 (0.026262) | 1,075/1,375 |
| availability-ecosystem | 0.634429 | 0.111214 | 0.571200 (0.028427) | 1,075/1,375 |

The controlled endpoint imputes 300 missing CRYCHIC ligand-target rows below
returned scores. Most current external arms return 408/1,375 rows, while the
CellChat composite returns 251/1,375. Differences in coverage are therefore
part of the measured result and remain an important interpretation limit.

## Historical reference

The checksum-frozen official Figure 6A table contains 112 rows: HER2 and TNBC,
eight historical methods and seven rank cutoffs. In the published HER2 top-250
table, Connectome has OR 2.292682, NATMI and Crosstalk each 2.088019, and
CellChat 1.894582. These values are reference checks only and must not be ranked
against the current results above because the truth multiplicity, LIANA version,
resource snapshot and returned interaction universe differ.

The historical LIANA 0.0.5 OmniPath snapshot and original processed Seurat
object were unavailable. Crosstalk is not exposed by the current Python runner
and was skipped. Current LIANA used 100 permutations rather than the paper's
1,000 permutations. No historical-exact performance claim is made.

## Runtime and memory

Every stage ran serially in a user cgroup with `MemoryMax=32G` and
`MemorySwapMax=0`; LIANA used 16 workers.

| Stage | Wall time | Main-process peak RSS | Aggregate observation |
|---|---:|---:|---:|
| Preparation and truth | 44.25 s | 1.30 GiB | below 32 GiB cap |
| CRYCHIC | 18.26 s | 2.44 GiB | below 32 GiB cap |
| LIANA, 100 permutations | 2 min 50.54 s | 3.11 GiB | sampled cgroup peak 8.77 GiB |
| Evaluation | 34.77 s | 2.60 GiB | below 32 GiB cap |

## Published artifacts

- [Ranking metrics](ranking_metrics.tsv)
- [Fisher rank curves](fisher_rank_curves.tsv)
- [Score inventory](score_inventory.tsv)
- [Sanitized provenance manifest](manifest.json)

Large inputs and score matrices are checksum-referenced in the manifest but intentionally excluded from this GitHub bundle.
