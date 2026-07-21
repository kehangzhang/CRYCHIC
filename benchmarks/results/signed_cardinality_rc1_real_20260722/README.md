# Frozen RC1 real-cohort benchmark

The RC1-v2 candidate was frozen on simulations at `42f06b6`; the checksum-bound
real-cohort runner was frozen at `bfdf178`.  Kuppe and MS were evaluated once
after those commits.  All methods use the same component-swap inputs,
ConnectomeDB resource, DES evaluator, and unordered non-self pair universe.

## Median DES ranking

| rank | Kuppe method | median DES | complete strata |
| ---: | --- | ---: | ---: |
| 1 | scSeqCommDiff native | 0.701 | 8/8 |
| 2 | LIANA+ native | 0.490 | 8/8 |
| 3 | CRYCHIC + Wilcoxon | 0.478 | 8/8 |
| 4 | CRYCHIC RC0 hard one-SE | 0.428 | 8/8 |
| 7 | CRYCHIC RC1 soft | 0.360 | 8/8 |

| rank | MS method | median DES | complete strata |
| ---: | --- | ---: | ---: |
| 1 | scSeqCommDiff native | 0.825 | 8/8 |
| 2 | CRYCHIC RC1 soft | 0.520 | 8/8 |
| incomplete | LIANA+ native | 0.500 | 6/8 |
| 4 | CRYCHIC Wald proxy | 0.393 | 8/8 |
| 8 | CRYCHIC RC0 hard one-SE | 0.050 | 8/8 |

RC1 fixes the catastrophic MS aggregate failure (+0.470 absolute median DES)
but does not overtake scSeqCommDiff and regresses on Kuppe (-0.069).  This is
therefore a diagnostic improvement, not the final benchmark head.

## Root-cause signal

The symmetric global working prior yields strongly imbalanced directional
minimum-effect counts.  Kuppe has 181 target-selected versus 3,077
reference-selected edges; MS has 610 versus 3.  The A-to-RC1 pair-rank
Spearman also changes sharply by direction: 0.777/0.199 for Kuppe and
0.440/-0.396 for MS.  A single symmetric slab scale is suppressing the
minority direction and changing pair rankings despite identical edge effects
and signs.

The next preregistered candidate should use positive and negative components
with separately fitted scales and weights, retain the null false-count gate,
and be selected on new asymmetric simulations.  No Kuppe/MS threshold scan is
authorized by this result.  RC1 probabilities remain benchmark-only and are
not released full-pipeline posteriors.
