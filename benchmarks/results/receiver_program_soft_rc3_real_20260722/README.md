# RC3 receiver-program soft evidence: Kuppe and MS

Status: complete benchmark-only evaluation. The real-cohort runner was frozen
at commit `6a7d6c0` after simulation selection and before either cohort ran.

RC3 keeps the exact RC2/LIANA call set and applies a bounded continuous weight
from sample-level receiver-family program effects. Missing program evidence has
weight 1.0. The source program artifacts are partial-pipeline cross-fit outputs
with `formal_inference_allowed=false`; these results are rankings, not formal
communication inference.

## Complete DES ranking

| Rank | Kuppe method | Median DES | Strata |
|---:|---|---:|---:|
| 1 | scSeqCommDiff native | 0.701 | 8/8 |
| 2 | CRYCHIC RC3 program soft + coverage fallback | 0.576 | 8/8 |
| 3 | CRYCHIC RC2 residual + coverage fallback | 0.557 | 8/8 |
| 4 | LIANA+ native | 0.490 | 8/8 |

| Rank | MS method | Median DES | Strata |
|---:|---|---:|---:|
| 1 | scSeqCommDiff native | 0.825 | 8/8 |
| 2 | CRYCHIC RC3 program soft + coverage fallback | 0.743 | 8/8 |
| 3 | CRYCHIC RC2 residual + coverage fallback | 0.700 | 8/8 |
| incomplete | CRYCHIC RC3 program soft strict | 0.736 | 6/8 |
| incomplete | LIANA+ native | 0.500 | 6/8 |

RC3 improves the complete RC2 median by `+0.019` on Kuppe and `+0.043` on MS,
but it remains second on both cohorts. No superiority claim is supported.

## Frozen fit and parity

| Dataset | Program coverage | Sign concordance | Effective alpha | RC2 calls | Alpha-zero parity |
|---|---:|---:|---:|---:|---|
| Kuppe | 86.0% | 0.706 | 1.500 | 3,203 | exact, 132 rows |
| MS | 90.9% | 0.630 | 1.041 | 782 | exact, 90 rows |

The cohort-level gain is heterogeneous. Kuppe improves in all four CTRL DES
strata but regresses in three of four IZ strata. MS improves most chronic-active
and control strata, while the control top-30% and top-40% gaps to scSeqCommDiff
remain large. Real outcomes must not be used to retune alpha or direction rules;
any next candidate requires independent simulation selection.

Full checksum-bound edge weights, receiver-program effects, rankings, coverage,
and manifests remain in the external benchmark run directory recorded in the
manifests copied here.
