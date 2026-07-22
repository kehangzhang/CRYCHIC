# RC9 bounded receiver-program blend: Kuppe and MS

Status: complete one-shot real-cohort benchmark. The runner was frozen at
commit `40a30f2` after independent simulation acceptance and before either
cohort was evaluated.

RC9 reconstructed the frozen RC3 rankings with exact rank order and numerical
tolerance `1e-13`, retained the exact LIANA call sets, and then applied the
frozen 20% bounded gate expert. Kuppe selected the sign-only fallback; MS
selected a cross-fold replicated enriched tail at program z `4.86071`.

## Complete DES ranking

| Rank | Kuppe method | Median DES | Mean DES | Strata |
|---:|---|---:|---:|---:|
| 1 | scSeqCommDiff native | 0.701 | 0.691 | 8/8 |
| 2 | CRYCHIC RC9 bounded program blend + fallback | 0.613 | 0.583 | 8/8 |
| 3 | CRYCHIC RC3 program soft + fallback | 0.576 | 0.563 | 8/8 |
| 4 | LIANA+ native | 0.490 | 0.470 | 8/8 |

| Rank | MS method | Median DES | Mean DES | Strata |
|---:|---|---:|---:|---:|
| 1 | scSeqCommDiff native | 0.825 | 0.673 | 8/8 |
| 2 | CRYCHIC RC9 bounded program blend + fallback | 0.796 | 0.754 | 8/8 |
| 3 | CRYCHIC RC4 adaptive program + fallback | 0.768 | 0.736 | 8/8 |
| 4 | CRYCHIC RC3 program soft + fallback | 0.743 | 0.734 | 8/8 |
| incomplete | LIANA+ native | 0.500 | 0.401 | 6/8 |

RC9 improves the prior complete CRYCHIC best by `+0.03671` on Kuppe and
`+0.02763` on MS. It remains below scSeqCommDiff by `0.08775` and `0.02895` in
the primary median DES ranking. RC9 has the higher MS mean (`0.754` versus
`0.673`), but this does not replace or reverse the preregistered median ranking.

## Frozen fit and parity

| Dataset | Gate mode | Program-call concordance | Gate threshold | Blend | Calls | RC3 rank parity |
|---|---|---:|---:|---:|---:|---|
| Kuppe | sign only | 0.710 | 0.000 | 0.20 | 3,203 | exact |
| MS | enriched tail | 0.749 | 4.861 | 0.20 | 782 | exact |

For MS, 49 positive-tail calls were retained and the minimum validation-fold
concordance gain was `+0.0541`. Soft and gate expert weights were normalized on
the call set; the blended call mean was exactly 1.0 in both cohorts. Runtime was
0.55 seconds for Kuppe and 0.50 seconds for MS, with peak RSS below 341 MB.

These are benchmark rankings only. The receiver-program source remains partial
pipeline cross-fit evidence, and formal communication inference is disabled.
