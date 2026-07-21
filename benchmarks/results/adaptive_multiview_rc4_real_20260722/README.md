# RC4 adaptive multi-view residual: Kuppe and MS

Status: complete one-shot benchmark. The real runner was frozen at commit
`8e5f6cc` after simulation acceptance and before either cohort ran.

The runner reproduced RC2 theta, final signs, call counts, and pair rankings
exactly before fitting RC4. It then compared the adaptive residual alone and in
combination with the independently frozen RC3 receiver-program evidence.

## Complete DES ranking

| Rank | Kuppe method | Median DES | Strata |
|---:|---|---:|---:|
| 1 | scSeqCommDiff native | 0.701 | 8/8 |
| 2 | CRYCHIC RC3 program soft + fallback | 0.576 | 8/8 |
| 3 | CRYCHIC RC4 adaptive program + fallback | 0.560 | 8/8 |
| 4 | CRYCHIC RC2 residual + fallback | 0.557 | 8/8 |
| 5 | CRYCHIC RC4 adaptive residual + fallback | 0.552 | 8/8 |

| Rank | MS method | Median DES | Strata |
|---:|---|---:|---:|
| 1 | scSeqCommDiff native | 0.825 | 8/8 |
| 2 | CRYCHIC RC4 adaptive program + fallback | 0.768 | 8/8 |
| 3 | CRYCHIC RC3 program soft + fallback | 0.743 | 8/8 |
| 4 | CRYCHIC RC2 residual + fallback | 0.700 | 8/8 |

RC4 is an MS specialist, improving the best prior complete CRYCHIC result by
`+0.025`, but it regresses Kuppe by `-0.017` versus RC3. It therefore cannot
replace RC3 as a universal default and does not exceed scSeqCommDiff.

## Fit diagnostics

| Dataset | Profile CV gain over equal | Sender | Receiver | Ligand | Receptor | Interaction | Sign flips |
|---|---:|---:|---:|---:|---:|---:|---:|
| Kuppe | 15.2% | 0.109 | 0.119 | 1.684 | 1.144 | 1.943 | 106 |
| MS | 16.5% | 0.061 | 0.070 | 1.678 | 1.350 | 1.841 | 81 |

Total penalty mass remains five and the LIANA call sets remain exactly 3,203
Kuppe and 782 MS calls. The adaptive signs slightly reduce agreement with the
independent receiver-program effect in both cohorts, so the MS gain cannot be
explained as a simple program-concordance gate.

These are benchmark-ranking heads only. Adaptive view reliability is predictive
edge CV, the program source is partial-pipeline cross-fit evidence, and formal
inference remains disabled.
