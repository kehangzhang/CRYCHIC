# RC6 two-part hurdle: Kuppe and MS

Status: complete one-shot benchmark; **not accepted as a replacement for RC3
or RC4**. The real-cohort runner was frozen at `b6318ea`; `bc18809` corrected
the pre-evaluation RC3 parity check to tolerate serialization-scale floating
differences while requiring exact pair rank order.

RC6 used the simulation-selected `two_part_20_05` candidate. Technical samples
were collapsed within biological subject and cell type. LR presence was the
Jeffreys-Beta posterior probability that expression proportion exceeded the
paper-matched LIANA threshold of 0.1. Positive magnitude was evaluated only for
active subjects. Structural absence remained missing.

## Complete DES comparison

| Rank | Kuppe method | Median DES | Strata |
|---:|---|---:|---:|
| 1 | scSeqCommDiff native | 0.701 | 8/8 |
| 2 | CRYCHIC RC3 program soft + fallback | 0.576 | 8/8 |
| 3 | CRYCHIC RC2 residual + fallback | 0.557 | 8/8 |
| 4 | CRYCHIC RC6 two-part + fallback | 0.514 | 8/8 |
| 5 | LIANA+ native | 0.490 | 8/8 |

| Rank | MS method | Median DES | Strata |
|---:|---|---:|---:|
| 1 | scSeqCommDiff native | 0.825 | 8/8 |
| 2 | CRYCHIC RC4 adaptive program + fallback | 0.768 | 8/8 |
| 3 | CRYCHIC RC3 program soft + fallback | 0.743 | 8/8 |
| 3 | CRYCHIC RC6 two-part + fallback | 0.743 | 8/8 |
| 5 | CRYCHIC RC2 residual + fallback | 0.700 | 8/8 |

RC6 therefore regressed Kuppe by `-0.062` versus RC3. On MS its median tied
RC3, but mean DES decreased from `0.734` to `0.705`; it also remained below the
RC4 MS specialist.

## Fit and parity

| Dataset | Occurrence coverage / alpha | Magnitude coverage / alpha | Calls | RC3 rank parity |
|---|---:|---:|---:|---|
| Kuppe | 82.5% / 0.483 | 43.0% / 0.299 | 3,203 | exact order |
| MS | 100.0% / 0.642 | 39.6% / 0.334 | 782 | exact order |

The RC3 call set, final sign, and pair template were unchanged. Kuppe strict
DES increased slightly from `0.461` to `0.466`, while the complete fallback
ranking regressed. RC3/RC6 fallback pair-rank Spearman remained 0.969/0.963 for
Kuppe CTRL/IZ and 0.997/0.969 for MS CA/Ctrl. The Kuppe CTRL top-10 overlap was
only 7/10, showing that small reordering near the top materially affects DES.

## Decision

The subject hurdle effects are useful diagnostics but are not a universal
ranking improvement. RC6 must remain benchmark-only and must not replace RC3
or RC4. The next iteration should calibrate how strict and fallback pair ranks
are combined under simulated structural coverage loss, rather than add another
edge multiplier. Any blend must preserve exact native behavior at its fallback
boundary and be selected without Kuppe/MS spatial outcomes.

Raw edge-by-subject hurdle tables, edge effects, weights, and complete rankings
remain in the checksum-bound external run directories referenced by the copied
manifests.
