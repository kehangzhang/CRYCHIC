# RC2 exact LIANA baseline plus hypergraph residual

The RC2 simulation candidate was accepted at `387622d`, and the checksum-bound
real runner was frozen at `681cc5c` before Kuppe/MS execution.

The runner reproduced the native LIANA baseline exactly before applying any
residual: Kuppe 3,203 calls/132 pair rows and MS 782 calls/90 pair rows all
matched.  CRYCHIC anchors covered 95.3%/90.9% of LIANA edges.  Internal edge-CV
selected lambda 0.05 for both cohorts, retained a weak anchor only for Kuppe
(0.1), and shrank the MS anchor to zero.  CV relative improvement was 43.7% for
Kuppe and 23.7% for MS.

## Complete DES ranking

| rank | Kuppe method | median DES | strata |
| ---: | --- | ---: | ---: |
| 1 | scSeqCommDiff native | 0.701 | 8/8 |
| 2 | CRYCHIC RC2 residual + coverage fallback | 0.557 | 8/8 |
| 3 | LIANA+ native | 0.490 | 8/8 |
| 4 | CRYCHIC + Wilcoxon | 0.478 | 8/8 |
| 6 | CRYCHIC RC0 hard one-SE | 0.428 | 8/8 |

| rank | MS method | median DES | strata |
| ---: | --- | ---: | ---: |
| 1 | scSeqCommDiff native | 0.825 | 8/8 |
| 2 | CRYCHIC RC2 residual + coverage fallback | 0.700 | 8/8 |
| incomplete | CRYCHIC RC2 strict | 0.571 | 6/8 |
| incomplete | LIANA+ native | 0.500 | 6/8 |
| 5 | CRYCHIC Wald proxy | 0.393 | 8/8 |
| 9 | CRYCHIC RC0 hard one-SE | 0.050 | 8/8 |

RC2 is the best complete CRYCHIC result so far: it improves the original head
by +0.129 on Kuppe and +0.650 on MS, and ranks second on both cohorts.  It does
not exceed scSeqCommDiff, so no superiority claim is made.

The strict method keeps exact LIANA eligibility.  The complete method uses the
CRYCHIC Wald rank only for cell pairs where LIANA pseudobulk is not estimable;
it never converts structural absence to a zero.  The residual and fallback are
benchmark-ranking outputs, not released calibrated inference.

Post hoc diagnostics found no defensible one-parameter shortcut to close the
remaining gap.  Opportunity amplification could raise DES but would reward
detection opportunity rather than communication reliability and was rejected.
A receiver-program gate helped MS but not Kuppe and was threshold-sensitive;
it should be developed only as a separately calibrated soft evidence channel.
