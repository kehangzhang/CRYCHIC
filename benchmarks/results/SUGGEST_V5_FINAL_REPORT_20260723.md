# Suggest-v5 optimization and benchmark audit

Status: **two of three critical evidence pillars established**.

## Result

| Gate | Status | Main result |
|---|---|---|
| A: signed estimands | PASS | 15/15 hand cases and 11/11 invariants |
| B: null calibration | PASS | type-I 3.41%; 95% CI coverage 96.59%; BH false-discovery replicates 0/50 |
| C: canonical noninferiority | MIXED | PASS versus complete scSeqCommDiff; conservative REJECT versus incomplete but stronger CellChat |
| D: native hypergraph truth | PASS | exact AP 1.00 versus 0.50 clique/min and 0.20 pairwise union |
| E: exact performance | PASS | wall 4.89x faster with byte-identical persisted score tables |

The critical objective is met by **statistical calibration plus synthetic native
hypergraph exact-truth superiority**. Canonical all-baseline noninferiority is
not established and is not counted as a successful pillar.

## Canonical score

The frozen canonical score is `mechanistic_sender_lr_score`:

```text
availability * prior_quality * sender_component
```

Downstream evidence is an independent annotation for canonical analysis. The
strict downstream-integrated score remains available for explicitly high-order
mechanism tasks. This fixes the BRCA failure mode in which the old geometric
score was zero for 89.5% of events because downstream support was zero for
75.9%, despite informative availability and sender components.

On the 20-seed BRCA holdout, CRYCHIC AUPRC was 0.9292, versus 0.9002 for
scSeqCommDiff and 0.6146 for LIANA. CellChat was 0.9851 on 19/20 estimable
replicates; CRYCHIC minus CellChat was -0.0578 with 95% CI [-0.0947, -0.0209].
The conservative all-baseline claim therefore remains rejected.

## Calibration

Fifty independently seeded fixtures preserved paired subjects and randomized
the E/NE assignment across subjects while retaining group sizes. Counts were
not injected. The released inference track used one On-minus-Pre delta per
biological subject, a between-arm Welch test, a Welch 95% interval, and BH
within dataset, method, view, and engine.

- Formal tests: 13,500/13,500.
- Mean unadjusted type-I at 0.05: 0.0341; one-sided upper 95%: 0.0378.
- Mean CI coverage: 0.9659; one-sided lower 95%: 0.9622.
- Replicates with any BH false discovery: 0/50; upper 95%: 0.0582.

## Hypergraph truth

The preregistered holdout used 200 new seeds, three registered LR edges, and
seven mechanism scenarios per edge. Every method scored the same 21 candidates
per seed at complete coverage. Missing candidates failed the campaign.

CRYCHIC exact hyperedge AP was 1.00. Clique expansion and simple complex-min
were both 0.50, and pairwise union was 0.20. The smallest paired-difference CI
lower bound was 0.50, above the frozen 0.20 threshold.

Partial canonical-LR AP was deliberately reported as a tradeoff: CRYCHIC was
0.643, clique/min 1.00, and pairwise union 0.40. Thus the exact-task advantage
comes from requiring the full downstream mechanism, not from broader output or
higher coverage.

## Limits

The calibration result applies to the released subject-level Welch DID track,
not a refit-per-permutation exact test. The hypergraph result is synthetic,
uses three edges and structural hard-zero near-misses, and is not real-data
accuracy evidence. It is not valid to claim CRYCHIC beats CellChat overall.

Machine-readable metrics, artifact paths, checksums, commits, and claim limits
are stored in `suggest_v5_final_summary_20260723.json`.
