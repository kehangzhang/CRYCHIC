# Suggest-next2 v7 component smoke report

This report is compact descriptive evidence from the frozen 20-seed smoke
campaign: 900/900 datasets completed with zero process failures. It is not
formal release evidence; analytic I1/I2 p-values remain diagnostic.

## Main estimator

- G3-I1 sender effect Spearman: 0.5913
- G3-I1 event AUPRC: 0.6382
- G0-I1 event AUPRC: 0.6444
- G3 minus G0 event AUPRC: -0.0063

G3 improves effect recovery substantially over G0 and remains within the
pre-registered 0.02 AUPRC non-inferiority margin, but its mean effect Spearman
does not yet clear the 0.60 release target.

## E2 hard-gate swaps

All four annotation-only arms are exact copies of G3 for every persisted
metric row. Ligand-contrast, family-selection, and downstream hard gates cause
large zero/tie inflation and loss of estimability. The downstream gate has a
negative paired AUPRC interval; receptor eligibility changes coverage geometry
without changing observed effect rankings in this smoke suite.

## E3 sender swaps

Nonconserving detection is the strongest sender-identity score. Null-sender
attribution controls diagnostic false positives, but its top-1 accuracy drops
at 10 and 20 candidates. Adding EB-shrunken M2 coupling does not materially
change rankings and fails the all-cardinality top-1 >= 0.80 gate.

## Gate status

- e2_annotation_arms_exact: PASS
- e2_downstream_hard_gate_harms_auprc: PASS
- m0_auprc_regression_within_0_02: PASS
- m0_effect_spearman_at_least_0_60: FAIL
- m1_direction_accuracy_at_least_0_85: PASS
- m1_generic_state_diagnostic_fpr_not_above_0_05: PASS
- m2_top1_at_least_0_80_at_every_candidate_count: FAIL

See the TSV files for counts, descriptive intervals, paired deltas, and stage
timings. Raw per-dataset outputs remain outside Git.
