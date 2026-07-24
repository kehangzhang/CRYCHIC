# Suggestions-next M1 locked mechanism-family validation

## Status

M1 (`signed_geometric_program_concordance_m1_v1`) passed every locked
validation gate. It remains an experimental contrast-level mechanism-support
head beside M0. It does not replace `parent_activity_raw`, does not claim active
inhibition, and emits no probability, p-value, or q-value.

## Frozen inputs

| Item | Value |
|---|---|
| Fixture generator commit | `7ef15d4923ade272a0579277242562ec4b308760` |
| Validation configuration commit | `8fc2a0575d0e64c5e50b87a6279b761ebf3277e7` |
| Final evaluated code commit | `c5b1ec3300737d7dd43e908e717588b07c918810` |
| Validation fixture | 20 new root seeds (20270501--20270520), 10 paired subjects, mean 180 cells/sample |
| Mechanism families | active, global null, abundance-only, receiver-autonomous, ligand-only, target-only, receptor-knockout |
| H-common resource | 5 harmonized LR interactions |
| Fixture manifest SHA256 | `7660c9a63fb4f8cabd2be54acc869cd46783be9a717b1ddbd369a85f646fd640` |
| Configuration SHA256 | `80356bbe81f554f2085019b865e962134b023f77d9030b118ecc81c41c440422` |
| Validation manifest SHA256 | `a2a1b9943feadd721cb5b0efb3c91f80c1cce6d80be782ed0a62a03f60a96ad2` |

The validation run completed from a clean worktree. Seeds and the validation
fixture checksum were frozen before evaluation. No parameter was fitted: M1
uses a fixed target-program map and a fixed concordance formula.

## Estimand separation

M0 remains the unsigned absolute LR activity head. For one contrast, M1 takes
the signed M0 LR effect `d_lr` and the signed receiver target-program effect
`d_program`. After aligning the target-program effect to the frozen expected
program direction, it reports

```text
sign(d_lr) * sqrt(abs(d_lr) * abs(d_program_aligned))
```

only when both effects have the same direction; otherwise it reports zero.
This is a local biochemical/mechanistic concordance diagnostic. It is not a
hard replacement for M0 strength and is not formal inference.

## Locked result

The positive unit is the active mechanism family; the other six families are
partial-mechanism or null controls. Metrics are paired over simulation root
seed, not over event rows.

| Internal arm | Mechanism AP | AUROC | Median active rank | Active retention | Mean active margin |
|---|---:|---:|---:|---:|---:|
| CRYCHIC M1 signed concordance | 1.0000 | 1.0000 | 1.0 | 1.000 | 0.05842 |
| CRYCHIC M0 absolute activity effect | 0.3500 | 0.6833 | 3.0 | 1.000 | -0.34398 |

M1 beat M0 on all 20 seeds. The paired AP delta was +0.6500 (95% bootstrap CI
[0.6250, 0.6667]); the AUROC delta was +0.3167 (95% CI [0.2917, 0.3333]).

## Mechanism stress

| Scenario | M0 absolute effect | Target-program absolute effect | M1 concordance | M1 nonzero fraction |
|---|---:|---:|---:|---:|
| active | 0.05509 | 0.11590 | 0.07983 | 1.00 |
| global null | 0.00338 | 0.00759 | 0.00236 | 0.50 |
| abundance-only | 0.00349 | 0.10441 | 0.00986 | 0.50 |
| receiver-autonomous | 0.00537 | 0.01235 | 0.00671 | 0.85 |
| ligand-only | 0.06082 | 0.00944 | 0.01572 | 0.65 |
| target-only | 0.00626 | 0.11356 | 0.00000 | 0.00 |
| receptor-knockout | 0.39907 | 0.11570 | 0.00000 | 0.00 |

The result explains why M0 cannot be interpreted as integrated communication:
receptor knockout creates the largest absolute M0 contrast because receptor
loss is a real LR-state change, but its direction conflicts with the increased
receiver program. M1 correctly leaves M0 visible while assigning zero
concordance. The largest partial-mechanism M1 mean was 19.7% of the active mean,
below the frozen 25% gate.

Abundance-only showed a broad target-program shift caused by the small-cell
composition realization in some paired samples. Directional concordance kept
its integrated score low; this also motivates residualized coupling and
composition stress in M2 rather than adding a wider generic program to M1.

## Robustness fix discovered by the benchmark

One development sample lacked a sender cell type. The context-level legacy
assignment was normalized over three senders while that sample contained two,
so its represented weights summed to 0.72--0.77. M0 detection was valid, but
the optional attribution was not fully representable. Commit `c5b1ec3` now:

- preserves nonconserved `sender_detection`;
- marks the incomplete sample attribution as typed not-estimable;
- does not renormalize the remaining senders;
- preserves original missing-evidence reasons when the candidate set is intact.

Regression tests cover both incomplete candidate coverage and complete but
missing assignment evidence.

## Core API migration parity

The frozen concordance formula was subsequently moved from the benchmark
evaluator into the public scoring module at commit `28197e7`, without changing
its version string, formula string, inputs, or output. The original locked
fixture and configuration were rerun from a clean worktree with 16 workers.

The five scientific tables (`mechanism_effects`, `method_summary`,
`paired_comparisons`, `scenario_summary`, and `seed_metrics`) were byte-for-byte
identical to the `c5b1ec3` validation outputs. Timing and provenance were
regenerated rather than compared. The core-migration replay manifest SHA256 is
`f5f2bbec0a236a3a745a926e9bfe6c0030fd96054b2b10938dd74a305658663d`.
This proves implementation relocation parity; it does not broaden the original
mechanism-family claim.

## Runtime and boundary

The validation evaluated 140 datasets and 503,479 cells in 13.6 seconds wall
time with 24 process workers and one BLAS thread per worker. Mean per-dataset
elapsed time was 1.77 seconds (maximum 2.02 seconds).

This benchmark compares M1 with M0 inside CRYCHIC. External methods were not
rerun on these seven mechanism families, so the result is not an external SOTA
rank. The earlier expression-driven H-common panel remains the external
comparison: M0 ranked 1/5 there. Future external mechanism-family comparisons
must match each method's legitimate estimand and report unavailable scenarios
as not estimable.

## Next isolated change

M2 should add residualized sender-receiver coupling without changing M0 or M1.
It should be tested on decoy-sender count, composition imbalance, batch/context
confounding, and receiver-autonomous families. Occurrence/prevalence remains a
separate later head, and frozen hypergraph shrinkage should only be attempted
after M2 has a locked sample-level contract.
