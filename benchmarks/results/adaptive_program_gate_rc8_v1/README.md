# RC8 adaptive receiver-program gate simulation

Status: **rejected before real-cohort benchmarking**. The model, scenarios,
candidate grid, seeds, and gates were frozen at commit `563d3b7`. Commit
`1a496e9` changed only the benchmark harness so that a development rejection is
persisted instead of raising an exception; it did not change any score or
selection rule.

RC8 tested an outcome-free counterpart of the scSeqCommDiff intracellular
gate. It retained the frozen RC2 LIANA call set and RC3 sign, treated missing
receiver-program evidence as neutral, and compared positive oriented program
tails with mirrored negative tails. A stronger binary gate required a frozen
relative mirror-FDP reduction, a Wilson lower-bound margin, and replication in
two stable edge folds. Unreliable evidence fell back to the unweighted RC2
head; reliable but nonseparable evidence used a sign-only gate.

## Development decision

No candidate passed every development gate. The top-ranked candidate was
`mirror_reduction_075`, but it was not eligible:

| Development metric | Delta | Gate | Pass |
|---|---:|---:|---|
| Primary scenarios versus RC3 | -0.00146 | at least +0.002 | no |
| Tail scenarios versus RC3 | +0.00043 | at least +0.005 | no |
| Safety scenarios versus RC3 | -0.00409 | at least -0.003 | no |
| Flat program versus sign-only | +0.00000 | at least -0.003 | yes |
| Global-null gate activation | 0.0% | at most 10% | yes |

The development eligibility failure is itself sufficient to reject RC8.
Holdout values below are retained only because the frozen runner evaluates all
replicates before producing its final manifest.

## Independent holdout audit

| Check | Result | Gate | Pass |
|---|---:|---:|---|
| Development candidate eligible | no | required | no |
| Primary-scenario delta versus RC3 | +0.00821 | at least +0.005 | yes |
| Tail-scenario delta versus RC3 | +0.01479 | at least +0.010 | yes |
| Safety-scenario delta versus RC3 | -0.01638 | at least -0.003 | no |
| Flat-program delta versus sign-only | +0.00000 | at least -0.003 | yes |
| Missing-program delta versus RC3 | +0.00471 | at least -0.003 | yes |
| Global-null gate activation | 6.25% | at most 10% | yes |

The main safety failures were diffuse program evidence (`-0.04961`) and
isolated edge changes (`-0.01480`). The stricter `mirror_reduction_050`
alternative still regressed diffuse evidence by `-0.03740`. The result therefore
rejects hard downstream gating as a general replacement for RC3 soft evidence.
Kuppe and MS were not run for RC8, and their DES outcomes were not used for the
formal simulation decision.

## Reproduction

```bash
PYTHONPATH=src:. python -m benchmarks.simulation.adaptive_program_gate_benchmark \
  --config benchmarks/configs/adaptive_program_gate_rc8_v1.json \
  --output-dir benchmarks/results/adaptive_program_gate_rc8_v1
```

These outputs are benchmark-ranking diagnostics, not calibrated communication
probabilities or formal differential-inference results.
