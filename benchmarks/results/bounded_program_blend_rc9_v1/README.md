# RC9 bounded receiver-program blend simulation

Status: **accepted for one-shot real-cohort benchmarking**. The implementation,
candidate grid, development and holdout seeds, scenarios, and acceptance gates
were frozen at commit `fc5089a` before the formal run.

RC9 retains the exact RC2 residual sign, LIANA call set, and RC3 soft program
expert. It adds the RC8 mirror-tail gate only as a bounded convex expert. Soft
and gated weights are normalized to equal mean on the frozen call set before
mixing, so the candidate cannot gain by changing total score mass. Missing
program evidence remains neutral and formal inference remains disabled.

## Frozen candidate

Development selected `blend_r050_e020`: mirror-tail FDP reduction 0.50 and gate
blend fraction 0.20. More aggressive blends had higher primary gains but failed
the frozen diffuse-program safety gate.

| Development metric | Delta versus RC3 |
|---|---:|
| Primary scenarios | +0.01084 |
| Tail scenarios | +0.01459 |
| Safety scenarios | -0.00011 |
| Diffuse program | -0.00147 |
| Isolated edges | +0.00103 |
| Missing program | +0.00932 |

The global-null pair-score ratio q95 was `1.00107`.

## Independent holdout acceptance

| Check | Result | Gate | Pass |
|---|---:|---:|---|
| Primary-scenario delta versus RC3 | +0.01179 | at least +0.002 | yes |
| Tail-scenario delta versus RC3 | +0.01687 | at least +0.003 | yes |
| Safety-scenario delta versus RC3 | +0.00025 | at least -0.003 | yes |
| Diffuse-program delta versus RC3 | -0.00018 | at least -0.005 | yes |
| Isolated-edge delta versus RC3 | +0.00140 | at least -0.005 | yes |
| Missing-program delta versus RC3 | +0.00926 | at least -0.003 | yes |
| Global-null pair-score ratio q95 | 1.00059 | at most 1.02 | yes |

Holdout seeds were not used for candidate selection. Kuppe and MS outcomes were
not used in the formal simulation selection or acceptance decision.

## Reproduction

```bash
PYTHONPATH=src:. python -m benchmarks.simulation.bounded_program_blend_benchmark \
  --config benchmarks/configs/bounded_program_blend_rc9_v1.json \
  --output-dir benchmarks/results/bounded_program_blend_rc9_v1
```

These are benchmark-ranking diagnostics, not calibrated communication
probabilities or formal differential-inference results.
