# RC10 paired rank-contrast simulation

Status: **rejected before real-cohort benchmarking**. The scale-invariant pair
contrast, candidate grid, scenarios, development and holdout seeds, and gates
were frozen at commit `50d843c` before the formal run.

RC10 converted each condition's pair scores to internal percentiles and tested
`max(rank_condition - lambda * rank_other, 0)`. Structural absence remained
missing. The simulation deliberately included condition-specific shared
opportunity as well as legitimate independent, balanced, and asymmetric
bidirectional remodeling.

## Development decision

No candidate passed all development gates. The top primary candidate was
`contrast_lambda_075`:

| Development metric | Delta versus RC9 | Gate | Pass |
|---|---:|---:|---|
| Primary scenarios | +0.12259 | at least +0.005 | yes |
| Safety scenarios | -0.32241 | at least -0.003 | no |
| Independent bidirectional | -0.18655 | at least -0.005 | no |
| Asymmetric bidirectional | -0.45600 | at least -0.005 | no |
| Structural missingness | +0.08656 | at least -0.003 | yes |

Even the weakest candidate, `contrast_lambda_010`, regressed the safety mean by
`-0.00795` and asymmetric bidirectional remodeling by `-0.01365`.

## Independent holdout audit

| Check | Result | Gate | Pass |
|---|---:|---:|---|
| Development candidate eligible | no | required | no |
| Primary-scenario delta versus RC9 | +0.11893 | at least +0.005 | yes |
| Safety-scenario delta versus RC9 | -0.33259 | at least -0.003 | no |
| Independent-bidirectional delta | -0.20090 | at least -0.005 | no |
| Asymmetric-bidirectional delta | -0.45675 | at least -0.005 | no |
| Structural-missingness delta | +0.07275 | at least -0.003 | yes |
| Global-null score ratio q95 | 0.47103 | at most 1.02 | yes |

The contrast can strongly improve a condition-specific spatial benchmark when
opposite directions behave as nuisance, but it destroys valid bidirectional
communication when both directions carry biology. That ambiguity is not
identifiable from pair scores alone. RC10 is therefore rejected and was not run
on Kuppe or MS. The exploratory MS sensitivity value is not an algorithm result
and must not be included in the formal leaderboard.

## Reproduction

```bash
PYTHONPATH=src:. python -m benchmarks.simulation.paired_rank_contrast_benchmark \
  --config benchmarks/configs/paired_rank_contrast_rc10_v1.json \
  --output-dir benchmarks/results/paired_rank_contrast_rc10_v1
```

These are benchmark-ranking diagnostics, not calibrated communication
probabilities or formal differential-inference results.
