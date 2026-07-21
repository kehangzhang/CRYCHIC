# RC1 signed expected-cardinality simulation v2

Status: **accepted for one-time Kuppe/MS evaluation as an unreleased benchmark
candidate**.

The implementation, candidate grid, null gate, and new development/holdout
seeds were committed at `20c136c` before this run.  Kuppe and MS were not used
in candidate selection or validation.

## Frozen candidate

`eb_delta_0.5` uses a zero-spike/symmetric-normal-slab working prior, an
identifiable slab SD floor of 1.5 times the robust observed scale, and a minimum
effect of 0.5 fitted slab SD.  Pair scores are sums of target- or
reference-oriented working probabilities.

| split | candidate | pair Spearman | top-quartile AUROC | tie fraction |
| --- | --- | ---: | ---: | ---: |
| development | eb_delta_0.5 | 0.882 | 0.954 | 0.000 |
| development | hard_z1 | 0.791 | 0.917 | 0.575 |
| holdout | eb_delta_0.5 | 0.885 | 0.955 | 0.000 |
| holdout | hard_z1 | 0.801 | 0.923 | 0.590 |
| holdout | wald_p05 | 0.855 | 0.940 | 0.824 |

Across 240 independent holdout scenario-seed-direction evaluations, the frozen
candidate improved pair Spearman over `hard_z1` by 0.084 on average, won 97.9%
of comparisons, and had paired Wilcoxon p=2.85e-40.  It improved every non-null
scenario family.

## Null gate

The preregistered development gate required the 95th percentile false expected
count per LR opportunity to be at most 0.001.  The frozen candidate reached
0.000209 on development and 0.000701 on untouched holdout.  The corresponding
holdout rates were 0.159 for `hard_z1`, 0.025 for nominal Wald p<0.05, and 0.250
for uncentered continuous sign evidence.

The remaining null scores are tiny but still ordered partly by opportunity, so
real-data outputs must continue to report `n_estimable_lr` and score/opportunity
correlation.  These analytic working-model probabilities are not calibrated
full-pipeline scientific posteriors and cannot be exposed as such.
