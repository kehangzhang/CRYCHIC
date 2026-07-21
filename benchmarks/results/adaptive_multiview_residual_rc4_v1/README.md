# RC4 adaptive multi-view signed residual simulation

Status: accepted for one-shot real-cohort benchmarking; benchmark-only and
unreleased. The implementation and policy were frozen at commit `758d4bc`
before the preregistered run.

Each view receives a nonnegative reliability from its out-of-fold group-mean
prediction improvement over the global training mean. Negative reliability is
clipped to zero, the total penalty mass is fixed, and the resulting profile is
continuously shrunk toward all-equal weights. The reliability profile replaces
all-equal only when its edge-CV MSE improves by at least 0.5%.

## Preregistered holdout acceptance

| Check | Result | Gate | Pass |
|---|---:|---:|---|
| Heterogeneous-scenario mean Spearman delta | +0.00166 | at least +0.001 | yes |
| Heterogeneous adaptive selection rate | 43.8% | at least 20% | yes |
| Coherent/noisy/antagonistic safety delta | -0.00014 | at least -0.003 | yes |
| Topology-jump safety delta | -0.00004 | at least -0.005 | yes |
| Wrong-topology safety delta | +0.00000 | at least -0.005 | yes |
| Global-null fallback gate q95 | 0.000 | at most 0.25 | yes |

## Holdout delta versus all-equal RC2

| Scenario | Pair-rank Spearman delta |
|---|---:|
| Cell-dominant | +0.00898 |
| Corrupted interaction view | +0.00054 |
| Corrupted sender view | -0.00040 |
| Molecular-dominant | -0.00248 |
| Coherent mixed | -0.00078 |
| Noisy anchor | +0.00018 |
| Antagonistic anchor | +0.00018 |
| Topology jump | -0.00004 |
| Wrong topology | +0.00000 |

The candidate passes the frozen aggregate contract, but the gain is small and
mostly cell-dominant. It is not a general superiority result. Holdout seeds
were used only for final acceptance; each replicate still performs the same
within-replicate edge-CV fitting that will be used on real cohorts.

## Reproduction

```bash
PYTHONPATH=src:. python -m benchmarks.simulation.adaptive_multiview_residual_benchmark \
  --config benchmarks/configs/adaptive_multiview_residual_rc4_v1.json \
  --output-dir benchmarks/results/adaptive_multiview_residual_rc4_v1
```
