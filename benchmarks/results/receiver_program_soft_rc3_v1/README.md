# RC3 receiver-program soft evidence simulation

Status: accepted for held-out real-cohort benchmarking; benchmark-only and
unreleased.

The candidate was selected on 20 development seeds across helpful, weak,
45%-missing helpful, noisy, antagonistic, and global-null scenarios. Twenty
separate holdout seeds were not used for selection. Kuppe and MS were not used
for either simulation design selection or candidate selection.

## Frozen candidate

- Candidate: `program_max_alpha_4`
- Reliability: `max(0, 2 * (LR/program sign concordance - 0.5))`
- Effective alpha: `min(1.5, 4 * reliability)`
- Missing receiver-program evidence: neutral weight `1.0`
- Global-null development alpha q95: `0.2052` (gate: at most `0.25`)

## Pair-rank Spearman delta versus alpha zero

| Scenario | Development | Holdout |
|---|---:|---:|
| Helpful program | +0.0376 | +0.0385 |
| 45%-missing helpful program | +0.0318 | +0.0195 |
| Weak program | +0.0118 | +0.0092 |
| Noisy program | +0.0009 | +0.0026 |
| Antagonistic program | +0.0000 | +0.0000 |

The antagonistic evidence is closed by the reliability gate rather than being
allowed to reverse LR evidence. The global-null truth is constant, so its rank
metric is undefined; its preregistered safety endpoint is effective alpha.

## Reproduction

```bash
PYTHONPATH=src:. python -m benchmarks.simulation.receiver_program_soft_benchmark \
  --config benchmarks/configs/receiver_program_soft_rc3_v1.json \
  --output-dir benchmarks/results/receiver_program_soft_rc3_v1
```

Machine-readable selection, replicate metrics, aggregate metrics, and SHA-256
provenance are in this directory.
