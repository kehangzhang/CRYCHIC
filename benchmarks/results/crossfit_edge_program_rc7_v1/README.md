# RC7 cross-fitted edge-program simulation

Status: **rejected before real-cohort benchmarking**. The implementation,
candidate grid, development and holdout seeds, scenarios, and acceptance gates
were frozen at commit `7b97331` before the formal run.

RC7 tested the program-first direction from `suggestions_v4.md` without using
condition labels to learn programs. For every held-out subject, rank-one to
rank-three edge programs were fitted on the remaining subjects' occurrence
matrix and projected into the held-out subject. Reconstruction weights were
calibrated only from held-out MSE improvement. The frozen RC6 magnitude head,
residual sign, receiver-program evidence, structural-missingness semantics, and
pair aggregation were otherwise unchanged.

## Development selection

Development selected `rank1_shrink05`, a rank-one reconstruction with 0.5
projection shrinkage. Its gains over the raw RC6 reference were small:

| Development metric | Delta versus RC6 |
|---|---:|
| Primary scenarios | +0.00031 |
| Coordinated scenarios | +0.00036 |
| Safety scenarios | +0.00006 |

The candidate was novel and passed the development safety and global-null
gates. Holdout seeds were not used for candidate selection.

## Independent holdout decision

| Check | Result | Gate | Pass |
|---|---:|---:|---|
| Primary-scenario delta versus RC6 | -0.00139 | at least +0.002 | no |
| Coordinated-scenario delta versus RC6 | -0.00093 | at least +0.002 | no |
| Structural-missingness delta versus RC6 | -0.00325 | at least -0.003 | no |
| Safety-scenario delta versus RC6 | +0.00015 | at least -0.003 | yes |
| Isolated-occurrence delta versus RC6 | +0.00076 | at least -0.003 | yes |
| Global-null reconstruction-reliability q95 | 0.000 | at most 0.1 | yes |

The candidate failed three frozen holdout gates. It is therefore rejected and
was not run on Kuppe or MS. No real-cohort result was inspected or used to make
this decision.

## Interpretation

Label-free subject reconstruction MSE was not a useful proxy for differential
cell-pair ranking. Even the lightly shrunk rank-one program lost signal in the
coordinated and structural-missingness scenarios. Further tuning of rank or
projection shrinkage is not supported by these results.

The next high-information direction is dataset-adaptive expert routing among
already frozen heads, using only outcome-free dataset diagnostics and
independent simulation for selection. A router should first demonstrate that
it can retain the RC3-like head in continuous, program-concordant settings and
the RC4-like adaptive head when view reliability is heterogeneous. Kuppe/MS DES
outcomes must remain final tests rather than routing labels.

## Reproduction

```bash
PYTHONPATH=src:. python -m benchmarks.simulation.crossfit_edge_program_benchmark \
  --config benchmarks/configs/crossfit_edge_program_rc7_v1.json \
  --output-dir benchmarks/results/crossfit_edge_program_rc7_v1
```

These are benchmark ranking diagnostics, not released communication
probabilities or formal differential-inference results.
