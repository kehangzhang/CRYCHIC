# Suggest-next2 v7 PR10 process-backend scaling

Status: **safe-spawn process backend accepted for PR10 execution; formal
calibration not evaluated**.

## Evidence

- Source commit: `898cac47d3eb11323253e7eb829c6c67073247d6`
  (`dirty=false` in every dataset manifest).
- Protocol SHA-256:
  `89ca902e09e4828ba910f8641e613b764fed72bfdcc2efcedc7419c670e1aae5`.
- Scaling output:
  `/media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/pr10_process_scaling_v1_20260726`.
- Six-design regression output:
  `/media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/pr10_six_design_process_smoke_v1_20260726`.
- Six-design campaign manifest SHA-256:
  `b6070999150980f3b85de915ec732443796f9960e6044105c0e1c87669da348a`.
- Six-design `runs.tsv` SHA-256:
  `77929a15034f890623012f171fd4951388c6eab9597f1856721e91249cbe4df3`.
- Focused core/runner/public-API tests: 18 passed. Ruff, formatter, import,
  and diff checks passed.

The process backend uses `multiprocessing` safe spawn rather than POSIX fork.
Each persistent worker receives the read-only raw snapshot, frozen resources,
design, exchangeability map, and point hypothesis axis once at initialization.
Each subsequent task transmits one small resampling plan and returns one compact
effect record. This avoids Python 3.14's unsafe multithreaded-fork path and does
not add a runtime dependency.

## Eight-plan scaling

The exact same continuous global-null dataset used four bootstraps and four
condition permutations without LOSO.

| Backend | Inner workers | Total seconds | Resampling seconds | Total speedup | Peak RSS |
|---|---:|---:|---:|---:|---:|
| serial | 1 | 135.26 | 132.39 | 1.000x | 279.9 MiB |
| process | 2 | 77.52 | 74.54 | 1.745x | 801.4 MiB |
| process | 4 | 49.34 | 46.29 | 2.741x | 1.287 GiB |
| process | 8 | 36.63 | 33.30 | 3.692x | 2.314 GiB |

All four runs had the same scientific result ID
`v7_full_pipeline_resampling_result_b437a7d43180b5a368d904671e268114`,
the same ordered record IDs, and identical SHA-256 values for every persisted
scientific output table. Only execution metadata differed.

## 64-plan scaling

The larger fixed input used 32 bootstraps and 32 permutations. It was designed
to expose persistent-pool throughput rather than letting an eight-task batch
cap useful concurrency.

| Inner workers | Total seconds | Resampling seconds | Plans/minute | Speedup vs 8 workers | Peak RSS | Max system memory |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 140.34 | 133.95 | 28.67 | 1.000x | 2.319 GiB | 18.85% |
| 16 | 83.23 | 76.38 | 50.27 | 1.686x | 4.323 GiB | 19.44% |
| 32 | 58.99 | 51.91 | 73.98 | 2.379x | 8.459 GiB | 20.57% |
| 64 | 50.82 | 44.22 | 86.83 | 2.761x | 16.563 GiB | 22.86% |

All four 64-plan runs had the same scientific result ID
`v7_full_pipeline_resampling_result_b0029e94e3c052a27137ee7c534970bd`,
the same 64 ordered records, identical output-table SHA-256 values, and 64/64
successful full refits. A serial 64-plan run was not spent solely for timing.
Using the measured serial first-completion time and median later-resample time
projects 970.18 seconds; against that explicitly labeled projection, 64 workers
are about 19.09x faster.

Doubling from 32 to 64 workers increases throughput by 17.4% while nearly
doubling process-tree RSS. Therefore 32 workers are the resource-efficiency
choice and 64 workers are the wall-time choice on this 192-core, 251-GiB host.
The user requirement prioritizes wall time while allowing up to 80% memory, so
64 is the selected formal-campaign candidate. The runner's CPU guard then caps
dataset-level concurrency at three, preventing more than 192 requested inner
workers at once.

## Six-design deterministic regression

The new process campaign was compared against the completed serial six-design
smoke at the same dataset IDs, seeds, protocol, two bootstraps, two
permutations, and complete LOSO.

| Design | Serial seconds | Process seconds | Speedup | Process peak RSS | Full refits |
|---|---:|---:|---:|---:|---:|
| continuous | 309.59 | 39.15 | 7.908x | 5.322 GiB | 20/20 |
| independent multi-group | 967.10 | 82.29 | 11.752x | 7.860 GiB | 28/28 |
| independent two-group | 302.40 | 37.59 | 8.045x | 5.328 GiB | 20/20 |
| multi-cohort | 307.42 | 38.84 | 7.915x | 5.320 GiB | 20/20 |
| paired | 194.60 | 37.07 | 5.249x | 3.318 GiB | 12/12 |
| repeated | 422.78 | 77.56 | 5.451x | 3.534 GiB | 12/12 |

The campaign completed 6/6 datasets and 112/112 full refits in 158.11 seconds
of wall time, versus about 21.3 minutes for the serial campaign. Dataset-level
parallelism was three, the maximum observed system-memory fraction was 21.46%,
and every design observed all ten required stages.

For every design, process and serial runs had identical scientific result IDs,
ordered record IDs, and output-table SHA-256 values. Thus the speedup did not
change the point estimator, resampling plans, seed lineage, estimability state,
or finalized diagnostic values.

## Boundary and next gate

- **Deterministic process execution gate: pass.**
- **Six-design integration regression: pass.**
- **Memory gate: pass for the measured workload.** The largest single dataset
  used 16.563 GiB and nested six-design execution reached 21.46% system memory,
  both below 80%.
- **Formal calibration gate: not evaluated.** These development timing runs
  still have too few resamples, no calibration gate, and no released p/q rows.
- Every new design/scale must retain a pilot memory measurement. A 64-worker
  choice must fall back to fewer inner workers when the pilot or current system
  headroom cannot satisfy the 80% campaign boundary.
- The next evidence run is a frozen full-pipeline global-null calibration
  campaign using the process backend, followed by locked DGP families and then
  untouched real/spatial cohorts.
