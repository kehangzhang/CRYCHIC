# Suggestions-next M0 locked validation

## Status

M0 (`log_reference_additive_m0_v1`) passed its preregistered locked validation
gate. It remains an experimental benchmark-only absolute activity head. It
does not replace the public communication-strength score and emits neither a
calibrated active probability nor formal p/q values.

## Frozen inputs

| Item | Value |
|---|---|
| Implementation commit | `209e6a323da0449ef0d7c7c42114b5ed0aa04340` |
| Validation configuration commit | `937abd5674a27ec7582e94e7bfbc55b1d9fc2d6a` |
| Fixture | 20 new seeds (20261201--20261220), 8/10/12 independent subjects, mean 180 cells/sample |
| Fixture manifest SHA256 | `afbf43f29ed2e9214ef0e920efb4751dc05f97bc3eaff506be0efea48e28ac6c` |
| Source-evaluation manifest SHA256 | `a725ef7a854dfda7a201d24b7c2139bad87670a0fd6e7cc4e4024f2b0669efd6` |
| M0 validation manifest SHA256 | `b0378ed07dab013991b5c613473fa5344f66bcfb7f1d6239992d69a63bd90c31` |
| External panel manifest SHA256 | `7388cb40a9219e9b5e4b7fe450bb1c39089e6d06e89f29bc54babc765630dd97` |
| External evaluation manifest SHA256 | `85e0de732dd1c147b9137685187d5c29481c150381cd9f3c66e0ab78b39f3db8` |

All 40 legacy CRYCHIC inputs completed with clean code provenance at
`209e6a3`. The external evaluation discovered 240 method runs with no missing
run or excluded run.

## Internal gate

| Head | AUPRC | AUROC | Localization AP | Direction | Zero | Tie | Coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| M0 absolute sender detection | 0.6112 | 0.9472 | 0.6578 | 1.0000 | 0.0000 | 0.0074 | 1.000 |
| RC12 sender/response | 0.1659 | 0.5492 | 0.1443 | 0.5143 | 0.0000 | 0.4856 | 1.000 |
| Canonical mechanistic | 0.0833 | 0.4667 | 0.0826 | 0.7286 | 0.0000 | 0.0000 | 1.000 |
| Legacy strict geometric | 0.0693 | 0.3804 | 0.0616 | 0.4643 | 0.4856 | 0.4856 | 1.000 |

All seven checks passed: 20 paired seeds, AUPRC noninferiority, AUROC
superiority, retained direction, coverage, reduced zero fraction, and reduced
tie fraction. M0 minus RC12 was +0.4452 AUPRC (95% paired bootstrap CI
[0.3927, 0.4912]) and +0.3980 AUROC (95% CI [0.3188, 0.4752]), with 20/0/0
wins/ties/losses for both metrics.

## H-common external comparison

All methods used the same five-interaction H-common resource and the same 20
active validation seeds. M0 effects use the same native raw-mean contrast
operator and metric implementation as the external evaluation. The primary
external-panel ranking metric is prevalence-adjusted AP at 10% target
prevalence.

| Rank | Method | Adjusted AP | AUPRC | AUROC | Localization AP | Direction | Effect Spearman |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | CRYCHIC M0 | 0.6859 | 0.6112 | 0.9472 | 0.6578 | 1.0000 | 0.3614 |
| 2 | scSeqCommDiff | 0.2036 | 0.1426 | 0.7048 | 0.1283 | 0.1714 | -0.1800 |
| 3 | CellChat | 0.1297 | 0.0881 | 0.4937 | 0.0779 | 0.8214 | 0.1474 |
| 4 | CRYCHIC generic baseline | 0.0786 | 0.0521 | 0.1444 | 0.0493 | 0.6143 | 0.0434 |
| 5 | LIANA | 0.0759 | 0.0503 | 0.0887 | 0.0466 | 0.8214 | 0.0721 |

M0 beat scSeqCommDiff on all 20 seeds: adjusted-AP delta +0.4822 (95% CI
[0.4524, 0.5049]), AUPRC delta +0.4686 (95% CI [0.4404, 0.4902]), and AUROC
delta +0.2425 (95% CI [0.2202, 0.2627]). It also won all 20 seeds against
CellChat, LIANA, and the generic CRYCHIC baseline on each of these three
ranking metrics.

## Runtime

M0 validation completed in 281.8 seconds using 16 processes. The 40 baseline
fits consumed 138.6 CPU-seconds in aggregate, with a mean of 3.46 seconds per
dataset. The external panel used 16 task workers: CellChat completed 40 tasks
at 61.9 seconds mean wall time and 4.58 GiB mean peak RSS; scSeqCommDiff
completed 120 pairwise tasks at 31.1 seconds mean wall time and 0.82 GiB mean
peak RSS. LIANA completed all 40 tasks with correct manifests during the first
resumable scheduler attempt, but its per-task timing was not persisted after
that scheduler stopped; its timing is therefore reported as NE rather than
reconstructed from file timestamps.

## Boundary and next test

The simulator plants ligand/receptor expression changes and aligned receiver
program changes. It therefore tests recovery on an expression-driven mechanism
family, not a universal definition of cell communication. The next cycle is a
factorial full-pipeline stress suite with ligand-only, receptor-only,
receiver-autonomous/generic-downstream, inhibitory, occurrence, decoy-sender,
complex-AND, composition, and confounding families. That suite will determine
whether the next isolated change is M1 signed downstream or an M0b local
ligand/receptor joint-evidence correction.
