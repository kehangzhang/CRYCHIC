# Suggest-next2 v7 paired-intervention protocol

Status: implementation and read-only input preflight complete; full refits are
queued behind the active PR10 full-pipeline calibration campaign.

## Why these runs are new

The existing Kang result is an exploratory in-sample generic baseline rather
than a v7 out-of-fold fit. The existing BRCA workflow fits 58 sample units and
then uses an external patient-level evaluator. It is not an internal v7 paired
fit. This protocol therefore fits three explicit paired analyses from raw
counts:

| Dataset | Cells | Subjects | Paired contrast | Preferred folds |
| --- | ---: | ---: | --- | --- |
| Kang IFN-beta | 24,673 | 8 | stim - ctrl | 4, fallback 2 |
| BRCA expander | 18,465 | 9 | OnE - PreE | 3, fallback 2 |
| BRCA non-expander | 30,243 | 20 | OnNE - PreNE | 3, fallback 2 |

Every subject has exactly one sample in each declared condition. Receiver axes
are preregistered from the input manifests. Outer folds are subject-blocked,
the common-sender training minimum is four subjects, and each selected fold
requires at least four training and two test subjects per context.

## Frozen estimator and resources

- Protocol: `benchmarks/configs/suggest_next2_v7_intervention_v1.json`
- Protocol SHA256: `ce9c22f0c065fc619563021068ab78ff93f2eebfdb2f613f5e4190365ef79079`
- Estimator source: `5dc22aaa87896eafbeb687aea8d8e96e3654af01`
- Score projection: `35db184d62653ff8c960e2b135fa8e3060e1d394`
- H-common: 455 simple human LR interactions; table SHA256
  `4707a67c304cf6c7e17d667e42af730f0ac52dc4952b23fefb0e41a0cbac0463`
- NicheNet v2_2021: 1,225 ligand drivers and 14,315 targets; native manifest
  SHA256 `4e2a107768e5dee7457fdb5b73ff3659d1f5dbb0a1f82f09dd5ddfc12749f146`

The runner uses the v7-primary G0/G2/G3/G4/G5 matrix and paired I1 effects.
G1 is unavailable because no checksum-bound legacy full-profile run exists for
these inputs. No parameter may be selected on Kang or either BRCA subtype.

## Endpoints and claim limits

Kang uses the preregistered supportive-silver biology file. It reports paired
ISG and antigen-presentation direction, donor direction consistency, cell-type
composition stability, the G4 signed-program direction, and M0/program
concordance. Recombinant IFN-beta is exogenous, so recovery of endogenous
IFNB1 sender expression is explicitly not required.

BRCA has no event-level truth. The E and NE cohorts are fit separately with
true `subject_id`. A checksum-bound postprocessor reports
`E(On-Pre) - NE(On-Pre)` only as a descriptive interaction. The current sender
and resampling contracts do not support a native mixed repeated-by-between-
subject 2x2 test, so the postprocessor withholds p/q and does not relabel the
result as formal inference.

No real-data edge AUROC, calibrated significance, or causal sender identity is
released for any of these cohorts.

## Implemented entrypoints

- `benchmarks/literature/run_v7_intervention_validation.py`
- `benchmarks/literature/summarize_v7_brca_response.py`
- `tests/benchmarks/test_run_v7_intervention_validation.py`

The fit runner writes `run.jsonl` after resource load, input validation,
cross-fit completion, paired inference, supportive-truth evaluation, and final
serialization. It persists score views, effects, score geometry, effect
coverage, and mechanism summaries. Kang additionally persists paired gene and
composition tables.

## Verification

- Ruff format/check: passed.
- Python compile and JSON parse: passed.
- New focused tests: 10 passed.
- New plus existing Kuppe/MS runner tests: 17 passed.
- Real input checksum, dimensions, pairing, counts layer, resource, and target
  prior preflight: passed for all three datasets.

The full fits are intentionally not launched while the 192-core PR10
calibration campaign is active. This keeps the campaign throughput stable and
prevents simultaneous fold-local matrices from competing for memory.

## Frozen execution commands

The three jobs may run concurrently after PR10 releases the host. Kang uses four
outer-fold workers and each BRCA subtype uses three; every fold is limited to eight
BLAS threads. Their maximum requested concurrency is therefore 80 logical threads.
Set `CRYCHIC_PROGRESS=1` so fold-stage progress and ETA are present in the captured
stderr log.

```bash
CRYCHIC_PROGRESS=1 python -m benchmarks.literature.run_v7_intervention_validation \
  --dataset kang \
  --input-h5ad /media/subunit/bioinfo/crychic_dev/benchmark_work/kang2018_batch2.h5ad \
  --input-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/comprehensive_multicontext_20260722/runs/expanded_full_20260723/real_tracks/kang2018_ifnb_paired/input_manifest.json \
  --output-dir /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/intervention_kang_v7_v1_20260726 \
  --resource /media/subunit/bioinfo/crychic_dev/benchmark_work/comprehensive_multicontext_20260722/runs/multigroup_headtohead_20260723/misc_prepared/resource/harmonized_lr.tsv \
  --resource-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/comprehensive_multicontext_20260722/runs/multigroup_headtohead_20260723/misc_prepared/resource/manifest.json \
  --database-root /media/subunit/bioinfo/crychic_dev/databases \
  --nichenet-manifest /media/subunit/bioinfo/crychic_dev/databases/nichenet/v2_2021/manifest.json \
  --truth /media/subunit/bioinfo/crychic_dev/.worktrees/suggest_next2_v7_real/benchmarks/truth/kang2018_expected_biology.yaml \
  --fold-jobs 4 \
  --threads 8
```

```bash
CRYCHIC_PROGRESS=1 python -m benchmarks.literature.run_v7_intervention_validation \
  --dataset brca_e \
  --input-h5ad /media/subunit/bioinfo/crychic_dev/benchmark_work/comprehensive_multicontext_20260722/runs/expanded_full_20260723/real_tracks/brca_anti_pd1_2x2/benchmark_input/inputs/brca_anti_pd1.E_On_vs_Pre.h5ad \
  --input-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/comprehensive_multicontext_20260722/runs/expanded_full_20260723/real_tracks/brca_anti_pd1_2x2/benchmark_input/inputs/E_On_vs_Pre.manifest.json \
  --output-dir /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/intervention_brca_e_v7_v1_20260726 \
  --resource /media/subunit/bioinfo/crychic_dev/benchmark_work/comprehensive_multicontext_20260722/runs/multigroup_headtohead_20260723/misc_prepared/resource/harmonized_lr.tsv \
  --resource-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/comprehensive_multicontext_20260722/runs/multigroup_headtohead_20260723/misc_prepared/resource/manifest.json \
  --database-root /media/subunit/bioinfo/crychic_dev/databases \
  --nichenet-manifest /media/subunit/bioinfo/crychic_dev/databases/nichenet/v2_2021/manifest.json \
  --fold-jobs 3 \
  --threads 8
```

```bash
CRYCHIC_PROGRESS=1 python -m benchmarks.literature.run_v7_intervention_validation \
  --dataset brca_ne \
  --input-h5ad /media/subunit/bioinfo/crychic_dev/benchmark_work/comprehensive_multicontext_20260722/runs/expanded_full_20260723/real_tracks/brca_anti_pd1_2x2/benchmark_input/inputs/brca_anti_pd1.NE_On_vs_Pre.h5ad \
  --input-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/comprehensive_multicontext_20260722/runs/expanded_full_20260723/real_tracks/brca_anti_pd1_2x2/benchmark_input/inputs/NE_On_vs_Pre.manifest.json \
  --output-dir /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/intervention_brca_ne_v7_v1_20260726 \
  --resource /media/subunit/bioinfo/crychic_dev/benchmark_work/comprehensive_multicontext_20260722/runs/multigroup_headtohead_20260723/misc_prepared/resource/harmonized_lr.tsv \
  --resource-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/comprehensive_multicontext_20260722/runs/multigroup_headtohead_20260723/misc_prepared/resource/manifest.json \
  --database-root /media/subunit/bioinfo/crychic_dev/databases \
  --nichenet-manifest /media/subunit/bioinfo/crychic_dev/databases/nichenet/v2_2021/manifest.json \
  --fold-jobs 3 \
  --threads 8
```

After both BRCA fits complete, construct the bounded cross-subtype descriptive
contrast with:

```bash
python -m benchmarks.literature.summarize_v7_brca_response \
  --expander-root /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/intervention_brca_e_v7_v1_20260726 \
  --nonexpander-root /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/intervention_brca_ne_v7_v1_20260726 \
  --output-dir /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/intervention_brca_response_v7_v1_20260726
```
