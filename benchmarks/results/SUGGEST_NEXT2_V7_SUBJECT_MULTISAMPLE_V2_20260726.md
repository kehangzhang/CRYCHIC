# suggest_next2 v7 subject multi-sample v2

Date: 2026-07-26

Status: subject-level truth and Kuppe scSeqCommDiff/LIANA replays complete;
Kuppe/MS CRYCHIC v7 refits, current-version CellChat and paper-matched MS
external refits are pending the active PR10 calibration campaign. Paper-matched
MS scSeqCommDiff and LIANA+ actual-data preflights are complete. No complete
cross-method rank is claimed.

## Frozen comparison contract

- analysis unit: biological `subject_id`;
- repeated sections: arithmetic mean within subject and condition;
- truth top sets: floor of 10%, 20%, 30% and 40% rankable pairs;
- cell-pair universe: canonical unordered pairs, self-pairs excluded;
- endpoint: weighted fgsea positive running sum, `gseaParam=1`, native stable
  tie order;
- ranking statistic: `raw_cardinality` for native event counts and
  `raw_strength` for continuous common-score effects;
- primary spatial variant for Kuppe: `spatial_neighbor_max`;
- resource: 2,293-row ConnectomeDB2020 payload SHA256 `e7813632...`, manifest
  SHA256 `3dd10324...`;
- native-cardinality and continuous common-score sensitivity panels are never
  pooled into one leaderboard.

Truth artifacts:

| Cohort | Unit support | Manifest SHA256 | Expected-set SHA256 |
|---|---|---|---|
| Kuppe | 4 CTRL / 7 IZ subjects | `4fe12433c953edaab760c64f369bb44762335346b2ba7a57fe10549ae71fd9b9` | `b883f58e1df3c773d09f80a32f966a5f8db5dee91740f10e2db70d6cd4e7119b` |
| MS | 5 Ctrl / 5 CA subjects | `f93042a947ce6dd0260a40f58ebc3155890c03b44d8674b38ffe275fbb2c20bd` | `9c91513503b7715949d247a558e23e9801d33cf56a327a0df52a87b1a2e5342d` |

## Kuppe completed external replay

These values use all eight condition-by-top-fraction strata. The two native
methods are directly comparable to each other. LIANA rank aggregate is a
separate continuous common-score sensitivity arm.

| Panel | Method | Median DES | Mean DES | Observed strata |
|---|---|---:|---:|---:|
| native cardinality | scSeqCommDiff 2.0.0 | 0.6596 | 0.6720 | 8/8 |
| native cardinality | LIANA+ DE 1.5.0 | 0.3798 | 0.3858 | 8/8 |
| continuous sensitivity | scSeqCommDiff 2.0.0 event strength | 0.7713 | 0.7403 | 8/8 |
| continuous sensitivity | LIANA rank aggregate 1.7.3 | 0.1631 | 0.1958 | 8/8 |
| continuous sensitivity | legacy CellChat 2.2-dev source | NE | NE | 0/8 |

Results are under
`benchmark_work/suggest_next2_v7/subject_multi_sample_v2_20260726/kuppe`.
Each evaluation manifest binds its absolute ranking path, ranking checksum,
truth checksum and source manifest. The legacy CellChat source had multiple
sample failures, leaving fewer than three valid subjects per condition for every
pair; NE is not converted to zero and is excluded from ranks. CRYCHIC v7 and a
clean CellChat 2.1.2 rerun are not yet present, so this table is not a final
algorithm ranking.

Track-labeled v3 replays are complete for scSeqCommDiff
(`native_cardinality`) and LIANA rank aggregate
(`continuous_strength`). Their score and method-summary checksums are
identical to the preceding evaluations. The existing LIANA+ run persisted the
full resource-manifest content but predates persistence of the resource
manifest file checksum, so it cannot satisfy the new strict three-checksum
gate. Its reported DES remains descriptive evidence; promotion into the final
`native_cardinality` track requires a short current-adapter Kuppe rerun after
the PR10 campaign. No missing checksum is imputed.

The checksum-bound external-only `continuous_strength` summary currently ranks
scSeqCommDiff 1/2 and LIANA 2/2. scSeqCommDiff has complete rank-universe
coverage; LIANA has 0.8 rank-universe coverage, reported separately rather than
used as a score penalty. The summary is under
`kuppe/continuous_strength_external_summary_v1` and has output SHA256
`e3335fe6f5ea3baf01c3e864c2e9db9974974bc6080225107b061141365e0485`.
Its panel digest includes the frozen input, resource payload and resource
manifest checksums.
CRYCHIC v7 and current CellChat are still absent, so this is not a final
cross-algorithm rank.

The new subject-level scSeqCommDiff event producer exported 234,256 events from
the frozen RDS and selected exact global budgets across both effect directions.
All fixed-K tracks have 8/8 observed strata:

| Track | Median DES | Mean DES |
|---|---:|---:|
| fixed K=100 | 0.1862 | 0.1778 |
| fixed K=250 | 0.1862 | 0.1778 |
| fixed K=500 | 0.6363 | 0.6115 |
| fixed K=1000 | 0.8513 | 0.6787 |

The derived track manifest is
`kuppe/scseqcommdiff_subject_event_tracks_v1/manifest.json`, SHA256
`fd2a65fde138f101e4272ee2b56ef2ab23ac727f778872bfb8bb4b01368818ea`.
Its 1,517-second runtime, including a 1,210-second RDS export, is preprocessing
time and is not counted as scSeqCommDiff algorithm runtime. Fixed-K ranks remain
pending the identically budgeted CRYCHIC v7 results.

## MS exclusion and required reruns

The existing subject outputs under `benchmark_work/multi-group/runs/ms` use the
older 75,004-cell object, SHA256 `612fe9c4...`. They are excluded from v2.
The required paper-matched input has 69,168 cells, 11 sections and 10 subjects:

```text
benchmark_work/multi-group/prepared/ms_figure3_5ctrl_6ca/
  UCSC_Lerma_Martin_MS_CA_vs_Ctrl_5ctrl_6ca.h5ad
```

H5AD SHA256: `433717d9fd98e57e15a444a338a6e1002f7ca3224022321d28a386c0a8498e1c`.
Preparation-manifest SHA256:
`35644aec92e6e383ef982e18bb3c6aff24c2e5faf8ae52ea921c01c440f0c15d`.

After the calibration campaign releases the 192 logical cores, run the native
scSeqCommDiff and LIANA+ jobs and the LIANA/CellChat sample-score sensitivity
jobs against this exact input. The old MS CellChat sample job ended through an
external `KeyboardInterrupt`, not an algorithm-level non-estimability result.

### Paper-matched MS actual-data preflights

Both native subject-level adapters were executed through their full input,
checksum, resource and environment preflight paths against the exact 69,168-cell
object. These are execution-readiness artifacts, not benchmark outcomes.

| Adapter | Status | Key checks | Elapsed under calibration load |
|---|---|---|---:|
| scSeqCommDiff 2.0.0 | preflight complete | input/resource hashes; R 4.3.3; 5 CA/5 Ctrl subjects; 68,858 eligible cells and 7 eligible cell types | 548.5 s |
| LIANA+ DE 1.5.0 | prepared | input/resource hashes; frozen Python package set; 5 CA/5 Ctrl subjects; 2,054/2,293 resource interactions matched | 178.7 s |

The scSeqCommDiff paper-compatible support filter excludes BC and SC before
inference because they lack the required cross-condition pseudobulk support;
this is persisted as method-specific estimability rather than converted to zero.
LIANA+ retains all nine source cell types at preparation and applies its own
frozen pseudobulk support rules in the exact worker.

Artifacts are under:

```text
benchmark_work/suggest_next2_v7/subject_multi_sample_v2_20260726/ms/preflight/
  scseqcommdiff_native_20260726/
  liana_plus_de_native_20260726/
```

The spatial DES evaluator now explicitly accepts both checksum-bound
scSeqCommDiff manifest schemas v1 and v2. This is a schema compatibility fix;
status, ranking checksum, input checksum, resource checksum and subject-unit
checks remain fail closed. It also resolves the v7 real E1 manifest's explicit
preparation-manifest and LR-resource input records, so CRYCHIC and external
methods can be replayed through the same checksum-bound evaluator. CRYCHIC's
multi-endpoint ranking payload is filtered only after that full-payload check:
`diagnostic_one_se_native_count_des` enters the native-cardinality panel,
`continuous_weighted_des` enters the continuous panel, and each
`top_k_count_des` event budget remains a separate fixed-K sensitivity. These
are frozen as `native_cardinality`, `continuous_strength`, and
`fixed_k_100/250/500/1000` comparison tracks; the summarizer ranks methods
only within an identical track.

### scSeqCommDiff native subject arm

```bash
PYTHONPATH=src python -m benchmarks.adapters.scseqcommdiff.run \
  /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/prepared/ms_figure3_5ctrl_6ca/UCSC_Lerma_Martin_MS_CA_vs_Ctrl_5ctrl_6ca.h5ad \
  /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/subject_multi_sample_v2_20260726/ms/runs/scseqcommdiff_native \
  --input-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/prepared/ms_figure3_5ctrl_6ca/UCSC_Lerma_Martin_MS_CA_vs_Ctrl_5ctrl_6ca.manifest.json \
  --resource /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/resources/connectomedb2020/connectomedb2020.tsv \
  --resource-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/resources/connectomedb2020/manifest.json \
  --rscript /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/envs/scseqcomm_2_0_0_conda/bin/Rscript \
  --python-executable /media/subunit/bioinfo/crychic_dev/.worktrees/suggest_next2_v7_real/.venv/bin/python \
  --dataset-id LermaMartin_MS_CA_vs_Ctrl --scenario multi-sample \
  --condition-key lesion_type --sample-unit-key subject_id \
  --target CA --reference Ctrl --cores 8 --nrep 1000 --min-cells 30
```

### LIANA+ native subject arm

```bash
PYTHONPATH=src python -m benchmarks.adapters.liana.run_condition_aware \
  --input-h5ad /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/prepared/ms_figure3_5ctrl_6ca/UCSC_Lerma_Martin_MS_CA_vs_Ctrl_5ctrl_6ca.h5ad \
  --input-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/prepared/ms_figure3_5ctrl_6ca/UCSC_Lerma_Martin_MS_CA_vs_Ctrl_5ctrl_6ca.manifest.json \
  --resource /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/resources/connectomedb2020/connectomedb2020.tsv \
  --resource-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/resources/connectomedb2020/manifest.json \
  --exact-python /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/envs/liana_1_5_0/bin/python \
  --output-dir /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/subject_multi_sample_v2_20260726/ms/runs/liana_plus_de_native \
  --dataset-id LermaMartin_MS_CA_vs_Ctrl --replicate-key subject_id \
  --condition-key lesion_type --target CA --reference Ctrl --cores 8
```

### LIANA sample-score sensitivity

```bash
PYTHONPATH=src /home/agent1/miniforge3/envs/liana_env/bin/python \
  -m benchmarks.adapters.liana.run_by_sample \
  /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/prepared/ms_figure3_5ctrl_6ca/UCSC_Lerma_Martin_MS_CA_vs_Ctrl_5ctrl_6ca.h5ad \
  /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/subject_multi_sample_v2_20260726/ms/runs/liana_rank_aggregate \
  --dataset-id LermaMartin_MS_CA_vs_Ctrl --context-key lesion_type \
  --resource-mode H-common \
  --harmonized-resource /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/resources/connectomedb2020/connectomedb2020.tsv \
  --harmonized-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/resources/connectomedb2020/manifest.json \
  --n-perms 100 --n-jobs 16 --min-cells 10 --expr-prop 0.1 \
  --seed 20260717
```

Feed its `interactions_long.parquet` to `sample_effect_des` with
the following lineage-bound aggregation:

```bash
PYTHONPATH=src python -m benchmarks.literature.sample_effect_des \
  /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/subject_multi_sample_v2_20260726/ms/runs/liana_rank_aggregate/interactions_long.parquet \
  /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/subject_multi_sample_v2_20260726/ms/rankings/liana_rank_aggregate \
  --context-key lesion_type --reference Ctrl --target CA \
  --min-subjects-per-context 3 \
  --source-run-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/subject_multi_sample_v2_20260726/ms/runs/liana_rank_aggregate/manifest.json
```

### CellChat sample-score sensitivity

```bash
PYTHONPATH=src python -m benchmarks.adapters.cellchat.run_by_sample \
  /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/prepared/ms_figure3_5ctrl_6ca/UCSC_Lerma_Martin_MS_CA_vs_Ctrl_5ctrl_6ca.h5ad \
  /media/subunit/bioinfo/crychic_dev/benchmark_work/suggest_next2_v7/subject_multi_sample_v2_20260726/ms/runs/cellchat_sample \
  --dataset-id LermaMartin_MS_CA_vs_Ctrl \
  --database-root /media/subunit/bioinfo/crychic_dev/databases \
  --context-key lesion_type --resource-mode H-common \
  --harmonized-resource /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/resources/connectomedb2020/connectomedb2020.tsv \
  --harmonized-manifest /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/resources/connectomedb2020/manifest.json \
  --custom-database-rds /media/subunit/bioinfo/crychic_dev/benchmark_work/multi-group/resources/connectomedb2020/CellChatDB_ConnectomeDB2020.rds \
  --environment r_cellchat_2_1_2 --nboot 100 --min-cells 10 \
  --sample-jobs 11 --threads 1 --seed 20260717
```

Its output is also passed through the same lineage-bound `sample_effect_des`
aggregation. It is a continuous subject-effect sensitivity result, not the
pooled condition-level CellChat native arm. CellChat condition-aware and
scDiffCom remain in their separate condition-level panel and are never included
in the subject ranking.

Every MS v2 evaluation must additionally pass these fail-closed bindings:

```bash
--expected-input-sha256 433717d9fd98e57e15a444a338a6e1002f7ca3224022321d28a386c0a8498e1c \
--expected-resource-sha256 e781363288a26c15e03246500111bfecb818eef997f5ebe1b936aaa465151c3a \
--expected-resource-manifest-sha256 3dd10324ae0fc3b903ee09fece3fbeb1933db94c92fe1fbf4eb644d407d6b7af
```

The v2 evaluator rejects the ranking if any supplied source manifest lacks or
disagrees with these values.

## CellChat 2.1.2 execution smoke

The current worker was executed, not only syntax-checked, on a deterministic
MS377I subset with at most 50 cells per observed cell type:

| Item | Result |
|---|---|
| input | 392 cells x 32,115 genes; 8 cell types |
| input SHA256 | `756e37ee52a5bf6d0d87d1bb0f25721c330afc45e5ed7d2ee78c19aab538e7b7` |
| CellChat | 2.1.2, R 4.5.3 |
| execution | 1 sample job, 1 thread, 2 bootstraps |
| elapsed | 623.625 seconds under concurrent calibration load |
| output | 146,752 fixed-universe rows, SHA256 `0394227e...` |
| failures | 0 |

The run manifest records `future.globals.maxSize=8 GiB`, sequential future
execution and a clean code commit. This closes the two legacy source failures:
the 500 MiB future-global limit and the obsolete R output path. Formal CellChat
therefore uses one sequential R process per sample and maximum safe sample-level
parallelism: 11 jobs for MS and 15 for Kuppe, subject to the 80% memory stop.

## Resource waves

Do not oversubscribe the active 192-core calibration run. Once it completes:

1. Wave 1: Kuppe/MS CRYCHIC v7 refits plus intervention validation, at most
   approximately 176 requested threads.
2. Wave 2: MS external refits, the provenance-upgrade Kuppe LIANA+ rerun,
   Kuppe/MS CellChat 2.1.2 and Kuppe/MS sparse geometry, with memory monitoring
   and immediate stop at 80% system memory.
3. Evaluate native cardinality, continuous sensitivity, fixed-K and geometry
   as separate panels, then summarize ranks only within identical contracts.
