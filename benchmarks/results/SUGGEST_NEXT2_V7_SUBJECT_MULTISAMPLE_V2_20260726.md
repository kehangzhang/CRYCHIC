# suggest_next2 v7 subject multi-sample v2

Date: 2026-07-26

Status: subject-level truth and Kuppe scSeqCommDiff/LIANA replays complete;
Kuppe/MS CRYCHIC v7 refits, current-version CellChat and paper-matched MS
external refits are pending the active PR10 calibration campaign. No complete
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
  --sample-jobs 11 --threads 2 --seed 20260717
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

## Resource waves

Do not oversubscribe the active 192-core calibration run. Once it completes:

1. Wave 1: Kuppe/MS CRYCHIC v7 refits plus intervention validation, at most
   approximately 176 requested threads.
2. Wave 2: MS external refits, Kuppe/MS CellChat 2.1.2 and Kuppe/MS sparse
   geometry, with memory monitoring and immediate stop at 80% system memory.
3. Evaluate native cardinality, continuous sensitivity, fixed-K and geometry
   as separate panels, then summarize ranks only within identical contracts.
