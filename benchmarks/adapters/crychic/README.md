# CRYCHIC benchmark adapter

The CRYCHIC adapter emits the same
`crychic-external-interactions-long-v1` contract as the external methods. It
does not add a privileged CRYCHIC-only metric path. Each persisted scoring
functional becomes a separate benchmark `run_id`, and each run materializes
the complete sample-by-sender-by-receiver-by-resource LR universe.

`comm_strength` is an exploratory score where higher values indicate stronger
communication. It is not a probability. The adapter always leaves all
within-dataset and differential p/q fields null and records
`v0_1_inferential_disabled` on every row.

## H-common run

Use a new output directory. The command retains the native CRYCHIC result under
`result/` and writes the normalized table beside it.

```bash
uv run python -m benchmarks.adapters.crychic.run_hcommon \
  benchmarks/configs/multicondition_hcommon_nocap_v02.json cscc_native_cellchat \
  ../benchmark_work/multicondition_v02/crychic_hcommon_nocap/cscc \
  --harmonized-resource ../benchmark_work/multicondition_v01/resources/harmonized_simple_lr/harmonized_lr.tsv \
  --harmonized-manifest ../benchmark_work/multicondition_v01/resources/harmonized_simple_lr/manifest.json \
  --blas-threads 8
```

Replace `cscc_native_cellchat` with `ms_native_cellchat` and use a different
output directory for the MS arm. Historical `multicondition_v01_*` configs are
retained unchanged for their recorded native capped runs and are not valid
inputs to the no-cap H-common adapter.

## Native result readback

Run the native arm separately with `benchmarks.run_canonical_v01`, then convert
its completed result directory. The prepared H5AD must be the exact analyzed
object, including any cell-type subset used for fitting.

```bash
uv run python -m benchmarks.adapters.crychic.readback \
  ../benchmark_work/multicondition_v01/crychic_initial/cscc_native_cellchat/result \
  ../benchmark_work/multicondition_v01/prepared/GSE144236_cscc_paired.h5ad \
  ../benchmark_work/multicondition_v01/adapters/crychic_native/cscc \
  --dataset-id GSE144236_Ji_cSCC --resource-mode native \
  --native-adapter cellchat --database-root ../databases \
  --native-manifest resources/cellchatdb_human_v2.json
```

Use the MS result/input/dataset ID for MS. Omit
`--scoring-functional-id` to export every persisted scoring functional as a
separate fixed-universe run; pass the option one or more times to select exact
functional IDs.
