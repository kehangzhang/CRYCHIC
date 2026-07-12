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

Current results may persist one receiver-scoped child scoring functional per
contrast. Default readback treats these as a provenance-preserving row union:
each child must match exactly one persisted contrast, cover exactly one
receiver, and the receiver/edge partitions for a contrast must be disjoint and
complete. Scores are never averaged or rewritten. The adapter manifest records
the sorted child IDs, receiver partitions, source-table checksums,
`aggregation=none_row_union_by_receiver`, and
`common_functional_claim=false`, with
`provenance_status=legacy_reconstructed_fail_closed`. These bridge semantics,
the partitions, child membership, contrast, and source checksums are bound into
the benchmark run ID. Ambiguous, overlapping, or incomplete child sets fail
closed. Passing one or more `--scoring-functional-id` options retains
single-child diagnostic views instead.

This is a compatibility bridge, not evidence that the children form a common
`ScoringFunctional`. A future core `ScoringCollectionManifest` must provide an
explicit child-to-contrast registry before the adapter can stop using persisted
numeric matching. Historical results with one common functional remain one
view.

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

If fitting completed but an earlier readback emitted one full-universe view per
receiver child, keep the persisted `result/` unchanged and re-export to a
disjoint sibling directory with `benchmarks.adapters.crychic.readback`. Never
use `--overwrite` on the parent directory containing `result/`.

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
