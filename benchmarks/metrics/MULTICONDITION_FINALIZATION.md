# Multi-condition benchmark finalization

`benchmarks.metrics.finalize_multicondition` is the single metrics/provenance
stage between sample-level adapters and the publication report generator. It
accepts a JSON specification with schema
`crychic-multicondition-finalize-v1`:

```json
{
  "schema_version": "crychic-multicondition-finalize-v1",
  "generated_at": "2026-07-12T12:00:00+08:00",
  "supportive_biology_truth": "truth/multicondition_supportive_biology.yaml",
  "supportive_biology_dataset_aliases": {
    "UCSC_Lerma_Martin_MS_snRNA": "UCSC_Lerma_Martin_MS_snRNA_CA_vs_Ctrl"
  },
  "biology_support": "inputs/biology_support.tsv",
  "coverage_records": "inputs/native_coverage_summary.tsv",
  "simulation_records": "inputs/track_b_simulation_truth.tsv",
  "performance_records": "inputs/performance_runs.tsv",
  "iteration_comparison": "inputs/iteration_comparison.tsv",
  "parameters": {
    "top_k": 100,
    "minimum_shared_edges": 20,
    "n_bootstrap": 2000,
    "n_split_repeats": 200,
    "min_subjects_per_half": 2,
    "random_seed": 20260712
  },
  "datasets": [
    {
      "dataset_id": "GSE144236_Ji_cSCC",
      "context_key": "condition",
      "truth_scope": "real_data",
      "design": {
        "n_cells": 47068,
        "n_samples": 20,
        "n_subjects": 10,
        "n_contexts": 2,
        "n_cell_types": 14
      },
      "comparison": {
        "design": "paired",
        "reference": "Normal",
        "target": "Tumor",
        "contrast": "Tumor_vs_Normal",
        "min_subjects": 5,
        "primary_score_view": "global:'Tumor'"
      },
      "dataset_manifest": "inputs/cscc_dataset_manifest.json",
      "adapter_runs": [
        {
          "manifest": "runs/cellchat/manifest.json",
          "long_table": "runs/cellchat/interactions_long.parquet"
        },
        {
          "manifest": "runs/crychic/manifest.json",
          "long_table": "runs/crychic/interactions_long.parquet",
          "score_views": [
            {
              "run_id": "crychic_benchmark_run_tumor",
              "contrast_candidate": "global:'Tumor'",
              "label": "tumor_functional",
              "role": "primary",
              "contrast": "Tumor_vs_Normal"
            },
            {
              "run_id": "crychic_benchmark_run_normal",
              "contrast_candidate": "global:'Normal'",
              "label": "normal_functional",
              "role": "sensitivity",
              "contrast": "Tumor_vs_Normal"
            }
          ]
        }
      ]
    }
  ]
}
```

Paths are relative to the specification. `biology_support`, `coverage_records`,
`simulation_records`, `performance_records`, `iteration_comparison`,
`dataset_manifest`, and each dataset's `simulation_truth` are optional.
`coverage_records` is intended for
precomputed coverage-only sensitivity arms such as native resources whose
large, incompatible universes must not enter the H-common edge-effect loop.
`simulation_records` appends separately computed synthetic-track metrics, such
as ligand-target program recovery, and accepts only the frozen synthetic truth
scopes.
The locked supportive-biology YAML is required. When biology evidence or
iteration comparisons are absent, the corresponding report table is still
emitted with `not_estimable` and a stable `reason_code`.

`supportive_biology_dataset_aliases` is an optional one-to-one mapping from a
locked atlas/cohort identifier to the exact analyzed subset identifier. It may
change only dataset identity metadata, never observation IDs or expectations.
The final table retains the original identifier in `truth_dataset`, while the
locked YAML and its checksum remain unchanged.

Supported comparison designs are `paired`, `unpaired`, and `unsupported`.
Paired datasets use edge effects, differential LOSO, subject bootstrap CIs,
and the cross-dataset nested bootstrap. Independent groups use subject-equal
edge effects, repeated stratified split-half stability, and a separately
labelled leave-one-subject influence diagnostic. Track B ligand-target outputs
are copied to `track_b_tables/`, but receive explicit LR-comparison NE rows;
they never enter LR coverage, LOSO, concordance, or edge truth metrics.

An adapter table with more than one `run_id` must provide a complete
`score_views` mapping. Exactly one included view must have `role: primary`, and
its `contrast_candidate` must equal the dataset's preregistered
`comparison.primary_score_view`. Other included views use `role: sensitivity`;
they are normalized and evaluated into `derived/sensitivity_*` artifacts but
do not enter the nine primary report tables or cross-method concordance. A
view with `role: excluded` must set (or inherit) `include: false` and remains in
the score index with an explicit exclusion reason. For the frozen analyses,
the primary CRYCHIC views are `global:'Tumor'` for cSCC and `global:'CA'` for
MS; the reverse global views are sensitivity analyses.

For a simulated dataset, set `truth_scope` to `simulation`, `synthetic`, or
`perturbation_with_known_truth` and provide an exact frozen-universe edge truth
table. Providing a truth table for `real_data` is rejected. Real-data
AUROC/AUPRC-like iteration metrics are also rejected.

Run:

```bash
uv run python -m benchmarks.metrics.finalize_multicondition \
  --spec /path/to/finalize_run.json \
  --output-dir /path/to/frozen_final \
  --overwrite
```

The output contains normalized LR score parquets, Track B tables, derived edge
effects/folds/diagnostics, the nine report metric TSVs, `report_inputs.json`, a
finalization manifest, and `SHA256SUMS.tsv`. Output installation is atomic;
an existing directory is replaced only when `--overwrite` is explicit.
