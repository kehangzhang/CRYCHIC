# Multi-condition benchmark report input contract

`generate_multicondition_report.py` is a presentation-only stage. It does not
run methods, rerank scores, calculate missing inferential statistics, or infer
that an absent adapter row is zero. The preferred input is a frozen result
directory containing `report_inputs.json` with this schema:

```json
{
  "schema_version": "multicondition-report-inputs.v1",
  "adapter_manifests": [
    "adapters/cellchat/adapter_manifest.json",
    "adapters/cellphonedb/adapter_manifest.json"
  ],
  "score_tables": [
    "adapters/cellchat/scores_long.parquet",
    "adapters/cellphonedb/scores_long.parquet"
  ],
  "dataset_manifests": [
    "datasets/cscc/dataset_manifest.json"
  ],
  "truth_yaml": "truth/multicondition_supportive_biology.yaml",
  "metrics": {
    "dataset_design": "metrics/dataset_design.tsv",
    "coverage": "metrics/coverage_summary.tsv",
    "loso_primary": "metrics/loso_primary_endpoint.tsv",
    "stability": "metrics/stability_summary.tsv",
    "concordance": "metrics/concordance_summary.tsv",
    "performance": "metrics/performance_summary.tsv",
    "biology_support": "metrics/biology_support.tsv",
    "simulation_truth": "metrics/simulation_truth_metrics.tsv",
    "iteration_comparison": "metrics/iteration_comparison.tsv"
  }
}
```

All paths are relative to the frozen result directory unless absolute. Missing
optional metric tables produce explicit `not_estimable` panels with reason
codes. When the JSON file is absent, the generator discovers the standard file
names above recursively; multiple matches are rejected so a report cannot
silently choose one iteration.

## Minimum table fields

- `dataset_design`: `dataset`, scale fields (`n_cells`, `n_samples`,
  `n_subjects`, `n_contexts`, `n_cell_types`), `design_type`, `status`, and
  `reason_code`.
- `coverage`: method identity and resource arm, resource/comparison coverage,
  explicit status counts, `status`, and `reason_code`. If absent, only status
  coverage is summarized from the long adapter tables.
- `loso_primary`: `dataset`, `method`, `resource_mode`, `contrast`, `estimate`,
  `ci_lower`, `ci_upper`, `n_subjects_estimable`, `status`, and `reason_code`.
  Observed rows without ordered confidence bounds are rejected.
- `stability`: identity columns plus either long `metric`/`estimate` columns or
  standard stability fields such as `median_spearman` and
  `median_top_k_jaccard`.
- `concordance`: `method_left`, `method_right`, `effect_spearman`, coverage,
  `status`, and `reason_code`. The report labels concordance as agreement, never
  accuracy.
- `performance`: method identity, `median_wall_time_seconds`,
  `median_peak_rss_mb`, success/failure fields, `status`, and `reason_code`.
- `biology_support`: locked YAML `observation_id`, method/resource identity,
  `support_status`, observed direction, `status`, and `reason_code`. Rows not in
  the locked YAML are rejected.
- `simulation_truth`: `truth_scope`, method identity, and either long
  `metric`/`estimate` columns or standard truth fields. The only accepted
  scopes are `simulation`, `synthetic`, and
  `perturbation_with_known_truth`; therefore real-data AUROC/AUPRC cannot enter
  the report.
- `iteration_comparison`: `dataset`, `method`, `metric`, before/after values,
  version labels, `metric_direction` (`higher` or `lower`), `status`, and
  `reason_code`.

## Command

```bash
uv run python -m benchmarks.report.generate_multicondition_report \
  --final-dir /path/to/frozen/multicondition_run \
  --output-dir reports/multicondition_v01
```

The command writes self-contained `REPORT.html`, paginated `REPORT.pdf`, the
Markdown source, an SHA256 input manifest, a machine-readable report manifest,
source-data CSV files, 300 dpi PNG figures, and SVG/PDF vector versions.
