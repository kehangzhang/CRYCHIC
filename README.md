# CRYCHIC

CRYCHIC: Cellular Relational dYnamics for Contextual Hypergraph Inference of Communication

CRYCHIC currently provides a runnable v0.1 exploratory baseline for
sample-aware cell-cell communication analysis. It validates count or explicitly
declared normalized AnnData, aggregates sample x cell-type pseudobulks, builds
balanced context contrasts, scores LR availability, optionally attributes
receiver responses with a versioned NicheNet prior, and writes atomic,
queryable results.

This is not the completed roadmap or a calibrated inference release. v0.1
strengths are descriptive and in-sample; p-values, q-values,
`comm_probability`, posterior quantities, and causal sender claims are disabled.

## Install

Python 3.11-3.13 is supported. From this repository:

```bash
uv sync --extra dev --extra resources --extra plotting --extra benchmark
```

## Repository Scope

This branch tracks the core package, benchmark code and configurations, tests,
schemas, method documentation, and checksum-pinned resource manifests. Large or
locally generated assets are intentionally not versioned:

- input `.h5ad`/HDF5 matrices;
- database payloads under the workspace-level `databases/` directory;
- `benchmark_work/` intermediate and finalized run directories;
- generated `reports/` documents and figures.

The resource manifests in `resources/` identify the expected database versions
and checksums. The benchmark and report commands below require those external
assets at the paths supplied by the caller.

## Minimal Run

```python
from pathlib import Path

import anndata as ad
import crychic

database_root = Path("../databases")
bundle = crychic.load_cellchat_resource(
    database_root,
    "human",
    manifest_path="resources/cellchatdb_human_v2.json",
)
prior = crychic.load_nichenet_target_prior(
    database_root,
    manifest_path="resources/nichenet_human_v2_2021.json",
)

adata = ad.read_h5ad("cohort.h5ad")
config = crychic.CrychicConfig(
    context_keys=["condition"],
    counts_layer="counts",
    sample_key="sample_id",
    subject_key="subject_id",
    cell_type_key="cell_type",
    random_seed=20260712,
)
model = crychic.Crychic(
    config,
    resource_bundle=bundle,
    target_prior=prior,
)
plan = model.dry_run(adata)
result = model.fit(
    adata,
    output_dir="crychic_result",
    input_digest="<sha256-of-input-h5ad>",
)
```

`model.fit()` returns in-memory `BaselineArtifacts` when `output_dir` is
omitted. With an output directory it returns `CrychicResult`, which supports
validated table reads and ranked interaction queries.

## Reproduce Benchmarks

```bash
uv run --extra benchmark python benchmarks/run_canonical_v01.py
uv run --extra benchmark python -m benchmarks.simulation.run_negative_controls \
  --output-dir ../benchmark_work/synthetic_v01 --seed 20260712
uv run --extra resources --extra benchmark --extra plotting \
  python benchmarks/report/generate_canonical_report.py --workspace-root ..
```

## Quality Checks

```bash
uv run --extra dev --extra resources --extra plotting --extra benchmark pytest -q
uv run --extra dev ruff check src tests benchmarks scripts
uv run --extra dev mypy
uv run --extra dev python -m build
```

- [Detailed development plan](DEVELOPMENT_PLAN.md)
- [v0.1 method specification](docs/methods/v0.1-exploratory-baseline.md)
- [Repository development instructions](AGENTS.md)
