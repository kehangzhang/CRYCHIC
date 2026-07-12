# External benchmark adapters

These runners execute each method independently for every biological sample and
export `crychic-external-interactions-long-v1`. They do not perform a
between-condition test. CellChat and CellPhoneDB native p-values retain their
within-sample semantics and must not be interpreted as condition effects.

## Frozen universe and statuses

Every sample materializes the same Cartesian universe of frozen interactions
and the dataset-wide cell-type ontology. `universe_size` is therefore the exact
number of `sender x receiver x interaction` rows per sample, and `universe_id`
is checksum-stable across methods in the same harmonized arm.

- `ok`: the method emitted a finite native score.
- `not_returned`: eligible edge absent from sparse native output; native score is
  `NA`, while comparison code may rank it below emitted edges.
- `resource_unavailable`: edge is in H-covered but unavailable to the method.
- `insufficient_cells`: sender or receiver has fewer than `--min-cells` in this
  sample.
- `missing`: required input genes are absent.
- `method_failed`: the sample-specific external call failed.
- `unsupported_resource`: the method cannot execute the requested resource arm.

The resource arms are `H-common`, `H-covered`, and `native`. H-common is the
exact simple-LR intersection. H-covered is the frozen simple-LR union with
method coverage retained as status. Native uses each released method-resource
system. Never average arms into one score.

## Build resources

Define the workspace locations once; the examples below reuse these variables:

```bash
WORKSPACE_ROOT=/path/to/crychic_dev
REPO_ROOT="$WORKSPACE_ROOT/CRYCHIC"
DATABASE_ROOT="$WORKSPACE_ROOT/databases"
RESULT_ROOT="$WORKSPACE_ROOT/benchmark_work/multicondition_v01"

uv run python -m benchmarks.adapters.build_harmonized_resource \
  --database-root "$DATABASE_ROOT" \
  --repo-root "$REPO_ROOT" \
  --output-dir "$RESULT_ROOT/resources/harmonized_simple_lr" \
  --overwrite
```

This writes `harmonized_lr.tsv` (H-common), `covered_lr.tsv` (H-covered), and a
checksum manifest.

## Runners

CellChat uses the `r_cellchat` conda environment and its frozen RDS database:

```bash
uv run python -m benchmarks.adapters.cellchat.run_by_sample INPUT.h5ad OUTPUT \
  --dataset-id DATASET --database-root "$DATABASE_ROOT" \
  --context-key condition --resource-mode H-common \
  --harmonized-resource RESOURCE_DIR/harmonized_lr.tsv \
  --harmonized-manifest RESOURCE_DIR/manifest.json --min-cells 10
```

CellPhoneDB must run inside `ov_2`. Its statistical mode requires at least 100
permutations because CellPhoneDB 5.0.1 otherwise divides by a zero progress
step. The adapter generates a checksum-recorded filtered CPDB archive for each
harmonized arm.

```bash
PYTHONPATH="$REPO_ROOT/src" \
conda run -n ov_2 python -m benchmarks.adapters.cellphonedb.run_by_sample \
  INPUT.h5ad OUTPUT --dataset-id DATASET \
  --database-root "$DATABASE_ROOT" \
  --context-key condition --resource-mode H-common \
  --harmonized-resource RESOURCE_DIR/harmonized_lr.tsv \
  --harmonized-manifest RESOURCE_DIR/manifest.json --iterations 1000 --threads 1
```

LIANA uses `liana_env`. The runner calls `rank_aggregate` in an isolated
per-sample loop equivalent to `by_sample`, so one failed sample remains an
explicit status instead of aborting the cohort.

```bash
conda run -n liana_env python -m benchmarks.adapters.liana.run_by_sample \
  INPUT.h5ad OUTPUT --dataset-id DATASET --context-key condition \
  --resource-mode H-common --harmonized-resource RESOURCE_DIR/harmonized_lr.tsv \
  --harmonized-manifest RESOURCE_DIR/manifest.json --n-perms 1000
```

The NicheNet package is not installed locally. Its runner therefore consumes
the checksum-pinned v2_2021 top-250 ligand-target matrix directly and exports a
separate `ligand_target_program` Track B score. It is a sample-level proxy equal
to sender ligand mean expression times the receiver's weighted target-program
mean. It is explicitly not `predict_ligand_activities`, which requires a
receiver DE gene set and background universe.

```bash
uv run python -m benchmarks.adapters.nichenet.run_by_sample INPUT.h5ad OUTPUT \
  --dataset-id DATASET --database-root "$DATABASE_ROOT" \
  --context-key condition --min-cells 10 --min-targets 25
```

Each output directory contains `manifest.json`, `interactions_long.parquet`,
and method-native raw files where the external API exposes them.
