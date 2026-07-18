# LIANA+ condition-aware S4 adapter

`run_condition_aware.py` reproduces the LIANA+ multi-sample arm described in
Cesaro et al. Section S4. It keeps the existing LIANA 1.7.3 descriptive adapter
separate from the inferential paper arm.

The exact worker is pinned to LIANA+ 1.5.0, decoupler 1.8.0, PyDESeq2 0.5.0,
anndata 0.10.8, numba 0.60.0, NumPy 1.26.4, pandas 2.2.3, and Scanpy 1.10.4.
The LIANA tag is `v1.5.0` at commit
`8f8f3d6617b190aaaf0d50fdff68aa16426abaf8`. Known PyPI wheel hashes are
recorded in every run manifest. The targeted notebook is Git blob
`105c49abcab2a278f0f5e88ade5a57d0e85a7d48`, SHA256
`56712d7c35ce9fb558f3adfa066dad904b0bbdbce728fe006696454f7eac621e`.

The primary Kuppe and MS panels use `subject_id` as the pseudobulk replicate.
This collapses repeated tissue sections before differential expression and
prevents pseudoreplication. A `sample_id` run, if produced, is a separate paper
sensitivity panel and must not be mixed into the subject-level leaderboard.

The protocol follows the LIANA 1.5 targeted differential-expression vignette:

- decoupler summed pseudobulk with `min_cells=10` and `min_counts=10000`;
- `filter_by_expr(min_count=5, min_total_count=10)` per cell type;
- PyDESeq2 Wald target-versus-reference tests and LFC shrinkage;
- `df_to_lr(expr_prop=0.1, complex_col="stat")` with Wald statistic, raw
  p-value, and adjusted p-value passed through `stat_keys`;
- raw `interaction_pvalue < 0.05` for the paper's small-sample, no-additional-
  correction rule; the gene-level adjusted p-values remain in the output;
- positive `interaction_stat` assigned to the target condition and negative
  values assigned to the reference condition.

The modern repository environment reads only the H5AD axes. The exact worker
reads the frozen raw-count CSR arrays directly from HDF5. This is required for
MS because anndata 0.10.8 cannot decode the newer nullable-string metadata
encoding; expression values are not converted or approximated.

decoupler 1.8.0 passes pandas boolean Series directly to SciPy CSR indexing.
The exact worker installs a process-local bridge only around `get_pseudobulk`
that converts the same boolean values to NumPy arrays. The bridge does not
modify installed files or aggregation semantics and is recorded in each run
manifest.

Example primary run:

```bash
uv run python -m benchmarks.adapters.liana.run_condition_aware \
  --input-h5ad INPUT.h5ad \
  --input-manifest INPUT.json \
  --resource CONNECTOMEDB.tsv \
  --resource-manifest CONNECTOMEDB.manifest.json \
  --exact-python /path/to/liana_1_5_0/bin/python \
  --output-dir OUTPUT \
  --dataset-id DATASET \
  --replicate-key subject_id \
  --condition-key condition \
  --target TARGET \
  --reference REFERENCE \
  --cores 8
```
