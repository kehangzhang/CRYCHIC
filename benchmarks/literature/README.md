# Literature evidence for the multi-condition benchmark

This directory contains the auditable manifest and preregistered design for the
multi-condition cell-cell communication benchmark. Large or publisher-controlled
full text is stored outside Git under `benchmark_work/literature/fulltext/` at
the workspace root.

## Retrieval method

The user requested the `scansci` skill. It was not present in the skills catalog
available to this run, so it could not be invoked. The documented fallback was:

1. read `../../../benchmark.md` and `../../DEVELOPMENT_PLAN.md` in full;
2. resolve DOI metadata and licenses with Crossref;
3. verify bibliographic records and open-access status with Europe PMC;
4. download open full-text XML with the Europe PMC REST API and publisher PDFs
   only from public article URLs;
5. verify XML parsing and PDF page counts, then compute SHA256 digests;
6. record access and redistribution restrictions rather than bypassing them.

The snapshot was retrieved on 2026-07-12. The manifest deliberately includes
the peer-reviewed 2026 Genome Biology benchmark, DOI
`10.1186/s13059-026-04063-5`. The later `SpatialCCCbench` article remains a
preprint and is supplementary evidence only.

## Files

- `literature_manifest.tsv`: DOI, source, license, local cache status and SHA256.
- `MULTICONDITION_BENCHMARK_PROTOCOL.md`: frozen task definitions and metrics.
- `DATASET_ELIGIBILITY.tsv`: current local dataset audit and analysis eligibility.
- `DOWNLOAD_TARGETS.tsv`: prioritized supplement, code and data retrieval queue.
- `benchmark_work/literature/FULLTEXT_SHA256.tsv`: checksum for every cached file.

The full-text cache currently contains 19 unique papers, including 12 PDFs, 12
OA XML files and checksum-pinned supplementary archives/tables for the cSCC and
MS validation cohorts; some articles have more than one representation. The XML
files were validated with `xmllint`, and all PDFs passed `pdfinfo` with nonzero
page counts.

## Dimitrov 2022 multi-modal validation

The scripts below reproduce three validation tracks from Dimitrov et al.,
Nature Communications 2022 (`10.1038/s41467-022-30755-0`) with current method
implementations:

- `evaluate_cytokine_activity.py`: TNBC CytoSig Fisher odds-ratio curves plus a
  controlled unique ligand-target AUROC/AP arm;
- `evaluate_citeseq.py`: receptor-protein AUROC and the paper's repeated 1:1
  negative-sampling AUPRC on four complete CITE-seq datasets;
- `prepare_pbmc3k_robustness.py`, `run_robustness_liana.py`,
  `run_robustness_crychic.py`, and `evaluate_robustness.py`: PBMC3k cell
  subsampling, label reshuffling, selective LR replacement, and non-selective
  LR replacement at 5% increments through 40%; and
- `report_dimitrov_validation.py`: publication figures, source-data tables and
  a Chinese summary report spanning all three tracks.

The robustness primary metric is recovery of each method's own unperturbed
top 250. It is a reproducibility endpoint, not biological accuracy. The report
also retains the paper code's so-called TPR (intersection divided by the
perturbed top-set size) and Jaccard. Resource perturbations are replacements,
not additions, and the selective arm preserves the union of baseline top LR
pairs across CRYCHIC and the external methods.

CRYCHIC is evaluated here only through its public single-sample static
`availability_state`. These single-condition datasets cannot estimate the full
multi-condition differential `comm_strength` workflow.

Typical execution uses the dedicated LIANA and CRYCHIC environments:

```bash
PYTHONPATH=. /path/to/liana_env/bin/python \
  -m benchmarks.literature.prepare_pbmc3k_robustness \
  /path/to/pbmc3k_filtered_gene_bc_matrices.tar.gz \
  /path/to/robustness/pbmc3k_prepared.h5ad

PYTHONPATH=. .venv/bin/python \
  -m benchmarks.literature.run_robustness_crychic \
  /path/to/robustness/pbmc3k_prepared.h5ad \
  /path/to/liana_consensus_human.parquet \
  /path/to/robustness/crychic_baseline --baseline-only

PYTHONPATH=. /path/to/liana_env/bin/python \
  -m benchmarks.literature.run_robustness_liana \
  /path/to/robustness/pbmc3k_prepared.h5ad \
  /path/to/robustness/liana_full \
  --crychic-baseline-top \
  /path/to/robustness/crychic_baseline/baseline_top_predictions.parquet \
  --n-perms 20 --n-jobs 8 --replicates 5

PYTHONPATH=. .venv/bin/python \
  -m benchmarks.literature.run_robustness_crychic \
  /path/to/robustness/pbmc3k_prepared.h5ad \
  /path/to/liana_consensus_human.parquet \
  /path/to/robustness/crychic_full \
  --liana-run-dir /path/to/robustness/liana_full
```

If a long LIANA run is interrupted after writing both partial files, rerun the
same LIANA command with `--resume`. The runner validates the frozen design,
baseline resource, baseline ranking, completed method sets and
prediction/audit keys before continuing. `--resume` and `--overwrite` are
mutually exclusive. An advisory process lock prevents two runners from writing
the same output directory concurrently.

NicheNet is intentionally not forced into these direct LR rankings. As in the
paper, its ligand-to-target regulatory potential is a complementary task with
a different estimand.
