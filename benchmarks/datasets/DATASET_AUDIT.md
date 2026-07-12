# Multi-condition dataset and design audit

Audit date: 2026-07-12. Paths in the inventory are relative to the shared
`crychic_dev` workspace unless stated otherwise. The machine-readable source of
record is `dataset_inventory.json`; the flat execution matrix is
`dataset_inventory.tsv`.

## Execution priority

| Priority | Dataset | Current status | Statistical role |
| --- | --- | --- | --- |
| 1 | GSE144236 Ji cSCC | Prepared and CRYCHIC-validated | Primary paired real-data development cohort |
| 2 | Lerma-Martin MS snRNA | Exact UCSC matrix prepared and CRYCHIC dry-run validated | Locked CA-vs-Ctrl holdout; CI/three-group descriptive; spatial extension separate |
| 3 | Kuppe MI | Converter and design audit implemented; full counts write deferred | Descriptive availability and scalability validation; response blocked by mixed design |
| 4 | Panc02 PancVAX/ICI | Eight treatments and two object types available | Descriptive multigroup software stress test only |
| Controls | Kang, AD skin, trophoblast, embryonic skin | Previously prepared/reference objects | Positive-control, tutorial, and structural-absence checks |

This ordering is based on independent biological replication and input
readiness, not dataset size.

## GSE144236 Ji cSCC

The converter `prepare_cscc.py` reads the 3.16 GB uncompressed wide count table
as a stream and never materializes the dense cell-by-gene matrix. It verifies
exact cell-header alignment against the corrected metadata, per-cell library
size, detected-gene count, non-negative integer counts, unique identifiers, and
the full paired design.

Prepared artifact outside Git:

```text
benchmark_work/multicondition_v01/prepared/GSE144236_cscc_paired.h5ad
```

- Shape: 47,068 cells by 32,738 unique gene symbols.
- Filter: 1,096 `Multiplet` cells removed before every method.
- Primary annotation: `level1_celltype`, 14 retained cell types.
- Independent unit: 10 patients.
- Samples: 20, defined exactly as `patient + "_" + condition`.
- Contexts: every patient has both `Normal` and `Tumor`.
- Expression: `layers["counts"]` contains raw `int32` counts; `X` is log1p
  library-normalized to 10,000.
- Artifact SHA256:
  `b15759def47df2c2aa5e1936398c9e57fab92dac6814b77d31839ae50b675b81`.

At a minimum of 10 cells in each patient-context, paired subject support is 10
for Epithelial, 8 for CD1C, 7 for LC, 6 for Mac and Tcell, and 5 for Fibroblast
and CLEC9A. MDSC has 4 pairs; Endothelial Cell and Melanocyte have 3; rarer
types have fewer. Do not freeze one global seven-cell-type subset and silently
apply it to all edges. Freeze the full annotation, retain per-sample eligibility,
and require a prespecified number of complete subject pairs for each
sender-receiver edge. Report excluded edges with their support reason.

Run all requested human methods on the same prepared cells and primary labels.
CellChat, CellPhoneDB, and LIANA belong to the LR/STLR track. NicheNet or
MultiNicheNet belongs to the ligand-target/receiver-response track. CRYCHIC may
enter both only with the native field semantics intact. Method-internal p-values
are not interchangeable condition-difference p-values.

## Lerma-Martin MS

GEO records for GSM8563681 through GSM8563696 explicitly identify the 16 local
10x matrices as snRNA-seq. They contain 134,118 raw barcodes by 36,601 features.
They are not the spatial matrices. However, no cell-type annotation is present
in the local directory.

The official UCSC processed snRNA matrix and metadata were checksum-validated,
exactly aligned by barcode order, and prepared outside Git:

```text
benchmark_work/multicondition_v01/prepared/ms_snrna_ucsc/
benchmark_work/multicondition_v01/prepared/UCSC_Lerma_Martin_MS_snRNA.h5ad
benchmark_work/multicondition_v01/prepared/UCSC_Lerma_Martin_MS_CA_vs_Ctrl.h5ad
```

The full object contains 103,794 nuclei by 32,115 unique gene symbols, 220,543,010
nonzero raw integer counts, 13 subjects, 16 samples, 9 major cell types and 84
nonmissing subtypes. `X` is log1p library-normalized to 10,000 and `counts` is
`int32`. The full-object SHA256 is
`3019c05759439ceacff29db2b80dae79beb1b38687f574581803d1da51a2d3ba`.
Conversion took 108 seconds on the benchmark host with a 6.82 GiB peak RSS.
Independent-subject support is:

| Context | Subjects | Samples | Nuclei |
| --- | ---: | ---: | ---: |
| Ctrl | 6 | 6 | 29,104 |
| CA | 5 | 6 | 45,900 |
| CI | 2 | 4 | 28,790 |

The primary contrast must therefore be CA versus Ctrl. CI contrasts and the
three-group result are secondary/descriptive because four CI samples arise from
only two independent subjects.

The frozen CA-versus-Ctrl object contains 75,004 nuclei, 12 samples and 11
independent subjects. Its SHA256 is
`612fe9c4cdaf88694a47e13ba4458c828f206cd945eba9e9c72f46e9bd0196c7`.
The reviewed primary formula is `~ batch + lesion_type`; its design matrix is
full rank (5/5), and the context contrast is estimable. Six CA libraries map to
five subjects, so same-subject libraries are collapsed before equal-subject
estimation. AS, EC, MG, NEU, OL and OPC meet four-subject support in both groups;
rare BC/SC/TC results retain explicit receiver-level support states.

The UCSC processed barcodes overlap the local raw matrices by a sample-dependent
fraction, approximately 50% to 99.9% in the audit. The benchmark therefore uses
the exact UCSC matrix rather than joining annotations to the local GEO matrices.
Matched spatial inputs remain a separate orthogonal DES validation layer.

## Kuppe MI

The local h5ad contains 191,795 nuclei, 29,126 genes, 20 patients, 29 samples,
and 11 broad cell types. `X` is log-normalized, while `raw.X` contains integer
counts. `prepare_kuppe.py` strictly validates these dimensions and metadata,
copies `raw.X` to an `int32` `counts` layer, preserves normalized `X`, and writes
source/output SHA256 lineage. The full matrix write remains intentionally
deferred for a separately scheduled execution. Its metadata-only audit is at:

```text
benchmark_work/multicondition_v01/prepared/kuppe_design_audit/
```

Region support is CTRL 4 subjects, RZ 5, BZ 3, IZ 7 and FZ 6. P2, P3 and P9
provide multiple regions; most patients contribute one region. P9 has three IZ
samples, while P15 and P16 have two IZ samples each. Consequently this is
neither fully paired nor a simple independent five-group design. RZ-BZ, RZ-IZ,
and BZ-IZ mix paired and unpaired subjects; the remaining region pairs are
between-subject candidates only after explicit, preregistered subsetting.

The current CRYCHIC response backend cannot combine the paired, unpaired, and
nested within-region contributions without changing the estimand. The complete
five-region response analysis is therefore `not_estimable`, with reason code
`mixed_paired_unpaired_multi_region_design_unsupported`. Do not pool cells or
silently discard unpaired subjects. Per-sample/per-region availability and
runtime analysis may run as descriptive-only outputs, with every sample from a
patient kept in one resampling block. BZ and CTRL cannot support broad formal
claims under typical small-cluster thresholds.

## Panc02 PancVAX/ICI

The 16 local processed objects are double-gzip Seurat RDS files, not corrupt
downloads. After opening both compression layers, they contain 56,674 cells,
24,337 genes, raw integer counts, 14 cell types, eight treatment arms, and a
`tumor` or `immune` object for each treatment.

No independent mouse identifier is available, and the study/benchmark use
pooling or cell bootstrap in relevant analyses. The 16 objects are not 16
biological replicates. This cohort can test multigroup ingestion, contrasts,
failure handling, runtime and memory, but cannot evaluate type-I error, FDR,
coverage, subject variability or a method's inferential superiority. Mouse
resource and orthology policies must also be frozen before comparing methods.

## Fair run matrix

| Dataset | CellChat | CellPhoneDB | NicheNet | LIANA | CRYCHIC |
| --- | --- | --- | --- | --- | --- |
| cSCC | Run per sample/condition; LR track | Run per sample; LR track | MultiNicheNet/paired target track | `by_sample`; LR track | Counts path; exploratory until gates pass |
| MS | Ready on exact UCSC object | Ready on exact UCSC object | NicheNet target track | Ready with `by_sample` | Counts path; CA-vs-Ctrl design validated |
| MI | Per-sample descriptive run after counts conversion | Same | Response track blocked for the full mixed design | Per-sample descriptive run with patient blocks | Availability-only; response `not_estimable` |
| PancVAX | Descriptive mouse run | Only with frozen mouse mapping | Only with frozen mouse prior | Only with frozen mouse policy | Availability/descriptive stress test |

For every eligible dataset run two resource arms without averaging them:

1. A harmonized, exactly mappable LR universe with explicit coverage states.
2. Each method's version-locked native resource.

Compare only native quantities with the same estimand. Keep unavailable,
filtered, missing-cell-type, failed and not-estimable states distinct from a
zero score. Real-data agreement is consensus, not ground truth; causal,
probability and calibrated significance claims remain disabled until their
development-plan gates pass.
