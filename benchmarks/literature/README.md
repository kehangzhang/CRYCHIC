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
