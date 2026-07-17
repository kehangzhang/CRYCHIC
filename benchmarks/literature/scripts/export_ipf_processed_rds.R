#!/usr/bin/env Rscript

# Export the four Xie et al. processed Seurat objects to checksum-bound,
# per-patient sparse bundles.  The study-specific label vectors below are a
# literal transcription of the commit-pinned author preprocessing scripts.

suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(digest))
suppressPackageStartupMessages(library(parallel))

usage <- function() {
  cat(
    "Usage: export_ipf_processed_rds.R --study GSE... --input FILE.rds\n",
    "       --roster ipf_sample_manifest.tsv --output-root DIR [--workers N]\n",
    sep = ""
  )
}

options <- list(
  study = NULL, input = NULL, roster = NULL, output_root = NULL, workers = "1"
)
args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 1L && args[[1L]] %in% c("-h", "--help")) {
  usage()
  quit(status = 0L)
}
if (length(args) %% 2L != 0L) {
  usage()
  stop("arguments must be supplied as --name value pairs", call. = FALSE)
}
for (index in seq.int(1L, length(args), by = 2L)) {
  key <- sub("^--", "", args[[index]])
  key <- gsub("-", "_", key, fixed = TRUE)
  if (!key %in% names(options)) stop("unknown argument: ", args[[index]], call. = FALSE)
  options[[key]] <- args[[index + 1L]]
}
if (any(vapply(options[c("study", "input", "roster", "output_root")], is.null, logical(1)))) {
  usage()
  stop("--study, --input, --roster, and --output-root are required", call. = FALSE)
}
workers <- suppressWarnings(as.integer(options$workers))
if (is.na(workers) || workers < 1L) stop("--workers must be a positive integer")
if (workers > 1L && .Platform$OS.type == "windows") {
  stop("--workers > 1 requires a fork-capable non-Windows platform")
}
if (!file.exists(options$input)) stop("input RDS does not exist: ", options$input)
if (!file.exists(options$roster)) stop("roster does not exist: ", options$roster)

canonical_types <- c(
  "AT1", "AT2", "Endothelial", "Fibroblast",
  "Macrophage", "Mast", "Monocyte", "Tcell"
)

label_maps <- list(
  GSE122960 = c(
    "Macrophage", "Macrophage", "AT2", "AT2", "AT2", "Plasma",
    "Club/Basal", "Dendritic cell", "Monocyte", "Macrophage", "Ciliated",
    "Macrophage", "AT1", "Tcell", "Endothelial", "Macrophage", "Unknown",
    "B cell", "Macrophage", "Fibroblast", "Mast", "AT2", "B cell",
    "Endothelial", "AT2", "Macrophage", "AT2"
  ),
  GSE128033 = c(
    "Macrophage", "Macrophage", "Macrophage", "Unknown", "Unknown", "Tcell",
    "Fibroblast", "Endothelial", "Monocyte", "AT2", "Ciliated cell",
    "Dendritic cell", "Club/Goblet cell", "Pericyte", "AT1", "Unknown",
    "NK cell", "Mast", "Unknown", "Fibroblast", "Smooth muscle cell",
    "B cell", "Ciliated cell", "Basal cell", "Dendritic cell", "Unknown",
    "Lymphatic endothelial", "Monocyte", "Unknown", "AT1"
  ),
  GSE135893 = c(
    "AT1", "AT2", "B Cells", "Basal", "Dendritic cell", "Ciliated",
    "Differentiating Ciliated", "Endothelial", "Fibroblast",
    "HAS1 High Fibroblasts", "KRT5-/KRT17+", "Lymphatic Endothelial Cells",
    "Macrophage", "Mast", "Mesothelial Cells", "Monocyte", "MUC5AC+ High",
    "MUC5B+", "Myofibroblasts", "NK Cells", "Dendritic cell", "Plasma Cells",
    "PLIN2+ Fibroblasts", "Proliferating Epithelial Cells",
    "Proliferating Macrophages", "Proliferating T Cells", "SCGB3A2+",
    "SCGB3A2+ SCGB1A1+", "Smooth Muscle Cells", "Tcell", "Transitional AT2"
  ),
  GSE136831 = c(
    "Monocyte", "Alveolar macrophage", "NK cell", "Monocyte", "Lymphatic",
    "Macrophage", "Fibroblast", "Basal cell", "Myofibroblast", "Multiplet",
    "Dendritic cell", "B cell", "Endothelial", "Tcell", "AT2", "AT1",
    "Endothelial", "Endothelial", "Ciliated cell", "Club cell", "Endothelial",
    "Plasma", "ILC", "T cytotoxic", "Goblet cell", "T regulatory",
    "Mesothelial", "ILC", "SMC", "Dendritic cell", "Dendritic cell", "Mast",
    "Dendritic cell", "Dendritic cell", "Endothelial", "Pericyte",
    "Aberrant basaloid", "Ionocyte", "PNEC"
  )
)
sample_columns <- c(
  GSE122960 = "orig.ident",
  GSE128033 = "orig.ident",
  GSE135893 = "Sample_Name",
  GSE136831 = "orig.ident"
)
if (!options$study %in% names(label_maps)) {
  stop("unsupported study: ", options$study, call. = FALSE)
}

sha256_file <- function(path) {
  digest(path, algo = "sha256", file = TRUE, serialize = FALSE)
}

record_file <- function(path) {
  list(
    filename = basename(path),
    bytes = unname(file.info(path)$size),
    sha256 = sha256_file(path)
  )
}

gzip_file <- function(source, destination) {
  input <- file(source, open = "rb")
  output <- gzfile(destination, open = "wb", compression = 9)
  on.exit(close(input), add = TRUE)
  on.exit(close(output), add = TRUE)
  repeat {
    chunk <- readBin(input, what = "raw", n = 1024L * 1024L)
    if (!length(chunk)) break
    writeBin(chunk, output)
  }
}

write_bundle <- function(
  counts, cells, study_id, sample_id, output_root, source, source_sha256
) {
  final <- file.path(output_root, study_id, sample_id)
  manifest_path <- file.path(final, "manifest.json")
  if (dir.exists(final)) {
    if (!file.exists(manifest_path)) stop("partial bundle already exists: ", final)
    manifest <- fromJSON(manifest_path, simplifyVector = FALSE)
    if (!identical(manifest$status, "complete")) stop("incomplete bundle exists: ", final)
    for (key in c("counts", "features", "cells")) {
      path <- file.path(final, manifest$outputs[[key]]$filename)
      if (!file.exists(path) || sha256_file(path) != manifest$outputs[[key]]$sha256) {
        stop("existing bundle checksum mismatch: ", path)
      }
    }
    message("verified existing bundle ", study_id, "/", sample_id)
    return(invisible(manifest))
  }
  dir.create(dirname(final), recursive = TRUE, showWarnings = FALSE)
  staging <- paste0(final, ".tmp-", Sys.getpid())
  if (dir.exists(staging)) stop("stale staging directory exists: ", staging)
  dir.create(staging, recursive = FALSE)
  complete <- FALSE
  on.exit(if (!complete && dir.exists(staging)) unlink(staging, recursive = TRUE), add = TRUE)

  counts_path <- file.path(staging, "counts.mtx.gz")
  matrix_market <- file.path(staging, "counts.mtx")
  features_path <- file.path(staging, "features.tsv")
  cells_path <- file.path(staging, "cells.tsv")
  Matrix::writeMM(counts, matrix_market)
  gzip_file(matrix_market, counts_path)
  unlink(matrix_market)
  write.table(
    data.frame(gene_symbol = rownames(counts), check.names = FALSE),
    features_path, sep = "\t", quote = FALSE, row.names = FALSE
  )
  write.table(cells, cells_path, sep = "\t", quote = FALSE, row.names = FALSE)
  manifest <- list(
    schema_version = "xie-ipf-processed-rds-sample-bundle-v1",
    status = "complete",
    study_id = study_id,
    sample_id = sample_id,
    shape_features_by_cells = unname(dim(counts)),
    nnz = length(counts@x),
    cell_type_counts = as.list(table(cells$cell_type)),
    source = list(
      filename = basename(source),
      sha256 = source_sha256,
      zenodo_record = "10.5281/zenodo.6497091",
      author_workflow_commit = "279718d539b47dc890c6f3f2e4c03f5a5df33c3e"
    ),
    outputs = list(
      counts = record_file(counts_path),
      features = record_file(features_path),
      cells = record_file(cells_path)
    )
  )
  write_json(manifest, file.path(staging, "manifest.json"), pretty = TRUE, auto_unbox = TRUE)
  if (!file.rename(staging, final)) stop("failed to atomically publish bundle: ", final)
  complete <- TRUE
  invisible(manifest)
}

roster <- read.delim(options$roster, check.names = FALSE, stringsAsFactors = FALSE)
required_roster <- c("geo_accession", "sample_id")
if (!all(required_roster %in% names(roster))) stop("roster lacks required columns")
expected_samples <- roster$sample_id[roster$geo_accession == options$study]
if (!length(expected_samples) || anyDuplicated(expected_samples)) {
  stop("study roster is empty or duplicated: ", options$study)
}

source_path <- normalizePath(options$input, mustWork = TRUE)
source_sha256 <- sha256_file(source_path)
object <- readRDS(source_path)
if (!inherits(object, "Seurat")) stop("processed RDS is not a Seurat object")
metadata <- object[[]]
sample_column <- unname(sample_columns[[options$study]])
if (!sample_column %in% names(metadata)) {
  stop("processed object lacks sample column ", sample_column)
}
sample_ids <- as.character(metadata[[sample_column]])

if ("cell.type" %in% names(metadata)) {
  candidate <- as.character(metadata$cell.type)
  supplied_labels_are_author_mapped <- all(unique(candidate) %in% unique(label_maps[[options$study]]))
} else {
  candidate <- rep(NA_character_, nrow(metadata))
  supplied_labels_are_author_mapped <- FALSE
}
if (!supplied_labels_are_author_mapped) {
  identities <- Idents(object)
  identity_levels <- levels(identities)
  frozen_map <- label_maps[[options$study]]
  if (length(identity_levels) != length(frozen_map)) {
    stop(
      "identity level count differs from frozen author mapping for ", options$study,
      ": observed ", length(identity_levels), ", expected ", length(frozen_map)
    )
  }
  names(frozen_map) <- identity_levels
  candidate <- unname(frozen_map[as.character(identities)])
}
if (anyNA(candidate)) stop("one or more cells lack an author-mapped cell label")

selected <- sample_ids %in% expected_samples & candidate %in% canonical_types
observed_samples <- sort(unique(sample_ids[selected]))
if (!identical(observed_samples, sort(expected_samples))) {
  missing <- setdiff(expected_samples, observed_samples)
  extra <- setdiff(observed_samples, expected_samples)
  stop(
    "processed object/roster sample mismatch; missing=", paste(missing, collapse = ","),
    ", extra=", paste(extra, collapse = ",")
  )
}

assay <- if ("RNA" %in% Assays(object)) "RNA" else DefaultAssay(object)
counts <- tryCatch(
  GetAssayData(object, assay = assay, layer = "counts"),
  error = function(error) GetAssayData(object, assay = assay, slot = "counts")
)
if (!nrow(counts) || !ncol(counts)) {
  stop("selected assay has an empty count layer: ", assay)
}
if (!inherits(counts, "sparseMatrix")) counts <- Matrix(counts, sparse = TRUE)
counts <- as(counts, "dgCMatrix")
if (is.null(rownames(counts)) || anyDuplicated(rownames(counts))) {
  stop("count matrix gene identifiers must be present and unique")
}
if (is.null(colnames(counts)) || !identical(colnames(counts), rownames(metadata))) {
  stop("count matrix columns are not aligned to Seurat metadata")
}
if (length(counts@x) && any(!is.finite(counts@x) | counts@x < 0 | counts@x != round(counts@x))) {
  stop("count layer must contain finite non-negative integers")
}

export_sample <- function(sample_id) {
  tryCatch({
  keep <- selected & sample_ids == sample_id
  sample_counts <- counts[, keep, drop = FALSE]
  cells <- data.frame(
    barcode = paste(options$study, sample_id, colnames(sample_counts), sep = "_"),
    study_id = options$study,
    sample_id = sample_id,
    cell_type = candidate[keep],
    check.names = FALSE,
    stringsAsFactors = FALSE
  )
  if (anyDuplicated(cells$barcode)) stop("generated sample barcodes are duplicated")
  write_bundle(
    sample_counts, cells, options$study, sample_id,
    options$output_root, source_path, source_sha256
  )
    list(sample_id = sample_id, status = "complete", error = NULL)
  }, error = function(error) {
    list(sample_id = sample_id, status = "failed", error = conditionMessage(error))
  })
}

if (workers == 1L) {
  results <- lapply(expected_samples, export_sample)
} else {
  results <- parallel::mclapply(
    expected_samples,
    export_sample,
    mc.cores = workers,
    mc.preschedule = FALSE,
    mc.set.seed = FALSE
  )
}
failed <- Filter(function(item) !identical(item$status, "complete"), results)
if (length(failed)) {
  details <- vapply(
    failed,
    function(item) paste0(item$sample_id, ": ", item$error),
    character(1)
  )
  stop("one or more sample exports failed: ", paste(details, collapse = "; "))
}

cat(toJSON(list(
  status = "complete",
  study = options$study,
  samples = length(expected_samples),
  workers = workers
), auto_unbox = TRUE), "\n")
