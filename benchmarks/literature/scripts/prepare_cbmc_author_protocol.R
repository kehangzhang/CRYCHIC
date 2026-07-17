#!/usr/bin/env Rscript

# Export the public CBMC SeuratData object using the labels and QC in the
# released Dimitrov/LIANA benchmark. Protein is retained only in truth tables.

suppressPackageStartupMessages({
  library(Matrix)
  library(Seurat)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop(
    "Usage: prepare_cbmc_author_protocol.R INPUT_RDA OUTPUT_DIR DATASET_ID",
    call. = FALSE
  )
}

input_rda <- normalizePath(args[[1L]], mustWork = TRUE)
output_dir <- normalizePath(args[[2L]], mustWork = FALSE)
dataset_id <- args[[3L]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
started <- Sys.time()

objects <- new.env(parent = globalenv())
loaded <- load(input_rda, envir = objects)
if (length(loaded) != 1L) {
  stop("Expected exactly one object in the CBMC RDA", call. = FALSE)
}
cbmc <- objects[[loaded[[1L]]]]
if (!all(c("RNA", "ADT") %in% names(cbmc@assays))) {
  stop("CBMC object must contain RNA and ADT assays", call. = FALSE)
}
if (!("rna_annotations" %in% colnames(cbmc@meta.data))) {
  stop("CBMC metadata lacks rna_annotations", call. = FALSE)
}

# This is the exact cleanup in analysis/comparison/cbmc_prep.R. The paper's
# display label says 3kCBMCs, but the public object yields 7,713 retained cells.
cell_type <- as.character(cbmc$rna_annotations)
cell_type <- gsub("[+]", "", cell_type)
cell_type <- sub("/", ".", cell_type, fixed = TRUE)
cell_type[cell_type == "B"] <- "B cell"
keep <- !(cell_type %in% c("Mouse", "Multiplets", "T.Mono doublets"))
cell_type <- sub(" ", ".", cell_type, fixed = TRUE)
retained_cells <- colnames(cbmc)[keep]
cbmc <- subset(cbmc, cells = retained_cells)
cbmc$seurat_clusters <- factor(cell_type[keep])
Seurat::Idents(cbmc) <- cbmc$seurat_clusters

rna_counts <- Seurat::GetAssayData(cbmc, assay = "RNA", slot = "counts")
adt_counts <- Seurat::GetAssayData(cbmc, assay = "ADT", slot = "counts")
if (any(rna_counts@x < 0) || any(rna_counts@x != round(rna_counts@x))) {
  stop("CBMC RNA assay is not a non-negative integer count matrix", call. = FALSE)
}
if (any(adt_counts@x < 0) || any(adt_counts@x != round(adt_counts@x))) {
  stop("CBMC ADT assay is not a non-negative integer count matrix", call. = FALSE)
}

# The packaged Seurat v3 object can expose counts in its data slot under newer
# Seurat versions. Recompute the author-specified CLR transform from counts.
cbmc <- Seurat::NormalizeData(
  cbmc,
  assay = "ADT",
  normalization.method = "CLR",
  margin = 1,
  verbose = FALSE
)
adt_clr <- Seurat::GetAssayData(cbmc, assay = "ADT", slot = "data")

cell_clusters <- data.frame(
  dataset_id = dataset_id,
  barcode = colnames(cbmc),
  cluster_id = as.character(cbmc$seurat_clusters),
  stringsAsFactors = FALSE
)
write.table(
  cell_clusters,
  file = file.path(output_dir, "cell_clusters.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

cluster_levels <- levels(cbmc$seurat_clusters)
adt_means <- do.call(rbind, lapply(cluster_levels, function(cluster_id) {
  cells <- colnames(cbmc)[cbmc$seurat_clusters == cluster_id]
  data.frame(
    dataset_id = dataset_id,
    cluster_id = cluster_id,
    adt_feature = rownames(adt_clr),
    mean_clr = Matrix::rowMeans(adt_clr[, cells, drop = FALSE]),
    n_cells = length(cells),
    stringsAsFactors = FALSE
  )
}))
adt_means$z_across_clusters <- ave(
  adt_means$mean_clr,
  adt_means$adt_feature,
  FUN = function(values) as.numeric(scale(values))
)
if (anyNA(adt_means$z_across_clusters)) {
  stop("CBMC has an undefined across-cluster ADT z score", call. = FALSE)
}
write.table(
  adt_means,
  file = file.path(output_dir, "adt_cluster_means.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

matrix_path <- file.path(output_dir, "rna_counts.mtx")
Matrix::writeMM(rna_counts, matrix_path)
write_gzip_table <- function(value, path) {
  connection <- gzfile(path, "wt")
  on.exit(close(connection), add = TRUE)
  write.table(
    value,
    connection,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE,
    col.names = FALSE
  )
}
write_gzip_table(
  data.frame(
    gene_id = rownames(rna_counts),
    gene_symbol = rownames(rna_counts),
    feature_type = "Gene Expression"
  ),
  file.path(output_dir, "rna_features.tsv.gz")
)
write_gzip_table(colnames(rna_counts), file.path(output_dir, "barcodes.tsv.gz"))
if (system2("gzip", c("-f", matrix_path)) != 0L) {
  stop("gzip failed for CBMC RNA Matrix Market export", call. = FALSE)
}

metadata <- data.frame(
  key = c(
    "dataset_id", "input_rda", "started_at", "finished_at", "r_version",
    "seurat_version", "matrix_version", "raw_cells", "author_qc_cells",
    "n_rna_features", "n_adt_features", "n_cell_types", "label_discrepancy",
    "adt_normalization", "protein_used_as_algorithm_input"
  ),
  value = c(
    dataset_id,
    input_rda,
    format(started, tz = "UTC", usetz = TRUE),
    format(Sys.time(), tz = "UTC", usetz = TRUE),
    R.version.string,
    as.character(utils::packageVersion("Seurat")),
    as.character(utils::packageVersion("Matrix")),
    length(keep),
    sum(keep),
    nrow(rna_counts),
    nrow(adt_counts),
    length(cluster_levels),
    "paper display name 3kCBMCs; public SeuratData object retains 7713 cells",
    "Seurat CLR margin=1 recomputed from retained-cell raw ADT counts",
    "false"
  ),
  stringsAsFactors = FALSE
)
write.table(
  metadata,
  file = file.path(output_dir, "run_metadata.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

message(
  "Prepared ", dataset_id, ": ", ncol(rna_counts), " cells, ",
  length(cluster_levels), " cell types, ", nrow(adt_counts), " ADTs"
)
