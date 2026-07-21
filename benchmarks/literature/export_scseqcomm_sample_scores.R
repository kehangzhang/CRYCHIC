#!/usr/bin/env Rscript

# Export the native per-sample S_inter matrix retained inside scSeqComm 2.0.0.
# The published adapter persisted only differential_comm.rds; component-swap
# arm D needs the sample distributions that generated that object.

suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(reticulate))
suppressPackageStartupMessages(library(scSeqComm))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 8L) {
  stop(paste(
    "usage: export_scseqcomm_sample_scores.R INPUT_H5AD METADATA RESOURCE",
    "OUTPUT TARGET REFERENCE CORES MIN_CELLS"
  ))
}

input_h5ad <- normalizePath(args[[1]], mustWork = TRUE)
metadata_path <- normalizePath(args[[2]], mustWork = TRUE)
resource_path <- normalizePath(args[[3]], mustWork = TRUE)
output_path <- normalizePath(args[[4]], mustWork = FALSE)
target <- args[[5]]
reference <- args[[6]]
cores <- as.integer(args[[7]])
min_cells <- as.integer(args[[8]])

if (as.character(packageVersion("scSeqComm")) != "2.0.0") {
  stop("scSeqComm 2.0.0 is required")
}
if (is.na(cores) || cores < 1L || is.na(min_cells) || min_cells < 1L) {
  stop("CORES and MIN_CELLS must be positive integers")
}
if (target == reference) {
  stop("TARGET and REFERENCE must differ")
}

metadata <- read.delim(
  metadata_path,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
required_metadata <- c("Cell_ID", "Cluster_ID", "Condition_ID", "Sample_ID")
if (!all(required_metadata %in% names(metadata))) {
  stop("metadata is missing required scSeqComm columns")
}
if (!setequal(unique(metadata$Condition_ID), c(target, reference))) {
  stop("metadata conditions disagree with TARGET/REFERENCE")
}
sample_design <- unique(metadata[c("Sample_ID", "Condition_ID")])
if (anyDuplicated(sample_design$Sample_ID)) {
  stop("one Sample_ID maps to multiple conditions")
}

anndata <- import("anndata", convert = FALSE)
adata <- anndata$read_h5ad(input_h5ad)
cell_gene <- py_to_r(adata$X$tocsc())
gene_expr <- as(Matrix::t(cell_gene), "dgCMatrix")
rownames(gene_expr) <- py_to_r(adata$var_names$to_list())
colnames(gene_expr) <- py_to_r(adata$obs_names$to_list())
rm(cell_gene, adata)
invisible(gc())

cell_index <- match(metadata$Cell_ID, colnames(gene_expr))
if (anyNA(cell_index)) {
  stop("analysis metadata contains cells absent from the h5ad")
}
gene_expr <- gene_expr[, cell_index, drop = FALSE]
if (!identical(colnames(gene_expr), metadata$Cell_ID)) {
  stop("h5ad cell order does not match analysis metadata")
}

resource <- read.delim(
  resource_path,
  check.names = FALSE,
  stringsAsFactors = FALSE
)
required_resource <- c("harmonized_interaction_id", "ligand", "receptor")
if (!all(required_resource %in% names(resource))) {
  stop("resource is missing harmonized interaction identifiers")
}
if (nrow(resource) != 2293L || anyDuplicated(resource[c("ligand", "receptor")])) {
  stop("ConnectomeDB2020 resource mismatch")
}
lr_db <- resource[c("ligand", "receptor")]

condition_scores <- list()
for (condition_name in c(reference, target)) {
  condition_metadata <- metadata[
    metadata$Condition_ID == condition_name,
    ,
    drop = FALSE
  ]
  sample_order <- unique(condition_metadata$Sample_ID)
  message(
    "Exporting ", condition_name, " S_inter for ", length(sample_order),
    " samples"
  )
  result <- suppressMessages(scSeqComm_analyze(
    gene_expr = gene_expr,
    cell_metadata = condition_metadata,
    LR_pairs_DB = lr_db,
    intra_signaling = FALSE,
    inter_scores = "scSeqComm",
    sampling = "by_sample",
    N_cores = cores,
    backend = "doMC",
    bigmatrix = FALSE,
    min_cells = min_cells
  ))
  distribution <- result$distribution_results
  score_columns <- grep("^S_inter_", names(distribution), value = TRUE)
  if (length(score_columns) != length(sample_order)) {
    stop("native S_inter distribution does not match sample order")
  }
  names(distribution)[match(score_columns, names(distribution))] <- sample_order
  identity <- c(
    "ligand", "receptor", "LR_pair", "cluster_L", "cluster_R", "interaction"
  )
  keep <- c(identity, sample_order)
  if (!all(keep %in% names(distribution))) {
    stop("native S_inter distribution is missing identity or score columns")
  }
  condition_scores[[condition_name]] <- distribution[keep]
  rm(result, distribution)
  invisible(gc())
}

identity <- c(
  "ligand", "receptor", "LR_pair", "cluster_L", "cluster_R", "interaction"
)
wide <- merge(
  condition_scores[[reference]],
  condition_scores[[target]],
  by = identity,
  all = TRUE,
  sort = FALSE
)
wide <- merge(
  wide,
  resource[c("harmonized_interaction_id", "ligand", "receptor")],
  by = c("ligand", "receptor"),
  all.x = TRUE,
  sort = FALSE
)
names(wide)[names(wide) == "harmonized_interaction_id"] <- "interaction_id"
names(wide)[names(wide) == "cluster_L"] <- "sender"
names(wide)[names(wide) == "cluster_R"] <- "receiver"
edge_key <- c("sender", "receiver", "interaction_id")
if (anyNA(wide$interaction_id) || anyDuplicated(wide[edge_key])) {
  stop("exported native S_inter rows do not map one-to-one to the resource")
}

sample_columns <- sample_design$Sample_ID
wide <- wide[c(
  "sender", "receiver", "interaction_id", "ligand", "receptor", "LR_pair",
  "interaction", sample_columns
)]
wide <- wide[do.call(order, c(wide[edge_key], list(method = "radix"))), ]

output_parent <- dirname(output_path)
if (!dir.exists(output_parent)) {
  dir.create(output_parent, recursive = TRUE)
}
connection <- gzfile(output_path, "wt")
write.table(
  wide,
  connection,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = "NA"
)
close(connection)
message("Wrote ", nrow(wide), " directed LR rows to ", output_path)
