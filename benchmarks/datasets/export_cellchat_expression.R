#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop("usage: export_cellchat_expression.R INPUT_RDS OUTPUT_DIR", call. = FALSE)
}
if (!requireNamespace("CellChat", quietly = TRUE)) {
  stop("R package 'CellChat' is required to deserialize the object", call. = FALSE)
}
if (!requireNamespace("Matrix", quietly = TRUE)) {
  stop("R package 'Matrix' is required", call. = FALSE)
}

object <- readRDS(args[[1L]])
expression <- object@data
if (is.null(expression) || length(expression) == 0L) {
  stop("CellChat object has no normalized expression in @data", call. = FALSE)
}
metadata <- object@meta
if (nrow(metadata) != ncol(expression)) {
  stop("CellChat metadata and expression columns are not aligned", call. = FALSE)
}
if (!identical(rownames(metadata), colnames(expression))) {
  stop("CellChat metadata row names differ from expression column names", call. = FALSE)
}

output <- args[[2L]]
dir.create(output, recursive = TRUE, showWarnings = FALSE)
Matrix::writeMM(expression, file.path(output, "expression.mtx"))
writeLines(rownames(expression), file.path(output, "genes.txt"), useBytes = TRUE)
metadata$cell_id <- rownames(metadata)
utils::write.table(
  metadata,
  file.path(output, "metadata.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)

cat(
  sprintf(
    "exported %d genes x %d cells with %d metadata fields\n",
    nrow(expression),
    ncol(expression),
    ncol(metadata)
  )
)
