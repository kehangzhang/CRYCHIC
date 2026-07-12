#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop(
    "usage: prepare_nichenet_prior.R INPUT_RDS OUTPUT_TSV_GZ TOP_N",
    call. = FALSE
  )
}

input_path <- args[[1L]]
output_path <- args[[2L]]
top_n <- as.integer(args[[3L]])
if (is.na(top_n) || top_n < 1L) {
  stop("TOP_N must be a positive integer", call. = FALSE)
}
if (!requireNamespace("data.table", quietly = TRUE)) {
  stop("R package 'data.table' is required", call. = FALSE)
}

prior <- readRDS(input_path)
if (!is.matrix(prior) || is.null(rownames(prior)) || is.null(colnames(prior))) {
  stop("input must be a named target-by-ligand matrix", call. = FALSE)
}

top_n <- min(top_n, nrow(prior))
parts <- lapply(seq_len(ncol(prior)), function(column_index) {
  weights <- prior[, column_index]
  keep <- order(weights, decreasing = TRUE, na.last = NA)[seq_len(top_n)]
  data.table::data.table(
    ligand = colnames(prior)[[column_index]],
    target = rownames(prior)[keep],
    weight = as.numeric(weights[keep]),
    rank = seq_along(keep)
  )
})

result <- data.table::rbindlist(parts)
result <- result[weight > 0]
data.table::setorder(result, ligand, rank, target)
data.table::fwrite(result, output_path, sep = "\t", compress = "gzip")

cat(
  sprintf(
    "exported %d positive links for %d ligands (top_n=%d)\n",
    nrow(result),
    data.table::uniqueN(result$ligand),
    top_n
  )
)
