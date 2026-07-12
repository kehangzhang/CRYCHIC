#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop("usage: summarize_reference.R INPUT_RDS CONTEXT OUTPUT_DIR", call. = FALSE)
}
if (!requireNamespace("CellChat", quietly = TRUE)) {
  stop("R package 'CellChat' is required", call. = FALSE)
}

object <- readRDS(args[[1L]])
context <- args[[2L]]
output <- args[[3L]]
dir.create(output, recursive = TRUE, showWarnings = FALSE)

cell_counts <- as.data.frame(table(object@idents), stringsAsFactors = FALSE)
colnames(cell_counts) <- c("cell_type", "n_cells")
cell_counts$context <- context
cell_counts$cell_fraction <- cell_counts$n_cells / sum(cell_counts$n_cells)
cell_counts <- cell_counts[, c("context", "cell_type", "n_cells", "cell_fraction")]

probability <- object@netP$prob
if (length(dim(probability)) != 3L) {
  stop("CellChat object has no three-dimensional netP probability array", call. = FALSE)
}
edges <- as.data.frame(as.table(probability), stringsAsFactors = FALSE)
colnames(edges) <- c("sender", "receiver", "pathway", "strength")
edges <- edges[is.finite(edges$strength) & edges$strength > 0, , drop = FALSE]
edges$context <- context
edges <- edges[, c("context", "pathway", "sender", "receiver", "strength")]
edges <- edges[order(edges$pathway, -edges$strength, edges$sender, edges$receiver), ]

pathways <- aggregate(
  edges$strength,
  by = list(context = edges$context, pathway = edges$pathway),
  FUN = function(values) c(total_strength = sum(values), nonzero_edges = length(values))
)
pathways <- data.frame(
  context = pathways$context,
  pathway = pathways$pathway,
  total_strength = pathways$x[, "total_strength"],
  nonzero_edges = as.integer(pathways$x[, "nonzero_edges"]),
  stringsAsFactors = FALSE
)
pathways <- pathways[order(-pathways$total_strength, pathways$pathway), ]

utils::write.csv(cell_counts, file.path(output, "cell_composition.csv"), row.names = FALSE)
utils::write.csv(pathways, file.path(output, "pathway_summary.csv"), row.names = FALSE)
utils::write.csv(edges, file.path(output, "pathway_edges.csv"), row.names = FALSE)

cat(
  sprintf(
    "summarized context=%s cells=%d cell_types=%d pathways=%d edges=%d\n",
    context,
    sum(cell_counts$n_cells),
    nrow(cell_counts),
    nrow(pathways),
    nrow(edges)
  )
)
