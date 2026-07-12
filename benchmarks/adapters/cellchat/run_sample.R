#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 9L) {
  stop(
    paste(
      "usage: run_sample.R INPUT_DIR CELLCHAT_DB_RDS SOURCE_IDS OUTPUT_CSV",
      "MIN_CELLS NBOOT SEED TRIMEAN_TRIM POPULATION_SIZE"
    ),
    call. = FALSE
  )
}
suppressPackageStartupMessages(library(CellChat))
suppressPackageStartupMessages(library(Matrix))

input_dir <- args[[1L]]
database_path <- args[[2L]]
source_ids_path <- args[[3L]]
output_path <- args[[4L]]
min_cells <- as.integer(args[[5L]])
nboot <- as.integer(args[[6L]])
seed <- as.integer(args[[7L]])
trim <- as.numeric(args[[8L]])
population_size <- as.logical(args[[9L]])

matrix <- Matrix::readMM(file.path(input_dir, "matrix.mtx"))
genes <- readLines(file.path(input_dir, "genes.txt"), warn = FALSE)
cells <- readLines(file.path(input_dir, "cells.txt"), warn = FALSE)
metadata <- utils::read.delim(
  file.path(input_dir, "metadata.tsv"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
if (nrow(matrix) != length(genes) || ncol(matrix) != length(cells)) {
  stop("matrix dimensions do not match gene/cell identifiers", call. = FALSE)
}
if (!identical(as.character(metadata$cell), cells)) {
  stop("metadata cell order does not match matrix columns", call. = FALSE)
}
rownames(matrix) <- genes
colnames(matrix) <- cells
rownames(metadata) <- metadata$cell

database <- readRDS(database_path)
source_ids <- readLines(source_ids_path, warn = FALSE)
if (length(source_ids) > 0L) {
  keep <- rownames(database$interaction) %in% source_ids |
    database$interaction$interaction_name %in% source_ids
  database$interaction <- database$interaction[keep, , drop = FALSE]
}
if (nrow(database$interaction) == 0L) {
  stop("selected CellChat resource has no interactions", call. = FALSE)
}

object <- createCellChat(
  object = matrix,
  meta = metadata,
  group.by = "cell_type",
  datatype = "RNA",
  do.sparse = TRUE
)
object@DB <- database
object <- subsetData(object)
object <- identifyOverExpressedGenes(
  object,
  min.cells = min_cells,
  thresh.pc = 0,
  thresh.fc = 0,
  thresh.p = 1
)
object <- identifyOverExpressedInteractions(object)
object <- computeCommunProb(
  object,
  type = "triMean",
  trim = trim,
  raw.use = TRUE,
  population.size = population_size,
  distance.use = FALSE,
  nboot = nboot,
  seed.use = seed
)
object <- filterCommunication(object, min.cells = min_cells)
result <- subsetCommunication(object, thresh = 1)
if (is.null(result) || nrow(result) == 0L) {
  result <- data.frame(
    source = character(),
    target = character(),
    ligand = character(),
    receptor = character(),
    interaction_name = character(),
    prob = numeric(),
    pval = numeric(),
    stringsAsFactors = FALSE
  )
}
utils::write.csv(result, output_path, row.names = FALSE, quote = TRUE)
