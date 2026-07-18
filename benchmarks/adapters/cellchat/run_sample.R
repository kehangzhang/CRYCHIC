#!/usr/bin/env Rscript

Sys.setenv(R_FUTURE_PLAN = "sequential")
options(future.plan = "sequential")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 10L) {
  stop(
    paste(
      "usage: run_sample.R INPUT_DIR CELLCHAT_DB_RDS SOURCE_IDS OUTPUT_CSV",
      "MIN_CELLS NBOOT SEED TRIMEAN_TRIM POPULATION_SIZE THREADS"
    ),
    call. = FALSE
  )
}
suppressPackageStartupMessages(library(CellChat))
suppressPackageStartupMessages(library(Matrix))

main <- function(arguments) {
  input_dir <- arguments[[1L]]
  database_path <- arguments[[2L]]
  source_ids_path <- arguments[[3L]]
  output_path <- arguments[[4L]]
  output_tmp <- paste0(output_path, ".tmp-", Sys.getpid())
  min_cells <- as.integer(arguments[[5L]])
  nboot <- as.integer(arguments[[6L]])
  seed <- as.integer(arguments[[7L]])
  trim <- as.numeric(arguments[[8L]])
  population_size <- as.logical(arguments[[9L]])
  threads <- as.integer(arguments[[10L]])
  if (is.na(threads) || threads < 1L) {
    stop("THREADS must be a positive integer", call. = FALSE)
  }

  future::plan(future::sequential)
  run_metadata_path <- Sys.getenv("CRYCHIC_CELLCHAT_RUN_METADATA", unset = "")
  write_run_metadata <- function(stage, destination = run_metadata_path) {
    if (!nzchar(run_metadata_path)) return(invisible(NULL))
    payload <- list(
      schema_version = "crychic-cellchat-r-runtime-v1",
      stage = stage,
      rng_kind = RNGkind(),
      future_plan = class(future::plan("next"))[[1L]],
      future_workers = future::nbrOfWorkers(),
      future_globals_max_size_bytes = getOption("future.globals.maxSize"),
      future_rng_on_misuse = getOption("future.rng.onMisuse")
    )
    json <- jsonlite::toJSON(payload, auto_unbox = TRUE, pretty = TRUE)
    temporary <- paste0(destination, ".tmp-", Sys.getpid())
    on.exit(unlink(temporary, force = TRUE), add = TRUE)
    writeLines(json, temporary, useBytes = TRUE)
    if (!file.rename(temporary, destination)) {
      stop("failed to atomically publish CellChat runtime metadata", call. = FALSE)
    }
  }
  on.exit({
    future::plan(future::sequential)
    write_run_metadata("exit")
  }, add = TRUE)
  on.exit(unlink(output_tmp, force = TRUE), add = TRUE)
  options(
    future.globals.maxSize = 8 * 1024^3,
    future.rng.onMisuse = "error"
  )
  if (threads > 1L) {
    future::plan(future::multisession, workers = threads)
  }
  if (nzchar(run_metadata_path)) {
    write_run_metadata("configured", paste0(run_metadata_path, ".configured"))
  }

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
  result <- tryCatch(
    subsetCommunication(object, thresh = 1),
    error = function(error) {
      message <- conditionMessage(error)
      valid_empty <- grepl(
        "No significant signaling interactions are inferred based on the input!",
        message,
        fixed = TRUE
      )
      if (valid_empty) return(NULL)
      stop(error)
    }
  )
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
  utils::write.csv(result, output_tmp, row.names = FALSE, quote = TRUE)
  if (!file.rename(output_tmp, output_path)) {
    stop("failed to atomically publish CellChat output", call. = FALSE)
  }
}

main(args)
