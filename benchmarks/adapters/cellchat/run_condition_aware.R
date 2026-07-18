#!/usr/bin/env Rscript

Sys.setenv(R_FUTURE_PLAN = "sequential")
options(future.plan = "sequential")

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 12L) {
  stop(
    paste(
      "usage: run_condition_aware.R INPUT_H5AD METADATA_TSV SUPPORT_TSV",
      "RESOURCE_TSV CELLCHAT_DB_RDS OUTPUT_DIR DATASET_ID TARGET REFERENCE",
      "THREADS MIN_CELLS EMIT_RECEPTOR_SENSITIVITY"
    ),
    call. = FALSE
  )
}

suppressPackageStartupMessages(library(CellChat))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(reticulate))

no_interactions_error <- function(error) {
  message <- conditionMessage(error)
  any(vapply(
    c(
      "No significant signaling interactions are inferred based on the input!",
      "No significant signaling interactions are inferred in all conditions!"
    ),
    grepl,
    logical(1L),
    x = message,
    fixed = TRUE
  ))
}

parse_boolean <- function(value, field) {
  text <- tolower(as.character(value))
  if (length(text) != 1L || !text %in% c("true", "false")) {
    stop(paste(field, "must be TRUE or FALSE"), call. = FALSE)
  }
  identical(text, "true")
}

parse_boolean_column <- function(values, field) {
  text <- tolower(as.character(values))
  if (any(!text %in% c("true", "false"))) {
    stop(paste(field, "must contain only TRUE or FALSE"), call. = FALSE)
  }
  text == "true"
}

write_gzip_table <- function(table, path) {
  connection <- gzfile(path, open = "wt")
  on.exit(close(connection), add = TRUE)
  utils::write.table(
    table,
    connection,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE,
    na = ""
  )
}

empty_mapped_network <- function() {
  data.frame(
    source = character(),
    target = character(),
    interaction_name = character(),
    pathway_name = character(),
    ligand = character(),
    receptor = character(),
    prob = numeric(),
    pval = numeric(),
    datasets = character(),
    ligand.pvalues = numeric(),
    ligand.logFC = numeric(),
    ligand.pct.1 = numeric(),
    ligand.pct.2 = numeric(),
    receptor.pvalues = numeric(),
    receptor.logFC = numeric(),
    receptor.pct.1 = numeric(),
    receptor.pct.2 = numeric(),
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
}

select_differential_network <- function(
    object,
    mapped_network,
    dataset,
    ligand_logfc,
    receptor_logfc = NULL) {
  if (nrow(mapped_network) == 0L) {
    return(mapped_network)
  }
  tryCatch(
    subsetCommunication(
      object,
      net = mapped_network,
      datasets = dataset,
      ligand.logFC = ligand_logfc,
      receptor.logFC = receptor_logfc
    ),
    error = function(error) {
      if (no_interactions_error(error)) {
        return(mapped_network[0L, , drop = FALSE])
      }
      stop(error)
    }
  )
}

annotate_selected <- function(table, condition, resource) {
  if (!all(c(
    "source", "target", "interaction_name", "ligand", "receptor", "datasets"
  ) %in% names(table))) {
    stop("selected CellChat table schema mismatch", call. = FALSE)
  }
  interaction_index <- match(
    as.character(table$interaction_name),
    resource$cellchat_source_interaction_id
  )
  if (anyNA(interaction_index)) {
    stop("CellChat selected an interaction outside ConnectomeDB2020", call. = FALSE)
  }
  if (nrow(table) > 0L && any(as.character(table$datasets) != condition)) {
    stop("CellChat selected rows from the wrong condition", call. = FALSE)
  }
  table$condition <- rep(condition, nrow(table))
  table$harmonized_interaction_id <-
    resource$harmonized_interaction_id[interaction_index]
  table$resource_ligand <- resource$ligand[interaction_index]
  table$resource_receptor <- resource$receptor[interaction_index]
  if (nrow(table) > 0L && (
    any(as.character(table$ligand) != table$resource_ligand) ||
      any(as.character(table$receptor) != table$resource_receptor)
  )) {
    stop("CellChat ligand/receptor symbols do not match the resource", call. = FALSE)
  }
  leading <- c(
    "condition", "source", "target", "harmonized_interaction_id",
    "interaction_name", "ligand", "receptor"
  )
  table[c(leading, setdiff(names(table), leading))]
}

combine_selected <- function(target_table, reference_table) {
  all_columns <- union(names(target_table), names(reference_table))
  add_missing <- function(table) {
    for (column in setdiff(all_columns, names(table))) {
      table[[column]] <- NA
    }
    table[all_columns]
  }
  combined <- rbind(add_missing(target_table), add_missing(reference_table))
  keys <- c("condition", "source", "target", "harmonized_interaction_id")
  if (nrow(combined) > 0L && anyDuplicated(combined[keys])) {
    stop("CellChat selected duplicate directed interaction rows", call. = FALSE)
  }
  rownames(combined) <- NULL
  combined
}

make_pair_axis <- function(cell_types, conditions) {
  pairs <- do.call(
    rbind,
    lapply(seq_along(cell_types), function(index) {
      data.frame(
        sender = cell_types[[index]],
        receiver = cell_types[index:length(cell_types)],
        stringsAsFactors = FALSE
      )
    })
  )
  do.call(
    rbind,
    lapply(conditions, function(condition) {
      data.frame(
        condition = condition,
        sender = pairs$sender,
        receiver = pairs$receiver,
        stringsAsFactors = FALSE
      )
    })
  )
}

make_rankings <- function(
    selected,
    support,
    dataset_id,
    conditions,
    ranking_semantics,
    selection_mode) {
  if (!selection_mode %in% c("ligand_only", "ligand_receptor_concordant")) {
    stop("unknown differential selection mode", call. = FALSE)
  }
  cell_types <- sort(unique(as.character(support$cell_type)), method = "radix")
  rankings <- make_pair_axis(cell_types, conditions)

  if (nrow(selected) > 0L) {
    selected$sender_unordered <- ifelse(
      as.character(selected$source) <= as.character(selected$target),
      as.character(selected$source),
      as.character(selected$target)
    )
    selected$receiver_unordered <- ifelse(
      as.character(selected$source) <= as.character(selected$target),
      as.character(selected$target),
      as.character(selected$source)
    )
  }
  rankings$ranked_strength <- vapply(
    seq_len(nrow(rankings)),
    function(index) {
      if (nrow(selected) == 0L) {
        return(0L)
      }
      sum(
        selected$condition == rankings$condition[[index]] &
          selected$sender_unordered == rankings$sender[[index]] &
          selected$receiver_unordered == rankings$receiver[[index]]
      )
    },
    integer(1L)
  )
  support_key <- paste(support$condition, support$cell_type, sep = "\r")
  if (anyDuplicated(support_key)) {
    stop("condition/cell-type support must be unique", call. = FALSE)
  }
  ranking_sender_key <- paste(
    rankings$condition, rankings$sender, sep = "\r"
  )
  ranking_receiver_key <- paste(
    rankings$condition, rankings$receiver, sep = "\r"
  )
  sender_index <- match(ranking_sender_key, support_key)
  receiver_index <- match(ranking_receiver_key, support_key)
  if (anyNA(sender_index) || anyNA(receiver_index)) {
    stop("ranking axis is absent from condition support", call. = FALSE)
  }
  sender_present <- support$condition_present[sender_index]
  receiver_present <- support$condition_present[receiver_index]
  sender_network_supported <- support$native_network_supported[sender_index]
  receiver_network_supported <- support$native_network_supported[receiver_index]
  sender_de <- support$de_comparable[sender_index]
  receiver_de <- support$de_comparable[receiver_index]
  if (selection_mode == "ligand_only") {
    pair_eligible <- sender_present & receiver_present & (sender_de | receiver_de)
  } else {
    pair_eligible <- sender_present & receiver_present & sender_de & receiver_de
  }
  structural_zero <- (
    pair_eligible & (!sender_network_supported | !receiver_network_supported)
  )
  if (any(rankings$ranked_strength[structural_zero] != 0L)) {
    stop("native low-support structural zero has selected rows", call. = FALSE)
  }
  rankings$ranked_strength[!pair_eligible] <- NA_integer_
  rankings$dataset <- dataset_id
  rankings$method <- "cellchat_condition_aware"
  rankings$method_version <- "2.1.2"
  rankings$resource <- "ConnectomeDB2020_Hou_2020_human"
  rankings$ranking_semantics <- ranking_semantics
  rankings$status <- ifelse(pair_eligible, "observed", "not_estimable")
  rankings$reason_code <- ""
  rankings$reason_code[structural_zero] <-
    "native_structural_zero_low_cell_support"
  if (selection_mode == "ligand_only") {
    partial <- pair_eligible & xor(sender_de, receiver_de) & !structural_zero
    rankings$reason_code[partial] <- "partial_directional_de_support"
  }
  absent <- !pair_eligible & (!sender_present | !receiver_present)
  rankings$reason_code[absent] <- "cell_type_absent_in_condition"
  no_de <- !pair_eligible & !absent
  rankings$reason_code[no_de] <- if (selection_mode == "ligand_only") {
    "no_de_comparable_ligand_direction"
  } else {
    "ligand_or_receptor_de_not_comparable"
  }
  rankings[c(
    "dataset", "method", "method_version", "resource", "ranking_semantics",
    "condition", "sender", "receiver", "ranked_strength", "status",
    "reason_code"
  )]
}

main <- function(arguments) {
  input_h5ad <- arguments[[1L]]
  metadata_path <- arguments[[2L]]
  support_path <- arguments[[3L]]
  resource_path <- arguments[[4L]]
  database_path <- arguments[[5L]]
  output_dir <- arguments[[6L]]
  dataset_id <- arguments[[7L]]
  target <- arguments[[8L]]
  reference <- arguments[[9L]]
  threads <- as.integer(arguments[[10L]])
  min_cells <- as.integer(arguments[[11L]])
  emit_receptor_sensitivity <- parse_boolean(
    arguments[[12L]],
    "EMIT_RECEPTOR_SENSITIVITY"
  )
  if (target == reference) {
    stop("TARGET and REFERENCE must differ", call. = FALSE)
  }
  if (is.na(threads) || threads < 1L || is.na(min_cells) || min_cells < 1L) {
    stop("THREADS and MIN_CELLS must be positive integers", call. = FALSE)
  }
  if (as.character(packageVersion("CellChat")) != "2.1.2") {
    stop("CellChat 2.1.2 is required", call. = FALSE)
  }

  future::plan(future::sequential)
  on.exit(future::plan(future::sequential), add = TRUE)
  options(
    future.globals.maxSize = 8 * 1024^3,
    future.rng.onMisuse = "error"
  )
  if (threads > 1L) {
    future::plan(future::multisession, workers = threads)
  }

  output_dir <- normalizePath(output_dir, mustWork = TRUE)
  staging_dir <- file.path(
    output_dir,
    paste0(".condition-aware.tmp-", Sys.getpid())
  )
  if (dir.exists(staging_dir) || file.exists(staging_dir)) {
    unlink(staging_dir, recursive = TRUE, force = TRUE)
  }
  if (!dir.create(staging_dir, recursive = FALSE, showWarnings = FALSE)) {
    stop("failed to create the CellChat staging directory", call. = FALSE)
  }
  on.exit(unlink(staging_dir, recursive = TRUE, force = TRUE), add = TRUE)
  published <- character()
  publish_complete <- FALSE
  on.exit({
    if (!publish_complete && length(published) > 0L) {
      unlink(file.path(output_dir, published), force = TRUE)
    }
  }, add = TRUE)

  metadata <- utils::read.delim(
    metadata_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  if (!identical(names(metadata), c("cell", "cell_type", "condition"))) {
    stop("condition metadata schema mismatch", call. = FALSE)
  }
  support <- utils::read.delim(
    support_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  support_columns <- c(
    "condition", "cell_type", "n_cells", "condition_present",
    "native_network_supported", "de_comparable", "reason_code"
  )
  if (!identical(names(support), support_columns)) {
    stop("condition support schema mismatch", call. = FALSE)
  }
  support$condition_present <- parse_boolean_column(
    support$condition_present,
    "condition_present"
  )
  support$native_network_supported <- parse_boolean_column(
    support$native_network_supported,
    "native_network_supported"
  )
  support$de_comparable <- parse_boolean_column(
    support$de_comparable,
    "de_comparable"
  )
  support$reason_code[is.na(support$reason_code)] <- ""
  conditions <- c(reference, target)
  if (!setequal(unique(as.character(metadata$condition)), conditions)) {
    stop("analysis metadata condition axis mismatch", call. = FALSE)
  }
  if (anyDuplicated(metadata$cell)) {
    stop("analysis metadata cell identifiers must be unique", call. = FALSE)
  }
  all_cell_types <- sort(unique(as.character(support$cell_type)), method = "radix")
  de_comparable_cell_types <- sort(unique(as.character(
    support$cell_type[support$de_comparable]
  )), method = "radix")
  if (!setequal(unique(as.character(metadata$cell_type)), all_cell_types)) {
    stop("analysis metadata does not match the source cell-type axis", call. = FALSE)
  }

  resource <- utils::read.delim(
    resource_path,
    stringsAsFactors = FALSE,
    check.names = FALSE
  )
  resource_columns <- c(
    "harmonized_interaction_id", "ligand", "receptor",
    "cellchat_source_interaction_id", "cellchat_covered"
  )
  if (nrow(resource) != 2293L || !all(resource_columns %in% names(resource))) {
    stop("ConnectomeDB2020 resource mismatch", call. = FALSE)
  }
  if (
    anyDuplicated(resource$harmonized_interaction_id) ||
      anyDuplicated(resource$cellchat_source_interaction_id) ||
      anyDuplicated(resource[c("ligand", "receptor")])
  ) {
    stop("ConnectomeDB2020 identifiers must be unique", call. = FALSE)
  }
  if (any(tolower(as.character(resource$cellchat_covered)) != "true")) {
    stop("all ConnectomeDB2020 rows must be covered by CellChat", call. = FALSE)
  }

  database <- readRDS(database_path)
  if (!is.list(database) || !all(c(
    "interaction", "complex", "cofactor", "geneInfo"
  ) %in% names(database))) {
    stop("CellChat database object schema mismatch", call. = FALSE)
  }
  database_ids <- rownames(database$interaction)
  if (is.null(database_ids) || any(!nzchar(database_ids))) {
    database_ids <- as.character(database$interaction$interaction_name)
  }
  database_index <- match(resource$cellchat_source_interaction_id, database_ids)
  if (anyNA(database_index)) {
    stop("CellChat database does not cover the frozen resource", call. = FALSE)
  }
  database$interaction <- database$interaction[database_index, , drop = FALSE]
  rownames(database$interaction) <- resource$cellchat_source_interaction_id
  if (
    !identical(
      as.character(database$interaction$interaction_name),
      resource$cellchat_source_interaction_id
    ) ||
      any(as.character(database$interaction$ligand) != resource$ligand) ||
      any(as.character(database$interaction$receptor) != resource$receptor)
  ) {
    stop("CellChat database rows do not match ConnectomeDB2020", call. = FALSE)
  }

  anndata <- import("anndata", convert = FALSE)
  numpy <- import("numpy", convert = FALSE)
  adata <- anndata$read_h5ad(input_h5ad)
  if (!py_has_attr(adata$X, "tocsc")) {
    stop("CellChat input X must be a scipy sparse matrix", call. = FALSE)
  }
  resource_genes <- sort(
    unique(c(as.character(resource$ligand), as.character(resource$receptor))),
    method = "radix"
  )
  cell_index <- as.integer(py_to_r(
    adata$obs_names$get_indexer(r_to_py(as.character(metadata$cell)))
  ))
  gene_index <- as.integer(py_to_r(
    adata$var_names$get_indexer(r_to_py(resource_genes))
  ))
  if (any(cell_index < 0L)) {
    stop("analysis metadata contains cells absent from the h5ad", call. = FALSE)
  }
  genes_present <- gene_index >= 0L
  if (sum(genes_present) < 3L) {
    stop("fewer than three ConnectomeDB2020 genes are present", call. = FALSE)
  }
  matrix_index <- numpy$ix_(
    r_to_py(cell_index),
    r_to_py(gene_index[genes_present])
  )
  cell_gene_python <- py_get_item(adata$X, matrix_index)
  cell_gene <- py_to_r(cell_gene_python$tocsc())
  gene_expr <- as(Matrix::t(cell_gene), "dgCMatrix")
  rownames(gene_expr) <- resource_genes[genes_present]
  colnames(gene_expr) <- as.character(metadata$cell)
  rm(cell_gene, cell_gene_python, adata)
  invisible(gc())
  if (any(!is.finite(gene_expr@x)) || any(gene_expr@x < 0)) {
    stop("CellChat expression matrix must be finite and non-negative", call. = FALSE)
  }
  if (!identical(colnames(gene_expr), as.character(metadata$cell))) {
    stop("h5ad cell order does not match analysis metadata", call. = FALSE)
  }

  object_list <- vector("list", length(conditions))
  names(object_list) <- conditions
  for (condition in conditions) {
    condition_index <- which(metadata$condition == condition)
    condition_cell_types <- sort(unique(as.character(
      metadata$cell_type[condition_index]
    )), method = "radix")
    condition_metadata <- data.frame(
      cell_type = factor(
        as.character(metadata$cell_type[condition_index]),
        levels = condition_cell_types
      ),
      row.names = metadata$cell[condition_index],
      stringsAsFactors = FALSE
    )
    object <- createCellChat(
      object = gene_expr[, condition_index, drop = FALSE],
      meta = condition_metadata,
      group.by = "cell_type",
      datatype = "RNA",
      do.sparse = TRUE
    )
    object@DB <- database
    object <- subsetData(object)
    object <- identifyOverExpressedGenes(object, min.cells = min_cells)
    object <- identifyOverExpressedInteractions(object)
    if (nrow(object@LR$LRsig) == 0L) {
      stop(
        paste("CellChat found no testable interactions in", condition),
        call. = FALSE
      )
    }
    object <- computeCommunProb(object, type = "triMean")
    object <- filterCommunication(object, min.cells = min_cells)
    object_list[[condition]] <- object
  }

  object_list <- lapply(object_list, function(object) {
    liftCellChat(object, group.new = all_cell_types)
  })
  names(object_list) <- conditions

  merged <- mergeCellChat(
    object_list,
    add.names = conditions,
    merge.data = FALSE,
    cell.prefix = FALSE
  )
  features_name <- "target.merged"
  merged <- identifyOverExpressedGenes(
    merged,
    group.dataset = "datasets",
    pos.dataset = target,
    idents.use = de_comparable_cell_types,
    features.name = features_name,
    only.pos = FALSE,
    thresh.pc = 0.1,
    thresh.fc = 0.05,
    thresh.p = 0.05,
    group.DE.combined = FALSE
  )
  mapped_network <- tryCatch(
    netMappingDEG(
      merged,
      features.name = features_name,
      variable.all = TRUE
    ),
    error = function(error) {
      if (no_interactions_error(error)) {
        return(empty_mapped_network())
      }
      stop(error)
    }
  )

  target_selected <- annotate_selected(
    select_differential_network(
      merged,
      mapped_network,
      dataset = target,
      ligand_logfc = 0.05,
      receptor_logfc = NULL
    ),
    target,
    resource
  )
  reference_selected <- annotate_selected(
    select_differential_network(
      merged,
      mapped_network,
      dataset = reference,
      ligand_logfc = -0.05,
      receptor_logfc = NULL
    ),
    reference,
    resource
  )
  primary_selected <- combine_selected(target_selected, reference_selected)
  primary_semantics <- paste0(
    "cardinality_of_cellchat_v2.1.2_official_vignette_ligand_only_",
    "logFC_selected_directed_lr_",
    "after_unordered_cell_pair_collapse;target_ligand_logFC>=0.05;",
    "reference_ligand_logFC<=-0.05;receptor_logFC_unrestricted;",
    "full_condition_by_cell_pair_axis"
  )
  primary_rankings <- make_rankings(
    primary_selected,
    support,
    dataset_id,
    c(target, reference),
    primary_semantics,
    "ligand_only"
  )

  condition_networks <- lapply(object_list, function(object) {
    list(
      net = object@net,
      LR = object@LR,
      cell_groups = levels(object@idents),
      parameters = object@options$parameter,
      run_time_seconds = object@options$run.time
    )
  })
  saveRDS(
    condition_networks,
    file.path(staging_dir, "condition_communication_networks.rds"),
    compress = "gzip"
  )
  saveRDS(
    mapped_network,
    file.path(staging_dir, "mapped_differential_network.rds"),
    compress = "gzip"
  )
  write_gzip_table(
    primary_selected,
    file.path(staging_dir, "selected_differential_interactions.tsv.gz")
  )
  utils::write.table(
    primary_rankings,
    file.path(staging_dir, "condition_cell_pair_rankings.tsv"),
    sep = "\t",
    quote = FALSE,
    row.names = FALSE,
    na = ""
  )

  if (emit_receptor_sensitivity) {
    target_receptor_selected <- annotate_selected(
      select_differential_network(
        merged,
        mapped_network,
        dataset = target,
        ligand_logfc = 0.05,
        receptor_logfc = 0.05
      ),
      target,
      resource
    )
    reference_receptor_selected <- annotate_selected(
      select_differential_network(
        merged,
        mapped_network,
        dataset = reference,
        ligand_logfc = -0.05,
        receptor_logfc = -0.05
      ),
      reference,
      resource
    )
    receptor_selected <- combine_selected(
      target_receptor_selected,
      reference_receptor_selected
    )
    receptor_semantics <- paste0(
      "cardinality_of_cellchat_v2.1.2_S4_literal_",
      "concordant_ligand_receptor_logFC_",
      "selected_directed_lr_after_unordered_cell_pair_collapse;",
      "target_ligand_and_receptor_logFC>=0.05;",
      "reference_ligand_and_receptor_logFC<=-0.05;",
      "full_condition_by_cell_pair_axis"
    )
    receptor_rankings <- make_rankings(
      receptor_selected,
      support,
      dataset_id,
      c(target, reference),
      receptor_semantics,
      "ligand_receptor_concordant"
    )
    write_gzip_table(
      receptor_selected,
      file.path(
        staging_dir,
        "receptor_sensitivity_selected_interactions.tsv.gz"
      )
    )
    utils::write.table(
      receptor_rankings,
      file.path(
        staging_dir,
        "receptor_sensitivity_condition_cell_pair_rankings.tsv"
      ),
      sep = "\t",
      quote = FALSE,
      row.names = FALSE,
      na = ""
    )
  }
  writeLines(
    capture.output(sessionInfo()),
    file.path(staging_dir, "session_info.txt"),
    useBytes = TRUE
  )

  filenames <- list.files(staging_dir, all.files = FALSE, no.. = TRUE)
  if (length(filenames) == 0L) {
    stop("CellChat produced no staged outputs", call. = FALSE)
  }
  existing <- filenames[file.exists(file.path(output_dir, filenames))]
  if (length(existing) > 0L) {
    stop(
      paste("refusing to overwrite CellChat outputs:", paste(existing, collapse = ",")),
      call. = FALSE
    )
  }
  for (filename in filenames) {
    if (!file.rename(
      file.path(staging_dir, filename),
      file.path(output_dir, filename)
    )) {
      stop(paste("failed to publish", filename), call. = FALSE)
    }
    published <- c(published, filename)
  }
  publish_complete <- TRUE
}

main(args)
