#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(basilisk)
  library(digest)
  library(jsonlite)
  library(SingleCellExperiment)
  library(zellkonverter)
})

schema_version <- "crychic-misc-prepared-input-v1"

parse_args <- function(args) {
  values <- list(
    input_rds = NULL,
    olink = NULL,
    output_dir = NULL,
    python_env = NULL,
    overwrite = FALSE
  )
  index <- 1L
  while (index <= length(args)) {
    argument <- args[[index]]
    if (argument == "--overwrite") {
      values$overwrite <- TRUE
      index <- index + 1L
    } else if (argument %in% c("--input-rds", "--olink", "--output-dir", "--python-env")) {
      if (index == length(args)) {
        stop(sprintf("%s requires a value", argument), call. = FALSE)
      }
      key <- gsub("-", "_", sub("^--", "", argument))
      values[[key]] <- args[[index + 1L]]
      index <- index + 2L
    } else {
      stop(sprintf("unknown argument: %s", argument), call. = FALSE)
    }
  }
  required <- c("input_rds", "olink", "output_dir", "python_env")
  missing <- required[vapply(values[required], is.null, logical(1))]
  if (length(missing)) {
    stop(sprintf("missing required arguments: %s", paste(missing, collapse = ", ")), call. = FALSE)
  }
  values
}

sha256_file <- function(path) {
  digest(path, algo = "sha256", file = TRUE, serialize = FALSE)
}

validate_one_to_one <- function(frame, key, values) {
  unique_rows <- unique(frame[c(key, values)])
  counts <- table(unique_rows[[key]])
  if (any(counts != 1L)) {
    affected <- names(counts[counts != 1L])
    stop(
      sprintf(
        "%s does not map to one %s tuple: %s",
        key,
        paste(values, collapse = "/"),
        paste(head(affected, 5L), collapse = ", ")
      ),
      call. = FALSE
    )
  }
}

atomic_install <- function(staging, output_dir, overwrite) {
  if (!dir.exists(output_dir)) {
    if (!file.rename(staging, output_dir)) {
      stop("failed to install prepared output directory", call. = FALSE)
    }
    return(invisible(NULL))
  }
  if (!overwrite) {
    stop(
      sprintf("output directory exists: %s; pass --overwrite to replace it", output_dir),
      call. = FALSE
    )
  }
  backup <- paste0(output_dir, ".previous")
  if (dir.exists(backup)) {
    unlink(backup, recursive = TRUE, force = TRUE)
  }
  if (!file.rename(output_dir, backup)) {
    stop("failed to move the previous output directory", call. = FALSE)
  }
  if (!file.rename(staging, output_dir)) {
    file.rename(backup, output_dir)
    stop("failed to install prepared output; previous output restored", call. = FALSE)
  }
  unlink(backup, recursive = TRUE, force = TRUE)
  invisible(NULL)
}

build_design_audit <- function(metadata) {
  base <- unique(metadata[c(
    "sample_id", "subject_id", "family_id", "family_cluster_is_observed",
    "condition", "condition_code", "sequencing_batch"
  )])
  validate_one_to_one(
    base,
    "sample_id",
    c("subject_id", "family_id", "condition", "sequencing_batch")
  )
  counts <- as.data.frame(
    table(metadata$sample_id, metadata$cell_type),
    stringsAsFactors = FALSE
  )
  names(counts) <- c("sample_id", "cell_type", "n_cells")
  wide <- reshape(
    counts,
    idvar = "sample_id",
    timevar = "cell_type",
    direction = "wide"
  )
  names(wide) <- sub("^n_cells\\.", "cells_", names(wide))
  result <- merge(base, wide, by = "sample_id", all.x = TRUE, sort = FALSE)
  cell_columns <- grep("^cells_", names(result), value = TRUE)
  result$total_cells <- rowSums(result[cell_columns])
  result$n_observed_cell_types <- rowSums(result[cell_columns] > 0L)
  result <- result[order(result$condition, result$sample_id), ]
  rownames(result) <- NULL
  result
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
input_rds <- normalizePath(args$input_rds, mustWork = TRUE)
olink_path <- normalizePath(args$olink, mustWork = TRUE)
output_dir <- normalizePath(args$output_dir, mustWork = FALSE)
python_env <- normalizePath(args$python_env, mustWork = TRUE)
python_executable <- file.path(python_env, "bin", "python")
if (!file.exists(python_executable)) {
  stop("--python-env does not contain bin/python", call. = FALSE)
}
parent_dir <- dirname(output_dir)
dir.create(parent_dir, recursive = TRUE, showWarnings = FALSE)
staging <- tempfile(pattern = paste0(".", basename(output_dir), ".staging-"), tmpdir = parent_dir)
dir.create(staging, recursive = FALSE)
installed <- FALSE
on.exit({
  if (!installed && dir.exists(staging)) {
    unlink(staging, recursive = TRUE, force = TRUE)
  }
}, add = TRUE)

sce <- readRDS(input_rds)
if (!is(sce, "SingleCellExperiment")) {
  stop("--input-rds must contain a SingleCellExperiment", call. = FALSE)
}
required_assays <- c("counts", "logcounts")
missing_assays <- setdiff(required_assays, assayNames(sce))
if (length(missing_assays)) {
  stop(sprintf("input is missing assays: %s", paste(missing_assays, collapse = ", ")), call. = FALSE)
}
required_columns <- c("ShortID", "PatientID", "FamilyCode", "orig.ident", "Condition", "LEVEL2")
missing_columns <- setdiff(required_columns, colnames(colData(sce)))
if (length(missing_columns)) {
  stop(sprintf("input is missing colData: %s", paste(missing_columns, collapse = ", ")), call. = FALSE)
}
if (anyDuplicated(rownames(sce)) || anyDuplicated(colnames(sce))) {
  stop("gene and cell identifiers must be unique", call. = FALSE)
}

source_metadata <- as.data.frame(colData(sce))[required_columns]
if (anyNA(source_metadata)) {
  stop("required MIS-C metadata contain missing values", call. = FALSE)
}
condition_map <- c(M = "MIS-C", S = "healthy_sibling", C = "adult_severe_COVID19")
cell_type_map <- c(L_NKcell = "NK", L_Tcell = "T", M_Monocyte = "Monocyte")
unknown_conditions <- setdiff(unique(source_metadata$Condition), names(condition_map))
unknown_cell_types <- setdiff(unique(source_metadata$LEVEL2), names(cell_type_map))
if (length(unknown_conditions) || length(unknown_cell_types)) {
  stop(
    sprintf(
      "unmapped values; conditions=%s, cell_types=%s",
      paste(unknown_conditions, collapse = ","),
      paste(unknown_cell_types, collapse = ",")
    ),
    call. = FALSE
  )
}

family_code <- as.character(source_metadata$FamilyCode)
subject_id <- as.character(source_metadata$PatientID)
is_observed_family <- family_code != "UNR"
family_id <- ifelse(
  is_observed_family,
  family_code,
  paste0("UNRELATED_", subject_id)
)
metadata <- DataFrame(
  cell_id = colnames(sce),
  sample_id = as.character(source_metadata$ShortID),
  subject_id = subject_id,
  family_id = family_id,
  family_cluster_is_observed = is_observed_family,
  sequencing_batch = as.character(source_metadata$orig.ident),
  condition = unname(condition_map[as.character(source_metadata$Condition)]),
  condition_code = as.character(source_metadata$Condition),
  cell_type = unname(cell_type_map[as.character(source_metadata$LEVEL2)]),
  row.names = colnames(sce)
)
metadata_frame <- as.data.frame(metadata)
validate_one_to_one(
  metadata_frame,
  "sample_id",
  c("subject_id", "family_id", "condition", "sequencing_batch")
)
validate_one_to_one(metadata_frame, "subject_id", c("sample_id", "condition"))

prepared <- SingleCellExperiment(
  assays = list(counts = assay(sce, "counts"), logcounts = assay(sce, "logcounts")),
  rowData = DataFrame(feature_id = rownames(sce), row.names = rownames(sce)),
  colData = metadata,
  metadata = list(
    schema_version = schema_version,
    dataset_id = "misc_olink",
    inferential_unit = "subject_id",
    family_block = "family_id",
    structural_absence_is_zero = FALSE
  )
)

design <- build_design_audit(metadata_frame)
design_path <- file.path(staging, "design_audit.tsv")
write.table(
  design,
  design_path,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = ""
)

h5ad_path <- file.path(staging, "misc_olink.h5ad")
basiliskRun(
  env = python_env,
  fun = getFromNamespace(".H5ADwriter", "zellkonverter"),
  sce = prepared,
  file = h5ad_path,
  X_name = "logcounts",
  skip_assays = FALSE,
  compression = "gzip"
)

anndata_version <- system2(
  python_executable,
  c("-c", shQuote("import importlib.metadata; print(importlib.metadata.version('anndata'))")),
  stdout = TRUE
)

manifest <- list(
  schema_version = schema_version,
  created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  dataset_id = "misc_olink",
  inputs = list(
    sce_rds = list(filename = basename(input_rds), sha256 = sha256_file(input_rds)),
    olink_xlsx = list(filename = basename(olink_path), sha256 = sha256_file(olink_path))
  ),
  dimensions = list(
    n_genes = nrow(prepared),
    n_cells = ncol(prepared),
    n_samples = length(unique(prepared$sample_id)),
    n_subjects = length(unique(prepared$subject_id)),
    n_family_blocks = length(unique(prepared$family_id)),
    n_conditions = length(unique(prepared$condition)),
    n_cell_types = length(unique(prepared$cell_type))
  ),
  condition_counts = as.list(table(design$condition)),
  cell_type_counts = as.list(table(metadata_frame$cell_type)),
  contracts = list(
    inferential_unit = "subject_id",
    sample_key = "sample_id",
    family_block = "family_id",
    condition_key = "condition",
    cell_type_key = "cell_type",
    x_assay = "logcounts",
    counts_layer = "counts",
    structural_absence_is_missing = TRUE,
    unrelated_family_code_split_by_subject = TRUE,
    olink_withheld_until_predictions_frozen = TRUE
  ),
  outputs = list(
    h5ad = list(
      filename = basename(h5ad_path),
      bytes = file.info(h5ad_path)$size,
      sha256 = sha256_file(h5ad_path)
    ),
    design_audit = list(
      filename = basename(design_path),
      rows = nrow(design),
      sha256 = sha256_file(design_path)
    )
  ),
  software = list(
    R = as.character(getRversion()),
    SingleCellExperiment = as.character(packageVersion("SingleCellExperiment")),
    zellkonverter = as.character(packageVersion("zellkonverter")),
    basilisk = as.character(packageVersion("basilisk")),
    anndata = anndata_version[[1]],
    python_environment = basename(python_env)
  )
)
write_json(
  manifest,
  file.path(staging, "preparation_manifest.json"),
  auto_unbox = TRUE,
  pretty = TRUE,
  null = "null",
  na = "null"
)
cat("\n", file = file.path(staging, "preparation_manifest.json"), append = TRUE)

atomic_install(staging, output_dir, args$overwrite)
installed <- TRUE
cat(toJSON(manifest$dimensions, auto_unbox = TRUE), "\n")
