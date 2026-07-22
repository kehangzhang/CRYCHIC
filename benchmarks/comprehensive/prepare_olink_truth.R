#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(digest)
  library(jsonlite)
  library(readxl)
})

schema_version <- "crychic-misc-olink-truth-v1"
dataset_id <- "misc_olink"
canonical_contrast <- "misc_m_vs_s"
contrast_label <- "M-vs-S"
effect_semantics <- "mean_diff_d0 = MIS-C day 0 minus Diorio healthy controls"
q_value_semantics <- paste(
  "adj_p_values is the source BH-adjusted p-value used as q",
  "without re-adjustment"
)

parse_args <- function(arguments) {
  values <- list(
    input_xlsx = NULL,
    output_dir = NULL,
    source_rmd = NULL,
    overwrite = FALSE
  )
  index <- 1L
  while (index <= length(arguments)) {
    argument <- arguments[[index]]
    if (argument == "--overwrite") {
      values$overwrite <- TRUE
      index <- index + 1L
    } else if (argument %in% c("--input-xlsx", "--output-dir", "--source-rmd")) {
      if (index == length(arguments)) {
        stop(sprintf("%s requires a value", argument), call. = FALSE)
      }
      key <- gsub("-", "_", sub("^--", "", argument))
      values[[key]] <- arguments[[index + 1L]]
      index <- index + 2L
    } else {
      stop(sprintf("unknown argument: %s", argument), call. = FALSE)
    }
  }
  required <- c("input_xlsx", "output_dir")
  missing <- required[vapply(values[required], is.null, logical(1))]
  if (length(missing)) {
    stop(
      sprintf("missing required arguments: %s", paste(missing, collapse = ", ")),
      call. = FALSE
    )
  }
  values
}

sha256_file <- function(path) {
  digest(path, algo = "sha256", file = TRUE, serialize = FALSE)
}

strict_json <- function(value, path) {
  writeLines(
    toJSON(
      value,
      auto_unbox = TRUE,
      pretty = TRUE,
      null = "null",
      na = "null",
      digits = NA
    ),
    path,
    useBytes = TRUE
  )
}

atomic_install <- function(staging, output_dir, overwrite) {
  output_exists <- dir.exists(output_dir) || file.exists(output_dir)
  if (!output_exists) {
    if (!file.rename(staging, output_dir)) {
      stop("failed to atomically install Olink truth directory", call. = FALSE)
    }
    return(invisible(NULL))
  }
  if (!overwrite) {
    stop(
      sprintf("output directory exists: %s; pass --overwrite to replace it", output_dir),
      call. = FALSE
    )
  }
  backup <- tempfile(
    pattern = paste0(".", basename(output_dir), ".previous-"),
    tmpdir = dirname(output_dir)
  )
  if (!file.rename(output_dir, backup)) {
    stop("failed to move the previous Olink truth directory", call. = FALSE)
  }
  if (!file.rename(staging, output_dir)) {
    file.rename(backup, output_dir)
    stop(
      "failed to install Olink truth; previous output restored",
      call. = FALSE
    )
  }
  unlink(backup, recursive = TRUE, force = TRUE)
  invisible(NULL)
}

validate_numeric <- function(frame, columns) {
  for (column in columns) {
    values <- frame[[column]]
    if (!is.numeric(values) || anyNA(values) || any(!is.finite(values))) {
      stop(
        sprintf("source column %s must contain finite numeric values", column),
        call. = FALSE
      )
    }
  }
}

args <- parse_args(commandArgs(trailingOnly = TRUE))
input_xlsx <- normalizePath(args$input_xlsx, mustWork = TRUE)
if (!grepl("\\.xlsx$", input_xlsx, ignore.case = TRUE)) {
  stop("--input-xlsx must be an .xlsx workbook", call. = FALSE)
}
source_rmd <- if (is.null(args$source_rmd)) {
  NULL
} else {
  normalizePath(args$source_rmd, mustWork = TRUE)
}
dir.create(dirname(args$output_dir), recursive = TRUE, showWarnings = FALSE)
parent_dir <- normalizePath(dirname(args$output_dir), mustWork = TRUE)
output_dir <- file.path(parent_dir, basename(args$output_dir))
if ((dir.exists(output_dir) || file.exists(output_dir)) && !args$overwrite) {
  stop(
    sprintf("output directory exists: %s; pass --overwrite to replace it", output_dir),
    call. = FALSE
  )
}

staging <- tempfile(
  pattern = paste0(".", basename(output_dir), ".staging-"),
  tmpdir = parent_dir
)
dir.create(staging, recursive = FALSE)
installed <- FALSE
on.exit({
  if (!installed && dir.exists(staging)) {
    unlink(staging, recursive = TRUE, force = TRUE)
  }
}, add = TRUE)

sheets <- excel_sheets(input_xlsx)
if (length(sheets) < 1L) {
  stop("Olink workbook contains no worksheets", call. = FALSE)
}
source <- as.data.frame(
  read_excel(input_xlsx, sheet = 1L, .name_repair = "minimal"),
  stringsAsFactors = FALSE,
  check.names = FALSE
)
required_columns <- c(
  "variables",
  "mean_d0",
  "mean_hc",
  "mean_diff_d0",
  "unadj_p_values",
  "adj_p_values"
)
missing_columns <- setdiff(required_columns, names(source))
extra_columns <- setdiff(names(source), required_columns)
if (length(missing_columns) || length(extra_columns)) {
  stop(
    sprintf(
      "Olink worksheet columns differ: missing=%s; extra=%s",
      paste(missing_columns, collapse = ","),
      paste(extra_columns, collapse = ",")
    ),
    call. = FALSE
  )
}
if (!nrow(source)) {
  stop("Olink worksheet contains no analytes", call. = FALSE)
}

analyte_original <- as.character(source$variables)
if (
  anyNA(analyte_original) ||
    any(!nzchar(analyte_original)) ||
    any(grepl("\t", analyte_original, fixed = TRUE)) ||
    any(grepl("\r", analyte_original, fixed = TRUE)) ||
    any(grepl("\n", analyte_original, fixed = TRUE))
) {
  stop("source analyte identifiers must be complete one-line strings", call. = FALSE)
}
if (anyDuplicated(analyte_original)) {
  stop("source analyte identifiers must be unique", call. = FALSE)
}
analyte_make_names <- make.names(analyte_original)
if (anyDuplicated(analyte_make_names)) {
  stop("make.names creates ambiguous analyte identifiers", call. = FALSE)
}

numeric_columns <- setdiff(required_columns, "variables")
validate_numeric(source, numeric_columns)
for (column in c("unadj_p_values", "adj_p_values")) {
  if (any(source[[column]] < 0 | source[[column]] > 1)) {
    stop(sprintf("source column %s must lie in [0, 1]", column), call. = FALSE)
  }
}
effect_residual <- source$mean_diff_d0 - (source$mean_d0 - source$mean_hc)
if (max(abs(effect_residual)) > 1e-10) {
  stop(
    "mean_diff_d0 is inconsistent with mean_d0 minus mean_hc",
    call. = FALSE
  )
}

truth <- data.frame(
  schema_version = rep(schema_version, nrow(source)),
  dataset_id = rep(dataset_id, nrow(source)),
  source_row = seq_len(nrow(source)),
  contrast_id = rep(canonical_contrast, nrow(source)),
  contrast_label = rep(contrast_label, nrow(source)),
  contrast_numerator = rep("MIS-C", nrow(source)),
  contrast_denominator = rep("Diorio_healthy_control", nrow(source)),
  effect_semantics = rep(effect_semantics, nrow(source)),
  q_value_semantics = rep(q_value_semantics, nrow(source)),
  analyte_original = analyte_original,
  analyte_make_names = analyte_make_names,
  variables = analyte_original,
  mean_d0 = source$mean_d0,
  mean_hc = source$mean_hc,
  mean_diff_d0 = source$mean_diff_d0,
  unadj_p_values = source$unadj_p_values,
  adj_p_values = source$adj_p_values,
  stringsAsFactors = FALSE,
  check.names = FALSE
)
truth_path <- file.path(staging, "olink_truth.tsv")
write.table(
  truth,
  truth_path,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  col.names = TRUE,
  na = "",
  eol = "\n"
)

source_rmd_record <- if (is.null(source_rmd)) {
  list(
    filename = "add_proteomics_MISC.Rmd",
    sha256 = NULL,
    supplied = FALSE
  )
} else {
  list(
    filename = basename(source_rmd),
    bytes = unname(file.info(source_rmd)$size),
    sha256 = sha256_file(source_rmd),
    supplied = TRUE
  )
}
file_arguments <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
if (length(file_arguments) != 1L) {
  stop("cannot identify the running R script for provenance", call. = FALSE)
}
script_path <- normalizePath(sub("^--file=", "", file_arguments[[1]]), mustWork = TRUE)
manifest <- list(
  schema_version = schema_version,
  status = "complete",
  created_at_utc = format(Sys.time(), tz = "UTC", usetz = TRUE),
  dataset_id = dataset_id,
  source = list(
    workbook = list(
      filename = basename(input_xlsx),
      bytes = unname(file.info(input_xlsx)$size),
      sha256 = sha256_file(input_xlsx),
      zenodo_doi = "10.5281/zenodo.10908003",
      publication_doi = "10.1038/s41467-021-27544-6",
      worksheet_index = 1L,
      worksheet_name = sheets[[1]]
    ),
    specimen_wording = list(
      vignette_and_zenodo = "serum",
      Diorio_publication_methods = "plasma",
      policy = "retain the source discrepancy; do not relabel as one specimen type"
    ),
    official_direction_reference = source_rmd_record
  ),
  truth_contract = list(
    canonical_contrast_id = canonical_contrast,
    contrast_label = contrast_label,
    source_contrast = "MIS-C-vs-healthy-control",
    numerator = "MIS-C",
    denominator = "Diorio_healthy_control",
    external_healthy_controls_proxy_the_Hoste_sibling_group = TRUE,
    effect_column = "mean_diff_d0",
    effect_semantics = effect_semantics,
    effect_scale = "difference of group-mean log2-scale Olink NPX values",
    effect_is_not_a_covariate_adjusted_model_coefficient = TRUE,
    reverse_contrast_id = "misc_s_vs_m",
    reverse_contrast_label = "S-vs-M",
    reverse_is_evaluator_derived = TRUE,
    reverse_effect = "-mean_diff_d0",
    q_value_column = "adj_p_values",
    q_value_semantics = q_value_semantics,
    q_values_are_not_readjusted = TRUE,
    significance_threshold = 0.05,
    analyte_source_column = "variables",
    analyte_mapping = "analyte_original -> base::make.names(analyte_original)",
    underscore_analytes_are_retained_unsplit = TRUE,
    underscore_analytes = as.list(analyte_original[grepl("_", analyte_original)]),
    missing_analytes_are_not_zero = TRUE,
    olink_is_held_out_evaluation_only = TRUE,
    olink_must_not_enter_method_scores = TRUE
  ),
  dimensions = list(
    analytes = nrow(truth),
    q_le_0_05 = sum(truth$adj_p_values <= 0.05),
    q_le_0_05_positive_misc_minus_healthy = sum(
      truth$adj_p_values <= 0.05 & truth$mean_diff_d0 > 0
    ),
    q_le_0_05_negative_misc_minus_healthy = sum(
      truth$adj_p_values <= 0.05 & truth$mean_diff_d0 < 0
    ),
    make_names_changed = sum(truth$analyte_original != truth$analyte_make_names),
    make_names_unchanged = sum(
      truth$analyte_original == truth$analyte_make_names
    ),
    analyte_duplicates_original = sum(duplicated(truth$analyte_original)),
    analyte_duplicates_make_names = sum(duplicated(truth$analyte_make_names)),
    underscore_analytes_retained_unsplit = sum(grepl("_", truth$analyte_original))
  ),
  output = list(
    truth_tsv = list(
      filename = basename(truth_path),
      rows = nrow(truth),
      bytes = unname(file.info(truth_path)$size),
      sha256 = sha256_file(truth_path)
    )
  ),
  software = list(
    R = as.character(getRversion()),
    readxl = as.character(packageVersion("readxl")),
    digest = as.character(packageVersion("digest")),
    jsonlite = as.character(packageVersion("jsonlite"))
  ),
  code = list(
    filename = basename(script_path),
    sha256 = sha256_file(script_path)
  )
)
strict_json(manifest, file.path(staging, "manifest.json"))
atomic_install(staging, output_dir, args$overwrite)
installed <- TRUE
cat(toJSON(manifest$dimensions, auto_unbox = TRUE), "\n")
