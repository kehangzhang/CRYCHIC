#!/usr/bin/env Rscript

# Protocol-level reconstruction of Kuppe spatial MISTy results. The output is
# recomputed from public CELLxGENE files and is not the authors' importance table.

usage <- paste(
  "Usage: run_kuppe_spatial_misty.R --manifest INPUT_MANIFEST --output-dir DIR",
  "[--threads N] [--seed N] [--cv-folds N] [--cache] [--resume]",
  "[--samples ID1,ID2]"
)

parse_args <- function(args) {
  result <- list(
    threads = 1L,
    seed = 42L,
    cv_folds = 10L,
    cache = FALSE,
    resume = FALSE,
    samples = NULL
  )
  index <- 1L
  while (index <= length(args)) {
    key <- args[[index]]
    if (key %in% c("--cache", "--resume")) {
      result[[substring(key, 3L)]] <- TRUE
      index <- index + 1L
      next
    }
    if (!key %in% c(
      "--manifest", "--output-dir", "--threads", "--seed", "--cv-folds",
      "--samples"
    )) stop("Unknown argument: ", key, "\n", usage)
    if (index == length(args)) stop("Missing value for ", key, "\n", usage)
    value <- args[[index + 1L]]
    name <- gsub("-", "_", substring(key, 3L), fixed = TRUE)
    result[[name]] <- value
    index <- index + 2L
  }
  if (is.null(result$manifest) || is.null(result$output_dir)) stop(usage)
  for (name in c("threads", "seed", "cv_folds")) {
    result[[name]] <- suppressWarnings(as.integer(result[[name]]))
    if (is.na(result[[name]])) stop("--", gsub("_", "-", name), " must be an integer")
  }
  if (result$threads < 1L) stop("--threads must be >= 1")
  if (result$cv_folds < 2L) stop("--cv-folds must be >= 2")
  if (!is.null(result$samples)) {
    result$samples <- Filter(nzchar, strsplit(result$samples, ",", fixed = TRUE)[[1L]])
  }
  result
}

required_packages <- c("digest", "future", "jsonlite", "mistyR", "readr")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1L), quietly = TRUE)
]
if (length(missing_packages) > 0L) {
  stop("Missing required R packages: ", paste(missing_packages, collapse = ", "))
}
if (as.character(utils::packageVersion("mistyR")) != "1.3.5") {
  stop(
    "This reconstruction is pinned to mistyR 1.3.5; found ",
    as.character(utils::packageVersion("mistyR"))
  )
}

EXPECTED_INPUT_SCHEMA <- "crychic-kuppe-spatial-misty-input-v1"
RUN_SCHEMA <- "crychic-kuppe-spatial-misty-run-v1"
DATASET_ID <- "Kuppe_MI_spatial_CTRL_vs_IZ"
MISTYR_TAG_COMMIT <- "19248ea7e02803063d1e1112a8af6c3f06c59e03"
ABUNDANCE_COLUMNS <- c(
  "Adipocyte", "Cardiomyocyte", "Endothelial", "Fibroblast", "Lymphoid",
  "Mast", "Myeloid", "Neuronal", "Pericyte", "Cycling.cells", "vSMCs"
)
EXPECTED_SAMPLE_CONDITIONS <- c(
  control_P1 = "CTRL",
  control_P17 = "CTRL",
  control_P7 = "CTRL",
  control_P8 = "CTRL",
  GT_IZ_P13 = "IZ",
  GT_IZ_P15 = "IZ",
  GT_IZ_P9 = "IZ",
  GT_IZ_P9_rep2 = "IZ",
  IZ_BZ_P2 = "IZ",
  IZ_P10 = "IZ",
  IZ_P15 = "IZ",
  IZ_P16 = "IZ",
  IZ_P3 = "IZ"
)
JUXTA_NEIGHBOR_THRESHOLD <- 5
PARA_LENGTH_SCALE <- 15
RECOMPUTATION_STATUS <- paste(
  "protocol-level recomputation from public CELLxGENE inputs;",
  "not the authors' published MISTy importance table"
)
mistyr_description <- utils::packageDescription("mistyR")
mistyr_remote_sha <- mistyr_description$RemoteSha
if (!is.null(mistyr_remote_sha) && !identical(mistyr_remote_sha, MISTYR_TAG_COMMIT)) {
  stop(
    "mistyR 1.3.5 RemoteSha mismatch: ", mistyr_remote_sha,
    " != ", MISTYR_TAG_COMMIT
  )
}

sha256_file <- function(path) digest::digest(file = path, algo = "sha256", serialize = FALSE)

write_json_atomic <- function(value, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp-", Sys.getpid())
  on.exit(if (file.exists(temporary)) unlink(temporary), add = TRUE)
  jsonlite::write_json(value, temporary, pretty = TRUE, auto_unbox = TRUE, null = "null")
  if (!file.rename(temporary, path)) stop("Failed to finalize JSON file: ", path)
}

write_tsv_gz <- function(table, path) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  temporary <- paste0(path, ".tmp-", Sys.getpid())
  on.exit(if (file.exists(temporary)) unlink(temporary), add = TRUE)
  # readr delegates connection output to vroom, which requires a binary
  # connection; gzfile still writes a standards-compliant gzip stream.
  connection <- gzfile(temporary, open = "wb", encoding = "UTF-8")
  tryCatch(
    readr::write_tsv(table, connection, na = "NA"),
    finally = close(connection)
  )
  if (!file.rename(temporary, path)) stop("Failed to finalize TSV file: ", path)
  checksum <- sha256_file(path)
  writeLines(
    paste(checksum, basename(path), sep = "  "),
    paste0(path, ".sha256"),
    useBytes = TRUE
  )
  list(filename = basename(path), sha256 = checksum, rows = nrow(table))
}

read_prepared_slide <- function(path, expected_sha256) {
  if (!file.exists(path)) stop("Prepared slide input does not exist: ", path)
  observed_sha256 <- sha256_file(path)
  if (!identical(observed_sha256, expected_sha256)) {
    stop(
      "Prepared slide checksum mismatch for ", path, ": ", observed_sha256,
      " != ", expected_sha256
    )
  }
  table <- readr::read_tsv(
    path,
    col_types = readr::cols(.default = readr::col_double(), spot_id = readr::col_character()),
    name_repair = "minimal",
    progress = FALSE
  )
  expected_columns <- c("spot_id", "array_row", "array_col", ABUNDANCE_COLUMNS)
  if (!identical(names(table), expected_columns)) {
    stop("Unexpected prepared slide columns for ", path)
  }
  if (nrow(table) < 10L) stop("Prepared slide has fewer than 10 retained spots: ", path)
  if (anyDuplicated(table$spot_id)) stop("Duplicate spot_id values in ", path)
  if (anyDuplicated(table[c("array_row", "array_col")])) {
    stop("Duplicate array geometry in ", path)
  }
  abundance <- as.data.frame(table[ABUNDANCE_COLUMNS], check.names = FALSE)
  geometry <- data.frame(
    row = table$array_row,
    col = table$array_col,
    row.names = table$spot_id,
    check.names = FALSE
  )
  rownames(abundance) <- table$spot_id
  if (any(!is.finite(as.matrix(abundance))) || any(as.matrix(abundance) < 0)) {
    stop("Invalid abundance values in ", path)
  }
  if (any(!is.finite(as.matrix(geometry)))) stop("Invalid array geometry in ", path)
  list(abundance = abundance, geometry = geometry, sha256 = observed_sha256)
}

compose_author_views <- function(abundance, geometry, cache_id, cache) {
  # This mirrors Kuppe's run_misty_seurat helper after assay extraction. The
  # custom view names deliberately use underscores, matching its output labels.
  main <- mistyR::create_initial_view(abundance, unique.id = cache_id)
  juxta_source <- mistyR::create_initial_view(abundance, unique.id = cache_id)
  juxta_source <- mistyR::add_juxtaview(
    juxta_source,
    positions = geometry,
    neighbor.thr = JUXTA_NEIGHBOR_THRESHOLD,
    cached = cache,
    verbose = TRUE
  )
  para_source <- mistyR::create_initial_view(abundance, unique.id = cache_id)
  para_source <- mistyR::add_paraview(
    para_source,
    positions = geometry,
    l = PARA_LENGTH_SCALE,
    zoi = 0,
    family = "gaussian",
    approx = 1,
    cached = cache,
    verbose = TRUE
  )
  juxta <- mistyR::create_view(
    "juxta_5",
    juxta_source[[paste0("juxtaview.", JUXTA_NEIGHBOR_THRESHOLD)]]$data
  )
  para <- mistyR::create_view(
    "para_15",
    para_source[[paste0("paraview.", PARA_LENGTH_SCALE)]]$data
  )
  mistyR::add_views(main, c(juxta, para))
}

valid_completed_sample <- function(sample_dir, expected_input_sha256, options) {
  manifest_path <- file.path(sample_dir, "sample.manifest.json")
  if (!file.exists(manifest_path)) return(FALSE)
  manifest <- tryCatch(
    jsonlite::read_json(manifest_path, simplifyVector = FALSE),
    error = function(error) NULL
  )
  if (is.null(manifest) || !identical(manifest$status, "complete")) return(FALSE)
  if (!identical(manifest$input$sha256, expected_input_sha256)) return(FALSE)
  if (as.integer(manifest$protocol$seed) != options$seed) return(FALSE)
  if (as.integer(manifest$protocol$cv_folds) != options$cv_folds) return(FALSE)
  if (!identical(manifest$software$mistyR, "1.3.5")) return(FALSE)
  for (artifact in manifest$outputs) {
    path <- file.path(sample_dir, artifact$filename)
    if (!file.exists(path) || !identical(sha256_file(path), artifact$sha256)) return(FALSE)
  }
  TRUE
}

run_slide <- function(slide, input_root, output_root, options) {
  sample_id <- slide$sample_id
  condition <- slide$condition
  input_path <- file.path(input_root, slide$input$filename)
  sample_dir <- file.path(output_root, sample_id)
  sample_manifest_path <- file.path(sample_dir, "sample.manifest.json")

  if (
    options$resume &&
      valid_completed_sample(sample_dir, slide$input$sha256, options)
  ) {
    manifest <- jsonlite::read_json(sample_manifest_path, simplifyVector = FALSE)
    manifest$resume_status <- "reused_checksum_validated_result"
    return(manifest)
  }

  if (dir.exists(sample_dir)) unlink(sample_dir, recursive = TRUE)
  dir.create(sample_dir, recursive = TRUE, showWarnings = FALSE)
  started_at <- format(Sys.time(), tz = "UTC", usetz = TRUE)
  base_manifest <- list(
    schema_version = RUN_SCHEMA,
    dataset_id = DATASET_ID,
    sample_id = sample_id,
    condition = condition,
    status = "running",
    recomputation_status = RECOMPUTATION_STATUS,
    started_at = started_at,
    input = list(filename = basename(input_path), sha256 = slide$input$sha256),
    software = list(
      R = as.character(getRversion()),
      mistyR = as.character(utils::packageVersion("mistyR")),
      mistyR_tag_commit = MISTYR_TAG_COMMIT,
      mistyR_installed_remote_sha = mistyr_remote_sha
    ),
    protocol = list(
      geometry = "Visium array_row/array_col (not image pixels)",
      abundance_transform = "none",
      assay_compatibility_note = paste(
        "Author source ran c2l and c2l_props; the public H5AD exposes one",
        "normalized 11-column panel. This recomputation uses that panel",
        "unchanged and does not claim recovery of the private raw c2l assay."
      ),
      feature_compatibility_note = paste(
        "Author source excluded 'prolif'; the public CELLxGENE reconstruction",
        "uses the frozen 11-column panel including 'Cycling.cells' without an",
        "unverified alias mapping."
      ),
      views = list(
        intra = list(),
        juxta_5 = list(neighbor_thr = JUXTA_NEIGHBOR_THRESHOLD),
        para_15 = list(l = PARA_LENGTH_SCALE, family = "gaussian", approx = 1)
      ),
      seed = options$seed,
      cv_folds = options$cv_folds,
      future_workers = options$threads,
      ranger_threads_per_worker = 1L,
      cache = options$cache
    )
  )
  write_json_atomic(base_manifest, sample_manifest_path)

  tryCatch({
    prepared <- read_prepared_slide(input_path, slide$input$sha256)
    cache_id <- paste(
      sample_id,
      substr(prepared$sha256, 1L, 16L),
      paste0("seed", options$seed),
      paste0("cv", options$cv_folds),
      "juxta5_para15",
      sep = "_"
    )
    views <- compose_author_views(
      prepared$abundance, prepared$geometry, cache_id, options$cache
    )
    raw_dir <- file.path(sample_dir, "misty_raw")
    mistyR::run_misty(
      views,
      results.folder = raw_dir,
      seed = options$seed,
      cv.folds = options$cv_folds,
      cached = options$cache,
      append = FALSE,
      num.threads = 1L
    )
    collected <- mistyR::collect_results(raw_dir)
    importances <- as.data.frame(collected$importances)
    expected_importance_columns <- c("view", "Predictor", "Target", "Importance")
    missing <- setdiff(expected_importance_columns, names(importances))
    if (length(missing) > 0L) {
      stop("mistyR collected importance columns are missing: ", paste(missing, collapse = ", "))
    }
    importances <- importances[expected_importance_columns]
    importances <- cbind(
      data.frame(
        dataset = DATASET_ID,
        sample_id = sample_id,
        condition = condition,
        stringsAsFactors = FALSE
      ),
      importances,
      recomputation_status = RECOMPUTATION_STATUS
    )
    performance <- as.data.frame(collected$improvements)
    if ("sample" %in% names(performance)) performance$sample <- NULL
    performance <- cbind(
      data.frame(
        dataset = DATASET_ID,
        sample_id = sample_id,
        condition = condition,
        stringsAsFactors = FALSE
      ),
      performance,
      recomputation_status = RECOMPUTATION_STATUS
    )
    importance_artifact <- write_tsv_gz(
      importances, file.path(sample_dir, "sample_importances.tsv.gz")
    )
    performance_artifact <- write_tsv_gz(
      performance, file.path(sample_dir, "sample_performance.tsv.gz")
    )
    completed <- base_manifest
    completed$status <- "complete"
    completed$completed_at <- format(Sys.time(), tz = "UTC", usetz = TRUE)
    completed$input$sha256 <- prepared$sha256
    completed$n_spots <- nrow(prepared$abundance)
    completed$features <- ABUNDANCE_COLUMNS
    completed$outputs <- list(
      importances = importance_artifact,
      performance = performance_artifact
    )
    write_json_atomic(completed, sample_manifest_path)
    completed
  }, error = function(error) {
    failed <- base_manifest
    failed$status <- "failed"
    failed$failed_at <- format(Sys.time(), tz = "UTC", usetz = TRUE)
    failed$error <- list(
      class = class(error)[[1L]],
      message = conditionMessage(error)
    )
    write_json_atomic(failed, sample_manifest_path)
    failed
  })
}

options <- parse_args(commandArgs(trailingOnly = TRUE))
input_manifest_path <- normalizePath(options$manifest, mustWork = TRUE)
output_root <- normalizePath(options$output_dir, mustWork = FALSE)
dir.create(output_root, recursive = TRUE, showWarnings = FALSE)
output_root <- normalizePath(output_root, mustWork = TRUE)
input_root <- dirname(input_manifest_path)
input_manifest <- jsonlite::read_json(input_manifest_path, simplifyVector = FALSE)
if (!identical(input_manifest$schema_version, EXPECTED_INPUT_SCHEMA)) {
  stop("Unsupported input manifest schema: ", input_manifest$schema_version)
}
if (!identical(input_manifest$dataset_id, DATASET_ID)) {
  stop("Unexpected dataset_id: ", input_manifest$dataset_id)
}
if (isTRUE(input_manifest$protocol$pixel_coordinates_used)) {
  stop("Input manifest incorrectly declares pixel geometry")
}

slides <- input_manifest$slides
observed_sample_conditions <- vapply(
  slides,
  function(slide) slide$condition,
  character(1L)
)
names(observed_sample_conditions) <- vapply(
  slides,
  function(slide) slide$sample_id,
  character(1L)
)
observed_sample_conditions <- observed_sample_conditions[
  order(names(observed_sample_conditions))
]
expected_sample_conditions <- EXPECTED_SAMPLE_CONDITIONS[
  order(names(EXPECTED_SAMPLE_CONDITIONS))
]
if (!identical(observed_sample_conditions, expected_sample_conditions)) {
  stop("Input manifest does not match the frozen 4 CTRL / 9 IZ Kuppe roster")
}
if (!is.null(options$samples)) {
  known <- vapply(slides, function(slide) slide$sample_id, character(1L))
  unknown <- setdiff(options$samples, known)
  if (length(unknown) > 0L) stop("Unknown --samples IDs: ", paste(unknown, collapse = ", "))
  slides <- Filter(function(slide) slide$sample_id %in% options$samples, slides)
}
if (length(slides) == 0L) stop("No Kuppe slides selected for execution")

old_plan <- future::plan()
on.exit(future::plan(old_plan), add = TRUE)
if (options$threads == 1L) {
  future::plan(future::sequential)
} else {
  future::plan(future::multisession, workers = options$threads)
}
base::options(future.rng.onMisuse = "error")

# MISTy cache paths are relative to the working directory; keep them under the
# declared output root so cache ownership and cleanup are explicit.
old_working_directory <- getwd()
on.exit(setwd(old_working_directory), add = TRUE)
setwd(output_root)

sample_manifests <- lapply(
  slides,
  run_slide,
  input_root = input_root,
  output_root = output_root,
  options = options
)
statuses <- vapply(sample_manifests, function(manifest) manifest$status, character(1L))
completed <- sample_manifests[statuses == "complete"]

combined_artifacts <- list()
if (length(completed) > 0L) {
  importance_tables <- lapply(completed, function(manifest) {
    readr::read_tsv(
      file.path(output_root, manifest$sample_id, manifest$outputs$importances$filename),
      show_col_types = FALSE,
      progress = FALSE
    )
  })
  performance_tables <- lapply(completed, function(manifest) {
    readr::read_tsv(
      file.path(output_root, manifest$sample_id, manifest$outputs$performance$filename),
      show_col_types = FALSE,
      progress = FALSE
    )
  })
  combined_artifacts$importances <- write_tsv_gz(
    do.call(rbind, importance_tables),
    file.path(output_root, "kuppe_misty_sample_importances.tsv.gz")
  )
  combined_artifacts$performance <- write_tsv_gz(
    do.call(rbind, performance_tables),
    file.path(output_root, "kuppe_misty_sample_performance.tsv.gz")
  )
}

run_manifest <- list(
  schema_version = RUN_SCHEMA,
  dataset_id = DATASET_ID,
  status = if (all(statuses == "complete")) "complete" else "partial_failure",
  recomputation_status = RECOMPUTATION_STATUS,
  input_manifest = list(
    filename = basename(input_manifest_path),
    sha256 = sha256_file(input_manifest_path),
    payload_sha256 = input_manifest$manifest_payload_sha256
  ),
  software = list(
    R = as.character(getRversion()),
    mistyR = as.character(utils::packageVersion("mistyR")),
    mistyR_tag_commit = MISTYR_TAG_COMMIT,
    mistyR_installed_remote_sha = mistyr_remote_sha
  ),
  hardware = list(
    platform = R.version$platform,
    machine = unname(Sys.info()[["machine"]]),
    logical_cores = parallel::detectCores(logical = TRUE),
    requested_future_workers = options$threads,
    ranger_threads_per_worker = 1L
  ),
  options = list(
    seed = options$seed,
    cv_folds = options$cv_folds,
    cache = options$cache,
    resume = options$resume,
    selected_samples = options$samples
  ),
  samples = lapply(sample_manifests, function(manifest) {
    list(
      sample_id = manifest$sample_id,
      condition = manifest$condition,
      status = manifest$status,
      manifest = file.path(manifest$sample_id, "sample.manifest.json"),
      error = manifest$error
    )
  }),
  outputs = combined_artifacts,
  completed_at = format(Sys.time(), tz = "UTC", usetz = TRUE)
)
run_manifest_path <- file.path(output_root, "kuppe_spatial_misty_run.manifest.json")
write_json_atomic(run_manifest, run_manifest_path)

if (any(statuses != "complete")) {
  failed_ids <- vapply(
    sample_manifests[statuses != "complete"],
    function(manifest) manifest$sample_id,
    character(1L)
  )
  message("Failed slides: ", paste(failed_ids, collapse = ", "))
  quit(save = "no", status = 1L)
}
message("Completed ", length(completed), " Kuppe spatial MISTy slide(s).")
