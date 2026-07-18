#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 12) {
  stop("expected 12 arguments")
}

counts_path <- args[[1]]
genes_path <- args[[2]]
cells_path <- args[[3]]
metadata_path <- args[[4]]
lri_path <- args[[5]]
output_dir <- args[[6]]
dataset_id <- args[[7]]
target <- args[[8]]
reference <- args[[9]]
cores <- as.integer(args[[10]])
iterations <- as.integer(args[[11]])
seed <- as.integer(args[[12]])

suppressPackageStartupMessages(library(scDiffCom))
suppressPackageStartupMessages(library(Seurat))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(data.table))
suppressPackageStartupMessages(library(future))

if (as.character(packageVersion("scDiffCom")) != "1.1.1") {
  stop("scDiffCom 1.1.1 is required")
}
if (cores < 1L || cores > 8L) {
  stop("cores must be between 1 and 8")
}
if (iterations < 1L) {
  stop("iterations must be positive")
}

options(future.globals.maxSize = 8 * 1024^3)
data.table::setDTthreads(1L)
if (cores == 1L) {
  future::plan(future::sequential)
} else if (.Platform$OS.type == "unix") {
  future::plan(future::multicore, workers = cores)
} else {
  future::plan(future::multisession, workers = cores)
}

install_seurat5_getassaydata_bridge <- function() {
  if (packageVersion("SeuratObject") < "5.0.0") {
    return(invisible(FALSE))
  }
  seurat_namespace <- asNamespace("SeuratObject")
  original_get_assay_data <- get("GetAssayData.Seurat", envir = seurat_namespace)
  bridge <- function(object, assay = NULL, slot = NULL, layer = NULL, ...) {
    if (is.null(layer) && !is.null(slot)) {
      layer <- slot
    }
    original_get_assay_data(
      object = object,
      assay = assay,
      layer = layer,
      ...
    )
  }
  unlockBinding("GetAssayData.Seurat", seurat_namespace)
  assign("GetAssayData.Seurat", bridge, envir = seurat_namespace)
  lockBinding("GetAssayData.Seurat", seurat_namespace)
  registerS3method(
    "GetAssayData",
    "Seurat",
    bridge,
    envir = seurat_namespace
  )
  invisible(TRUE)
}
install_seurat5_getassaydata_bridge()

genes <- readLines(genes_path, warn = FALSE)
cells <- readLines(cells_path, warn = FALSE)
counts <- Matrix::readMM(counts_path)
counts <- methods::as(counts, "CsparseMatrix")
if (!identical(dim(counts), c(length(genes), length(cells)))) {
  stop("count matrix dimensions do not match genes and cells")
}
rownames(counts) <- genes
colnames(counts) <- cells

metadata <- data.table::fread(metadata_path, sep = "\t", data.table = TRUE)
if (!identical(names(metadata), c("cell_id", "cell_type", "condition"))) {
  stop("metadata schema mismatch")
}
if (!setequal(metadata$cell_id, cells)) {
  stop("metadata cell identifiers do not match the count matrix")
}
metadata <- metadata[match(cells, cell_id)]
if (!identical(metadata$cell_id, cells)) {
  stop("metadata could not be aligned to count matrix columns")
}
meta_frame <- as.data.frame(metadata[, .(cell_type, condition)])
rownames(meta_frame) <- metadata$cell_id

lri <- data.table::fread(
  lri_path,
  sep = "\t",
  na.strings = c("NA", ""),
  data.table = TRUE
)
required_lri <- c(
  "LRI", "LIGAND_1", "LIGAND_2", "RECEPTOR_1", "RECEPTOR_2", "RECEPTOR_3"
)
if (!identical(names(lri), required_lri)) {
  stop("custom LRI schema mismatch")
}
for (column in required_lri) {
  data.table::set(lri, j = column, value = as.character(lri[[column]]))
}

seurat_object <- Seurat::CreateSeuratObject(
  counts = counts,
  assay = "RNA",
  meta.data = meta_frame,
  min.cells = 0,
  min.features = 0
)
seurat_object <- Seurat::NormalizeData(
  seurat_object,
  normalization.method = "LogNormalize",
  scale.factor = 10000,
  verbose = FALSE
)
rm(counts)
invisible(gc())

scdiffcom_object <- scDiffCom::run_interaction_analysis(
  seurat_object = seurat_object,
  LRI_species = "custom",
  seurat_celltype_id = "cell_type",
  seurat_condition_id = list(
    column_name = "condition",
    cond1_name = reference,
    cond2_name = target
  ),
  iterations = iterations,
  threshold_p_value_de = 0.05,
  threshold_logfc = log(1.5),
  seed = seed,
  custom_LRI_tables = list(LRI = lri)
)

saveRDS(
  scdiffcom_object,
  file.path(output_dir, "scdiffcom_result.rds"),
  compress = "gzip"
)
detected <- data.table::as.data.table(
  scDiffCom::GetTableCCI(
    scdiffcom_object,
    type = "detected",
    simplified = FALSE
  )
)
required_detected <- c(
  "EMITTER_CELLTYPE", "RECEIVER_CELLTYPE", "LRI", "REGULATION",
  "LOGFC", "BH_P_VALUE_DE"
)
missing_detected <- setdiff(required_detected, names(detected))
if (length(missing_detected)) {
  stop(paste("detected CCI columns are missing:", paste(missing_detected, collapse = ",")))
}

write_tsv_gz <- function(table, path) {
  connection <- gzfile(path, "wt")
  on.exit(close(connection), add = TRUE)
  write.table(
    table,
    connection,
    sep = "\t",
    quote = FALSE,
    row.names = FALSE,
    na = ""
  )
}

write_tsv_gz(detected, file.path(output_dir, "detected_calls.tsv.gz"))
condition_specific <- data.table::copy(detected[REGULATION %in% c("UP", "DOWN")])
condition_specific[, condition := data.table::fifelse(REGULATION == "UP", target, reference)]
condition_specific[, sender := EMITTER_CELLTYPE]
condition_specific[, receiver := RECEIVER_CELLTYPE]
data.table::setcolorder(
  condition_specific,
  c("condition", "sender", "receiver", setdiff(names(condition_specific), c("condition", "sender", "receiver")))
)
write_tsv_gz(
  condition_specific,
  file.path(output_dir, "condition_specific_calls.tsv.gz")
)

raw_calls <- data.table::as.data.table(
  scDiffCom::GetTableCCI(scdiffcom_object, type = "raw", simplified = FALSE)
)
analysis_cell_types <- sort(unique(c(raw_calls$EMITTER_CELLTYPE, raw_calls$RECEIVER_CELLTYPE)))
write.table(
  data.frame(cell_type = analysis_cell_types),
  file.path(output_dir, "analysis_cell_types.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))
