#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 12) {
  stop("expected 12 arguments")
}

input_h5ad <- args[[1]]
metadata_path <- args[[2]]
resource_path <- args[[3]]
output_dir <- args[[4]]
scenario <- args[[5]]
target <- args[[6]]
reference <- args[[7]]
dataset_id <- args[[8]]
cores <- as.integer(args[[9]])
nrep <- as.integer(args[[10]])
min_cells <- as.integer(args[[11]])
support_path <- args[[12]]

suppressPackageStartupMessages(library(scSeqComm))
suppressPackageStartupMessages(library(doRNG))
suppressPackageStartupMessages(library(Matrix))
suppressPackageStartupMessages(library(dplyr))
suppressPackageStartupMessages(library(reticulate))

if (as.character(packageVersion("scSeqComm")) != "2.0.0") {
  stop("scSeqComm 2.0.0 is required")
}
if (.Platform$OS.type != "unix") {
  stop("the memory-preserving doMC benchmark runner requires Unix")
}

metadata <- read.delim(metadata_path, check.names = FALSE, stringsAsFactors = FALSE)
required_metadata <- c("Cell_ID", "Cluster_ID", "Condition_ID", "Sample_ID")
if (!identical(names(metadata), required_metadata)) {
  stop("metadata schema mismatch")
}
support <- read.delim(support_path, check.names = FALSE, stringsAsFactors = FALSE)
required_support <- c(
  "cell_type", "condition", "sample_id", "n_cells", "valid_pseudobulk",
  "valid_units_in_condition", "analysis_eligible", "reason_code"
)
if (!identical(names(support), required_support)) {
  stop("cell-type support schema mismatch")
}
parse_boolean_column <- function(values, field) {
  text <- tolower(as.character(values))
  if (any(!text %in% c("true", "false"))) {
    stop(paste(field, "must contain only true or false"))
  }
  text == "true"
}
support$valid_pseudobulk <- parse_boolean_column(
  support$valid_pseudobulk,
  "valid_pseudobulk"
)
support$analysis_eligible <- parse_boolean_column(
  support$analysis_eligible,
  "analysis_eligible"
)
support$reason_code[is.na(support$reason_code)] <- ""
eligibility <- unique(support[c("cell_type", "analysis_eligible", "reason_code")])
if (anyDuplicated(eligibility$cell_type)) {
  stop("cell-type eligibility must be unique")
}
eligible_cell_types <- eligibility$cell_type[eligibility$analysis_eligible]
if (!setequal(unique(metadata$Cluster_ID), eligible_cell_types)) {
  stop("analysis metadata does not match eligible cell types")
}

anndata <- import("anndata", convert = FALSE)
adata <- anndata$read_h5ad(input_h5ad)
cell_gene <- py_to_r(adata$X$tocsc())
gene_expr <- as(Matrix::t(cell_gene), "dgCMatrix")
rownames(gene_expr) <- py_to_r(adata$var_names$to_list())
colnames(gene_expr) <- py_to_r(adata$obs_names$to_list())
rm(cell_gene, adata)
invisible(gc())
cell_index <- match(metadata$Cell_ID, colnames(gene_expr))
if (anyNA(cell_index)) {
  stop("analysis metadata contains cells absent from the h5ad")
}
gene_expr <- gene_expr[, cell_index, drop = FALSE]
if (!identical(colnames(gene_expr), metadata$Cell_ID)) {
  stop("h5ad cell order does not match the analysis metadata")
}

resource <- read.delim(resource_path, check.names = FALSE, stringsAsFactors = FALSE)
if (nrow(resource) != 2293L || anyDuplicated(resource[c("ligand", "receptor")])) {
  stop("ConnectomeDB2020 resource mismatch")
}
LR_db <- resource[c("ligand", "receptor")]
data(TF_TG_TRRUSTv2_HTRIdb_RegNetwork_High)
data(TF_PPR_KEGG_human)

DEmethod <- if (scenario == "multi-condition") "wilcoxon" else "pseudo.wilcoxon"
set.seed(20260717)
result <- scSeqComm_differential(
  gene_expr = gene_expr,
  cell_metadata = metadata,
  scenario = scenario,
  cond_names = c(target, reference),
  inter_signaling = TRUE,
  LR_pairs_DB = LR_db,
  inter_scores = "scSeqComm",
  Nrep = nrep,
  alternative_inter = "two.sided",
  padj_method = "BH",
  intra_signaling = TRUE,
  TF_reg_DB = TF_TG_TRRUSTv2_HTRIdb_RegNetwork_High,
  DEmethod = DEmethod,
  only.pos = FALSE,
  aggregation_method = "mean",
  R_TF_association = TF_PPR_KEGG_human,
  N_cores = cores,
  backend = "doMC",
  bigmatrix = FALSE,
  min_cells = min_cells,
  count_thr = 1
)

differential <- result$differential_comm
saveRDS(differential, file.path(output_dir, "differential_comm.rds"), compress = "gzip")

target_score <- paste0("S_inter_", target)
reference_score <- paste0("S_inter_", reference)
p_column <- if (scenario == "multi-condition") {
  "pvalue_adj_S_inter"
} else {
  "pvalue_S_inter"
}
required_result <- c(
  "ligand", "receptor", "LR_pair", "cluster_L", "cluster_R", "interaction",
  target_score, reference_score, "logFC_S_inter", p_column, "S_intra"
)
missing_result <- setdiff(required_result, names(differential))
if (length(missing_result)) {
  stop(paste("result columns are missing:", paste(missing_result, collapse = ",")))
}

selected <- differential |>
  group_by(ligand, receptor, LR_pair, cluster_L, cluster_R, interaction) |>
  summarise(
    score_target = first(.data[[target_score]]),
    score_reference = first(.data[[reference_score]]),
    logFC = first(logFC_S_inter),
    p_value = first(.data[[p_column]]),
    max_S_intra = if (all(is.na(S_intra))) NA_real_ else max(S_intra, na.rm = TRUE),
    .groups = "drop"
  ) |>
  filter(is.finite(p_value), p_value < 0.05, is.na(max_S_intra) | max_S_intra > 0.5) |>
  mutate(condition = case_when(logFC > 0 ~ target, logFC < 0 ~ reference)) |>
  filter(!is.na(condition))

selected$sender_unordered <- ifelse(
  selected$cluster_L <= selected$cluster_R, selected$cluster_L, selected$cluster_R
)
selected$receiver_unordered <- ifelse(
  selected$cluster_L <= selected$cluster_R, selected$cluster_R, selected$cluster_L
)
selected_connection <- gzfile(
  file.path(output_dir, "selected_differential_interactions.tsv.gz"), "wt"
)
write.table(selected, selected_connection, sep = "\t", quote = FALSE, row.names = FALSE)
close(selected_connection)

cell_types <- sort(unique(eligibility$cell_type), method = "radix")
pairs <- do.call(
  rbind,
  lapply(seq_along(cell_types), function(i) {
    data.frame(
      sender = cell_types[[i]],
      receiver = cell_types[i:length(cell_types)],
      stringsAsFactors = FALSE
    )
  })
)
universe <- do.call(
  rbind,
  lapply(c(target, reference), function(condition_name) {
    transform(pairs, condition = condition_name)
  })
)
universe <- universe[c("condition", "sender", "receiver")]
counts <- selected |>
  count(condition, sender_unordered, receiver_unordered, name = "ranked_strength") |>
  rename(sender = sender_unordered, receiver = receiver_unordered)
rankings <- left_join(universe, counts, by = c("condition", "sender", "receiver"))
rankings$pair_eligible <- (
  rankings$sender %in% eligible_cell_types &
    rankings$receiver %in% eligible_cell_types
)
rankings$ranked_strength[
  rankings$pair_eligible & is.na(rankings$ranked_strength)
] <- 0
rankings$dataset <- dataset_id
rankings$method <- "scseqcommdiff"
rankings$method_version <- "2.0.0"
rankings$resource <- "ConnectomeDB2020_Hou_2020_human"
rankings$ranking_semantics <- paste0(
  "cardinality_of_native_significant_directed_lr_after_unordered_cell_pair_collapse;",
  ifelse(scenario == "multi-condition", "BH_q<0.05", "raw_p<0.05"),
  ";max_S_intra>0.5_or_all_NA;common_native_pseudobulk_estimability"
)
rankings$status <- ifelse(rankings$pair_eligible, "observed", "not_estimable")
rankings$reason_code <- ifelse(
  rankings$pair_eligible,
  "",
  ifelse(
    scenario == "multi-condition",
    "cell_type_not_eligible_for_native_condition_permutation",
    "cell_type_not_eligible_for_native_multisample"
  )
)
rankings <- rankings[c(
  "dataset", "method", "method_version", "resource", "ranking_semantics",
  "condition", "sender", "receiver", "ranked_strength", "status", "reason_code"
)]
write.table(
  rankings,
  file.path(output_dir, "condition_cell_pair_rankings.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))
