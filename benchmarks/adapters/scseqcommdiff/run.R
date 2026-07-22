#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 13) {
  stop("expected 13 arguments")
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
resource_id <- args[[13]]

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
# Matrix::dgCMatrix requires a double-valued x slot. Raw count H5ADs commonly
# store integer sparse values, so normalize only the storage dtype at the
# Python/R boundary without changing any count values.
cell_gene <- py_to_r(adata$X$astype("float64")$tocsc())
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
if (
  nrow(resource) < 1L ||
    !all(c("ligand", "receptor") %in% names(resource)) ||
    anyDuplicated(resource[c("ligand", "receptor")])
) {
  stop("validated ligand-receptor resource is empty, incomplete, or duplicated")
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

event_scores <- differential |>
  group_by(ligand, receptor, LR_pair, cluster_L, cluster_R) |>
  summarise(
    score_target = first(.data[[target_score]]),
    score_reference = first(.data[[reference_score]]),
    logFC = first(logFC_S_inter),
    p_value = {
      finite_p <- .data[[p_column]][is.finite(.data[[p_column]])]
      if (length(finite_p)) first(finite_p) else NA_real_
    },
    native_rows_collapsed = n(),
    .groups = "drop"
  )
event_scores$effect <- event_scores$score_target - event_scores$score_reference
event_scores$status <- ifelse(
  is.finite(event_scores$score_target) &
    is.finite(event_scores$score_reference) &
    is.finite(event_scores$effect),
  "observed",
  "not_estimable"
)
event_scores$reason_code <- ifelse(
  event_scores$status == "observed", "", "non_finite_native_intercellular_score"
)
event_scores$target <- target
event_scores$reference <- reference
event_scores$native_p_column <- p_column
event_connection <- gzfile(
  file.path(output_dir, "differential_event_scores.tsv.gz"), "wt"
)
write.table(
  event_scores,
  event_connection,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
close(event_connection)

selected <- differential |>
  group_by(ligand, receptor, LR_pair, cluster_L, cluster_R, interaction) |>
  summarise(
    score_target = first(.data[[target_score]]),
    score_reference = first(.data[[reference_score]]),
    logFC = first(logFC_S_inter),
    p_value = {
      finite_p <- .data[[p_column]][is.finite(.data[[p_column]])]
      if (length(finite_p)) first(finite_p) else NA_real_
    },
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
  transmute(
    condition = as.character(condition),
    sender = as.character(sender_unordered),
    receiver = as.character(receiver_unordered)
  ) |>
  count(condition, sender, receiver, name = "ranked_strength")
pair_support <- differential |>
  mutate(
    sender = ifelse(cluster_L <= cluster_R, cluster_L, cluster_R),
    receiver = ifelse(cluster_L <= cluster_R, cluster_R, cluster_L),
    finite_native_p = is.finite(.data[[p_column]]),
    native_interaction_id = paste(cluster_L, cluster_R, LR_pair, sep = "|")
  ) |>
  group_by(sender, receiver) |>
  summarise(
    native_result_rows = n(),
    finite_native_p_rows = sum(finite_native_p),
    finite_native_interactions = n_distinct(
      native_interaction_id[finite_native_p]
    ),
    pair_native_tested = any(finite_native_p),
    .groups = "drop"
  )
rankings <- left_join(
  universe,
  counts,
  by = c("condition", "sender", "receiver")
) |>
  left_join(pair_support, by = c("sender", "receiver"))
rankings$native_result_rows[is.na(rankings$native_result_rows)] <- 0L
rankings$finite_native_p_rows[is.na(rankings$finite_native_p_rows)] <- 0L
rankings$finite_native_interactions[
  is.na(rankings$finite_native_interactions)
] <- 0L
rankings$pair_native_tested[is.na(rankings$pair_native_tested)] <- FALSE
rankings$cell_type_eligible <- (
  rankings$sender %in% eligible_cell_types &
    rankings$receiver %in% eligible_cell_types
)
rankings$pair_eligible <- (
  rankings$cell_type_eligible & rankings$pair_native_tested
)
paper_rankings <- rankings
paper_rankings$ranked_strength[
  paper_rankings$cell_type_eligible & is.na(paper_rankings$ranked_strength)
] <- 0
paper_rankings$dataset <- dataset_id
paper_rankings$method <- "scseqcommdiff"
paper_rankings$method_version <- "2.0.0"
paper_rankings$resource <- resource_id
paper_rankings$ranking_semantics <- paste0(
  "cardinality_of_native_significant_directed_lr_after_unordered_cell_pair_collapse;",
  ifelse(scenario == "multi-condition", "BH_q<0.05", "raw_p<0.05"),
  ";max_S_intra>0.5_or_all_NA;paper_zero_completed_cell_type_eligible_universe"
)
paper_rankings$status <- ifelse(
  paper_rankings$cell_type_eligible, "observed", "not_estimable"
)
paper_rankings$reason_code <- ifelse(
  paper_rankings$cell_type_eligible,
  "",
  ifelse(
    scenario == "multi-condition",
    "cell_type_not_eligible_for_native_condition_permutation",
    "cell_type_not_eligible_for_native_multisample"
  )
)
paper_rankings <- paper_rankings[c(
  "dataset", "method", "method_version", "resource", "ranking_semantics",
  "condition", "sender", "receiver", "ranked_strength", "status", "reason_code"
)]
write.table(
  paper_rankings,
  file.path(
    output_dir,
    "condition_cell_pair_rankings_paper_zero_completed.tsv"
  ),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
rankings$ranked_strength[
  rankings$pair_eligible & is.na(rankings$ranked_strength)
] <- 0
rankings$dataset <- dataset_id
rankings$method <- "scseqcommdiff"
rankings$method_version <- "2.0.0"
rankings$resource <- resource_id
rankings$ranking_semantics <- paste0(
  "cardinality_of_native_significant_directed_lr_after_unordered_cell_pair_collapse;",
  ifelse(scenario == "multi-condition", "BH_q<0.05", "raw_p<0.05"),
  ";max_S_intra>0.5_or_all_NA;pair_level_finite_native_p_estimability"
)
rankings$status <- ifelse(rankings$pair_eligible, "observed", "not_estimable")
rankings$reason_code <- ifelse(
  rankings$pair_eligible,
  "",
  ifelse(
    !rankings$cell_type_eligible,
    ifelse(
      scenario == "multi-condition",
      "cell_type_not_eligible_for_native_condition_permutation",
      "cell_type_not_eligible_for_native_multisample"
    ),
    "no_finite_native_intercellular_test_for_cell_pair"
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

pair_audit <- left_join(pairs, pair_support, by = c("sender", "receiver"))
pair_audit$native_result_rows[is.na(pair_audit$native_result_rows)] <- 0L
pair_audit$finite_native_p_rows[is.na(pair_audit$finite_native_p_rows)] <- 0L
pair_audit$finite_native_interactions[
  is.na(pair_audit$finite_native_interactions)
] <- 0L
pair_audit$pair_native_tested[is.na(pair_audit$pair_native_tested)] <- FALSE
pair_audit$cell_type_eligible <- (
  pair_audit$sender %in% eligible_cell_types &
    pair_audit$receiver %in% eligible_cell_types
)
pair_audit$status <- ifelse(
  pair_audit$cell_type_eligible & pair_audit$pair_native_tested,
  "observed",
  "not_estimable"
)
pair_audit$reason_code <- ifelse(
  pair_audit$status == "observed",
  "",
  ifelse(
    !pair_audit$cell_type_eligible,
    "cell_type_not_eligible",
    "no_finite_native_intercellular_test_for_cell_pair"
  )
)
pair_audit$dataset <- dataset_id
pair_audit$scenario <- scenario
pair_audit <- pair_audit[c(
  "dataset", "scenario", "sender", "receiver", "native_result_rows",
  "finite_native_p_rows", "finite_native_interactions",
  "pair_native_tested", "cell_type_eligible", "status", "reason_code"
)]
write.table(
  pair_audit,
  file.path(output_dir, "cell_pair_native_test_support.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
writeLines(capture.output(sessionInfo()), file.path(output_dir, "session_info.txt"))
