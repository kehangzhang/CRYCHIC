#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 5) {
  stop("expected differential.rds, rankings.tsv, output_dir, scenario, dataset_id")
}

differential_path <- args[[1]]
ranking_path <- args[[2]]
output_dir <- args[[3]]
scenario <- args[[4]]
dataset_id <- args[[5]]

suppressPackageStartupMessages(library(dplyr))

if (!scenario %in% c("multi-condition", "multi-sample")) {
  stop("unsupported scenario")
}
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
differential <- readRDS(differential_path)
rankings <- read.delim(
  ranking_path,
  check.names = FALSE,
  stringsAsFactors = FALSE,
  na.strings = c("", "NA")
)
p_column <- if (scenario == "multi-condition") {
  "pvalue_adj_S_inter"
} else {
  "pvalue_S_inter"
}
required_differential <- c(
  "LR_pair", "cluster_L", "cluster_R", p_column
)
if (length(setdiff(required_differential, names(differential)))) {
  stop("differential result lacks the native p-value columns")
}
required_rankings <- c(
  "dataset", "condition", "sender", "receiver", "ranked_strength",
  "status", "reason_code"
)
if (length(setdiff(required_rankings, names(rankings)))) {
  stop("ranking table lacks the frozen output columns")
}

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

pair_base <- rankings |>
  group_by(sender, receiver) |>
  summarise(
    cell_type_eligible = all(status == "observed"),
    .groups = "drop"
  )
pair_universe <- distinct(rankings, dataset, sender, receiver) |>
  left_join(pair_base, by = c("sender", "receiver")) |>
  left_join(pair_support, by = c("sender", "receiver"))
for (column in c(
  "native_result_rows", "finite_native_p_rows", "finite_native_interactions"
)) {
  pair_universe[[column]][is.na(pair_universe[[column]])] <- 0L
}
pair_universe$pair_native_tested[
  is.na(pair_universe$pair_native_tested)
] <- FALSE
pair_universe$cell_type_eligible[
  is.na(pair_universe$cell_type_eligible)
] <- FALSE
pair_universe$status <- ifelse(
  pair_universe$cell_type_eligible & pair_universe$pair_native_tested,
  "observed",
  "not_estimable"
)
pair_universe$reason_code <- ifelse(
  pair_universe$status == "observed",
  "",
  ifelse(
    !pair_universe$cell_type_eligible,
    "cell_type_not_eligible_for_native_analysis",
    "no_finite_native_intercellular_test_for_cell_pair"
  )
)

paper <- rankings |>
  select(-any_of(c(
    "native_result_rows", "finite_native_p_rows",
    "finite_native_interactions", "pair_native_tested",
    "cell_type_eligible"
  ))) |>
  left_join(
    select(pair_universe, sender, receiver, cell_type_eligible),
    by = c("sender", "receiver")
  )
paper$ranked_strength[
  paper$cell_type_eligible & is.na(paper$ranked_strength)
] <- 0
paper$status <- ifelse(paper$cell_type_eligible, "observed", "not_estimable")
paper$reason_code <- ifelse(
  paper$cell_type_eligible,
  "",
  "cell_type_not_eligible_for_native_analysis"
)
paper$ranking_semantics <- paste0(
  "cardinality_of_native_significant_directed_lr_after_unordered_cell_pair_collapse;",
  ifelse(scenario == "multi-condition", "BH_q<0.05", "raw_p<0.05"),
  ";max_S_intra>0.5_or_all_NA;paper_zero_completed_cell_type_eligible_universe"
)
paper <- select(
  paper,
  dataset, method, method_version, resource, ranking_semantics, condition,
  sender, receiver, ranked_strength, status, reason_code
)

updated <- rankings |>
  select(-any_of(c(
    "native_result_rows", "finite_native_p_rows",
    "finite_native_interactions", "pair_native_tested",
    "cell_type_eligible"
  ))) |>
  left_join(
    select(
      pair_universe,
      sender, receiver, native_result_rows, finite_native_p_rows,
      finite_native_interactions, pair_native_tested, cell_type_eligible,
      pair_status = status, pair_reason_code = reason_code
    ),
    by = c("sender", "receiver")
  )
updated$ranked_strength[
  updated$pair_status != "observed"
] <- NA_real_
observed_missing <- updated$pair_status == "observed" &
  is.na(updated$ranked_strength)
updated$ranked_strength[observed_missing] <- 0
updated$status <- updated$pair_status
updated$reason_code <- updated$pair_reason_code
updated$ranking_semantics <- paste0(
  "cardinality_of_native_significant_directed_lr_after_unordered_cell_pair_collapse;",
  ifelse(scenario == "multi-condition", "BH_q<0.05", "raw_p<0.05"),
  ";max_S_intra>0.5_or_all_NA;pair_level_finite_native_p_estimability"
)
updated <- select(
  updated,
  dataset, method, method_version, resource, ranking_semantics, condition,
  sender, receiver, ranked_strength, status, reason_code
)

pair_universe$dataset <- dataset_id
pair_universe$scenario <- scenario
pair_universe <- select(
  pair_universe,
  dataset, scenario, sender, receiver, native_result_rows,
  finite_native_p_rows, finite_native_interactions, pair_native_tested,
  cell_type_eligible, status, reason_code
)
write.table(
  updated,
  file.path(output_dir, "condition_cell_pair_rankings.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
write.table(
  paper,
  file.path(
    output_dir,
    "condition_cell_pair_rankings_paper_zero_completed.tsv"
  ),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
write.table(
  pair_universe,
  file.path(output_dir, "cell_pair_native_test_support.tsv"),
  sep = "\t",
  quote = FALSE,
  row.names = FALSE
)
