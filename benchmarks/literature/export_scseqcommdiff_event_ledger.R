#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(dplyr))

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4L) {
  stop(
    "usage: export_scseqcommdiff_event_ledger.R INPUT_RDS OUTPUT_TSV_GZ TARGET REFERENCE"
  )
}

input_path <- args[[1L]]
output_path <- args[[2L]]
target <- args[[3L]]
reference <- args[[4L]]

source <- readRDS(input_path)
target_score <- paste0("S_inter_", target)
reference_score <- paste0("S_inter_", reference)
required <- c(
  "ligand", "receptor", "cluster_L", "cluster_R", target_score,
  reference_score, "logFC_S_inter", "pvalue_S_inter", "S_intra"
)
missing <- setdiff(required, names(source))
if (length(missing)) {
  stop(paste("scSeqCommDiff result columns are missing:", paste(missing, collapse = ",")))
}

ledger <- source |>
  group_by(ligand, receptor, cluster_L, cluster_R) |>
  summarise(
    score_target = first(.data[[target_score]]),
    score_reference = first(.data[[reference_score]]),
    logFC = first(logFC_S_inter),
    p_value = {
      finite_p <- pvalue_S_inter[is.finite(pvalue_S_inter)]
      if (length(finite_p)) first(finite_p) else NA_real_
    },
    max_S_intra = if (all(is.na(S_intra))) NA_real_ else max(S_intra, na.rm = TRUE),
    native_rows_collapsed = n(),
    .groups = "drop"
  ) |>
  rename(sender = cluster_L, receiver = cluster_R)

ledger$effect_target_minus_reference <- ledger$score_target - ledger$score_reference
ledger$target_condition <- target
ledger$reference_condition <- reference
ledger$status <- ifelse(
  !is.na(ledger$logFC) & is.finite(ledger$p_value),
  "observed",
  "not_estimable"
)
ledger$reason_code <- ifelse(
  ledger$status == "observed", "", "missing_native_differential_statistic"
)

connection <- gzfile(output_path, "wt")
write.table(
  ledger,
  connection,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  na = "NA"
)
close(connection)
