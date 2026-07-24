#!/usr/bin/env Rscript

# Reproduce the released scACCorDiON.su stage-adjusted TCGA-PAAD validation.

arguments <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag) {
  index <- match(flag, arguments)
  if (is.na(index) || index == length(arguments)) stop(paste("missing", flag))
  arguments[[index + 1L]]
}

source_root <- arg_value("--source-root")
differential_path <- arg_value("--differential")
output <- arg_value("--output")

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(tibble)
  library(survival)
  library(broom)
})

load(file.path(source_root, "data", "pdac_cci_data.rda"))
load(file.path(source_root, "data", "paad_tcga_clinical_data.rda"))
load(file.path(source_root, "data", "paad_tcga_expression_data.rda"))
load(file.path(source_root, "data", "comparison_of_interest.rda"))

lrsankey <- read.csv(differential_path, check.names = FALSE)
lrsankey <- lrsankey[, !is.na(names(lrsankey)) & nzchar(names(lrsankey)), drop = FALSE]
pairs <- lrsankey |>
  filter(grepl("Ductal cell type 2", source)) |>
  filter(grepl("Ductal cell type 1", target)) |>
  filter(!grepl("HLA-", gene_A)) |>
  mutate(ligand = gsub("\\|L", "", gene_A)) |>
  mutate(receptor = gsub("\\|R", "", gene_B)) |>
  filter(ligand %in% paad_tcga_expression_data$symbol) |>
  filter(receptor %in% paad_tcga_expression_data$symbol) |>
  slice_max(n = 10, order_by = LRScore, with_ties = FALSE) |>
  select(ligand, receptor, LRScore)

if (nrow(pairs) != 10L) stop("paper protocol did not select exactly ten LR pairs")
genes <- unique(c(pairs$ligand, pairs$receptor))
expression <- paad_tcga_expression_data |>
  filter(symbol %in% genes) |>
  column_to_rownames("symbol") |>
  t() |>
  as.data.frame() |>
  rownames_to_column("sample_id") |>
  as_tibble() |>
  mutate(across(where(is.numeric), function(x) log2(x + 1)))
model_data <- paad_tcga_clinical_data |> left_join(expression, by = "sample_id")

pair_features <- character()
for (index in seq_len(nrow(pairs))) {
  ligand <- pairs$ligand[[index]]
  receptor <- pairs$receptor[[index]]
  feature <- paste0(ligand, "_", receptor)
  model_data[[feature]] <- sqrt(model_data[[ligand]] * model_data[[receptor]])
  pair_features <- c(pair_features, feature)
}

fit_mode <- function(mode, features) {
  formula <- reformulate(
    termlabels = c("stage", features),
    response = "survival::Surv(time=time, event=event)"
  )
  fit <- coxph(formula, data = model_data, x = TRUE, model = TRUE)
  tidy <- broom::tidy(fit, exponentiate = TRUE, conf.int = TRUE)
  tidy$mode <- mode
  tidy$is_interaction_term <- tidy$term %in% features
  tidy$q.value.within.mode <- NA_real_
  selected <- which(tidy$is_interaction_term)
  tidy$q.value.within.mode[selected] <- p.adjust(tidy$p.value[selected], method = "BH")
  diagnostics <- data.frame(
    mode = mode,
    subjects = fit$n,
    events = fit$nevent,
    concordance = unname(summary(fit)$concordance[[1L]]),
    likelihood_ratio_statistic = unname(summary(fit)$logtest[[1L]]),
    likelihood_ratio_df = unname(summary(fit)$logtest[[2L]]),
    likelihood_ratio_p_value = unname(summary(fit)$logtest[[3L]])
  )
  proportional <- as.data.frame(cox.zph(fit)$table) |>
    rownames_to_column("term") |>
    mutate(mode = mode)
  list(tidy = tidy, diagnostics = diagnostics, proportional = proportional)
}

fits <- list(
  pair = fit_mode("ligand_receptor_geometric_mean", pair_features),
  ligand = fit_mode("ligand_expression", unique(pairs$ligand)),
  receptor = fit_mode("receptor_expression", unique(pairs$receptor))
)

dir.create(output, recursive = TRUE, showWarnings = FALSE)
write.table(pairs, file.path(output, "selected_lr_pairs.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(bind_rows(lapply(fits, `[[`, "tidy")), file.path(output, "cox_coefficients.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(bind_rows(lapply(fits, `[[`, "diagnostics")), file.path(output, "model_diagnostics.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(bind_rows(lapply(fits, `[[`, "proportional")), file.path(output, "proportional_hazards.tsv"), sep = "\t", row.names = FALSE, quote = FALSE)
write.table(
  data.frame(
    comparison_of_interest = comparison_of_interest,
    tcga_subjects = nrow(model_data),
    tcga_events = sum(model_data$event),
    selected_pairs = nrow(pairs),
    clinical_adjustment = "stage",
    pair_score = "geometric_mean_of_log2_expression"
  ),
  file.path(output, "run_summary.tsv"), sep = "\t", row.names = FALSE, quote = FALSE
)
