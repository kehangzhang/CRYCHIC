#!/usr/bin/env Rscript

# Native STACCato worker for the under-100k condition/batch simulation.
# It uses the public STACCato implementation and records the paper's three
# 60-subject designs. The ASD-fitted coefficient tensor was not distributed,
# so this is a documented low-rank protocol reconstruction rather than a
# byte-identical run of the original random draws.

arguments <- commandArgs(trailingOnly = TRUE)

arg_value <- function(flag) {
  index <- match(flag, arguments)
  if (is.na(index) || index == length(arguments)) {
    stop(paste("missing required argument", flag))
  }
  arguments[[index + 1]]
}

seed <- as.integer(arg_value("--seed"))
output <- arg_value("--output")
source_path <- arg_value("--staccato-source")

if (!file.exists(source_path)) {
  stop(paste("STACCato source is missing:", source_path))
}

suppressPackageStartupMessages({
  library(rTensor)
})
source(source_path)

number_lr <- 300L
number_sender <- 2L
number_receiver <- 2L
number_events <- number_lr * number_sender * number_receiver
top_k <- 100L

outer3 <- function(lr_weights, sender_weights, receiver_weights) {
  array(
    outer(
      as.vector(outer(lr_weights, sender_weights)),
      receiver_weights
    ),
    dim = c(number_lr, number_sender, number_receiver)
  )
}

build_truth <- function() {
  lr_one <- c(rep(1, 100), rep(0, 200))
  lr_two <- c(rep(0, 100), rep(1, 100), rep(0, 100))
  lr_three <- c(rep(0, 200), rep(1, 100))
  intercept <- array(0.50, dim = c(number_lr, number_sender, number_receiver)) +
    0.03 * outer3(lr_three, c(1, 0.5), c(0.5, 1))
  disease <-
    0.20 * outer3(lr_one, c(1, 0.25), c(1, 0.20)) -
    0.18 * outer3(lr_two, c(0.30, 1), c(0.25, 1))
  batch <-
    0.16 * outer3(lr_two, c(1, 0.35), c(0.35, 1)) +
    0.12 * outer3(lr_three, c(0.20, 1), c(1, 0.30))
  list(intercept = intercept, disease = disease, batch = batch)
}

build_design <- function(name) {
  cells <- switch(
    name,
    balanced = list(c(15L, 15L), c(15L, 15L)),
    moderate = list(c(20L, 10L), c(10L, 20L)),
    extreme = list(c(30L, 5L), c(0L, 25L)),
    stop(paste("unknown design", name))
  )
  rows <- list()
  for (batch_index in seq_along(cells)) {
    counts <- cells[[batch_index]]
    rows[[length(rows) + 1L]] <- data.frame(
      disease = c(rep(0, counts[[1L]]), rep(1, counts[[2L]])),
      batch = batch_index - 1L
    )
  }
  design <- do.call(rbind, rows)
  if (nrow(design) != 60L || sum(design$disease) != 30L) {
    stop("paper design did not produce 60 subjects with 30 disease subjects")
  }
  design
}

simulate_tensor <- function(design, truth) {
  number_subjects <- nrow(design)
  result <- array(
    0,
    dim = c(number_subjects, number_lr, number_sender, number_receiver)
  )
  for (subject in seq_len(number_subjects)) {
    result[subject, , , ] <-
      truth$intercept +
      design$disease[[subject]] * truth$disease +
      design$batch[[subject]] * truth$batch
  }
  result + array(rnorm(length(result), mean = 0, sd = 0.15), dim = dim(result))
}

metric_row <- function(design_name, method, estimate, truth, elapsed_seconds, status) {
  truth_vector <- as.vector(truth)
  if (!identical(status, "complete") || length(estimate) != length(truth_vector)) {
    return(data.frame(
      seed = seed,
      design = design_name,
      method = method,
      status = status,
      mse = NA_real_,
      rmse = NA_real_,
      sign_accuracy_nonzero = NA_real_,
      opposite_direction_rate = NA_real_,
      top100_recovery = NA_real_,
      elapsed_seconds = elapsed_seconds
    ))
  }
  nonzero <- abs(truth_vector) > 1e-12
  signs_correct <- sign(estimate[nonzero]) == sign(truth_vector[nonzero])
  truth_top <- order(abs(truth_vector), decreasing = TRUE)[seq_len(top_k)]
  estimate_top <- order(abs(estimate), decreasing = TRUE)[seq_len(top_k)]
  data.frame(
    seed = seed,
    design = design_name,
    method = method,
    status = "complete",
    mse = mean((estimate - truth_vector)^2),
    rmse = sqrt(mean((estimate - truth_vector)^2)),
    sign_accuracy_nonzero = mean(signs_correct),
    opposite_direction_rate = mean(!signs_correct),
    top100_recovery = length(intersect(truth_top, estimate_top)) / top_k,
    elapsed_seconds = elapsed_seconds
  )
}

ols_effect <- function(tensor, covariates, disease_column) {
  response <- matrix(tensor, nrow = dim(tensor)[[1L]])
  coefficients <- qr.solve(covariates, response)
  as.vector(coefficients[disease_column, ])
}

native_staccato_effect <- function(tensor, covariates, disease_column) {
  started <- proc.time()[["elapsed"]]
  result <- tryCatch(
    {
      # capture.output prevents rTensor's progress output from contaminating logs.
      fitted <- capture.output(
        fit <- staccato(
          tsr = as.tensor(tensor),
          X_covar1 = covariates,
          lr.names = paste0("LR", seq_len(number_lr)),
          sender.names = paste0("Sender", seq_len(number_sender)),
          receiver.names = paste0("Receiver", seq_len(number_receiver)),
          core_shape = c(ncol(covariates), 2L, 2L, 2L)
        )
      )
      list(
        status = "complete",
        estimate = as.vector(fit$C_ts[disease_column, , , ]),
        elapsed_seconds = proc.time()[["elapsed"]] - started
      )
    },
    error = function(error) {
      list(
        status = paste0("failed:", conditionMessage(error)),
        estimate = numeric(0),
        elapsed_seconds = proc.time()[["elapsed"]] - started
      )
    }
  )
  result
}

set.seed(seed)
truth <- build_truth()
rows <- list()
for (design_name in c("balanced", "moderate", "extreme")) {
  design <- build_design(design_name)
  tensor <- simulate_tensor(design, truth)
  adjusted <- cbind(
    intercept = 1,
    disease = design$disease,
    batch = design$batch
  )
  disease_only <- cbind(intercept = 1, disease = design$disease)

  started <- proc.time()[["elapsed"]]
  estimate <- ols_effect(tensor, adjusted, disease_column = 2L)
  rows[[length(rows) + 1L]] <- metric_row(
    design_name,
    "eventwise_ols_adjusted",
    estimate,
    truth$disease,
    proc.time()[["elapsed"]] - started,
    "complete"
  )
  started <- proc.time()[["elapsed"]]
  estimate <- ols_effect(tensor, disease_only, disease_column = 2L)
  rows[[length(rows) + 1L]] <- metric_row(
    design_name,
    "eventwise_ols_disease_only",
    estimate,
    truth$disease,
    proc.time()[["elapsed"]] - started,
    "complete"
  )
  native_adjusted <- native_staccato_effect(tensor, adjusted, disease_column = 2L)
  rows[[length(rows) + 1L]] <- metric_row(
    design_name,
    "staccato_adjusted",
    native_adjusted$estimate,
    truth$disease,
    native_adjusted$elapsed_seconds,
    native_adjusted$status
  )
  native_disease_only <- native_staccato_effect(
    tensor,
    disease_only,
    disease_column = 2L
  )
  rows[[length(rows) + 1L]] <- metric_row(
    design_name,
    "staccato_disease_only",
    native_disease_only$estimate,
    truth$disease,
    native_disease_only$elapsed_seconds,
    native_disease_only$status
  )
  # CRYCHIC's public OOF contrast API requires every subject in every context.
  # These unpaired 60-subject paper designs do not satisfy that estimand, so the
  # correct result is NE rather than an invented zero or a pseudo-paired test.
  rows[[length(rows) + 1L]] <- metric_row(
    design_name,
    "crychic_common_functional_oof",
    numeric(0),
    truth$disease,
    0,
    "NE_unpaired_subject_context_design"
  )
}

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write.table(
  do.call(rbind, rows),
  file = output,
  sep = "\t",
  row.names = FALSE,
  quote = FALSE
)
