#!/usr/bin/env Rscript

# Model-stage residual-bootstrap diagnostics for the frozen STACCato simulator.
# These P values and intervals are explicitly not full upstream-pipeline inference.

arguments <- commandArgs(trailingOnly = TRUE)
arg_value <- function(flag) {
  index <- match(flag, arguments)
  if (is.na(index) || index == length(arguments)) stop(paste("missing", flag))
  arguments[[index + 1L]]
}

seed <- as.integer(arg_value("--seed"))
output <- arg_value("--output")
source_path <- arg_value("--staccato-source")
design_name <- arg_value("--design")
number_subjects <- as.integer(arg_value("--subjects"))
effect_multiplier <- as.numeric(arg_value("--effect-multiplier"))
number_bootstrap <- as.integer(arg_value("--bootstrap"))

suppressPackageStartupMessages(library(rTensor))
source(source_path)

number_lr <- 300L
number_sender <- 2L
number_receiver <- 2L
number_events <- number_lr * number_sender * number_receiver

outer3 <- function(lr_weights, sender_weights, receiver_weights) {
  array(
    outer(as.vector(outer(lr_weights, sender_weights)), receiver_weights),
    dim = c(number_lr, number_sender, number_receiver)
  )
}

build_truth <- function(multiplier) {
  lr_one <- c(rep(1, 100), rep(0, 200))
  lr_two <- c(rep(0, 100), rep(1, 100), rep(0, 100))
  lr_three <- c(rep(0, 200), rep(1, 100))
  disease_base <-
    0.20 * outer3(lr_one, c(1, 0.25), c(1, 0.20)) -
    0.18 * outer3(lr_two, c(0.30, 1), c(0.25, 1))
  list(
    intercept = array(0.50, dim = c(number_lr, number_sender, number_receiver)) +
      0.03 * outer3(lr_three, c(1, 0.5), c(0.5, 1)),
    disease = multiplier * disease_base,
    batch = 0.16 * outer3(lr_two, c(1, 0.35), c(0.35, 1)) +
      0.12 * outer3(lr_three, c(0.20, 1), c(1, 0.30))
  )
}

build_design <- function(name, n) {
  if (name == "balanced") {
    if (n %% 4L != 0L) stop("balanced subject count must be divisible by four")
    cells <- list(c(n / 4L, n / 4L), c(n / 4L, n / 4L))
  } else {
    if (n != 60L) stop("moderate/extreme designs are frozen at 60 subjects")
    cells <- switch(
      name,
      moderate = list(c(20L, 10L), c(10L, 20L)),
      extreme = list(c(30L, 5L), c(0L, 25L)),
      stop(paste("unknown design", name))
    )
  }
  rows <- list()
  for (batch_index in seq_along(cells)) {
    counts <- cells[[batch_index]]
    rows[[length(rows) + 1L]] <- data.frame(
      disease = c(rep(0, counts[[1L]]), rep(1, counts[[2L]])),
      batch = batch_index - 1L
    )
  }
  do.call(rbind, rows)
}

simulate_tensor <- function(design, truth) {
  result <- array(0, dim = c(nrow(design), number_lr, number_sender, number_receiver))
  for (subject in seq_len(nrow(design))) {
    result[subject, , , ] <- truth$intercept +
      design$disease[[subject]] * truth$disease +
      design$batch[[subject]] * truth$batch
  }
  result + array(rnorm(length(result), sd = 0.15), dim = dim(result))
}

fit_staccato <- function(tensor, covariates) {
  capture.output(
    fitted <- staccato(
      tsr = as.tensor(tensor),
      X_covar1 = covariates,
      lr.names = paste0("LR", seq_len(number_lr)),
      sender.names = paste0("Sender", seq_len(number_sender)),
      receiver.names = paste0("Receiver", seq_len(number_receiver)),
      core_shape = c(ncol(covariates), 2L, 2L, 2L)
    )
  )
  fitted
}

bh <- function(values) p.adjust(values, method = "BH")
metric <- function(estimate, truth) {
  error <- estimate - truth
  c(mse = mean(error^2), rmse = sqrt(mean(error^2)), error_sd = sd(error))
}

set.seed(seed)
started <- proc.time()[["elapsed"]]
truth <- build_truth(effect_multiplier)
design <- build_design(design_name, number_subjects)
tensor <- simulate_tensor(design, truth)
covariates <- cbind(intercept = 1, disease = design$disease, batch = design$batch)
fit <- fit_staccato(tensor, covariates)
disease_estimate <- as.vector(fit$C_ts[2L, , , ])
batch_estimate <- as.vector(fit$C_ts[3L, , , ])
disease_truth <- as.vector(truth$disease)
batch_truth <- as.vector(truth$batch)

bootstrap_disease <- matrix(NA_real_, nrow = number_bootstrap, ncol = number_events)
bootstrap_batch <- matrix(NA_real_, nrow = number_bootstrap, ncol = number_events)
# The released boot() helper references dcomp_res$input, but staccato() does
# not return that field. The observed tensor is available here explicitly.
residuals <- as.vector(tensor - fit$U)
for (index in seq_len(number_bootstrap)) {
  set.seed(seed + 100003L * index)
  new_residuals <- array(
    sample(residuals, size = length(residuals), replace = TRUE),
    dim = dim(fit$U)
  )
  current <- fit_staccato(fit$U + new_residuals, covariates)
  bootstrap_disease[index, ] <- as.vector(current$C_ts[2L, , , ])
  bootstrap_batch[index, ] <- as.vector(current$C_ts[3L, , , ])
}

disease_p <- (
  colSums(abs(sweep(bootstrap_disease, 2L, disease_estimate, "-")) >
            matrix(abs(disease_estimate), nrow = number_bootstrap, ncol = number_events, byrow = TRUE)) + 1
) / (number_bootstrap + 1)
disease_q <- bh(disease_p)
lower <- apply(bootstrap_disease, 2L, quantile, probs = 0.025, names = FALSE)
upper <- apply(bootstrap_disease, 2L, quantile, probs = 0.975, names = FALSE)
active <- abs(disease_truth) > 1e-12
null <- !active
disease_metric <- metric(disease_estimate, disease_truth)
batch_metric <- metric(batch_estimate, batch_truth)

row <- data.frame(
  seed = seed,
  design = design_name,
  subjects = number_subjects,
  effect_multiplier = effect_multiplier,
  bootstrap_replicates = number_bootstrap,
  status = "complete",
  disease_mse = disease_metric[["mse"]],
  disease_rmse = disease_metric[["rmse"]],
  disease_error_sd = disease_metric[["error_sd"]],
  batch_mse = batch_metric[["mse"]],
  batch_rmse = batch_metric[["rmse"]],
  batch_error_sd = batch_metric[["error_sd"]],
  ci_coverage_all = mean(lower <= disease_truth & disease_truth <= upper),
  ci_coverage_active = if (any(active)) mean(lower[active] <= disease_truth[active] & disease_truth[active] <= upper[active]) else NA_real_,
  ci_mean_width = mean(upper - lower),
  elapsed_seconds = proc.time()[["elapsed"]] - started
)
for (alpha in c(0.01, 0.05, 0.10)) {
  suffix <- gsub("\\.", "", sprintf("%0.2f", alpha))
  row[[paste0("type1_p_", suffix)]] <- mean(disease_p[null] <= alpha)
  row[[paste0("type1_q_", suffix)]] <- mean(disease_q[null] <= alpha)
  row[[paste0("power_p_", suffix)]] <- if (any(active)) mean(disease_p[active] <= alpha) else NA_real_
  row[[paste0("power_q_", suffix)]] <- if (any(active)) mean(disease_q[active] <= alpha) else NA_real_
  calls <- disease_q <= alpha
  row[[paste0("empirical_fdr_q_", suffix)]] <- sum(calls & null) / max(1L, sum(calls))
}

dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write.table(row, file = output, sep = "\t", row.names = FALSE, quote = FALSE)
