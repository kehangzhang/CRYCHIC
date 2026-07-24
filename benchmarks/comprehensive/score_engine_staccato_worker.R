#!/usr/bin/env Rscript

arguments <- commandArgs(trailingOnly = TRUE)

arg_value <- function(flag) {
  index <- match(flag, arguments)
  if (is.na(index) || index == length(arguments)) {
    stop(paste("missing required argument", flag))
  }
  arguments[[index + 1L]]
}

input_path <- arg_value("--input")
output_path <- arg_value("--output")
source_path <- arg_value("--staccato-source")
bootstrap_replicates <- as.integer(arg_value("--bootstrap-replicates"))
bootstrap_cores <- as.integer(arg_value("--bootstrap-cores"))

if (bootstrap_replicates < 1L || bootstrap_cores < 1L) {
  stop("bootstrap settings must be positive")
}

suppressPackageStartupMessages(library(rTensor))
suppressPackageStartupMessages(capture.output(source(source_path)))

scores <- read.delim(gzfile(input_path), check.names = FALSE, stringsAsFactors = FALSE)
required <- c(
  "dataset_id", "seed", "score_generator", "contrast", "target", "reference",
  "sample_id", "subject_id", "condition", "sender", "receiver",
  "interaction_id", "score", "status"
)
missing_columns <- setdiff(required, colnames(scores))
if (length(missing_columns) > 0L) {
  stop(paste("score input lacks", paste(missing_columns, collapse = ",")))
}

quiet_boot <- function(index, fitted) {
  set.seed(index)
  fitted_mean <- fitted$U
  dimensions <- dim(fitted_mean)
  coefficient <- fitted$C_ts
  ranks <- dim(fitted$G)
  residuals <- as.vector(fitted$input - fitted_mean)
  sampled <- array(
    sample(residuals, size = length(residuals), replace = TRUE),
    dim = dimensions
  )
  invisible(capture.output(
    result <- suppressMessages(staccato(
      as.tensor(fitted_mean + sampled),
      fitted$X_covar1,
      core_shape = ranks
    ))
  ))
  abs(result$C_ts - coefficient) > abs(coefficient)
}

bootstrap_p <- function(fitted, replicates, cores) {
  draws <- parallel::mclapply(
    seq_len(replicates),
    quiet_boot,
    fitted = fitted,
    mc.cores = cores,
    mc.preschedule = TRUE
  )
  (Reduce("+", draws) + 1) / (replicates + 1)
}

empty_arm <- function(template, generator, contrast, reason, elapsed) {
  data.frame(
    dataset_id = template$dataset_id,
    seed = template$seed,
    score_generator = generator,
    differential_engine = "staccato",
    contrast = contrast,
    sender = template$sender,
    receiver = template$receiver,
    interaction_id = template$interaction_id,
    effect = NA_real_,
    p_value = NA_real_,
    q_value = NA_real_,
    status = "not_estimable",
    reason_code = reason,
    engine_elapsed_seconds = elapsed,
    bootstrap_replicates = bootstrap_replicates,
    stringsAsFactors = FALSE
  )
}

rows <- list()
generators <- sort(unique(scores$score_generator))
contrasts <- sort(unique(scores$contrast))
for (generator in generators) {
  for (contrast in contrasts) {
    arm_started <- proc.time()[["elapsed"]]
    arm <- scores[
      scores$score_generator == generator & scores$contrast == contrast,
      ,
      drop = FALSE
    ]
    target <- unique(arm$target)
    reference <- unique(arm$reference)
    if (length(target) != 1L || length(reference) != 1L) {
      stop(paste("invalid contrast labels", generator, contrast))
    }
    arm <- arm[arm$condition %in% c(reference, target), , drop = FALSE]
    event_template <- unique(arm[c(
      "dataset_id", "seed", "sender", "receiver", "interaction_id"
    )])
    event_template <- event_template[order(
      event_template$interaction_id,
      event_template$sender,
      event_template$receiver
    ), , drop = FALSE]
    samples <- sort(unique(arm$sample_id))
    interactions <- sort(unique(arm$interaction_id))
    senders <- sort(unique(arm$sender))
    receivers <- sort(unique(arm$receiver))
    expected_rows <- length(samples) * length(interactions) *
      length(senders) * length(receivers)
    duplicate_keys <- duplicated(arm[c(
      "sample_id", "sender", "receiver", "interaction_id"
    )])
    complete_tensor <- nrow(arm) == expected_rows && !any(duplicate_keys) &&
      all(arm$status == "observed") && all(is.finite(arm$score))
    group_counts <- table(arm[!duplicated(arm$sample_id), c("sample_id", "condition")][[2L]])
    enough_groups <- all(c(reference, target) %in% names(group_counts)) &&
      all(group_counts[c(reference, target)] >= 4L)
    if (!complete_tensor || !enough_groups) {
      reason <- if (!complete_tensor) {
        "structural_missing_score_tensor"
      } else {
        "fewer_than_four_subjects_per_condition"
      }
      rows[[length(rows) + 1L]] <- empty_arm(
        event_template,
        generator,
        contrast,
        reason,
        proc.time()[["elapsed"]] - arm_started
      )
      next
    }

    tensor <- array(
      NA_real_,
      dim = c(length(samples), length(interactions), length(senders), length(receivers))
    )
    sample_index <- match(arm$sample_id, samples)
    interaction_index <- match(arm$interaction_id, interactions)
    sender_index <- match(arm$sender, senders)
    receiver_index <- match(arm$receiver, receivers)
    tensor[cbind(sample_index, interaction_index, sender_index, receiver_index)] <- arm$score
    sample_rows <- arm[match(samples, arm$sample_id), c("sample_id", "condition")]
    design <- cbind(
      intercept = 1,
      target = as.integer(sample_rows$condition == target)
    )
    fitted <- tryCatch(
      {
        invisible(capture.output(
          result <- suppressMessages(staccato(
            tsr = as.tensor(tensor),
            X_covar1 = design,
            lr.names = interactions,
            sender.names = senders,
            receiver.names = receivers,
            core_shape = c(
              2L,
              min(2L, length(interactions)),
              min(2L, length(senders)),
              min(2L, length(receivers))
            )
          ))
        ))
        result
      },
      error = function(error) error
    )
    if (inherits(fitted, "error")) {
      rows[[length(rows) + 1L]] <- empty_arm(
        event_template,
        generator,
        contrast,
        paste0("staccato_fit_failed:", conditionMessage(fitted)),
        proc.time()[["elapsed"]] - arm_started
      )
      next
    }
    fitted$input <- tensor
    p_values <- tryCatch(
      bootstrap_p(fitted, bootstrap_replicates, bootstrap_cores),
      error = function(error) error
    )
    if (inherits(p_values, "error")) {
      rows[[length(rows) + 1L]] <- empty_arm(
        event_template,
        generator,
        contrast,
        paste0("staccato_bootstrap_failed:", conditionMessage(p_values)),
        proc.time()[["elapsed"]] - arm_started
      )
      next
    }
    dimnames(p_values) <- list(
      colnames(design), interactions, senders, receivers
    )
    fitted_rows <- event_template
    fitted_rows$score_generator <- generator
    fitted_rows$differential_engine <- "staccato"
    fitted_rows$contrast <- contrast
    fitted_rows$effect <- mapply(
      function(interaction, sender, receiver) {
        fitted$C_ts["target", interaction, sender, receiver]
      },
      fitted_rows$interaction_id,
      fitted_rows$sender,
      fitted_rows$receiver
    )
    fitted_rows$p_value <- mapply(
      function(interaction, sender, receiver) {
        p_values[2L, interaction, sender, receiver]
      },
      fitted_rows$interaction_id,
      fitted_rows$sender,
      fitted_rows$receiver
    )
    fitted_rows$q_value <- p.adjust(fitted_rows$p_value, method = "BH")
    fitted_rows$status <- "observed"
    fitted_rows$reason_code <- "staccato_residual_bootstrap"
    fitted_rows$engine_elapsed_seconds <- proc.time()[["elapsed"]] - arm_started
    fitted_rows$bootstrap_replicates <- bootstrap_replicates
    rows[[length(rows) + 1L]] <- fitted_rows
  }
}

result <- do.call(rbind, rows)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.table(
  result,
  file = output_path,
  sep = "\t",
  row.names = FALSE,
  quote = FALSE,
  na = "NA"
)
