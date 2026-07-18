#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop(
    "usage: export_connectomedb2020.R SOURCE_RDATA OUTPUT_TSV",
    call. = FALSE
  )
}

source_path <- normalizePath(args[[1L]], mustWork = TRUE)
output_path <- args[[2L]]
env <- new.env(parent = emptyenv())
loaded <- load(source_path, envir = env)

expected_object <- "LR_pairs_ConnectomeDB_2020"
if (!identical(loaded, expected_object)) {
  stop(
    sprintf(
      "source RData must contain only %s; found: %s",
      expected_object,
      paste(loaded, collapse = ", ")
    ),
    call. = FALSE
  )
}

resource <- env[[expected_object]]
required <- c(
  "ligand", "receptor", "Ligand.receptor.pair", "Source",
  "PMID.support", "Ligand.HGNC.ID", "Ligand.location",
  "Receptor.HGNC.ID", "HGNC.L.R", "secondary.source."
)
if (!is.data.frame(resource) || !identical(names(resource), required)) {
  stop("ConnectomeDB2020 source has an unexpected table schema", call. = FALSE)
}
if (nrow(resource) != 2293L) {
  stop(
    sprintf("ConnectomeDB2020 source has %d rows; expected 2293", nrow(resource)),
    call. = FALSE
  )
}

names(resource) <- c(
  "ligand", "receptor", "interaction_label", "source", "pmid_support",
  "ligand_hgnc_id", "ligand_location", "receptor_hgnc_id", "hgnc_pair",
  "secondary_source"
)
resource$ligand <- trimws(as.character(resource$ligand))
resource$receptor <- trimws(as.character(resource$receptor))
if (anyNA(resource[c("ligand", "receptor")]) ||
    any(resource$ligand == "") || any(resource$receptor == "")) {
  stop("ConnectomeDB2020 contains missing or empty ligand/receptor symbols", call. = FALSE)
}
if (anyDuplicated(resource[c("ligand", "receptor")])) {
  stop("ConnectomeDB2020 contains duplicate directed ligand-receptor pairs", call. = FALSE)
}

old_locale <- Sys.getlocale("LC_COLLATE")
on.exit(suppressWarnings(Sys.setlocale("LC_COLLATE", old_locale)), add = TRUE)
invisible(suppressWarnings(Sys.setlocale("LC_COLLATE", "C")))
resource <- resource[order(resource$ligand, resource$receptor), , drop = FALSE]
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.table(
  resource,
  file = output_path,
  sep = "\t",
  quote = FALSE,
  row.names = FALSE,
  col.names = TRUE,
  na = ""
)
