#!/usr/bin/env Rscript

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2L) {
  stop(
    "usage: build_cellchat_connectomedb.R CONNECTOMEDB_TSV OUTPUT_RDS",
    call. = FALSE
  )
}

suppressPackageStartupMessages(library(CellChat))
resource_path <- normalizePath(args[[1L]], mustWork = TRUE)
output_path <- args[[2L]]
resource <- utils::read.delim(
  resource_path,
  stringsAsFactors = FALSE,
  check.names = FALSE
)
required <- c(
  "harmonized_interaction_id", "ligand", "receptor", "ligand_location"
)
if (!all(required %in% names(resource)) || nrow(resource) != 2293L) {
  stop("ConnectomeDB2020 table must contain the frozen 2293-row axis", call. = FALSE)
}
if (anyNA(resource[required]) ||
    anyDuplicated(resource$harmonized_interaction_id) ||
    anyDuplicated(resource[c("ligand", "receptor")])) {
  stop("ConnectomeDB2020 identifiers must be complete and unique", call. = FALSE)
}

empty <- rep("", nrow(resource))
interaction <- data.frame(
  interaction_name = resource$harmonized_interaction_id,
  pathway_name = "ConnectomeDB2020",
  ligand = resource$ligand,
  receptor = resource$receptor,
  agonist = empty,
  antagonist = empty,
  co_A_receptor = empty,
  co_I_receptor = empty,
  annotation = "ConnectomeDB2020",
  interaction_name_2 = paste(resource$ligand, resource$receptor, sep = " - "),
  evidence = "Hou et al. 2020; doi:10.1038/s41467-020-18873-z",
  is_neurotransmitter = "FALSE",
  ligand.symbol = resource$ligand,
  ligand.family = empty,
  ligand.location = resource$ligand_location,
  ligand.keyword = empty,
  ligand.secreted_type = empty,
  ligand.transmembrane = empty,
  receptor.symbol = resource$receptor,
  receptor.family = empty,
  receptor.location = empty,
  receptor.keyword = empty,
  receptor.surfaceome_main = empty,
  receptor.surfaceome_sub = empty,
  receptor.adhesome = empty,
  receptor.secreted_type = empty,
  receptor.transmembrane = empty,
  version = "ConnectomeDB2020-scSeqComm-v2.0.0",
  stringsAsFactors = FALSE,
  check.names = FALSE
)
rownames(interaction) <- interaction$interaction_name
database <- list(
  interaction = interaction,
  complex = CellChatDB.human$complex[0, , drop = FALSE],
  cofactor = CellChatDB.human$cofactor[0, , drop = FALSE],
  geneInfo = CellChatDB.human$geneInfo
)
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
saveRDS(database, output_path, version = 3)
