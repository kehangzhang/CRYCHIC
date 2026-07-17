#!/usr/bin/env Rscript

# Freeze the released mouse ADT alias logic against a pinned receptor universe.

suppressPackageStartupMessages({
  library(AnnotationDbi)
  library(DBI)
  library(org.Mm.eg.db)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3L) {
  stop(
    "Usage: build_mouse_adt_aliases.R ADT_MEANS_TSV RECEPTOR_RESOURCE_TSV OUTPUT_TSV",
    call. = FALSE
  )
}
means_path <- normalizePath(args[[1L]], mustWork = TRUE)
resource_path <- normalizePath(args[[2L]], mustWork = TRUE)
output_path <- normalizePath(args[[3L]], mustWork = FALSE)

means <- read.delim(means_path, check.names = FALSE, stringsAsFactors = FALSE)
resource <- read.delim(resource_path, check.names = FALSE, stringsAsFactors = FALSE)
if (!("adt_feature" %in% colnames(means)) || !("receptor" %in% colnames(resource))) {
  stop("Input tables lack adt_feature or receptor", call. = FALSE)
}
features <- sort(unique(as.character(means$adt_feature)))
receptors <- unique(tolower(as.character(resource$receptor)))

connection <- org.Mm.eg_dbconn()
aliases <- DBI::dbGetQuery(
  connection,
  paste0(
    "SELECT alias.alias_symbol, gene_info.symbol ",
    "FROM alias, gene_info WHERE alias._id == gene_info._id;"
  )
)
aliases$alias_lower <- tolower(aliases$alias_symbol)

manual <- list(
  CD105 = c("Eng", "Endo"),
  CD107a = c("Lamp1", "Perk"),
  CD120b = c("Tnfrsf1b", "Tnfr2"),
  CD16.32 = "Fcgr3",
  CD198 = "Cxcr8",
  CD199 = c("Ccr9", "Cmkbr10"),
  CD201 = c("Procr", "Epcr"),
  CD21.CD35 = c("Cr2", "Cr1"),
  CD278.1 = c("Icos", "Ailim"),
  CD300c.d = c("Cd300c", "Clm6"),
  CD301a = c("Mgl", "Mgl1"),
  CD301b = "Mgl2",
  CD309.1 = c("Kdr", "Flk1"),
  CD326 = c("Epcam", "Tacstd1"),
  CD34.1 = "Cd34",
  CD45.1 = "Ptprc",
  CD45.2 = "Ptprc",
  CD45R.B220 = "Ptprc",
  CD49a = "Itga1",
  CD90.1 = c("Thy1", "Thy-1"),
  CD90.2 = c("Thy1", "Thy-1"),
  D62E = c("Sele", "Elam-1"),
  F4.80 = c("Adgre1", "Adgre4", "Emr4"),
  FceRIa = c("Fcer1a", "Fcer1g", "Fce1g"),
  FolateReceptorb = c("Folr2", "Fbp2", "Folbp2"),
  IL.21Receptor = c("Il21r", "Nilr"),
  IL.33Ra = c("Il1rl1", "Ly84", "St2", "Ste2"),
  Ly.49A = c("Klra1", "Ly-49", "Ly-49a", "Ly49", "Ly49A"),
  Ly.6A.E = c("Sca-1", "Ly6e"),
  Ly.6G = "Ly6g6e",
  Ly.6G.Ly.6C = "Gr-1",
  MAdCAM.1 = "Madcam1",
  Mac.2 = "Lgals3",
  NK.1.1 = c("Klrb1c", "Ly55c", "Nkrp1c"),
  P2X7R = "P2rx7",
  Tim.4 = c("Timd4", "Tim4"),
  anti.P2RY12 = c("P2ry12", "P2y12")
)

records <- list()
record_index <- 1L
for (feature in features) {
  matched <- unique(aliases$symbol[aliases$alias_lower == tolower(feature)])
  if (feature %in% names(manual)) {
    matched <- unique(c(matched, manual[[feature]]))
  }
  matched <- matched[tolower(matched) %in% receptors]
  for (symbol in sort(unique(matched))) {
    is_manual <- feature %in% names(manual) && symbol %in% manual[[feature]]
    records[[record_index]] <- data.frame(
      adt_feature = feature,
      receptor_gene = symbol,
      mapping_source = if (is_manual) {
        "paper_manual_list_canonicalized"
      } else {
        "org.Mm.eg.db_alias_receptor_filtered"
      },
      stringsAsFactors = FALSE
    )
    record_index <- record_index + 1L
  }
}
if (length(records) == 0L) {
  stop("No mouse ADTs map to the frozen receptor universe", call. = FALSE)
}
result <- unique(do.call(rbind, records))
result <- result[order(result$adt_feature, result$receptor_gene), ]
dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
write.table(result, output_path, sep = "\t", quote = FALSE, row.names = FALSE)

unmapped <- setdiff(features, result$adt_feature)
message(
  "Mapped ", length(unique(result$adt_feature)), "/", length(features),
  " ADTs to ", length(unique(result$receptor_gene)), " frozen receptors; unmapped=",
  paste(unmapped, collapse = ",")
)
