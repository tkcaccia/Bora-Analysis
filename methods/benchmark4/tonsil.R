#!/usr/bin/env Rscript
# Tonsil Visium: read 10x counts and positions into SpatialExperiment, group
# expression with KODAMA/Louvain, then refine with Bora.spatial to GeoJSON.
# Source: CytAssist_FFPE_Protein_Expression_Human_Tonsil_AddOns (10x).
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
here <- dirname(normalizePath(script))
source(file.path(here, "driver.R"), local = TRUE)
result <- bora_benchmark4_case(
  "spatialexperiment", "CytAssist_FFPE_Protein_Expression_Human_Tonsil_AddOns",
  "btf", commandArgs(trailingOnly = TRUE), here)
cat("Mask:", result$mask, "\nGeoJSON:", result$geojson,
    "\nReceipt:", result$manifest, "\n")
