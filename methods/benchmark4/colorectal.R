#!/usr/bin/env Rscript
# Colorectal cancer Visium: read 10x counts and positions with Seurat, group
# expression with KODAMA/Louvain, then refine with Bora.spatial to GeoJSON.
# Source: CytAssist_11mm_FFPE_Human_Colorectal_Cancer (10x Genomics).
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
here <- dirname(normalizePath(script))
source(file.path(here, "driver.R"), local = TRUE)
result <- bora_benchmark4_case(
  "seurat", "CytAssist_11mm_FFPE_Human_Colorectal_Cancer", "tif",
  commandArgs(trailingOnly = TRUE), here)
cat("Mask:", result$mask, "\nGeoJSON:", result$geojson,
    "\nReceipt:", result$manifest, "\n")
