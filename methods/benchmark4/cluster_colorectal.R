#!/usr/bin/env Rscript
# Generate the registered Seurat/KODAMA colorectal object for Bora.spatial().
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
here <- dirname(normalizePath(script))
source(file.path(here, "driver.R"), local = TRUE)
path <- bora_benchmark4_prepare_case(
  "seurat", "CytAssist_11mm_FFPE_Human_Colorectal_Cancer", "tif",
  commandArgs(trailingOnly = TRUE), here)
cat("Clustered Seurat object:", path, "\n")
