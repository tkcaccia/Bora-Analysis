#!/usr/bin/env Rscript
# Generate the registered SpatialExperiment/KODAMA tonsil object for Bora.spatial().
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
here <- dirname(normalizePath(script))
source(file.path(here, "driver.R"), local = TRUE)
path <- bora_benchmark4_prepare_case(
  "spatialexperiment", "CytAssist_FFPE_Protein_Expression_Human_Tonsil_AddOns",
  "btf", commandArgs(trailingOnly = TRUE), here)
cat("Clustered SpatialExperiment object:", path, "\n")
