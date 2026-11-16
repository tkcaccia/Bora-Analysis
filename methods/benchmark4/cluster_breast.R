#!/usr/bin/env Rscript
# Generate the registered Giotto/KODAMA breast object for Bora.spatial().
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
here <- dirname(normalizePath(script))
source(file.path(here, "driver.R"), local = TRUE)
path <- bora_benchmark4_prepare_case(
  "giotto", "CytAssist_FFPE_Human_Breast_Cancer", "tif",
  commandArgs(trailingOnly = TRUE), here)
cat("Clustered Giotto object:", path, "\n")
