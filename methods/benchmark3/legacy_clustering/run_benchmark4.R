#!/usr/bin/env Rscript
# Backward-compatible generic entry point for the three Benchmark 4 examples.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 8L)
  stop(paste("Usage: Rscript run_benchmark4.R",
             "giotto|seurat|spatialexperiment DATA_DIR IMAGE",
             "TISSUE_MASK MODEL_DIR NEW_OUTPUT_DIR MPP_X MPP_Y"))
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
here <- dirname(normalizePath(script))
source(file.path(here, "..", "..", "benchmark4", "driver.R"), local = TRUE)
result <- bora_benchmark4_run(
  args[[1L]], args[[2L]], args[[3L]], args[[4L]], args[[5L]],
  args[[6L]], as.numeric(args[7:8]), here)
cat("Mask:", result$mask, "\nGeoJSON:", result$geojson,
    "\nReceipt:", result$manifest, "\n")
