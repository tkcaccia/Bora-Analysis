#!/usr/bin/env Rscript
# New same-section expression-guided Bora run. The historical front ends are
# sourced without changing their archived code or published output hashes.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 8L) {
  stop(paste("Usage: Rscript run_benchmark4.R",
             "seurat|giotto|spatialexperiment DATA_DIR IMAGE.tif",
             "TISSUE_MASK MODEL_DIR OUTPUT_DIR MPP_X MPP_Y"))
}
route <- tolower(args[[1L]])
front <- c(seurat = "run_seurat.R", giotto = "run_giotto.R",
           spatialexperiment = "run_spatialexperiment.R")
if (!(route %in% names(front))) stop("Unknown expression workflow")
data_dir <- normalizePath(args[[2L]], mustWork = TRUE)
image <- normalizePath(args[[3L]], mustWork = TRUE)
tissue_mask <- normalizePath(args[[4L]], mustWork = TRUE)
model_dir <- normalizePath(args[[5L]], mustWork = TRUE)
out <- normalizePath(args[[6L]], mustWork = FALSE)
mpp <- as.numeric(args[7:8])
if (anyNA(mpp) || any(!is.finite(mpp)) || any(mpp <= 0))
  stop("MPP_X and MPP_Y must be measured positive image spacings")
if (file.exists(out) || !dir.exists(dirname(out)))
  stop("OUTPUT_DIR must be new and its parent must exist")
if (!nzchar(Sys.getenv("RETICULATE_PYTHON")))
  stop("Set RETICULATE_PYTHON to the matching Bora Python environment")
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
here <- dirname(normalizePath(script))
legacy <- file.path(here, front[[route]])
dir.create(out)
cluster_dir <- file.path(out, "expression-groups")
working <- new.env(parent = globalenv())
# The released front ends expect two CLI inputs and inspect their own script
# path. Supply those exact values within this sourced environment.
working$commandArgs <- function(trailingOnly = FALSE) {
  if (trailingOnly) c(data_dir, cluster_dir) else paste0("--file=", legacy)
}
sys.source(file.path(here, "common.R"), envir = working)
sys.source(legacy, envir = working)
if (route == "giotto") {
  # The historical Giotto front end wrote the cluster CSV but did not add its
  # selected membership to the object. Attach it by stable spot ID here.
  groups <- data.frame(cell_ID = rownames(working$pca),
                       KODAMA_Louvain = as.character(working$cluster$labels))
  working$obj <- Giotto::addCellMetadata(
    working$obj, new_metadata = groups, by_column = TRUE,
    column_cell_ID = "cell_ID")
  loc <- as.data.frame(Giotto::getSpatialLocations(
    working$obj, output = "data.table"))
  expected <- working$visium_positions(data_dir)
  indices <- match(as.character(loc$cell_ID), expected$barcode)
  if (anyNA(indices) ||
      any(abs(loc$sdimx - expected$pxl_col_in_fullres[indices]) > 1) ||
      any(abs(loc$sdimy - expected$pxl_row_in_fullres[indices]) > 1)) {
    stop("Giotto spatial coordinates do not match full-resolution H&E positions")
  }
}
object <- if (route == "spatialexperiment") working$spe else working$obj
if (is.null(object)) stop("Expression front end did not return its clustered object")
result <- Bora::Bora.spatial(
  image = image, object = object,
  cluster_column = "KODAMA_Louvain",
  tissue_mask = tissue_mask, model_dir = model_dir,
  source_mpp_xy = mpp,
  output_dir = file.path(out, "image-guided-mask"))
cat("Mask:", result$mask, "\nGeoJSON:", result$geojson,
    "\nReceipt:", result$manifest, "\n")
