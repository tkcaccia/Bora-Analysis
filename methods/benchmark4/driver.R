# End-to-end Benchmark 4 orchestration. The historical expression front ends
# read the 10x count matrix and Space Ranger positions, then generate groups.
# Bora.spatial fits to those groups on this section and exports the GeoJSON.
bora_benchmark4_run <- function(route, data_dir, image, tissue_mask,
                                model_dir, output_dir, mpp_xy,
                                front_dir, spatial_fun = NULL,
                                giotto_attach = NULL) {
  front <- c(giotto = "run_giotto.R", seurat = "run_seurat.R",
             spatialexperiment = "run_spatialexperiment.R")
  route <- tolower(route)
  if (length(route) != 1L || is.na(route) || !(route %in% names(front)))
    stop("Choose giotto, seurat or spatialexperiment", call. = FALSE)
  data_dir <- normalizePath(data_dir, mustWork = TRUE)
  image <- normalizePath(image, mustWork = TRUE)
  tissue_mask <- normalizePath(tissue_mask, mustWork = TRUE)
  model_dir <- normalizePath(model_dir, mustWork = TRUE)
  front_dir <- normalizePath(front_dir, mustWork = TRUE)
  if (length(mpp_xy) != 2L || anyNA(mpp_xy) ||
      any(!is.finite(mpp_xy)) || any(mpp_xy <= 0))
    stop("Measured positive X/Y image spacing is required", call. = FALSE)
  if (file.exists(output_dir) || dir.exists(output_dir))
    stop("OUTPUT_DIR must be new", call. = FALSE)
  if (!dir.exists(dirname(output_dir)))
    stop("OUTPUT_DIR parent does not exist", call. = FALSE)
  legacy <- normalizePath(file.path(front_dir, front[[route]]), mustWork = TRUE)
  normalizePath(file.path(front_dir, "common.R"), mustWork = TRUE)
  if (is.null(spatial_fun)) {
    if (!nzchar(Sys.getenv("RETICULATE_PYTHON")))
      stop("Set RETICULATE_PYTHON to the Bora Python environment", call. = FALSE)
    needed <- switch(route,
      giotto = c("Bora", "Giotto", "DropletUtils", "scuttle", "scran",
                 "SingleCellExperiment", "KODAMA", "KODAMAextra", "wsiTools"),
      seurat = c("Bora", "Seurat", "KODAMA", "wsiTools"),
      spatialexperiment = c("Bora", "SpatialExperiment", "DropletUtils",
                            "scuttle", "scran", "scater", "KODAMA", "igraph",
                            "RANN", "wsiTools", "HDF5Array"))
    missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
    if (length(missing))
      stop(paste("Install required R packages:", paste(missing, collapse = ", ")),
           call. = FALSE)
  }

  # The archived front ends call source("common.R") and commandArgs(). Keep
  # both inside one environment so their spot table and clustered R object
  # remain available to Bora.spatial after the front end finishes.
  work <- new.env(parent = globalenv())
  groups_dir <- file.path(output_dir, "expression-groups")
  work$commandArgs <- function(trailingOnly = FALSE) {
    if (trailingOnly) c(data_dir, groups_dir) else paste0("--file=", legacy)
  }
  work$source <- function(file, local = FALSE, ...) {
    base::source(file, local = work, ...)
  }
  dir.create(output_dir)
  sys.source(legacy, envir = work)

  object <- if (route == "spatialexperiment") work$spe else work$obj
  if (is.null(object)) stop("Expression front end returned no clustered object")
  if (route == "giotto") {
    if (is.null(giotto_attach)) {
      giotto_attach <- function(object, work, data_dir) {
        groups <- data.frame(cell_ID = rownames(work$pca),
                             KODAMA_Louvain = as.character(work$cluster$labels))
        if (anyDuplicated(groups$cell_ID) || anyNA(groups$KODAMA_Louvain))
          stop("Giotto group identities are invalid", call. = FALSE)
        object <- Giotto::addCellMetadata(
          object, new_metadata = groups, by_column = TRUE,
          column_cell_ID = "cell_ID")
        loc <- as.data.frame(Giotto::getSpatialLocations(
          object, output = "data.table"))
        expected <- work$visium_positions(data_dir)
        idx <- match(as.character(loc$cell_ID), expected$barcode)
        if (anyNA(idx) ||
            any(abs(loc$sdimx - expected$pxl_col_in_fullres[idx]) > 1) ||
            any(abs(loc$sdimy - expected$pxl_row_in_fullres[idx]) > 1))
          stop("Giotto coordinates do not match full-resolution H&E positions")
        object
      }
    }
    object <- giotto_attach(object, work, data_dir)
  }
  if (is.null(spatial_fun)) spatial_fun <- Bora::Bora.spatial
  result <- spatial_fun(
    image = image, object = object, cluster_column = "KODAMA_Louvain",
    tissue_mask = tissue_mask, model_dir = model_dir,
    source_mpp_xy = as.numeric(mpp_xy),
    output_dir = file.path(output_dir, "image-guided-mask"))
  paths <- unlist(result[c("mask", "geojson", "manifest")], use.names = FALSE)
  if (length(paths) != 3L || any(!file.exists(paths)))
    stop("Bora.spatial did not return its mask, GeoJSON and receipt")
  result
}

bora_benchmark4_case <- function(route, sample, extension, args,
                                 script_dir) {
  if (length(args) != 4L)
    stop(paste("Usage: Rscript CASE.R DATA_DIR TISSUE_MASK",
               "VIRCHOW2_MODEL_DIR NEW_OUTPUT_DIR"), call. = FALSE)
  data_dir <- normalizePath(args[[1L]], mustWork = TRUE)
  image <- file.path(data_dir, paste0(sample, "_tissue_image.", extension))
  image <- normalizePath(image, mustWork = TRUE)
  metadata <- file.path(data_dir, "image_metadata.json")
  if (!file.exists(metadata))
    stop("Verified image_metadata.json with OME X/Y spacing is required")
  meta <- jsonlite::fromJSON(metadata)
  if (!identical(meta$source, "OME Pixels PhysicalSizeX/Y") ||
      !identical(normalizePath(meta$image, mustWork = TRUE), image))
    stop("Image metadata must identify this exact OME H&E image")
  mpp <- as.numeric(c(meta$mpp_x, meta$mpp_y))
  bora_benchmark4_run(
    route = route, data_dir = data_dir, image = image,
    tissue_mask = args[[2L]], model_dir = args[[3L]],
    output_dir = args[[4L]], mpp_xy = mpp,
    front_dir = file.path(script_dir, "..", "benchmark3", "legacy_clustering"))
}
