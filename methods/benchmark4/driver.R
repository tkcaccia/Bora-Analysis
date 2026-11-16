# The same historical expression front end supplies both the reusable RDS
# handoff and the one-command end-to-end example.
.bora_benchmark4_frontend <- function(route, data_dir, output_dir, front_dir,
                                      giotto_attach = NULL,
                                      check_packages = TRUE) {
  front <- c(giotto = "run_giotto.R", seurat = "run_seurat.R",
             spatialexperiment = "run_spatialexperiment.R")
  route <- tolower(route)
  if (length(route) != 1L || is.na(route) || !(route %in% names(front)))
    stop("Choose giotto, seurat or spatialexperiment", call. = FALSE)
  data_dir <- normalizePath(data_dir, mustWork = TRUE)
  front_dir <- normalizePath(front_dir, mustWork = TRUE)
  if (file.exists(output_dir) || dir.exists(output_dir))
    stop("OUTPUT_DIR must be new", call. = FALSE)
  if (!dir.exists(dirname(output_dir)))
    stop("OUTPUT_DIR parent does not exist", call. = FALSE)
  legacy <- normalizePath(file.path(front_dir, front[[route]]), mustWork = TRUE)
  normalizePath(file.path(front_dir, "common.R"), mustWork = TRUE)
  if (check_packages) {
    needed <- switch(route,
      giotto = c("Giotto", "DropletUtils", "scuttle", "scran",
                 "SingleCellExperiment", "KODAMA", "KODAMAextra", "wsiTools"),
      seurat = c("Seurat", "KODAMA", "wsiTools"),
      spatialexperiment = c("SpatialExperiment", "DropletUtils",
                            "scuttle", "scran", "scater", "KODAMA", "igraph",
                            "RANN", "wsiTools", "HDF5Array"))
    missing <- needed[!vapply(needed, requireNamespace, logical(1), quietly = TRUE)]
    if (length(missing))
      stop(paste("Install required R packages:", paste(missing, collapse = ", ")),
           call. = FALSE)
  }

  # The archived front ends call source("common.R") and commandArgs(). Keep
  # both inside one environment so their spot table and clustered R object
  # remain available after the front end finishes.
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
  object
}

.bora_benchmark4_save_object <- function(object, output_dir) {
  path <- file.path(output_dir, "clustered-object.rds")
  partial <- tempfile(pattern = ".clustered-object-", tmpdir = output_dir)
  on.exit(unlink(partial), add = TRUE)
  saveRDS(object, partial, compress = "gzip")
  if (!isTRUE(file.rename(partial, path)))
    stop("Could not finalize the clustered RDS", call. = FALSE)
  path
}

bora_benchmark4_prepare <- function(route, data_dir, output_dir, front_dir,
                                    giotto_attach = NULL,
                                    check_packages = TRUE) {
  object <- .bora_benchmark4_frontend(
    route, data_dir, output_dir, front_dir,
    giotto_attach = giotto_attach, check_packages = check_packages)
  .bora_benchmark4_save_object(object, output_dir)
}

# Bora.spatial fits to the expression groups on this section and exports the
# image-guided mask and GeoJSON. This route also retains the reusable RDS.
bora_benchmark4_run <- function(route, data_dir, image, tissue_mask,
                                model_dir, output_dir, mpp_xy,
                                front_dir, spatial_fun = NULL,
                                giotto_attach = NULL) {
  image <- normalizePath(image, mustWork = TRUE)
  tissue_mask <- normalizePath(tissue_mask, mustWork = TRUE)
  model_dir <- normalizePath(model_dir, mustWork = TRUE)
  if (length(mpp_xy) != 2L || anyNA(mpp_xy) ||
      any(!is.finite(mpp_xy)) || any(mpp_xy <= 0))
    stop("Measured positive X/Y image spacing is required", call. = FALSE)
  if (is.null(spatial_fun)) {
    if (!nzchar(Sys.getenv("RETICULATE_PYTHON")))
      stop("Set RETICULATE_PYTHON to the Bora Python environment", call. = FALSE)
    if (!requireNamespace("Bora", quietly = TRUE))
      stop("Install the Bora R package", call. = FALSE)
  }
  object <- .bora_benchmark4_frontend(
    route, data_dir, output_dir, front_dir,
    giotto_attach = giotto_attach, check_packages = is.null(spatial_fun))
  .bora_benchmark4_save_object(object, output_dir)
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

.bora_benchmark4_image <- function(data_dir, sample, extension) {
  data_dir <- normalizePath(data_dir, mustWork = TRUE)
  image <- normalizePath(file.path(data_dir, paste0(sample, "_tissue_image.", extension)),
                         mustWork = TRUE)
  metadata <- file.path(data_dir, "image_metadata.json")
  if (!file.exists(metadata))
    stop("Verified image_metadata.json with OME X/Y spacing is required")
  meta <- jsonlite::fromJSON(metadata)
  if (!identical(meta$source, "OME Pixels PhysicalSizeX/Y") ||
      !identical(normalizePath(meta$image, mustWork = TRUE), image))
    stop("Image metadata must identify this exact OME H&E image")
  mpp <- as.numeric(c(meta$mpp_x, meta$mpp_y))
  if (length(mpp) != 2L || anyNA(mpp) ||
      any(!is.finite(mpp)) || any(mpp <= 0))
    stop("Measured positive X/Y image spacing is required", call. = FALSE)
  list(image = image, mpp = mpp)
}

bora_benchmark4_case <- function(route, sample, extension, args,
                                 script_dir) {
  if (length(args) != 4L)
    stop(paste("Usage: Rscript CASE.R DATA_DIR TISSUE_MASK",
               "VIRCHOW2_MODEL_DIR NEW_OUTPUT_DIR"), call. = FALSE)
  data_dir <- normalizePath(args[[1L]], mustWork = TRUE)
  image_info <- .bora_benchmark4_image(data_dir, sample, extension)
  bora_benchmark4_run(
    route = route, data_dir = data_dir, image = image_info$image,
    tissue_mask = args[[2L]], model_dir = args[[3L]],
    output_dir = args[[4L]], mpp_xy = image_info$mpp,
    front_dir = file.path(script_dir, "..", "benchmark3", "legacy_clustering"))
}

bora_benchmark4_prepare_case <- function(route, sample, extension, args,
                                         script_dir) {
  if (length(args) != 2L)
    stop("Usage: Rscript cluster_CASE.R DATA_DIR NEW_OUTPUT_DIR", call. = FALSE)
  data_dir <- normalizePath(args[[1L]], mustWork = TRUE)
  .bora_benchmark4_image(data_dir, sample, extension)
  bora_benchmark4_prepare(
    route = route, data_dir = data_dir, output_dir = args[[2L]],
    front_dir = file.path(script_dir, "..", "benchmark3", "legacy_clustering"))
}
