#!/usr/bin/env Rscript
# Build a calibrated Virchow2 tile plan without saving tile images.
args <- commandArgs(trailingOnly = TRUE)
if (!(length(args) %in% c(2L, 3L, 5L, 6L))) {
  stop("Usage: Rscript plan_virchow2_tiles.R IMAGE OUTPUT_DIR [WSITOOLS_API_SOURCE [MPP_X MPP_Y [ALLOW_UPSAMPLE]]]", call. = FALSE)
}
image_path <- normalizePath(args[[1L]], mustWork = TRUE)
output_dir <- args[[2L]]
if (dir.exists(output_dir) || file.exists(output_dir)) {
  stop("Output directory already exists; use a new path.", call. = FALSE)
}
suppressPackageStartupMessages(library(wsiTools))
if (!"wsi_virchow2_tile_plan" %in% getNamespaceExports("wsiTools")) {
  if (length(args) < 3L) {
    stop("Update wsiTools or provide the reviewed API source file.", call. = FALSE)
  }
  source(normalizePath(args[[3L]], mustWork = TRUE), local = FALSE)
} else {
  wsi_virchow2_tile_plan <- wsiTools::wsi_virchow2_tile_plan
  wsi_virchow2_geometry <- wsiTools::wsi_virchow2_geometry
}
slide <- wsiTools::wsi_open(image_path, backend = "vips")
spacing <- if (length(args) >= 5L) as.numeric(args[4:5]) else NULL
if (!is.null(spacing) && (any(!is.finite(spacing)) || any(spacing <= 0))) {
  stop("MPP_X and MPP_Y must be positive finite micrometres/pixel.", call. = FALSE)
}
allow_upsample <- length(args) == 6L && identical(tolower(args[[6L]]), "true")
if (length(args) == 6L && !tolower(args[[6L]]) %in% c("true", "false")) {
  stop("ALLOW_UPSAMPLE must be TRUE or FALSE.", call. = FALSE)
}
evidence <- if (is.null(spacing)) NULL else "Verified provider OME or pinned Space Ranger calibration; matches embedded rounded metadata"
geometry <- wsi_virchow2_geometry(slide, source_mpp_xy = spacing,
                                  spacing_evidence = evidence,
                                  allow_upsample = allow_upsample)
plan <- wsi_virchow2_tile_plan(slide, source_mpp_xy = spacing,
                               spacing_evidence = evidence,
                               allow_upsample = allow_upsample)
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
if (!dir.exists(output_dir)) stop("Could not create output directory.", call. = FALSE)
on.exit({
  if (!file.exists(file.path(output_dir, "PLAN_COMPLETE.json"))) {
    unlink(output_dir, recursive = TRUE)
  }
}, add = TRUE)
table_file <- file.path(output_dir, "tile_plan.tsv")
write.table(plan, table_file, sep = "\t", row.names = FALSE, quote = FALSE)
metadata <- list(
  format = "wsiTools-virchow2-plan-v1",
  image = image_path,
  image_size_wh = as.numeric(slide$dimensions[c("width", "height")]),
  source_mpp_xy = geometry$source_mpp_xy,
  target_mpp = geometry$target_mpp,
  tile_pixels = geometry$tile_pixels,
  native_tile_wh = geometry$native_tile_wh,
  effective_mpp_xy = geometry$effective_mpp_xy,
  spacing_evidence = geometry$spacing_evidence,
  source_upsampled = geometry$upsampled,
  tiles = nrow(plan),
  tile_table = basename(table_file),
  tile_images_saved = FALSE
)
jsonlite::write_json(metadata, file.path(output_dir, "PLAN_COMPLETE.json"),
                     auto_unbox = TRUE, pretty = TRUE, digits = 16)
cat(jsonlite::toJSON(metadata, auto_unbox = TRUE, digits = 16), "\n")
