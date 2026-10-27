#!/usr/bin/env Rscript
# Plan physically calibrated, spot-centered tiles without saving images.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 7L) {
  stop("Usage: Rscript plan_visium_spots.R IMAGE SPOTS_CSV MPP_X MPP_Y OUTPUT_DIR WSITOOLS_API_SOURCE ALLOW_UPSAMPLE",
       call. = FALSE)
}
image <- normalizePath(args[[1L]], mustWork = TRUE)
spots_path <- normalizePath(args[[2L]], mustWork = TRUE)
mpp <- as.numeric(args[3:4])
output <- args[[5L]]
source_file <- normalizePath(args[[6L]], mustWork = TRUE)
allow <- tolower(args[[7L]])
if (any(!is.finite(mpp)) || any(mpp <= 0) || !allow %in% c("true", "false")) {
  stop("MPP must be positive; ALLOW_UPSAMPLE must be TRUE or FALSE.", call. = FALSE)
}
if (file.exists(output) || dir.exists(output)) stop("Output must be a new directory.", call. = FALSE)
suppressPackageStartupMessages(library(wsiTools))
if (!"wsi_virchow2_spot_tile_plan" %in% getNamespaceExports("wsiTools")) {
  source(source_file, local = FALSE)
} else {
  wsi_virchow2_spot_tile_plan <- wsiTools::wsi_virchow2_spot_tile_plan
}
spots <- utils::read.csv(spots_path, stringsAsFactors = FALSE)
need <- c("barcode", "pxl_col_in_fullres", "pxl_row_in_fullres")
if (!all(need %in% names(spots)) || !nrow(spots) || anyDuplicated(spots$barcode)) {
  stop("Spot table requires unique barcodes and full-resolution coordinates.", call. = FALSE)
}
centers <- data.frame(x = spots$pxl_col_in_fullres,
                      y = spots$pxl_row_in_fullres, spot_id = spots$barcode)
slide <- wsiTools::wsi_open(image, backend = "vips")
on.exit(wsiTools::wsi_close(slide), add = TRUE)
plan <- wsi_virchow2_spot_tile_plan(slide, centers,
    source_mpp_xy = mpp,
    spacing_evidence = "verified original OME PhysicalSizeX/Y and Space Ranger scale",
    allow_upsample = identical(allow, "true"))
if (nrow(plan) != nrow(spots) || !identical(plan$spot_id, spots$barcode)) {
  stop("Spot-to-tile association changed.", call. = FALSE)
}
dir.create(output, recursive = TRUE)
on.exit({ if (!file.exists(file.path(output, "PLAN_COMPLETE.json")))
            unlink(output, recursive = TRUE) }, add = TRUE)
utils::write.csv(plan, file.path(output, "spot_tile_plan.csv"), row.names = FALSE)
meta <- list(format = "wsiTools-virchow2-spot-plan-v1", image = image,
             spots = spots_path, tile_table = "spot_tile_plan.csv",
             source_mpp_xy = mpp, effective_mpp_xy =
               as.numeric(plan[1L, c("effective_mpp_x", "effective_mpp_y")]),
             native_tile_wh = as.numeric(plan[1L, c("native_width", "native_height")]),
             source_upsampled = identical(allow, "true"),
             tile_pixels = 224L, target_mpp = 0.5,
             planned_spots = nrow(plan), tile_images_saved = FALSE)
jsonlite::write_json(meta, file.path(output, "PLAN_COMPLETE.json"),
                     auto_unbox = TRUE, pretty = TRUE, digits = 16)
cat(jsonlite::toJSON(meta, auto_unbox = TRUE, digits = 16), "\n")
