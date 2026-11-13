# Shared, source-image-coordinate handoff for the three Visium front ends.
options(stringsAsFactors = FALSE)

visium_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) != 2L) stop("Usage: Rscript run_<pipeline>.R DATA_DIR OUTPUT_DIR")
  out <- normalizePath(args[[2]], mustWork = FALSE)
  dir.create(out, recursive = TRUE, showWarnings = FALSE)
  list(data = normalizePath(args[[1]], mustWork = TRUE), out = out)
}

visium_positions <- function(data_dir) {
  candidates <- file.path(data_dir, "spatial", c("tissue_positions.csv", "tissue_positions_list.csv"))
  file <- candidates[file.exists(candidates)][1]
  if (is.na(file)) stop("No Space Ranger tissue positions file")
  rows <- utils::read.csv(file, header = FALSE, stringsAsFactors = FALSE)
  expected <- c("barcode", "in_tissue", "array_row", "array_col",
                "pxl_row_in_fullres", "pxl_col_in_fullres")
  if (ncol(rows) != 6L) stop("Expected six Space Ranger position columns")
  if (identical(as.character(rows[1, 1]), "barcode")) rows <- rows[-1, , drop = FALSE]
  names(rows) <- expected
  for (name in expected[-1]) rows[[name]] <- as.numeric(rows[[name]])
  rows <- rows[rows$in_tissue == 1 & is.finite(rows$pxl_row_in_fullres) &
                 is.finite(rows$pxl_col_in_fullres), , drop = FALSE]
  if (!nrow(rows) || anyDuplicated(rows$barcode)) stop("Invalid tissue spot positions")
  rows
}

visium_write_clusters <- function(barcodes, labels, data_dir, output_dir, pipeline) {
  if (length(barcodes) != length(labels) || anyNA(labels) || anyDuplicated(barcodes))
    stop("One nonmissing cluster per unique spot is required")
  positions <- visium_positions(data_dir)
  match_idx <- match(positions$barcode, barcodes)
  positions <- positions[!is.na(match_idx), , drop = FALSE]
  match_idx <- match_idx[!is.na(match_idx)]
  positions$cluster <- as.character(labels[match_idx])
  if (length(unique(positions$cluster)) < 2L) stop("At least two spatial domains are needed")
  positions <- positions[order(positions$barcode), , drop = FALSE]
  utils::write.csv(positions, file.path(output_dir, "spots.csv"), row.names = FALSE)
  writeLines(paste("pipeline", pipeline, "spots", nrow(positions),
                   "clusters", length(unique(positions$cluster))),
             file.path(output_dir, "cluster_summary.txt"))
  positions
}

visium_louvain <- function(embedding, target_groups = 6L) {
  embedding <- as.matrix(embedding)
  if (nrow(embedding) < 30L || ncol(embedding) < 2L || any(!is.finite(embedding)))
    stop("KODAMA embedding is too small or non-finite")
  neighbors <- RANN::nn2(embedding[, 1:2, drop = FALSE], k = 21L)$nn.idx[, -1,
                                                                       drop = FALSE]
  edges <- cbind(rep(seq_len(nrow(embedding)), each = ncol(neighbors)),
                 as.vector(t(neighbors)))
  graph <- igraph::simplify(igraph::graph_from_edgelist(edges, directed = FALSE),
                            remove.multiple = TRUE, remove.loops = TRUE)
  best <- NULL
  gap <- Inf
  for (resolution in seq(0.10, 1.50, by = 0.05)) {
    set.seed(1234)
    labels <- igraph::membership(igraph::cluster_louvain(graph,
                                                         resolution = resolution))
    current <- abs(length(unique(labels)) - target_groups)
    if (current < gap) {
      best <- labels
      gap <- current
      best_resolution <- resolution
    }
    if (current == 0L) break
  }
  list(labels = best, resolution = best_resolution)
}

visium_export_tiles <- function(spots, data_dir, sample, image_ext, output_dir) {
  # Historical name retained for the three front ends. No tile images are saved.
  if (!requireNamespace("wsiTools", quietly = TRUE)) stop("wsiTools is required")
  image <- file.path(data_dir, paste0(sample, "_tissue_image.", image_ext))
  slide <- wsiTools::wsi_open(image, backend = "vips")
  on.exit(wsiTools::wsi_close(slide), add = TRUE)
  info <- wsiTools::wsi_info(slide)
  dims <- as.numeric(unlist(info$dimensions))[1:2]
  probe_file <- file.path(data_dir, "image_metadata.json")
  if (!file.exists(probe_file)) stop("Verified OME image_metadata.json is required")
  probe <- jsonlite::fromJSON(probe_file)
  if (!identical(normalizePath(image), normalizePath(probe$image)) ||
      !identical(probe$source, "OME Pixels PhysicalSizeX/Y"))
    stop("OME microscopy calibration is not bound to this image")
  mpp <- c(probe$mpp_x, probe$mpp_y)
  if (any(!is.finite(mpp)) || any(mpp <= 0)) stop("Invalid microscopy spacing")
  if (any(spots$pxl_col_in_fullres < 0 | spots$pxl_col_in_fullres >= dims[1] |
          spots$pxl_row_in_fullres < 0 | spots$pxl_row_in_fullres >= dims[2]))
    stop("Spot coordinates are outside the original image")
  sf <- jsonlite::fromJSON(file.path(data_dir, "spatial", "scalefactors_json.json"))
  implied_spot_um <- as.numeric(sf$spot_diameter_fullres) * mean(mpp)
  if (!is.finite(implied_spot_um) || implied_spot_um < 55 || implied_spot_um > 75)
    stop("OME spacing and provider spot geometry disagree")
  if (!"wsi_virchow2_spot_tile_plan" %in% getNamespaceExports("wsiTools")) {
    script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))
    source(file.path(dirname(normalizePath(script)), "..", "..", "integrations",
                     "wsiTools", "R", "virchow2-tiles.R"), local = FALSE)
  } else {
    wsi_virchow2_spot_tile_plan <- wsiTools::wsi_virchow2_spot_tile_plan
  }
  centers <- data.frame(x = spots$pxl_col_in_fullres,
                        y = spots$pxl_row_in_fullres, spot_id = spots$barcode)
  allow_upsample <- any(mpp > .5)
  plan <- wsi_virchow2_spot_tile_plan(slide, centers,
      source_mpp_xy = mpp,
      spacing_evidence = "verified original OME PhysicalSizeX/Y",
      allow_upsample = allow_upsample)
  utils::write.csv(plan, file.path(output_dir, "spot_tile_plan.csv"), row.names = FALSE)
  report <- list(sample = sample, image = normalizePath(image),
                 dimensions_wh = dims, image_mpp_xy = mpp,
                 source_upsampled = allow_upsample,
                 implied_spot_diameter_um = implied_spot_um,
                 spot_diameter_is_only_a_crosscheck = TRUE,
                 native_tile_wh = as.numeric(plan[1, c("native_width", "native_height")]),
                 virchow_tile_size_px = 224L,
                 effective_virchow_mpp_xy = as.numeric(plan[1, c("effective_mpp_x", "effective_mpp_y")]),
                 original_spots = nrow(spots), planned_spot_tiles = nrow(plan),
                 tile_images_saved = FALSE,
                 coordinate_system = "Space Ranger original full-resolution pixels")
  jsonlite::write_json(report, file.path(output_dir, "calibration.json"),
                       auto_unbox = TRUE, pretty = TRUE, digits = 16)
  invisible(report)
}
