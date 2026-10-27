#' Calibrated Virchow2 tile geometry
#'
#' The 224-pixel Virchow2 input represents 112 micrometres at 0.5
#' micrometres per pixel. Native crop dimensions are computed separately for
#' each image axis from the slide's microscopy calibration. TIFF display DPI
#' is not accepted as a substitute for physical pixel spacing.
#'
#' @param slide A `wsi_slide` or path accepted by [wsi_open()].
#' @param target_mpp Target micrometres per output pixel (default 0.5).
#' @param tile_pixels Output tile width and height (default 224).
#' @param allow_upsample Allow source pixels larger than target pixels. The
#'   default rejects such images because interpolation cannot recover detail.
#' @param source_mpp_xy Optional explicit source spacing (`x`, `y`) for images
#'   without embedded microscopy calibration. Supply `spacing_evidence` too.
#' @param spacing_evidence Source of an explicit calibration, such as a pinned
#'   dataset metadata field. Bare TIFF display DPI is not sufficient evidence.
#' @return A list with source and effective pixel spacing, native crop width
#'   and height, output size, and physical footprint.
#' @export
wsi_virchow2_geometry <- function(slide, target_mpp = 0.5,
                                  tile_pixels = 224L, allow_upsample = FALSE,
                                  source_mpp_xy = NULL, spacing_evidence = NULL) {
  if (is.character(slide) && length(slide) == 1L) {
    slide <- wsi_open(slide)
  }
  if (!inherits(slide, "wsi_slide")) {
    stop("`slide` must be a wsi_slide or an image path.", call. = FALSE)
  }
  target_mpp <- as.numeric(target_mpp)
  tile_pixels <- as.integer(tile_pixels)
  if (length(target_mpp) != 1L || !is.finite(target_mpp) || target_mpp <= 0 ||
      length(tile_pixels) != 1L || is.na(tile_pixels) || tile_pixels < 1L) {
    stop("`target_mpp` and `tile_pixels` must be positive scalars.", call. = FALSE)
  }
  properties <- wsi_properties(slide)
  trusted_keys <- c("openslide.mpp-x", "openslide.mpp-y", "ome.physical-size-x",
                    "ome.physical-size-y", "PhysicalSizeX", "PhysicalSizeY",
                    "microns-per-pixel-x", "microns-per-pixel-y")
  embedded <- any(trusted_keys %in% names(properties))
  if (is.null(source_mpp_xy) && !embedded) {
    stop("No trustworthy microscopy calibration; supply `source_mpp_xy` and `spacing_evidence`.",
         call. = FALSE)
  }
  if (!is.null(source_mpp_xy)) {
    if (!is.character(spacing_evidence) || length(spacing_evidence) != 1L ||
        is.na(spacing_evidence) || !nzchar(trimws(spacing_evidence))) {
      stop("Explicit `source_mpp_xy` requires non-empty `spacing_evidence`.", call. = FALSE)
    }
    mpp <- as.numeric(source_mpp_xy)
    if (embedded) {
      measured <- as.numeric(wsi_mpp(slide))
      if (all(is.finite(measured)) && any(abs(mpp - measured) / measured > 0.01)) {
        stop("Explicit spacing disagrees with embedded microscopy spacing by over 1%.",
             call. = FALSE)
      }
    }
  } else {
    mpp <- wsi_mpp(slide)
  }
  if (length(mpp) != 2L || any(!is.finite(mpp)) || any(mpp <= 0)) {
    stop("Microscopy pixel spacing is missing or invalid; refusing to infer it from magnification.",
         call. = FALSE)
  }
  if (!isTRUE(allow_upsample) && any(mpp > target_mpp * (1 + 1e-6))) {
    stop("Source spacing is coarser than Virchow2 target spacing; explicit `allow_upsample = TRUE` is required.",
         call. = FALSE)
  }
  native <- pmax(1L, as.integer(round(tile_pixels * target_mpp / as.numeric(mpp))))
  names(native) <- c("width", "height")
  effective <- native * as.numeric(mpp) / tile_pixels
  names(effective) <- c("x", "y")
  list(source_mpp_xy = unname(as.numeric(mpp)), target_mpp = target_mpp,
       tile_pixels = tile_pixels, native_tile_wh = unname(native),
       effective_mpp_xy = unname(effective),
       footprint_um_xy = unname(native * as.numeric(mpp)),
       upsampled = any(mpp > target_mpp * (1 + 1e-6)),
       spacing_evidence = if (is.null(source_mpp_xy)) "embedded microscopy metadata" else spacing_evidence)
}

#' Plan nonoverlapping Virchow2 tiles at the correct physical resolution
#'
#' Returns level-zero source coordinates and output geometry without writing
#' tile files. The caller can read each crop, pad partial right/bottom edges
#' white to `native_width`/`native_height`, resize to 224 × 224 RGB pixels,
#' encode it, and release the in-memory tile immediately.
#'
#' @param slide A `wsi_slide` or image path.
#' @param region Optional named level-zero pixel region (`x`, `y`, `width`,
#'   `height`); the default is the entire image.
#' @param max_tiles Safety cap on the number of planned tiles.
#' @inheritParams wsi_virchow2_geometry
#' @return Data frame in row-major order. `tile_index` is zero based;
#'   `valid_width` and `valid_height` exclude white edge padding.
#' @export
wsi_virchow2_tile_plan <- function(slide, region = NULL, target_mpp = 0.5,
                                  tile_pixels = 224L, max_tiles = 1000000L,
                                  allow_upsample = FALSE, source_mpp_xy = NULL,
                                  spacing_evidence = NULL) {
  if (is.character(slide) && length(slide) == 1L) {
    slide <- wsi_open(slide)
  }
  geometry <- wsi_virchow2_geometry(slide, target_mpp, tile_pixels, allow_upsample,
                                   source_mpp_xy, spacing_evidence)
  dimensions <- as.numeric(slide$dimensions[c("width", "height")])
  if (length(dimensions) != 2L || any(!is.finite(dimensions)) || any(dimensions < 1)) {
    stop("Slide dimensions are invalid.", call. = FALSE)
  }
  if (is.null(region)) {
    region <- c(x = 0, y = 0, width = dimensions[1L], height = dimensions[2L])
  }
  if (is.list(region)) region <- unlist(region, use.names = TRUE)
  if (!all(c("x", "y", "width", "height") %in% names(region))) {
    stop("`region` must contain x, y, width and height.", call. = FALSE)
  }
  region <- as.numeric(region[c("x", "y", "width", "height")])
  if (any(!is.finite(region)) || any(region != round(region)) || any(region[1:2] < 0) ||
      any(region[3:4] < 1) || any(region[1:2] + region[3:4] > dimensions)) {
    stop("`region` must be an integer level-zero rectangle inside the image.", call. = FALSE)
  }
  max_tiles <- as.numeric(max_tiles)
  if (length(max_tiles) != 1L || !is.finite(max_tiles) || max_tiles < 1) {
    stop("`max_tiles` must be positive.", call. = FALSE)
  }
  native_w <- geometry$native_tile_wh[1L]
  native_h <- geometry$native_tile_wh[2L]
  nx <- ceiling(region[3L] / native_w)
  ny <- ceiling(region[4L] / native_h)
  if (nx * ny > max_tiles) {
    stop(sprintf("Tile plan would contain %.0f tiles; `max_tiles` is %.0f.", nx * ny, max_tiles),
         call. = FALSE)
  }
  col <- rep(seq.int(0L, as.integer(nx) - 1L), times = as.integer(ny))
  row <- rep(seq.int(0L, as.integer(ny) - 1L), each = as.integer(nx))
  x <- as.integer(region[1L] + col * native_w)
  y <- as.integer(region[2L] + row * native_h)
  valid_w <- as.integer(pmin(native_w, region[1L] + region[3L] - x))
  valid_h <- as.integer(pmin(native_h, region[2L] + region[4L] - y))
  data.frame(tile_index = seq_along(x) - 1L, row = row, col = col,
             x = x, y = y, valid_width = valid_w, valid_height = valid_h,
             native_width = native_w, native_height = native_h,
             pad_left = 0L, pad_top = 0L,
             pad_right = native_w - valid_w, pad_bottom = native_h - valid_h,
             output_width = geometry$tile_pixels,
             output_height = geometry$tile_pixels,
             source_mpp_x = geometry$source_mpp_xy[1L],
             source_mpp_y = geometry$source_mpp_xy[2L],
             effective_mpp_x = geometry$effective_mpp_xy[1L],
             effective_mpp_y = geometry$effective_mpp_xy[2L])
}

#' Plan Virchow2 tiles centered on spatial transcriptomics spots
#'
#' `centers_xy` contains level-zero source-image pixel coordinates of spot
#' centers. Crops are set by verified microscopy spacing, independently of
#' Visium spot diameter. Crops touching any image edge are white-padded.
#'
#' @param slide An open `wsi_slide` or path.
#' @param centers_xy Data frame with numeric `x` and `y` spot centers and an
#'   optional `spot_id` column.
#' @inheritParams wsi_virchow2_geometry
#' @return One plan row per spot in the input order.
#' @export
wsi_virchow2_spot_tile_plan <- function(slide, centers_xy, target_mpp = 0.5,
                                        tile_pixels = 224L,
                                        allow_upsample = FALSE,
                                        source_mpp_xy = NULL,
                                        spacing_evidence = NULL) {
  if (is.character(slide) && length(slide) == 1L) slide <- wsi_open(slide)
  if (!is.data.frame(centers_xy) ||
      !all(c("x", "y") %in% names(centers_xy)) || nrow(centers_xy) < 1L) {
    stop("`centers_xy` must be a nonempty data frame with x and y.", call. = FALSE)
  }
  xy <- as.matrix(centers_xy[, c("x", "y")])
  storage.mode(xy) <- "double"
  dims <- as.numeric(slide$dimensions[c("width", "height")])
  if (any(!is.finite(xy)) || any(xy < 0) ||
      any(xy[, 1L] >= dims[1L]) || any(xy[, 2L] >= dims[2L])) {
    stop("Spot centers must be finite and inside the source image.", call. = FALSE)
  }
  geometry <- wsi_virchow2_geometry(slide, target_mpp, tile_pixels,
                                   allow_upsample, source_mpp_xy,
                                   spacing_evidence)
  nw <- geometry$native_tile_wh[1L]
  nh <- geometry$native_tile_wh[2L]
  desired_x <- as.integer(round(xy[, 1L] - (nw - 1L) / 2))
  desired_y <- as.integer(round(xy[, 2L] - (nh - 1L) / 2))
  x <- pmax(0L, desired_x)
  y <- pmax(0L, desired_y)
  right <- pmin(as.integer(dims[1L]), desired_x + nw)
  bottom <- pmin(as.integer(dims[2L]), desired_y + nh)
  valid_w <- right - x
  valid_h <- bottom - y
  pad_left <- x - desired_x
  pad_top <- y - desired_y
  result <- data.frame(tile_index = seq_len(nrow(xy)) - 1L,
                       x = x, y = y, valid_width = valid_w,
                       valid_height = valid_h,
                       native_width = nw, native_height = nh,
                       pad_left = pad_left, pad_top = pad_top,
                       pad_right = nw - pad_left - valid_w,
                       pad_bottom = nh - pad_top - valid_h,
                       output_width = geometry$tile_pixels,
                       output_height = geometry$tile_pixels,
                       source_mpp_x = geometry$source_mpp_xy[1L],
                       source_mpp_y = geometry$source_mpp_xy[2L],
                       effective_mpp_x = geometry$effective_mpp_xy[1L],
                       effective_mpp_y = geometry$effective_mpp_xy[2L])
  if ("spot_id" %in% names(centers_xy)) {
    result$spot_id <- as.character(centers_xy$spot_id)
    if (anyNA(result$spot_id) || anyDuplicated(result$spot_id)) {
      stop("`spot_id` must be nonmissing and unique.", call. = FALSE)
    }
  }
  result
}

#' Read one calibrated Virchow2 tile into memory
#'
#' The native level-zero crop in one row of [wsi_virchow2_tile_plan()] is
#' read, white-padded at partial image edges, and resized to a 224 x 224 RGB
#' integer array. The result is returned in `height, width, channel` order;
#' no tile image is retained on disk. An optional temporary backend image
#' created by [wsi_read_region()] is deleted by wsiTools after each read.
#'
#' @param slide An open `wsi_slide`.
#' @param tile One row of a Virchow2 tile plan.
#' @return Integer array with dimensions `224 x 224 x 3` and values 0..255.
#' @export
wsi_virchow2_tile <- function(slide, tile) {
  if (!inherits(slide, "wsi_slide")) {
    stop("`slide` must be an open wsi_slide.", call. = FALSE)
  }
  if (!is.data.frame(tile) || nrow(tile) != 1L) {
    stop("`tile` must be exactly one row of a tile plan.", call. = FALSE)
  }
  fields <- c("x", "y", "valid_width", "valid_height", "native_width",
              "native_height", "output_width", "output_height")
  if (!all(fields %in% names(tile))) {
    stop("Tile row is missing required plan coordinates.", call. = FALSE)
  }
  values <- as.integer(unlist(tile[1L, fields], use.names = FALSE))
  names(values) <- fields
  if (anyNA(values) || any(values[c("x", "y")] < 0L) ||
      any(values[c("valid_width", "valid_height")] < 1L) ||
      values[["native_width"]] < values[["valid_width"]] ||
      values[["native_height"]] < values[["valid_height"]] ||
      values[["output_width"]] != 224L || values[["output_height"]] != 224L ||
      values[["x"]] + values[["valid_width"]] > slide$dimensions[["width"]] ||
      values[["y"]] + values[["valid_height"]] > slide$dimensions[["height"]]) {
    stop("Tile row does not describe a valid 224-pixel Virchow2 crop.", call. = FALSE)
  }
  if (!requireNamespace("magick", quietly = TRUE)) {
    stop("The optional magick package is needed for Lanczos tile resize.", call. = FALSE)
  }
  patch <- wsi_read_region(slide, values[["x"]], values[["y"]],
                           values[["valid_width"]], values[["valid_height"]],
                           level = 0, format = "array")
  if (length(dim(patch)) != 3L || dim(patch)[3L] < 3L ||
      any(dim(patch)[1:2] != values[c("valid_height", "valid_width")])) {
    stop("Backend returned an unexpected crop shape.", call. = FALSE)
  }
  rgb <- patch[, , 1:3, drop = FALSE]
  if (dim(patch)[3L] >= 4L) {
    alpha <- patch[, , 4L]
    for (channel in 1:3) rgb[, , channel] <- rgb[, , channel] * alpha + 1 - alpha
  }
  pad_left <- if ("pad_left" %in% names(tile)) as.integer(tile$pad_left[[1L]]) else 0L
  pad_top <- if ("pad_top" %in% names(tile)) as.integer(tile$pad_top[[1L]]) else 0L
  if (is.na(pad_left) || is.na(pad_top) || pad_left < 0L || pad_top < 0L ||
      pad_left + values[["valid_width"]] > values[["native_width"]] ||
      pad_top + values[["valid_height"]] > values[["native_height"]]) {
    stop("Tile padding is inconsistent with the native crop.", call. = FALSE)
  }
  canvas <- array(1, dim = c(values[["native_height"]], values[["native_width"]], 3L))
  canvas[pad_top + seq_len(values[["valid_height"]]),
         pad_left + seq_len(values[["valid_width"]]), ] <-
    pmax(0, pmin(1, rgb))
  image <- magick::image_read(grDevices::as.raster(canvas))
  image <- magick::image_resize(image, "224x224!", filter = "Lanczos")
  raw <- magick::image_data(image, channels = "rgb")
  pixels <- as.integer(raw)
  dim(pixels) <- dim(raw)
  aperm(pixels, c(3L, 2L, 1L))
}
