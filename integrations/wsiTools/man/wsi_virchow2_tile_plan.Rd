\name{wsi_virchow2_tile_plan}
\alias{wsi_virchow2_tile_plan}
\title{Plan nonoverlapping, physically calibrated Virchow2 tiles}
\usage{
wsi_virchow2_tile_plan(slide, region = NULL, target_mpp = 0.5,
  tile_pixels = 224L, max_tiles = 1000000L, allow_upsample = FALSE,
  source_mpp_xy = NULL, spacing_evidence = NULL)
}
\arguments{
\item{slide}{A \code{wsi_slide} or image path.}
\item{region}{Optional named level-zero rectangle with x, y, width and height.}
\item{target_mpp}{Target micrometres per output pixel.}
\item{tile_pixels}{Square output tile size, normally 224.}
\item{max_tiles}{Safety cap on tile count.}
\item{allow_upsample}{Whether a coarser source may be interpolated.}
\item{source_mpp_xy}{Optional explicit x/y source spacing.}
\item{spacing_evidence}{Required provenance string for explicit spacing.}
}
\value{A row-major data frame with zero-based tile indices, level-zero crop coordinates, valid and padded native dimensions, 224-pixel output dimensions and source/effective x/y spacing.}
\description{Plans a complete, nonoverlapping tile grid without writing image files. Partial right and bottom edges are represented with white-padding dimensions. A caller may read, resize and encode each tile in memory, releasing it immediately after inference.}
\seealso{\code{\link{wsi_virchow2_geometry}}}
