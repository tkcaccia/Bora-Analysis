\name{wsi_virchow2_spot_tile_plan}
\alias{wsi_virchow2_spot_tile_plan}
\title{Plan calibrated Virchow2 tiles around spatial spot centers}
\usage{
wsi_virchow2_spot_tile_plan(slide, centers_xy, target_mpp = 0.5,
  tile_pixels = 224L, allow_upsample = FALSE, source_mpp_xy = NULL,
  spacing_evidence = NULL)
}
\arguments{
\item{slide}{An open \code{wsi_slide} or image path.}
\item{centers_xy}{Data frame of level-zero \code{x}, \code{y} spot centers,
optionally with unique \code{spot_id}.}
\item{target_mpp}{Output micrometres per pixel; default 0.5.}
\item{tile_pixels}{Output width and height in pixels; default 224.}
\item{allow_upsample}{Explicitly permit source spacing coarser than target.}
\item{source_mpp_xy}{Optional verified source spacing for the x and y axes.}
\item{spacing_evidence}{Required source citation when explicit spacing is used.}
}
\value{One tile-plan row per spot, including level-zero crop coordinates and
white padding on all four sides.}
\description{Spot diameter does not determine crop size. The physical
Virchow2 field of view is set by verified source microscopy spacing.}
