\name{wsi_virchow2_geometry}
\alias{wsi_virchow2_geometry}
\title{Calibrate a Virchow2 tile to microscopy pixel spacing}
\usage{
wsi_virchow2_geometry(slide, target_mpp = 0.5, tile_pixels = 224L,
  allow_upsample = FALSE, source_mpp_xy = NULL, spacing_evidence = NULL)
}
\arguments{
\item{slide}{A \code{wsi_slide} or image path accepted by \code{wsi_open()}.}
\item{target_mpp}{Target micrometres per output pixel.}
\item{tile_pixels}{Output tile width and height in pixels.}
\item{allow_upsample}{Allow interpolation from a source coarser than the target. Default \code{FALSE}.}
\item{source_mpp_xy}{Optional explicit x/y source pixel spacing for an image without embedded microscopy metadata.}
\item{spacing_evidence}{Required provenance string when \code{source_mpp_xy} is supplied.}
}
\value{A list with source and effective x/y spacing, native crop dimensions, output tile dimensions and physical footprint.}
\description{Computes the native crop that covers approximately 112 by 112 micrometres and resamples to the Virchow2 224 by 224 input. Embedded microscopy metadata or a sourced explicit calibration is required; display DPI alone is rejected.}
\seealso{\code{\link{wsi_virchow2_tile_plan}}}
