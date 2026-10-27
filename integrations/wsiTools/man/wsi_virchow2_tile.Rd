\name{wsi_virchow2_tile}
\alias{wsi_virchow2_tile}
\title{Read one calibrated Virchow2 tile into memory}
\usage{
wsi_virchow2_tile(slide, tile)
}
\arguments{
\item{slide}{An open \code{wsi_slide}.}
\item{tile}{Exactly one row of \code{wsi_virchow2_tile_plan()}.}
}
\value{An integer array in height, width, RGB channel order, with shape
224 by 224 by 3 and values from 0 to 255.}
\description{Reads the native level-zero crop, pads partial image edges white,
and resizes to Virchow2's calibrated 224-pixel RGB input. No tile file is
retained. The optional \code{magick} package is required for resizing.}
