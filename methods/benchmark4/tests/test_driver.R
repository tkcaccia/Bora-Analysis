#!/usr/bin/env Rscript
# Dependency-free orchestration test. Real encoder inference needs licensed
# weights and the complete 10x source datasets; it is not performed here.
script <- sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE)[1L])
source(file.path(dirname(dirname(normalizePath(script))), "driver.R"))
test_lib <- Sys.getenv("BORA_TEST_R_LIB")
if (nzchar(test_lib)) .libPaths(c(test_lib, .libPaths()))
root <- tempfile("bora-b4-test-")
dir.create(root)
data_dir <- file.path(root, "data")
front_dir <- file.path(root, "front")
model <- file.path(root, "model")
dir.create(data_dir)
dir.create(front_dir)
dir.create(model)
image <- file.path(data_dir, "he.tif")
support <- file.path(data_dir, "tissue.geojson")
invisible(file.create(image, support))
writeLines('visium_args <- function() { a <- commandArgs(TRUE); stopifnot(length(a)==2L); dir.create(a[[2L]]); list(data=a[[1L]], out=a[[2L]]) }',
           file.path(front_dir, "common.R"))
for (route in c("giotto", "seurat", "spatialexperiment")) {
  name <- switch(route, giotto="run_giotto.R", seurat="run_seurat.R",
                 spatialexperiment="run_spatialexperiment.R")
  var <- if (route == "spatialexperiment") "spe" else "obj"
  writeLines(c(
    'source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE))), "common.R"))',
    'a <- visium_args()',
    sprintf('%s <- data.frame(spot_id=c("a","b"), x=c(10,30), y=c(11,31), KODAMA_Louvain=c("1","2"))', var),
    'stopifnot(dir.exists(a$out))'), file.path(front_dir, name))
  out <- file.path(root, paste0("result-", route))
  spatial_fun <- function(image, object, cluster_column, tissue_mask,
                          model_dir, source_mpp_xy, output_dir) {
    stopifnot(nrow(object)==2L, identical(cluster_column,"KODAMA_Louvain"),
              identical(source_mpp_xy,c(0.5,0.6)),
              identical(object$KODAMA_Louvain,c("1","2")))
    dir.create(output_dir)
    paths <- file.path(output_dir,c("mask.tif","result.geojson","receipt.json"))
    writeLines("mock-mask",paths[[1L]])
    writeLines('{"type":"FeatureCollection","features":[]}',paths[[2L]])
    writeLines('{"status":"COMPLETE"}',paths[[3L]])
    list(mask=paths[[1L]],geojson=paths[[2L]],manifest=paths[[3L]])
  }
  result <- bora_benchmark4_run(route,data_dir,image,support,model,out,
    c(0.5,0.6),front_dir,spatial_fun=spatial_fun,
    giotto_attach=function(object, work, data_dir) object)
  stopifnot(file.exists(result$geojson),
            file.exists(file.path(out,"expression-groups")))
}
cat("PASS: 3/3 Visium front-end handoffs reach Bora.spatial and GeoJSON\n")

# Exercise the real Giotto metadata attachment and pixel-coordinate check.
if (requireNamespace("Giotto", quietly = TRUE) &&
    requireNamespace("Bora", quietly = TRUE)) {
  writeLines(c(
    'visium_args <- function() { a <- commandArgs(TRUE); dir.create(a[[2L]]); list(data=a[[1L]], out=a[[2L]]) }',
    'visium_positions <- function(data_dir) data.frame(barcode=c("a","b","c"), pxl_col_in_fullres=c(10,20,30), pxl_row_in_fullres=c(11,21,31))'),
    file.path(front_dir, "common.R"))
  writeLines(c(
    'source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value=TRUE))), "common.R"))',
    'a <- visium_args()',
    'counts <- matrix(1L,2L,3L,dimnames=list(c("g1","g2"),c("a","b","c")))',
    'loc <- data.frame(sdimx=c(10,20,30),sdimy=c(11,21,31),cell_ID=c("a","b","c"))',
    'obj <- Giotto::createGiottoObject(expression=counts,spatial_locs=loc)',
    'pca <- matrix(1,3,2,dimnames=list(c("a","b","c"),NULL))',
    'cluster <- list(labels=c("1","2","1"))'),
    file.path(front_dir, "run_giotto.R"))
  verified <- FALSE
  actual_giotto <- function(image, object, cluster_column, tissue_mask,
                            model_dir, source_mpp_xy, output_dir) {
    spots <- Bora:::.bora_spots(object, cluster_column, NULL, NULL)
    stopifnot(identical(spots$spot_id,c("a","b","c")),
              identical(spots$cluster,c("1","2","1")),
              identical(spots$x,c(10,20,30)))
    verified <<- TRUE
    dir.create(output_dir)
    paths <- file.path(output_dir,c("mask.tif","result.geojson","receipt.json"))
    invisible(file.create(paths))
    list(mask=paths[[1L]],geojson=paths[[2L]],manifest=paths[[3L]])
  }
  bora_benchmark4_run("giotto",data_dir,image,support,model,
    file.path(root,"giotto-real"),c(0.5,0.6),front_dir,
    spatial_fun=actual_giotto)
  stopifnot(verified)
  cat("PASS: Giotto group attachment and full-resolution spot registration\n")
}
unlink(root, recursive = TRUE)
