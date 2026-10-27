#!/usr/bin/env Rscript
options(giotto.check_core = FALSE, giotto.check_version = FALSE, giotto.cores = 2L)
data.table::setDTthreads(2L)
library(Giotto)
library(DropletUtils)
library(scuttle)
library(scran)
library(SingleCellExperiment)
library(KODAMA)
library(KODAMAextra)
library(wsiTools)
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))), "common.R"))

set.seed(1234)
a <- visium_args()
sample <- "CytAssist_FFPE_Human_Breast_Cancer"
if (!dir.exists(file.path(a$data, "filtered_feature_bc_matrix")))
  stop("Extract the filtered_feature_bc_matrix.tar.gz archive before running Giotto")
instructions <- createGiottoInstructions(save_dir = a$out, save_plot = FALSE,
                                         show_plot = FALSE)
obj <- createGiottoVisiumObject(visium_dir = a$data, expr_data = "filter",
                               png_name = "tissue_lowres_image.png",
                               gene_column_index = 2,
                               instructions = instructions, cores = 2L,
                               verbose = FALSE)
meta <- pDataDT(obj)
if ("in_tissue" %in% names(meta))
  obj <- subsetGiotto(obj, cell_ids = as.character(meta$cell_ID[meta$in_tissue == 1]),
                      verbose = FALSE)
obj <- filterGiotto(obj, expression_threshold = 1, feat_det_in_min_cells = 20,
                    min_det_feats_per_cell = 200, expression_values = "raw",
                    verbose = FALSE)
obj <- normalizeGiotto(obj, scalefactor = 6000, verbose = FALSE)
sce <- DropletUtils::read10xCounts(file.path(a$data, "filtered_feature_bc_matrix"),
                                   col.names = TRUE)
if ("Type" %in% names(rowData(sce)))
  sce <- sce[rowData(sce)$Type == "Gene Expression", ]
sce <- scuttle::logNormCounts(sce)
fit <- scran::modelGeneVar(sce)
ranked <- order(fit$bio, decreasing = TRUE, na.last = NA)
symbols <- if ("Symbol" %in% names(rowData(sce)))
  as.character(rowData(sce)$Symbol[ranked]) else rownames(sce)[ranked]
available <- as.character(fDataDT(obj)$feat_ID)
genes <- head(unique(symbols[!is.na(symbols) & symbols %in% available]), 2000)
if (length(genes) < 2000L) stop("Fewer than 2000 genes match Giotto features")
obj <- subsetGiotto(obj, feat_ids = genes, verbose = FALSE)
obj <- runPCA(obj, feats_to_use = genes, ncp = 30, scale_unit = TRUE, center = TRUE)
pca <- as.matrix(getDimReduction(obj, reduction_method = "pca", name = "pca",
                                 output = "matrix"))
if (ncol(pca) < 30L || is.null(rownames(pca)))
  stop("Giotto PCA must retain named spot rows and 30 components")
positions <- visium_positions(a$data)
positions <- positions[match(rownames(pca), positions$barcode), , drop = FALSE]
if (anyNA(positions$barcode)) stop("Giotto PCA spot IDs do not match Space Ranger")
fit <- KODAMA::KODAMA.matrix(pca[, 1:30, drop = FALSE],
  spatial = as.matrix(positions[, c("pxl_col_in_fullres", "pxl_row_in_fullres")]),
  M = 100, spatial.resolution = 0.3, n.cores = 1, seed = 1234)
vis <- as.matrix(KODAMA::KODAMA.visualization(fit, method = "UMAP"))
if (nrow(vis) != nrow(pca) || ncol(vis) < 2L) stop("Giotto KODAMA row mismatch")
cluster <- visium_louvain(vis[, 1:2, drop = FALSE])
best_resolution <- cluster$resolution
spots <- visium_write_clusters(rownames(pca), cluster$labels,
                               a$data, a$out, "Giotto + KODAMA + Louvain")
write.csv(data.frame(barcode = rownames(pca), kodama_1 = vis[, 1],
                     kodama_2 = vis[, 2]),
          file.path(a$out, "kodama_embedding.csv"), row.names = FALSE)
saveRDS(list(barcodes = rownames(pca), embedding = vis,
             clusters = cluster$labels, resolution = best_resolution),
        file.path(a$out, "giotto_clusters.rds"), compress = "gzip")
if (!file.exists(file.path(a$out, "native_tiles_manifest.csv")))
  visium_export_tiles(spots, a$data, sample, "tif", a$out)
writeLines(capture.output(sessionInfo()), file.path(a$out, "r_session.txt"))
