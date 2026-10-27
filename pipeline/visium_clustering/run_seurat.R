#!/usr/bin/env Rscript
library(Seurat)
library(KODAMA)
library(wsiTools)
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))), "common.R"))

set.seed(1234)
a <- visium_args()
sample <- "CytAssist_11mm_FFPE_Human_Colorectal_Cancer"
h5 <- paste0(sample, "_filtered_feature_bc_matrix.h5")
obj <- Load10X_Spatial(data.dir = a$data, filename = h5,
                       assay = "Spatial", slice = "colorectal", filter.matrix = TRUE)
DefaultAssay(obj) <- "Spatial"
obj <- NormalizeData(obj, verbose = FALSE)
obj <- FindVariableFeatures(obj, selection.method = "vst", nfeatures = 2000,
                            verbose = FALSE)
genes <- VariableFeatures(obj)
obj <- ScaleData(obj, features = genes, verbose = FALSE)
obj <- RunPCA(obj, features = genes, npcs = 30, verbose = FALSE)
pca <- Embeddings(obj[["pca"]])[, 1:30, drop = FALSE]
positions <- visium_positions(a$data)
positions <- positions[match(rownames(pca), positions$barcode), , drop = FALSE]
if (anyNA(positions$barcode)) stop("Seurat PCA spot IDs do not match Space Ranger")
fit <- KODAMA::KODAMA.matrix(pca,
  spatial = as.matrix(positions[, c("pxl_col_in_fullres", "pxl_row_in_fullres")]),
  M = 100, spatial.resolution = 0.3, n.cores = 1, seed = 1234)
vis <- as.matrix(KODAMA::KODAMA.visualization(fit, method = "UMAP"))
if (nrow(vis) != nrow(pca) || ncol(vis) < 2L) stop("Seurat KODAMA row mismatch")
rownames(vis) <- rownames(pca)
obj[["KODAMA"]] <- CreateDimReducObject(embeddings = vis[, 1:2, drop = FALSE],
                                          key = "KODAMA_", assay = "Spatial")
obj <- FindNeighbors(obj, reduction = "KODAMA", dims = 1:2,
                     graph.name = c("KODAMA_nn", "KODAMA_snn"), verbose = FALSE)
best <- NULL
gap <- Inf
for (resolution in seq(0.10, 1.50, by = 0.05)) {
  set.seed(1234)
  trial <- FindClusters(obj, graph.name = "KODAMA_snn", algorithm = 1,
                        resolution = resolution, verbose = FALSE)
  labels <- as.character(Idents(trial))
  current <- abs(length(unique(labels)) - 6L)
  if (current < gap) { best <- labels; gap <- current; best_resolution <- resolution }
  if (current == 0L) break
}
obj$KODAMA_Louvain <- factor(best)
spots <- visium_write_clusters(colnames(obj), obj$KODAMA_Louvain,
                               a$data, a$out, "Seurat + KODAMA + Louvain")
write.csv(data.frame(barcode = rownames(vis), kodama_1 = vis[, 1],
                     kodama_2 = vis[, 2]),
          file.path(a$out, "kodama_embedding.csv"), row.names = FALSE)
saveRDS(list(barcodes = colnames(obj), embedding = vis,
             clusters = best, resolution = best_resolution),
        file.path(a$out, "seurat_clusters.rds"), compress = "gzip")
visium_export_tiles(spots, a$data, sample, "tif", a$out)
writeLines(capture.output(sessionInfo()), file.path(a$out, "r_session.txt"))
