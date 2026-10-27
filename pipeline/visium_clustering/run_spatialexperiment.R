#!/usr/bin/env Rscript
library(SpatialExperiment)
library(DropletUtils)
library(scuttle)
library(scran)
library(scater)
library(KODAMA)
library(igraph)
library(RANN)
library(wsiTools)
source(file.path(dirname(sub("^--file=", "", grep("^--file=", commandArgs(), value = TRUE))), "common.R"))

set.seed(1234)
a <- visium_args()
sample <- "CytAssist_FFPE_Protein_Expression_Human_Tonsil_AddOns"
sce <- DropletUtils::read10xCounts(file.path(a$data, paste0(sample,
                                    "_filtered_feature_bc_matrix.h5")),
                                    col.names = TRUE)
if ("Type" %in% names(rowData(sce)))
  sce <- sce[rowData(sce)$Type == "Gene Expression", ]
positions <- visium_positions(a$data)
positions <- positions[positions$barcode %in% colnames(sce), , drop = FALSE]
sce <- sce[, positions$barcode]
stopifnot(identical(colnames(sce), positions$barcode))
spe <- SpatialExperiment(
  assays = list(counts = counts(sce)), rowData = rowData(sce),
  colData = S4Vectors::DataFrame(positions[, c("in_tissue", "array_row", "array_col")],
                                row.names = positions$barcode),
  spatialCoords = as.matrix(positions[, c("pxl_col_in_fullres", "pxl_row_in_fullres")]),
  sample_id = "tonsil")
rownames(spe) <- make.unique(if ("Symbol" %in% names(rowData(sce)))
                               as.character(rowData(sce)$Symbol) else rownames(sce))
spe <- spe[, Matrix::colSums(counts(spe)) > 0]
spe <- scuttle::computeLibraryFactors(spe)
spe <- scuttle::logNormCounts(spe)
variable_fit <- scran::modelGeneVar(spe)
genes <- scran::getTopHVGs(variable_fit, n = 2000)
spe <- scater::runPCA(spe, ncomponents = 50,
                      subset_row = genes, scale = TRUE)
pca <- SingleCellExperiment::reducedDim(spe, "PCA")
xy <- SpatialExperiment::spatialCoords(spe)
fit <- KODAMA::KODAMA.matrix(pca, spatial = xy, M = 100,
                             spatial.resolution = 0.3, n.cores = 1, seed = 1234)
vis <- as.matrix(KODAMA::KODAMA.visualization(fit, method = "UMAP"))
if (nrow(vis) != ncol(spe) || ncol(vis) < 2L || any(!is.finite(vis)))
  stop("KODAMA embedding does not align with tonsil spots")
SingleCellExperiment::reducedDim(spe, "KODAMA") <- vis[, 1:2, drop = FALSE]

# Complete the original script's missing clustering step on the KODAMA map.
# An undirected 20-neighbor graph is held fixed while resolution is searched.
neighbors <- RANN::nn2(vis[, 1:2, drop = FALSE], k = 21)$nn.idx[, -1, drop = FALSE]
edges <- cbind(rep(seq_len(nrow(vis)), each = ncol(neighbors)),
               as.vector(t(neighbors)))
graph <- igraph::simplify(igraph::graph_from_edgelist(edges, directed = FALSE),
                          remove.multiple = TRUE, remove.loops = TRUE)
gap <- Inf
best <- NULL
for (resolution in seq(0.10, 1.50, by = 0.05)) {
  set.seed(1234)
  labels <- igraph::membership(igraph::cluster_louvain(graph,
                                                       resolution = resolution))
  current <- abs(length(unique(labels)) - 6L)
  if (current < gap) { best <- labels; gap <- current; best_resolution <- resolution }
  if (current == 0L) break
}
spe$KODAMA_Louvain <- factor(best)
spots <- visium_write_clusters(colnames(spe), spe$KODAMA_Louvain,
                               a$data, a$out, "SpatialExperiment + KODAMA + Louvain")
write.csv(data.frame(barcode = colnames(spe), kodama_1 = vis[, 1],
                     kodama_2 = vis[, 2]),
          file.path(a$out, "kodama_embedding.csv"), row.names = FALSE)
HDF5Array::saveHDF5SummarizedExperiment(spe,
    dir = file.path(a$out, "spatialexperiment_clusters_hdf5"), replace = TRUE)
writeLines(as.character(best_resolution), file.path(a$out, "louvain_resolution.txt"))
visium_export_tiles(spots, a$data, sample, "btf", a$out)
writeLines(capture.output(sessionInfo()), file.path(a$out, "r_session.txt"))
