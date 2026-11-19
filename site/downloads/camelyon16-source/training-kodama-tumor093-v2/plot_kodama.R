#!/usr/bin/env Rscript
# Unsupervised, CPU-only KODAMA layout, colored post hoc by fitting annotation.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop('Usage: Rscript plot_kodama.R INPUT_DIR')
folder <- normalizePath(args[[1]], mustWork = TRUE)
if (file.exists(file.path(folder, 'KODAMA_COMPLETE.R'))) stop('Use a new version for rerun')
for (package in c('KODAMA', 'fastEmbedR')) {
  if (!requireNamespace(package, quietly = TRUE)) stop('Missing pinned package: ', package)
}
if (as.character(utils::packageVersion('KODAMA')) != '0.99.7' ||
    as.character(utils::packageVersion('fastEmbedR')) != '0.99.5')
  stop('Pinned KODAMA 0.99.7 and fastEmbedR 0.99.5 required')
x <- as.matrix(utils::read.csv(file.path(folder, 'tile-means-1280.csv'),
                               header = FALSE, check.names = FALSE))
if (!is.numeric(x) || !identical(dim(x), c(900L, 1280L)) || any(!is.finite(x)))
  stop('Expected 900 finite tile means with 1280 channels')
set.seed(20261119)
pca <- KODAMA::kodama_pca(x, ncomp = 30L, center = TRUE, scale = TRUE,
                          backend = 'cpu', n.cores = 2L, seed = 20261119L)
pcs <- as.matrix(pca$scores)
if (!identical(dim(pcs), c(900L, 30L)) || any(!is.finite(pcs)))
  stop('PCA output differs from bounded sample')
set.seed(20261119)
fit <- KODAMA::KODAMA.matrix(pcs, spatial = NULL, M = 100L,
                             n.cores = 2L, backend = 'cpu', seed = 20261119L,
                             return.graph = TRUE, progress = FALSE)
set.seed(20261119)
embedding <- as.matrix(KODAMA::KODAMA.visualization(
  fit, method = 'UMAP', backend = 'cpu', n.cores = 2L,
  seed = 20261119L, n.epochs = 200L))
if (nrow(embedding) != nrow(x) || ncol(embedding) < 2L ||
    any(!is.finite(embedding))) stop('KODAMA layout differs from sample')
embedding <- embedding[, 1:2, drop = FALSE]
label_path <- file.path(folder, 'tile-expert-center-label.csv')
labels <- scan(label_path, what = integer(), quiet = TRUE, sep = ',')
if (length(labels) != nrow(x) || !all(labels %in% c(1L, 2L)) ||
    sum(labels == 2L) < 8L) stop('Training-center labels differ from sample')
grDevices::png(file.path(folder, 'tumor_093-kodama.png'),
               width = 1700, height = 1150, res = 170)
par(mar = c(5.2, 5.2, 4.5, 2.0), xaxs = 'i', yaxs = 'i')
plot(embedding[labels == 1L, , drop = FALSE],
     col = grDevices::adjustcolor('#207466', alpha.f = .67),
     pch = 16, cex = .82, xlab = 'KODAMA dimension 1',
     ylab = 'KODAMA dimension 2',
     main = 'tumor_093: training-tissue Virchow2 tile structure')
points(embedding[labels == 2L, , drop = FALSE], pch = 21, cex = 1.25,
       bg = '#d4475b', col = '#641c2b', lwd = 1.1)
legend('topright', legend = c('Nonmetastatic center', 'Metastatic center'),
       pt.bg = c('#207466', '#d4475b'), col = c('#207466', '#641c2b'),
       pch = c(16, 21), cex = .8, bg = 'white')
grDevices::dev.off()
utils::write.csv(data.frame(tile = seq_len(nrow(x)), kodama_1 = embedding[, 1],
                            kodama_2 = embedding[, 2], expert_center_label = labels),
                 file.path(folder, 'kodama-tile-layout.csv'), row.names = FALSE)
summary <- list(format = 'camelyon16-tumor093-kodama-plot-v1', status = 'COMPLETE',
                slide = 'tumor_093', sampled_tissue_tiles = nrow(x),
                full_virchow2_channels = ncol(x), pca_dimensions = ncol(pcs),
                spatial_coordinates_used = FALSE,
                expert_labels_used_in_fit = FALSE,
                expert_labels_used_for_plot_color = TRUE,
                kodama_version = as.character(utils::packageVersion('KODAMA')),
                fastEmbedR_version = as.character(utils::packageVersion('fastEmbedR')),
                backend = 'cpu', n.cores = 2L, seed = 20261119L,
                M = 100L, visualization = 'UMAP',
                metastatic_center_tiles = sum(labels == 2L),
                nonmetastatic_center_tiles = sum(labels == 1L),
                supervised_model_changed = FALSE, test_scores_changed = FALSE)
json <- paste0(capture.output(dput(summary)), collapse = '\n')
writeLines(json, file.path(folder, 'KODAMA_COMPLETE.R'))
writeLines(capture.output(sessionInfo()), file.path(folder, 'R_SESSION.txt'))
cat('KODAMA groups:', summary$groups, '\n')
