# Native fastPLS tissue-group evolution

The user-requested v2 cohort excludes PUMA from both held-out development
splits without changing training rows or moving excluded images into training.
It has 78 test and 38 validation images: BCSS42, TIGER26, GCSS24 and IGNITE24.
The preceding reference viewer contained 51 non-PUMA test images; the expansion
adds 27 test images. Evaluation uses the same 116 images in every generation.
The partial v1 run is archived, not combined with v2 scores. Identical frozen
classifier predictions may be reused because their training and inputs are
unchanged; all output geometry and quality scores are regenerated.

Final cleanup checks continuous polygon-part area as well as connected native
raster counts. Parts and enclosed holes strictly below20 native pixels of area
are removed/filled, while exactly20 remains. Adjacent tissue fills a removed
part by longest shared boundary; isolated parts surrounded by glass are removed
to background. No dilation of the exterior glass is used. The calibrated shared
diagonal fitting remains active.

This experiment identifies whole anonymous Bora groups, including all disconnected pieces. It does not replace the accepted expert-derived input tiling with clusters. The previous RGB9+19 conversion and Bora-AF refinement remain fixed. These are reference-conditioned refinement experiments, not autonomous segmentation results.

Ten evaluated generations contain20 candidates each. The first generation has20 distinct mechanisms. Each subsequent generation retains five recipes and adds three structural descendants of each parent. Changing a seed or component count is not counted as a new algorithm. Recipe trees, all predictions, all20 scorecards and the best-five visual masks are retained for every generation.

## Initial mechanisms

| ID | Mechanism | Actual implementation |
|---|---|---|
| P01 | Whole group discriminant | Native PLS-LDA on the full 1280-channel group mean. Accepted reference classifier. |
| P02 | Covariance geometry | Log-Euclidean covariance of training-only label-free token coordinates, combined with group mean. Models correlated within-group heterogeneity rather than only channel SD. |
| P03 | Tissue composition | Native multiresponse PLS predicts annotated training tissue fractions; response ordering identifies each whole group. |
| P04 | Shared interface context | Native PLS-LDA sees a group mean and its contrast to physically shared-boundary-weighted neighbors. |
| P05 | Tile rank consensus | Native tile PLS-LDA rankings are pooled with exact tile-to-group area weights using Borda consensus. |
| P06 | Multiple instance attention | A native PLS instance head determines confidence-weighted attention within each group before its tissue evidence is aggregated. |
| P07 | Hierarchical routing | Training class prototypes define a binary class tree. Native binary PLS heads provide soft path evidence to every leaf. |
| P08 | Pairwise tournament | A native binary PLS model distinguishes each training class pair. Pairwise margins are coupled to a multiclass ranking. |
| P09 | Error correcting codes | A native multiresponse PLS predicts a deterministic redundant class code. Decoding uses distance to valid codewords. |
| P10 | Relational tissue embedding | Native PLS regresses a low-dimensional class-prototype embedding; tissue identity is decoded by proximity in that learned label space. |
| P11 | Nonlinear kernel PLS | The native fastPLS kernelpls RBF route models nonlinear similarity to training groups before LDA. Kernel scale is estimated on training only. |
| P12 | Orthogonal nuisance separation | Native OPLS removes predictor variation orthogonal to the supervised response before class discrimination. |
| P13 | Residual response boosting | Sequential native PLS regressors fit residual class responses; stages form an additive discriminant. |
| P14 | Source bagged subspaces | Multiple native PLS heads use source-group bootstrap samples and different feature subspaces; ranks are aggregated across models. |
| P15 | Local expert mixture | Training-only feature clusters gate specialized native PLS heads, with a global head for sparse or unseen neighborhoods. |
| P16 | Distribution quantile profiles | Weighted quantiles of training-only label-free token projections describe multimodal tissue groups; a native PLS head learns their identity. |
| P17 | Spatial tissue pooling | Physically defined interior and interface feature summaries, with spatial moments, feed a native PLS group head. |
| P18 | Image composition transport | A group PLS head and a separate image-composition PLS head are reconciled by soft transport over the image groups. Allows repeated tissue roles. |
| P19 | Latent class densities | Native PLS supplies the supervised latent coordinates; shrinkage class-specific Gaussian densities yield a generative tissue ranking. |
| P20 | Tissue verification metric | A native PLS verifier learns same-tissue versus different-tissue relations from training pairs, then ranks training-derived tissue prototypes. |

All learned tissue decisions use the installed native C++ fastPLS CPU backend. Spatial/distribution descriptors use a shared label-free12-coordinate training PCA; this prevents class targets entering the descriptor projection used by internal teacher cross-fitting. Group means still retain all1280 original feature channels. Clustering, rank aggregation, density evaluation and code decoding are explicitly reported supporting operations, not separate hidden classifiers.

## Structural evolution

- **distillation**: Compress a parent response surface into a native PLS student using training-only cross-fitted teacher evidence.
- **residual_specialist**: Add a native PLS residual expert for training-only cross-fitted confusions, rather than changing component count.
- **context_transport**: Couple parent group evidence to an independently fitted native PLS image-composition prior.
- **distribution_expert**: Fuse parent evidence with a native PLS distribution-profile expert; the new source of evidence is token heterogeneity.
- **hierarchical_decoder**: Replace flat response decoding with native PLS class-tree routing.
- **verification_decoder**: Use a native PLS pair verifier to adjudicate parent tissue alternatives.
- **coded_residual**: Fit redundant error-correcting class-code residuals through a native PLS stage.
- **local_residual**: Learn parent residuals within training-only feature neighborhoods using native PLS experts.
- **source_bagged_student**: Distill source-group bagged parent evidence to a compact native PLS student.
- **latent_density_fusion**: Fuse parent class evidence with shrinkage densities in supervised native PLS coordinates.
- **orthogonal_student**: Use native OPLS to isolate class-relevant variation while distilling parent responses.
- **spatial_expert**: Add physically pooled within-group spatial evidence through a native PLS expert.

Teacher class heads are cross-fitted in three source-group folds. The fixed descriptor PCA uses training features without class targets. Source-bagged students additionally use eight training-source subsamples and out-of-bag teacher evidence. At graph depth four, the parent is compressed to a native PLS student before applying the child's named innovation, limiting deployment graph growth. This is a reproducible bounded search, not a claim of discovering20 new mathematical model families in every generation.

## Acquisition, eligibility and splits

The acquisition adds up to40 GCSS fields and24 IGNITE fields from previously unused source groups. Native and working micrometres per pixel, provider provenance, WSI/ROI status and download scope remain in acquisition plans. Existing physical field, tile-count and multiple-tissue eligibility rules are preserved. Unknown optical resolution is not invented. Additional cache and fields use Chiamaka's NVMe drive; the existing20GiB acquisition scratch guard is retained.

Existing training/validation/test source groups remain immutable. Provider test data stay reserved. The append-only ontology preserves all existing role IDs and meanings. Training targets and prototypes use training sources only. Every candidate predicts every evaluation case before quality scoring begins.

The fixed cohort consists of eligible previously reserved validation and test images plus qualified additions. The user explicitly requested adaptive test-driven selection. These are therefore **development evaluation results**, not an untouched final test or an unbiased estimate of final-model performance. The viewer labels each image's split. Source groups are not universally verified patient identities; known TCGA cases and exact duplicates are linked, and unknown cross-dataset links remain a limitation.

## Ranking and timing

The predeclared score is0.50 times mean final GeoJSON IoU, plus0.30 times group balanced accuracy, plus0.20 times group top-one accuracy. Subtract0.05 times the mean cached deployment time above one second per image. Also subtract0.002 log2(1+native head count) plus0.002 per additional structural stage. This complexity penalty is a transparent simplicity proxy, not an objective measure of elegance.

Deployment timing includes cached Bora refinement, group feature pooling, required additional descriptor work, warm prediction, microscopic-region cleanup and GeoJSON export. Encoder inference and fit/startup are reported separately. Descriptor families conservatively share the cost of constructing the distribution/spatial descriptor bundle. Warm tile voting includes tile-feature loading. A content-addressed vector cache avoids redundant execution, but the recorded actual export cost is still charged to every candidate.

## Final20-pixel rule

Connected tissue islands and enclosed empty holes with area strictly below20 **native-image pixels** are filled using surrounding labels. Exactly20 pixels is retained. An isolated tiny tissue island surrounded entirely by glass is assigned to that surrounding background. Exterior background is never painted.

The published `bora.multicpu_geojson.convert` accurate profile remains the main vectorizer: minimum component area500 native pixels, shared-border simplification tolerance2 native pixels. After simplification, a final raster check repairs any newly created tiny fragments or holes. Only affected tiny areas and numerical boundary ties receive local pixel-cell repairs; all other simplified boundaries retain their converter coordinates. Following the user’s diagonal-contour request, each shared boundary is fitted once with Ramer–Douglas–Peucker straight chords in calibrated micrometre coordinates. The maximum tolerance is3µm, adaptively reduced to preserve every predicted class, at least97% per-class shape IoU and at most1.5% tissue-area change relative to the preceding predicted geometry. No expert annotations enter these fitting decisions. Junctions remain shared. Final sub20-pixel fragments and holes are repaired after chord fitting. The final GeoJSON is rasterized and checked against the cleaned target. Quality scores use these actual final polygons, ignoring unannotated truth pixels.

## Execution and checkpoints

`acquire` → `features` → `prepare` → `descriptors` → `run --generations 10`.

`pipeline` and `supervise` orchestrate this chain. Saved launch receipts include PID and Linux process startticks. Verify both before resuming a job; never duplicate a live process after a polling timeout. Preserve failed attempts and resume immutable completed artifacts.

Each generation saves `CANDIDATES.json`, `ALL_PREDICTIONS_FROZEN.json`, `RESULTS.json`, `CYCLE_COMPLETE.json`, and `viewer/index.html`. `LATEST_COMPLETE.json` and the root index point to completed generations. The viewer displays five final masks side by side, input/expert masks, unchanged converted RGB, actual GeoJSON, microscopy sampling, all20 numerical scores and the mechanisms/parent lineage. No top-five class list is displayed.
