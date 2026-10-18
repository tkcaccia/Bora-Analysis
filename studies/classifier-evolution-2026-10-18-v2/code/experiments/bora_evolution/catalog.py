"""Mechanism registry; component-count or seed changes are not new algorithms."""

ALGORITHMS=[
 ('P01','Whole group discriminant','Native PLS-LDA on the full 1280-channel group mean. Accepted reference classifier.','mean','lda'),
 ('P02','Covariance geometry','Log-Euclidean covariance of training-only label-free token coordinates, combined with group mean. Models correlated within-group heterogeneity rather than only channel SD.','covariance','lda'),
 ('P03','Tissue composition','Native multiresponse PLS predicts annotated training tissue fractions; response ordering identifies each whole group.','mean','composition'),
 ('P04','Shared interface context','Native PLS-LDA sees a group mean and its contrast to physically shared-boundary-weighted neighbors.','neighbor','lda'),
 ('P05','Tile rank consensus','Native tile PLS-LDA rankings are pooled with exact tile-to-group area weights using Borda consensus.','tiles','tile_vote'),
 ('P06','Multiple instance attention','A native PLS instance head determines confidence-weighted attention within each group before its tissue evidence is aggregated.','tiles','attention'),
 ('P07','Hierarchical routing','Training class prototypes define a binary class tree. Native binary PLS heads provide soft path evidence to every leaf.','mean','tree'),
 ('P08','Pairwise tournament','A native binary PLS model distinguishes each training class pair. Pairwise margins are coupled to a multiclass ranking.','mean','pairwise'),
 ('P09','Error correcting codes','A native multiresponse PLS predicts a deterministic redundant class code. Decoding uses distance to valid codewords.','mean','ecoc'),
 ('P10','Relational tissue embedding','Native PLS regresses a low-dimensional class-prototype embedding; tissue identity is decoded by proximity in that learned label space.','mean','embedding'),
 ('P11','Nonlinear kernel PLS','The native fastPLS kernelpls RBF route models nonlinear similarity to training groups before LDA. Kernel scale is estimated on training only.','mean','kernel'),
 ('P12','Orthogonal nuisance separation','Native OPLS removes predictor variation orthogonal to the supervised response before class discrimination.','mean','opls'),
 ('P13','Residual response boosting','Sequential native PLS regressors fit residual class responses; stages form an additive discriminant.','mean','boost'),
 ('P14','Source bagged subspaces','Multiple native PLS heads use source-group bootstrap samples and different feature subspaces; ranks are aggregated across models.','mean','bag'),
 ('P15','Local expert mixture','Training-only feature clusters gate specialized native PLS heads, with a global head for sparse or unseen neighborhoods.','mean','experts'),
 ('P16','Distribution quantile profiles','Weighted quantiles of training-only label-free token projections describe multimodal tissue groups; a native PLS head learns their identity.','quantile','lda'),
 ('P17','Spatial tissue pooling','Physically defined interior and interface feature summaries, with spatial moments, feed a native PLS group head.','spatial','lda'),
 ('P18','Image composition transport','A group PLS head and a separate image-composition PLS head are reconciled by soft transport over the image groups. Allows repeated tissue roles.','mean','transport'),
 ('P19','Latent class densities','Native PLS supplies the supervised latent coordinates; shrinkage class-specific Gaussian densities yield a generative tissue ranking.','mean','density'),
 ('P20','Tissue verification metric','A native PLS verifier learns same-tissue versus different-tissue relations from training pairs, then ranks training-derived tissue prototypes.','mean','verification')
]


def initial():
    return [dict(id=code,name=name,description=description,view=view,engine=engine,
                 parents=[],generation=1,native_backend='fastPLS C++ CPU')
            for code,name,description,view,engine in ALGORITHMS]


EVOLUTION_DIRECTIONS=[
 ('distillation','Compress a parent response surface into a native PLS student using training-only cross-fitted teacher evidence.'),
 ('residual_specialist','Add a native PLS residual expert for training-only cross-fitted confusions, rather than changing component count.'),
 ('context_transport','Couple parent group evidence to an independently fitted native PLS image-composition prior.'),
 ('distribution_expert','Fuse parent evidence with a native PLS distribution-profile expert; the new source of evidence is token heterogeneity.'),
 ('hierarchical_decoder','Replace flat response decoding with native PLS class-tree routing.'),
 ('verification_decoder','Use a native PLS pair verifier to adjudicate parent tissue alternatives.'),
 ('coded_residual','Fit redundant error-correcting class-code residuals through a native PLS stage.'),
 ('local_residual','Learn parent residuals within training-only feature neighborhoods using native PLS experts.'),
 ('source_bagged_student','Distill source-group bagged parent evidence to a compact native PLS student.'),
 ('latent_density_fusion','Fuse parent class evidence with shrinkage densities in supervised native PLS coordinates.'),
 ('orthogonal_student','Use native OPLS to isolate class-relevant variation while distilling parent responses.'),
 ('spatial_expert','Add physically pooled within-group spatial evidence through a native PLS expert.')
]
