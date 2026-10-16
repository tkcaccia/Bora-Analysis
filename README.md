# Bora-extra

Reproducible **tissue-mask** acquisition, benchmarking, scoring, figure and manuscript-generation code for [Bora](https://github.com/tkcaccia/Bora).

The Bora segmentation library is maintained in the main repository. This repository holds research workflows and source locks; it contains no patient images, masks, benchmark outputs, or downloaded archives. The local data root used for the current study is `/mnt/sata_ssd/Bora-benchmark`.

Read [benchmark/README.md](benchmark/README.md) for supported tissue datasets, acquisition commands, the benchmark protocol, output locations and limitations. The figure and manuscript builders are in `manuscript/`. Set `BORA_ROOT` to the Bora checkout if the two repositories are not siblings.

The manuscript's main comparison uses BCSS, TIGER and WSSS4LUAD tissue regions. Separate, non-pooled tissue-region checks use RINGS prostate tumour and CAMELYON16 lymph-node tumour masks. Cell and gland-instance datasets are outside this study.
