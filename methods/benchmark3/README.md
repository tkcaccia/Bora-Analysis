# Benchmark 3: Visium expression-guided examples

Start with the [step-by-step public tutorial](../../site/visium/tutorial.html).

The three published examples are breast/Giotto, colorectal/Seurat, and
tonsil/SpatialExperiment. The files in `legacy_clustering/` are copies of the
historical front ends that generated expression spot groups. Their SHA256
identities are recorded in this Git revision. `download_visium.py` and
`visium_datasets.json` retain the corresponding 10x source URLs. The front
ends require their original R package environment and the historical
`wsiTools` integration for tile planning; they are **not** a one-command
reproduction of the final Bora field masks.

The currently reproducible public check is:

```bash
python3 methods/benchmark3/verify_published.py \
  --provenance site/visium/FULLSECTION_PROVENANCE.json
```

It downloads the three commit-pinned final GeoJSON files and verifies their
compressed and uncompressed SHA256 digests, sizes, geometry types, and group
counts. Use `--archive archive/visium-fullsection-2026-11-10-v1` to check a
local clone instead. This verifies saved outputs; it does not regenerate them
or recompute spot-centre concordance. The full-section feature grids, fitted
historical field model, and frozen expression-spot input tables are not all
in this public repository, so an exact end-to-end rerun cannot currently be
claimed. See the tutorial for this distinction and the required input schema.
