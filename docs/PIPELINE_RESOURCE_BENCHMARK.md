# Bora pipeline time and memory pilot

Four TIGER tissue regions (1.27–1.55 million working pixels each) were processed with Bora 0.2.1, commit `bfae24aba03b18f77133a7b9d695ad49f8edf22a`, on an Apple M3 Mac running macOS 14.5. Each case was run three times in a fresh process with two numerical threads. The same pretrained two-class Neighbor context model was loaded for every run. The input image, coarse tile mask, tissue support, and previously computed Virchow2 base and multiscale features were loaded from disk. **RGB9+19 was fitted and constructed anew for each run.** The final mask and corrected GeoJSON exporter were run. No GPU was used in this pilot.

| Stage | Median wall time across 12 runs | Largest observed wall time |
| --- | ---: | ---: |
| Image, mask, and feature loading with checksum checks | 0.088 s | 0.194 s |
| Neighbor context model loading and fastPLS refit | 0.019 s | 0.042 s |
| RGB9+19 selection, fitting, and image construction | 0.238 s | 0.283 s |
| Bora appearance-field refinement | 0.101 s | 0.111 s |
| Refined-group feature pooling and neighbor context | 0.114 s | 0.124 s |
| fastPLS class prediction | <0.001 s | <0.001 s |
| Native mask and compact GeoJSON export | 0.979 s | 1.618 s |
| Other pipeline work, including output serialization | 0.125 s | 0.144 s |
| **Measured total, after Python imports** | **1.666 s** | **2.354 s** |

The process peak resident set size (RSS), including Python imports and all stages, was **491 MB median** and **570 MB maximum**. The median cumulative high-water marks were 195 MB after imports, 257 MB after input loading, and 308 MB after model loading. These checkpoints are **not independent stage peaks or stage allocations**; the 491 MB is the only whole-run peak. The exporter occupied the largest share of the observed cached-feature run. It should be optimized only while preserving the native mask, class geometry, and 20-pixel component rule.

These values do **not** constitute a cold end-to-end time or a whole-slide memory measurement. Virchow2 inference, its image resampling and GPU memory are excluded because the spatial features were cached and the source extraction did not record stage resource telemetry. The existing colorectal whole-slide log records 3,005.8 seconds for encoder and refinement together and 1,407.4 seconds for a separate vector pass, but it does not separate encoder time, include all cold I/O, or record peak CPU or GPU memory. Those whole-slide numbers must not be added to this Mac pilot. The 39-region refinement comparison in the manuscript has times for coarse tiles, H&E graph diffusion, a bilateral reference notebook, and Bora appearance fields, but no method-specific memory measurements. A matched hardware rerun is required before making a memory comparison among them.

To complete a publication-grade resource analysis, run a fixed patient-independent cohort through a cold pipeline on a single specified host. Record image read and resolution resampling, Virchow2 inference for base and multiscale features, RGB9+19 construction, refinement, group pooling, model loading and prediction, mask and GeoJSON export, and disk output separately. Record process peak RSS and GPU peak allocated/reserved memory, input size, tile/window count, thread count, batch size, and whether caches were warm. Measure the four refinement comparators on identical inputs and hardware. Include WSI windows and report the total slide wall time and peak memory without summing per-window peaks.

The [reproducible benchmark script](../benchmark/pipeline_resource_benchmark.py) and [per-run evidence](../benchmark/cmpb_evidence/pipeline_resource_tiger_four.json) provide the exact stage and memory observations. The [GeoJSON resource pilot](GEOJSON_RESOURCE_BENCHMARK.md) measures three exporters on four identical pre-export masks and should be read separately from this full Bora run.
