# GeoJSON export resource pilot

Four saved TIGER **pre-export** tissue masks were converted with three CPU methods on one Apple M3 Mac (macOS 14.5, arm64). Each mask contained about 1.27–1.55 million pixels and two positive tissue labels. Each method ran three times in a fresh process; methods were run serially. Timings start after Python imports and stop after the GeoJSON (and, for complete Bora, the matching OME mask) is written. Peak resident set size (RSS) includes Python and imported libraries. The same 20-pixel component threshold was used by both Bora converters.

| Export method | Median wall time | Median peak RSS | Largest peak RSS | Median GeoJSON size | Median ring segments | Lowest class IoU against input mask |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rasterio polygonization and same-label union, no simplification | 0.077 s | 211 MB | 217 MB | 210 kB | 12,824 | 1.000 |
| Bora initial coverage converter, 2-pixel simplification | 0.102 s | 221 MB | 234 MB | 89 kB | 5,376 | 0.986 |
| Complete Bora 0.2.1 export, including physical boundary fit and final cleanup | 0.974 s | 273 MB | 285 MB | 161 kB | 1,854 | 0.973 |

The complete exporter used about 12.6 times the wall time and 62 MB more median peak RSS than lossless Rasterio polygonization in this small pilot. It produced about 86% fewer ring segments. Its GeoJSON was larger than the intermediate coverage file despite fewer segments because it also stores names, provenance and fitting diagnostics. All tested outputs had valid, nonoverlapping class polygons and preserved both class IDs.

These four input masks were saved immediately before the previous export. They provide a common pre-export prediction for resource and geometry comparison, but the four images are too few for a broad performance claim. The lowest class IoU compares each exported mask with its predicted input, not with expert tissue. These measurements do not replace a complete benchmark on other tissue datasets or whole slides. The prior whole-slide vector pass recorded 1,407.4 seconds for a 75,250 × 48,740-pixel slide, but peak memory was not recorded, so no whole-slide memory comparison is claimed.

The next benchmark should run all exporters on the same retained **pre-export** prediction masks, with hardware and numerical threads fixed. It should report time and peak RSS by image size and dataset, including 4,096-pixel whole-slide windows, along with changed pixels, per-class IoU, small-component retention, contour count, file size and seam continuity. Repeated export should also be checked for drift.

The [benchmark script](../benchmark/geojson_resource_benchmark.py) and [per-run results](../benchmark/cmpb_evidence/geojson_resource_tiger_four.json) reproduce this table. No GPU or remote computer was used.
