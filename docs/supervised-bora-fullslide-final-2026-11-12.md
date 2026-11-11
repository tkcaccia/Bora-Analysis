# Supervised Bora full-slide masks

The public Benchmark 2 pages show one final supervised mask per H&E whole
slide. The earlier display-only crop smoothing has been replaced by a
complete-slide raster and native-coordinate GeoJSON. Each page's overview and
nine zooms are rendered from that same audited mask. The external-model
predictions and the anonymous KODAMA–Bora geometry are unchanged.

The supervised head uses 256 channels selected from the verified 1,280-channel
Virchow2 cache and was fitted on the 397 Benchmark 1 training images. To
avoid trace labels on a whole slide, a class is retained when it wins at
least 2% of tissue score-grid tokens. Suppressed-class pixels are assigned
their next-highest retained-class score. Two source-group-disjoint training
splits selected that cutoff: across 149 held-out training cases, mean false
positive class count fell from 1.134 to 0.389 while mean case macro IoU
changed from 0.68196 to 0.67889. No Benchmark 1 validation masks or
whole-slide expert masks were used in that choice.

The [executed source and code audit](../methods/benchmark2/supervised_final/)
verify the calibrated source, score grid, model and Bora tissue geometry;
project scores globally; apply a 400 µm² preparatory island repair; smooth
contours at a 6 µm physical scale; then remove positive regions and enclosed
holes strictly below 2,000 µm². Global vectorization and a bounded repair of
previously uncovered vector gaps produce the final raster and native GeoJSON.
The independent audit requires exact GeoJSON-to-mask raster agreement, no
positive pixels outside Bora's reviewed tissue support, and no subthreshold
positive component or enclosed hole. TIFF and GeoJSON SHA256 values are
bound to each slide's local and public receipts.

The original 10x ovarian fresh-frozen slide now retains four supervised
classes rather than eight trace predictions. Other slides may retain more
classes when each exceeds the training-selected area rule. These are model
predictions; independent whole-slide expert masks are unavailable, so this
release makes no whole-slide accuracy or method-ranking claim. The private
PM016 slide and withdrawn BEETLE dataset are excluded.
