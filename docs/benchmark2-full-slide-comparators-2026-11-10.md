# Benchmark 2 full-slide comparator publication

As of 10 November 2026, [the public gallery](https://tkcaccia.github.io/Bora-Analysis/visium-hd/comparators/) contains nine verified public-source H&E whole-slide comparisons. Every published page shows the same calibrated H&E frame alongside class-aware U-Net, DeepLabV3+, MedSAM and VISTA-PATH outputs. The web images are downsampled previews of complete full-resolution prediction masks.

U-Net and DeepLabV3+ use checkpoints fitted on the 397 retained Benchmark 1 training images and expert masks. MedSAM receives tumor, stroma and necrosis boxes derived from an earlier U-Net in each tile; it is a prompted three-role output. VISTA-PATH uses an externally pretrained 13-class vocabulary, with matching roles mapped to shared display colors. The method legends list classes visible in each preview.

These slides have no independent whole-slide expert masks. The gallery is for visual inspection and does not establish Dice, IoU, accuracy, or a ranking of the four methods. Patch boundaries and predictions on glass remain visible where produced. The public comparator gallery is separate from the 13 existing Bora whole-slide pages.

The nine-page site release was verified in commit `2e6d8c73122f303058dcae029d97fffa30a56162` and deployed by [GitHub Pages run 42883681413](https://github.com/tkcaccia/Bora-Analysis/actions/runs/42883681413). The independent release audit matched 64 live page and image SHA256 values. The broader local comparator campaign remains incomplete; further public-source slides will be added after their full-slide receipts and web previews pass verification.
