# Benchmark 2 full-slide comparator publication

As of 11 November 2026, [the public gallery](https://tkcaccia.github.io/Bora-Analysis/visium-hd/comparators/) contains 11 verified public-source H&E whole-slide comparisons. Every published page shows the same calibrated H&E frame alongside class-aware U-Net, DeepLabV3+, MedSAM and VISTA-PATH outputs. The web images are downsampled previews of complete full-resolution prediction masks.

U-Net and DeepLabV3+ use checkpoints fitted on the 397 retained Benchmark 1 training images and expert masks. MedSAM receives tumor, stroma and necrosis boxes derived from an earlier U-Net in each tile; it is a prompted three-role output. VISTA-PATH uses an externally pretrained 13-class vocabulary, with matching roles mapped to shared display colors. The method legends list classes visible in each preview.

These slides have no independent whole-slide expert masks. The gallery is for visual inspection and does not establish Dice, IoU, accuracy, or a ranking of the four methods. Patch boundaries and predictions on glass remain visible where produced. The public comparator gallery is separate from the 13 existing Bora whole-slide pages.

All 12 slides in the frozen local comparator campaign passed the independent streaming source and mask audit: 48 full-resolution prediction TIFFs. The 11 public-source previews are published; the user-supplied PM016 slide remains local and is not included on this site or in the manuscript. The site publication has per-slide page and asset SHA256 receipts in the local Bora project. These results remain exploratory visual comparisons without whole-slide expert accuracy.
