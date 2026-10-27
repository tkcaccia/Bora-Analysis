# Manuscript state

The working CMPB Word article and full-resolution figure assets are maintained
outside Git. Its reported numerical results describe the inspected 116-image
development analysis; they are not results of the later full rerun.

The current document uses Neighbor context as Bora's sole tissue classifier.
It does not report classifier-control methods or the selected ROI pilot whose
final tissue names came from a different readout. Its source file is
`benchmark/manuscript/cmpb-2026-10-27-submission-edit/Bora_CMPB_manuscript_neighbor_context_only.docx`
in the local Bora workspace.

The article cites Bora 0.2.1 on GitHub main. Dataset landing pages belong in
Data availability, not in the numbered scientific references. Four additional
10x Visium HD whole-slide images have been staged as an extended source panel:
healthy FFPE pancreas, fresh-frozen reactive lymph node, fresh-frozen
hyperplastic tonsil and FFPE anaplastic oligodendroglioma. Their separately
provided GeoJSONs still require annotation-provenance and tissue-label review,
so no accuracy results for those four slides are reported.

The [full-rerun integration note](FULL_RERUN_INTEGRATION_2026-10-27.md)
specifies the frozen primary and secondary analyses. Rebuild the manuscript,
figures and tables only after complete, verified result receipts are available.
