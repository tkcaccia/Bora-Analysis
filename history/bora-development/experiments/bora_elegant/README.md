# Elegant whole-field Bora experiments

The frozen plan is in PLAN.json. The nine user-selected ROIs are development
cases, not a test set. RGB conversions #9, #10, #15, #19 and equal 9+19 remain
byte-identical. Develop on equal 9+19 only, using arithmetic mean per-image
native macro IoU. Every method operates on original coarse labels and converted
RGB; completed unknown labels do not train class models. No fine-reference mask
is read by prediction code. Background support and scoring domain are unchanged.

The core question is whether soft, global evidence can replace fixed margins,
hard anchors and repeated expensive local optimization. Six different initial
principles are specified before evaluating any of their scores. Follow-up
hypotheses and their outcomes are appended to DEVELOPMENT.json. Final methods
are frozen before full-corpus scoring, including the other four RGB conversions.

## Reproduce

Use the environment recorded in the result's ENVIRONMENT.json. On Chiamaka the
executed interpreter is
`/mnt/sata_ssd/Bora-benchmark/experiments/virchow2-inverse-2026-10-16/.venv/bin/python`.
The source snapshot is under the experiment's `code/` directory. Keep the input
folder immutable and use a fresh output directory for every changed protocol.

```bash
export PYTHONPATH=src:. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m experiments.bora_elegant.develop --source INPUTS --root NEW_ROUND \
  --keys E01 E02 E03 E04 E05 E06 --repeats 3
python -m experiments.bora_elegant.full --source INPUTS --root FINAL_OUTPUTS \
  --selection experiments/bora_elegant/SELECTION.json
```

`SELECTION.json` maps development E09/E11 to final B21/B22. B22 is selected
before full scoring; B21 is retained as the faster alternative. The other
active comparisons are B02/B03/B08/B19 revision 3. RGB conversion numbers are
separate from solver IDs. None of these steps runs Virchow2 again.

A reusable ROI API is available as `bora.evidence.refine_evidence`; the
`bora evidence` CLI accepts converted RGB, original coarse labels, independent
tissue support, actual per-axis pixel spacing, tile size and the known RGB
conversion palette. See the repository README for an example.

[Mathematics and evaluation design](../../benchmark/virchow2/BORA_ELEGANT_METHODS.md)
explain the approximation, soft supervision, class ambiguity and timing scope.
