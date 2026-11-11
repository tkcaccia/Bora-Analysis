# Benchmark 1 comparator methods

The supervised Bora mask begins with all 1,280 Virchow2 channels. On the 397
training images, classwise feature contrasts select 32 channels per class (444
unique channels) for a joint multinomial classifier. A separate classwise PLS
head uses 64 channels per class. The joint and PLS probabilities are blended at
75% and 25%. These methods and the comparator training scripts are in this
directory.

The final decision applies the frozen dataset label ontology in
[`ontology_gate.py`](ontology_gate.py): classes with no scorable mapping in the
known source dataset cannot win. The gate was selected over the unconstrained
argmax using two source-group-disjoint splits of the 397 training images. The
selected training score was 0.5896 mean case macro IoU across four multiclass
datasets, versus 0.5796 for the unconstrained rule. The mask then receives
1.5 µm one-hot contour smoothing and 400 µm² small-region repair in
[`supervised_contour.py`](supervised_contour.py).

The 87-image validation gallery is development evidence: those images were
inspected during earlier method work. The gate was frozen before its 87-image
scoring. Predictions are scored before H&E-derived background display clipping.
KODAMA–Bora + fastPLS remains the canonical unsupervised route in the gallery;
the earlier 256-channel alternative is retained in historical records but is
not shown as a second candidate.
