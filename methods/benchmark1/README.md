# Benchmark 1 comparator methods

The [current eight-class development head](eight_class_ridge/README.md) starts
with the full 1,280-channel Virchow2 cache. A training-only, source-group-
disjoint score and speed comparison selected 32 one-versus-rest contrast
channels per class (203 unique channels in the final fit) and multiclass ridge.
Its eight response maps and colored mask appear in the 87-image development
gallery. The earlier 24-class joint classifier, classwise PLS head and 75:25
blend remain in this directory as historical implementations.

The historical 24-class decision applied the frozen dataset label ontology in
[`ontology_gate.py`](ontology_gate.py): classes with no scorable mapping in the
known source dataset cannot win. The gate was selected over the unconstrained
argmax using two source-group-disjoint splits of the 397 training images. The
selected training score was 0.5896 mean case macro IoU across four multiclass
datasets, versus 0.5796 for the unconstrained rule. The mask then receives
1.5 µm one-hot contour smoothing and 400 µm² small-region repair in
[`supervised_contour.py`](supervised_contour.py).

The 87-image gallery is development evidence: those images were inspected
during earlier method work. The current eight-class head was selected on
training images only and scored before H&E-derived background display clipping.
KODAMA–Bora + fastPLS remains the canonical unsupervised route in the gallery;
the earlier 256-channel alternative is retained in historical records but is
not shown as a second candidate.
