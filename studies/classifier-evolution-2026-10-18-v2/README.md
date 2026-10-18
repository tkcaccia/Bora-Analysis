# Frozen revised classifier study

`code/` is the exact preserved scientific source copied from the stopped revised
study on Chiamaka. `SOURCE_SHA256.json` records every file's hash. It retains
the original Python namespace layout (`code/bora`, `code/experiments`,
`code/tests`). These are historical implementations, not the installed current
Bora application or extra supported classifier options.

The user stopped evolution after six completed generations. The partial seventh
generation was preserved on the research computer and was not counted as
complete. No controller is resumed by publishing this snapshot. The originally
requested ten generations were not all completed.

`records/` contains the recipes and aggregate metrics for completed generations,
with machine-local case paths omitted. The public viewer packages cycle-03,
the comparison previously inspected by the user. P04 predictions and the fixed
geometry are unchanged across completed generations; only the measured warm
head timing can fluctuate.

The revised evaluation cohort contains 116 images (78 original test and 38
original validation images), with PUMA excluded. All these results were
adaptively reused for development. Do not treat them as independent final
testing or compare scores across differently defined cohorts without matching
their inputs. The 967 training groups and their feature caches remain in the
local study store and are not distributed here.

The current application is https://github.com/tkcaccia/Bora . The current
research viewer is https://tkcaccia.github.io/Bora-Analysis/ . Full image
archives, gated encoder weights and large feature caches are obtained
separately under their original provider terms.

This archive provides no authorization to launch historical jobs. UCT HPC GPU
work remains paused under the project coordination policy. Static website
packaging and deployment execute no training or GPU workloads.
