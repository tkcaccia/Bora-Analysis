# Current article

The current article targets **Computer Methods and Programs in Biomedicine** and
describes the final **Neighbor context** method and its fixed evaluation. It
uses descriptive method names, a structured abstract, numbered references,
editable equations and tables, and separate highlights. It does not narrate
internal development history.

The builder is `build_cmpb_manuscript.py`. It reads the completed selected
viewer as numerical evidence and the retained author manuscript for author and
affiliation text. It never runs model inference or submits a paper.

The current revised document is stored locally at
`/Users/stefano/Documents/ChatGPT/Bora/benchmark/manuscript/cmpb-2026-10-18-v2/Bora_CMPB_manuscript.docx`.
Its methods distinguish the reported source-identity training targets from the
new user training path based on semantic TIFF or GeoJSON annotations. Its data
availability section points to the exact [349-field manuscript source
manifest](../docs/manuscript_model_training_sources.json) and the
[provider acquisition guide](../docs/MANUSCRIPT_MODEL_DATA.md). The reported
evaluation numbers have not been recomputed for this documentation change.

```bash
python manuscript/build_cmpb_manuscript.py \
  --site site \
  --source-doc /path/to/retained_author_manuscript.docx \
  --output /path/to/manuscript-output
```

Run with the document runtime containing python-docx and Pillow. Render and
inspect every page before delivery. The generated Word article and highlights
are retained in the local manuscript store; the public website provides the
benchmark, not an automatically submitted or approved paper. Author declarations
and institutional requirements must be confirmed by the authors before
submission. Previously inspected data are reported as exploratory evaluation,
and the proposed source-to-target WSI transfer is not presented as completed.
