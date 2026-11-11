"""Training-frozen dataset ontology gate for the supervised Virchow2 mask.

The permitted class IDs are derived only from the declared expert-label
harmonization for each source dataset. No evaluation labels are read here.
"""
import json
from pathlib import Path

import numpy as np


def eligible_classes(harmonization_path: str | Path, canonical_classes: dict[int, str]):
    mapping = json.loads(Path(harmonization_path).read_text())['mapping']
    by_name = {name: cid for cid, name in canonical_classes.items()}
    return {
        dataset: sorted({by_name[entry['canonical_role']]
                         for entry in labels.values()
                         if entry.get('use') == 'score' and entry.get('canonical_role') in by_name})
        for dataset, labels in mapping.items()
    }


def predict_eligible(blended_response: np.ndarray, dataset: str,
                     eligible: dict[str, list[int]]) -> np.ndarray:
    """Return 1-based class IDs from a 24-channel response array."""
    if blended_response.shape[-1] != 24:
        raise ValueError('Expected 24 tissue-class responses')
    allowed = np.zeros(24, bool)
    allowed[np.asarray(eligible[dataset], dtype=int) - 1] = True
    if not allowed.any():
        raise ValueError(f'No eligible classes for {dataset}')
    return np.argmax(np.where(allowed, blended_response, -np.inf), axis=-1).astype(np.uint8) + 1
