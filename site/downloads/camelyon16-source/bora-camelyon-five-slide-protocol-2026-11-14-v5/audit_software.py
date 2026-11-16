#!/usr/bin/env python3
"""Record exact source-only aggregator qualification and the pending use gate."""
from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
GATE = HERE.parent / 'bora-r-camelyon-2026-11-14-v1/PERMISSION_GATE_2026-11-14.json'
SCORER = HERE.parent / 'bora-camelyon-five-slide-protocol-2026-11-14-v4/binary_score.py'


def digest(path: Path) -> str:
    h = sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    gate = json.loads(GATE.read_text())
    if (gate['status'] != 'PENDING' or gate['cam_specific_training_allowed_now'] or
            gate['three_slide_cam_test_allowed_now'] or gate['new_cam_web_publication_allowed_now']):
        raise ValueError('CAM use gate differs from the user instruction')
    source = HERE / 'study_aggregate.py'
    tests = HERE / 'test_study_aggregate.py'
    result = subprocess.run([sys.executable, '-m', 'pytest', '-q', str(tests)],
                            text=True, capture_output=True, check=True)
    if '10 passed' not in result.stdout:
        raise ValueError('Expected source-free synthetic checks did not pass')
    receipt = {
        'format': 'bora-camelyon-study-aggregator-software-audit-v1',
        'status': 'SOURCE_FREE_SOFTWARE_VERIFIED',
        'aggregator_source_sha256': digest(source),
        'synthetic_tests_sha256': digest(tests),
        'per_slide_binary_scorer_v4_sha256': digest(SCORER),
        'current_permission_gate_sha256': digest(GATE),
        'synthetic_tests': '10 passed',
        'permission_gate_refuses_before_study_input': True,
        'cam_source_or_reference_opened': False,
        'cam_specific_training': False,
        'actual_test_cohort_frozen': False,
        'actual_study_score_computed': False,
        'website_published': False,
    }
    with (HERE / 'AGGREGATOR_SOFTWARE_AUDIT_VERIFIED.json').open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
