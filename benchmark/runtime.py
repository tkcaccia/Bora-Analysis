"""Resolve the installed Bora source used by external benchmark scripts."""
from __future__ import annotations

import os
from pathlib import Path


BORA_ROOT = Path(os.environ.get('BORA_ROOT', Path(__file__).resolve().parents[2] / 'Bora')).resolve()
BORA_SRC = BORA_ROOT / 'src'


def require_bora() -> Path:
    if not (BORA_SRC / 'bora/cli.py').is_file():
        raise FileNotFoundError(f'Bora source not found at {BORA_SRC}; set BORA_ROOT')
    return BORA_SRC
