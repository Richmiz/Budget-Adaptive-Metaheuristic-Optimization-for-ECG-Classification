"""Protocol gates: fold 10 stays closed until the analysis lock exists (docs/07_analysis_plan.md)."""

from __future__ import annotations

from .config import PROJECT_ROOT
from .utils import read_json

LOCK_FILE = PROJECT_ROOT / "protocol" / "analysis_lock.json"


def require_analysis_lock() -> dict:
    if not LOCK_FILE.exists():
        raise PermissionError(f"{LOCK_FILE} not found: fold 10 stays closed until the analysis is locked "
                              "(scripts/lock_protocol.py analysis)")
    return read_json(LOCK_FILE)
