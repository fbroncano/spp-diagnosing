"""Repository paths.

The analysis reuses the data-access layer and the constants of the processor
(`processor/dao.py`, `processor/CONSTANTS.py`) instead of duplicating them, so
`processor/` must be on `sys.path`.
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROCESSOR = REPO / "processor"
REFERENCE_DIR = REPO / "data" / "ngl"

if str(PROCESSOR) not in sys.path:
    sys.path.insert(0, str(PROCESSOR))
