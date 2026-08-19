"""Shared CLI bootstrap: make `wqrag` importable when scripts run from any cwd."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
