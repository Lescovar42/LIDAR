from pathlib import Path
import sys

OREGON_DIR = Path(__file__).resolve().parents[1]
for sub in [
    "utils",
    "scripts/data_prep",
    "scripts/training",
    "scripts/visualization",
    "data_boundaries",
]:
    p = OREGON_DIR / sub
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))
