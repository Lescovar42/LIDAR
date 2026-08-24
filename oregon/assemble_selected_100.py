from pathlib import Path
from urllib.parse import urlparse
import csv
import os
import shutil

ROOT = Path(r"F:\LIDAR\oregon")

SELECTION = ROOT / "selection_tillamook_expansion_100" / "proposed_tillamook_expansion.csv"

DEST = ROOT / "lidar_tiles" / "tillamook_expansion_100_complete"
DEST.mkdir(parents=True, exist_ok=True)

# Extract the selected LAZ filename from whatever column contains it.
selected = []

with SELECTION.open("r", encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        filename = None

        for value in row.values():
            if not value:
                continue

            value = str(value).strip()

            if ".laz" in value.lower():
                if value.lower().startswith(("http://", "https://")):
                    filename = Path(urlparse(value).path).name
                else:
                    filename = Path(value).name

                if filename.lower().endswith(".laz"):
                    break

        if filename is None:
            raise RuntimeError(
                f"Could not determine LAZ filename from selection row:\n{row}"
            )

        selected.append(filename)

selected = list(dict.fromkeys(selected))

print(f"Selected unique files: {len(selected)}")

if len(selected) != 100:
    raise RuntimeError(
        f"Expected 100 selected tiles, found {len(selected)}"
    )

# Find every already-downloaded LAZ anywhere in the Oregon workspace,
# except the destination we're currently assembling.
available = {}

for path in ROOT.rglob("*.laz"):
    try:
        if DEST in path.parents:
            continue
    except Exception:
        pass

    # Prefer the largest copy if duplicates exist.
    existing = available.get(path.name)

    if existing is None or path.stat().st_size > existing.stat().st_size:
        available[path.name] = path

missing = []

for i, filename in enumerate(selected, 1):
    source = available.get(filename)

    if source is None:
        missing.append(filename)
        print(f"[{i:03d}/100] MISSING  {filename}")
        continue

    target = DEST / filename

    if target.exists():
        print(f"[{i:03d}/100] EXISTS   {filename}")
        continue

    # Same F: drive => hard link uses essentially no extra disk space.
    try:
        os.link(source, target)
        action = "LINKED"
    except OSError:
        shutil.copy2(source, target)
        action = "COPIED"

    print(f"[{i:03d}/100] {action:8s} {filename}")
    print(f"          from: {source}")

print()
print(f"Complete selected files: {len(list(DEST.glob('*.laz')))}")
print(f"Missing selected files:  {len(missing)}")

if missing:
    print("\nMissing:")
    for name in missing:
        print(" -", name)
else:
    print("\nALL 100 SELECTED TILES ASSEMBLED")