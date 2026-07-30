# Tillamook matched-project downloader/probe fix

Applies the measured result without changing the 0.80 coverage requirement:

- ordinary deduplication stays IoU based;
- probe matching uses smaller-footprint coverage >= 0.80 explicitly;
- selected records retain IoU, smaller-overlap, pair ID, and intersection geometry;
- project diagnostics are calculated only over the identical intersection window;
- no project or cell size is pinned automatically.

## Apply from the repository root

```powershell
python .\apply_tillamook_probe_fix.py --repo-root .
```

## Test

```powershell
python -m unittest .\tests\test_select_tiles.py .\tests\test_probe_overlap_fix.py -v
python -m unittest discover -s .\tests -v
python -m unittest discover -s .\oregon\tests -v
python -m compileall -q .\oregon .\tests
```

## Run

```powershell
cd .\oregon
.\run_tillamook_probe.ps1
```
