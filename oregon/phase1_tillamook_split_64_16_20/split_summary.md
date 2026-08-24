# Tillamook 64/16/20 Phase 1 split

- Source tiles: **98**
- Source patches: **6051**
- Spatial guard: **500 m**
- Atomic leakage-safe components: **43**
- Largest component: **17 tile(s)**
- Nominal tile target: `{'train': 63, 'validation': 16, 'test': 19}`

| Split | Tiles | Tile % | Patches | Positive patches | Negative patches | Polygons | Hard negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 63 | 64.29% | 3950 | 1615 | 2335 | 227 | 2209 |
| validation | 16 | 16.33% | 891 | 313 | 578 | 80 | 542 |
| test | 19 | 19.39% | 1210 | 501 | 709 | 73 | 685 |

## Leakage constraints

- A LiDAR tile appears in exactly one split.
- Any two tiles with retained patch footprints < 500 m apart are assigned to the same split.
- Any two tiles sharing a manifest landslide ID are assigned to the same split.
- Cross-split polygon overlap after assignment: **0**.

Run `verify_splits.py` on `preview_manifest.csv` before freezing this split.
