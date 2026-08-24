# Phase 3 Tillamook Dataset QC

**Status: PASS**

| Split | Tiles | Patches | Positive-mask patches | Positive pixels | GT positive % | Polygon IDs | Hard negatives |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 63 | 3950 | 1615 | 46256004 | 17.869% | 227 | 2209 |
| validation | 16 | 891 | 313 | 8327071 | 14.260% | 80 | 542 |
| test | 19 | 1210 | 501 | 16868846 | 21.273% | 73 | 685 |

## Hard gates

- Strict binary mask values: **PASS — [0, 1]**
- Feature shape: **PASS — 7 × 256 × 256**
- Spatial split buffer: **PASS — 500 m, 0 violations**
- Cross-split tile overlap: **PASS — 0**
- Cross-split landslide-ID overlap: **PASS — 0**
- Internal test lock file: **PASS**

## Channels

- `local_relief`
- `slope_degrees`
- `aspect_sin`
- `aspect_cos`
- `curvature`
- `multidirectional_hillshade`
- `tri`

The test split remains locked until architecture, feature set, epoch/checkpoint, and validation threshold are frozen.
