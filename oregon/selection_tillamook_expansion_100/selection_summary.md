# Tillamook expansion selection summary

## Guardrails enforced

- Frozen train tiles excluded: **13**.
- Frozen validation tiles excluded: **3**.
- Candidate footprints within **500 m** of frozen validation were excluded.
- Only the Tillamook source-region A22 project is considered.
- No Buxton/Vernonia or Oregon City data are read.
- Existing training-covered positive polygons are treated as already represented at tile scale.
- Low/unknown-only SLIDO overlap tiles are not silently treated as clean negatives.

## Proposed selection

- Requested target: **100 tiles**.
- Selected: **100 tiles**.
- Positive-diversity tiles: **60**.
- Hard-negative tiles: **40**.
- Hard-negative target fraction: **40%**.
- Already downloaded and reusable: **9**.
- Still requiring download: **91**.
- Planned storage: **16.25 GB**.
- Unique new high/moderate SLIDO polygons represented by selected positive tiles: **0**.
- Positive polygons already represented by the frozen training tiles: **1**.

## Interpretation

This is a **pre-download selection**, not a semantic terrain classification. The selector can measure SLIDO novelty, spatial novelty, known probe quality, split safety, file reuse, and clean no-inventory hard-negative status. It cannot honestly identify unseen TNM tiles as ridge, drainage, road cut, forestry disturbance, or rough terrain before terrain derivatives/NAIP are available.

After download/preprocessing, prioritize semantic review of hard-negative patches in the measured error-review order: rough/dissected natural terrain, steep non-landslide slopes, ridge/convex terrain, and drainage/valley sides. Roads/cuts and forest-management disturbance require NAIP/context confirmation.

## Files

- `proposed_tillamook_expansion.csv`: reviewable planned manifest.
- `proposed_tillamook_expansion.json`: full TNM records with selection annotations.
- `candidate_scores.csv`: all eligible candidates and scores.
- `excluded_tiles.csv`: frozen split, validation-buffer, explicit, and label-quality exclusions.
- `download_urls.txt`: only selected files not already present locally.
