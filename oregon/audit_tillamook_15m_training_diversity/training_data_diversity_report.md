# Tillamook training-data diversity audit

## Scope

This report audits only the source-region training and validation manifest. It does not use Buxton/Vernonia or Oregon City data and does not change labels, ignore masks, normalization, or model architecture.

## Measured findings

- Train: **513 patches**, **13 tiles**, **272 positive patches**, and **23 unique positive polygon keys**.
- Validation: **54 patches**, **3 tiles**, **14 positive patches**, and **3 unique positive polygon keys**.
- Train/validation positive polygon-key overlap: **0**.
- The four training tiles with the most positive patches contain **67.28%** of all training positive patches.
- The four most repeated training polygon keys account for **57.77%** of positive polygon-patch memberships.
- Potential overlapping-window indicator: **322** same-polygon pairs have at least 50% overlap of the smaller patch. This indicates possible redundancy, not automatically invalid data.
- Feature-file audit status: **processed 3969 patch-channel records; missing patch files=0**.

### Coverage composition

- Train coverage classes: `{"full_positive": 18, "low_coverage_positive": 42, "mixed_positive": 190, "near_full_positive": 22, "negative": 241}`.
- Validation coverage classes: `{"low_coverage_positive": 7, "mixed_positive": 7, "negative": 40}`.

### Manifest-level train/validation support screening

| Metric | Validation outside train 1st–99th percentile |
|---|---:|
| positive_fraction | 0.00% |
| ignore_fraction | 9.26% |
| ground_fraction | 5.56% |
| mean_slope_degrees | 0.00% |
| distance_to_positive_m | 7.41% |
| boundary_pixel_fraction | 3.70% |
| boundary_of_positive_fraction | 5.56% |

## What is not measured automatically

The manifest does not identify roads, engineered cuts, drainage forms, ridges, forest-management disturbance, or uncertain/incomplete inventory labels. Counts of hard negatives therefore measure sampling status, **not semantic hard-negative diversity**. Those categories require visual review of the exported candidates and, where needed, ancillary imagery or vector layers.

## Hypotheses to test with the ablation and visual review

- Repeated overlapping windows around a small number of polygons may reduce effective morphological diversity.
- Numerically abundant hard negatives may still be too homogeneous, allowing systematic false positives on underrepresented terrain forms.
- Validation estimates may be unstable because the positive validation subset contains few polygons and lacks some positive-coverage classes present in training.

## Expansion priorities supported by this audit

1. Select spatially distinct tiles and new polygon groups before adding more windows around already dominant polygons.
2. Add visually verified hard negatives across drainage/valley sides, ridges/convex slopes, rough natural terrain, roads/cuts, and forest-management disturbance.
3. Add underrepresented landslide sizes and coverage patterns, especially classes missing from validation, while keeping validation frozen for the current ablation.
4. Track per-tile and per-polygon caps so nominal patch count is not confused with independent terrain or morphology diversity.
