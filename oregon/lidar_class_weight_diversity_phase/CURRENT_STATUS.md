# Current execution status — 2026-08-01

## Completed here

- Inspected the current GitHub trainer and confirmed that `--pos-weight` was absent.
- Built a safe trainer patcher that adds `--pos-weight auto|<positive number>` and run provenance.
- Built independent validation threshold sweeping from 0.30 through 0.90.
- Built direct terrain/GT/prediction/FP-FN comparison image export.
- Built source-only training-diversity and provenance auditing.
- Built semantic error-review candidate export and post-review summarization.
- Built a fingerprint-guarded baseline-versus-ablation summary.
- Added exact PowerShell installation and execution workflows.
- Passed six unit tests, Python syntax compilation, and an end-to-end synthetic integration test covering evaluation, images, NPZ feature auditing, error summary, and ablation summary.
- Ran the manifest-only audit on the repository's checked-in `patches_boundary_aware.csv`.

## Not executed here

The real 30-epoch auto-control and `pos_weight=1.0` training jobs were not executed because this environment does not have access to `F:\LIDAR\oregon`, its NPZ patch files, or the user's training GPU. Therefore no real ablation metric, selected threshold, comparison map, or semantic error-category conclusion is fabricated in this pack.

Run `run_controlled_phase.ps1` locally to generate those outputs.

## Measured manifest findings

| Item | Train | Validation |
|---|---:|---:|
| Patches | 513 | 54 |
| LiDAR tiles | 13 | 3 |
| Positive patches | 272 | 14 |
| Negative patches | 241 | 40 |
| Unique positive polygon keys | 23 | 3 |
| Boundary patches | 263 | 14 |
| Hard negatives | 225 | 35 |

Additional measured findings:

- Train/validation positive polygon-key overlap is zero.
- Four tiles contain 67.28% of positive training patches.
- Four polygons contain 57.77% of positive polygon-patch memberships.
- 322 same-polygon positive-patch pairs overlap by at least 50% of the smaller patch.
- Validation positive coverage consists only of seven low-coverage and seven mixed-positive patches; it has no near-full or full-positive patches.
- The manifest labels all source rows with `region_id=legacy` and `region_role=train_val`.
- The manifest's acquisition field is 2020, while its project-name-derived hint field is 2022. The current repository evidence file says 2020 is owner-attested for the selected Tillamook tiles, but also states that a machine-readable USGS project citation is still missing. The paper should not infer 2022 from `A22`, and should distinguish owner attestation from independent external verification.

## Current recommendation status

A class-weight decision is **not yet measured**. The current defensible interim recommendation is:

1. run the paired fresh auto-control and `pos_weight=1.0` experiment;
2. independently select each validation threshold;
3. do not permanently change the weight from the old baseline alone;
4. improve data diversity regardless of which weight wins, because the concentration/redundancy limitation is already measured;
5. prioritize new tiles and polygon groups, then visually verified hard negatives and underrepresented landslide coverage/size/morphology—not additional overlapping windows from dominant polygons.

Semantic claims such as “roads cause the false positives” remain hypotheses until the exported error candidates are reviewed.
