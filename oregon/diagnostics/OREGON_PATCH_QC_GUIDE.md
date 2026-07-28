# Oregon LiDAR / SLIDO Patch QC Guide

This guide defines how to review LiDAR-derived terrain patches and their SLIDO ground-truth masks using:

```powershell
python .\diagnostics\qc_patch_viewer.py --dataset-dir .\dataset_pilot
```

The goal is to remove clearly unusable samples while preserving difficult but valid examples. QC should improve label reliability without turning the dataset into a collection of only obvious, easy landslides.

---

## 1. What the viewer shows

Each patch contains four main panels:

1. **Multidirectional hillshade**  
   Use this to inspect terrain shape, scarps, hummocky surfaces, channels, benches, roads, and artificial cuts.

2. **Local relief**  
   Use this to identify smaller-scale elevation differences that may be less obvious in hillshade.

3. **Slope in degrees**  
   Use this to confirm breaks in slope, steep scarps, road cuts, channels, and abrupt terrain transitions.

4. **SLIDO ground-truth overlay**  
   The red area is the rasterized SLIDO landslide label.

Also review the metadata shown above the panels:

- Dataset split
- Patch category
- Positive-pixel fraction
- Mean slope
- Ground-point coverage
- Hard-negative status
- Tile and patch identifiers
- SLIDO reference identifiers

Do not judge a patch from the slope panel alone. Use all terrain panels together.

---

## 2. QC decisions and keyboard shortcuts

| Key | QC status | Use when |
|---|---|---|
| `A` | `accept` | The patch and label are usable as-is. |
| `B` | `accept_approximate_boundary` | The landslide is visible, but the polygon boundary is somewhat coarse or uncertain. |
| `M` | `reject_misaligned` | The mask is clearly displaced, rotated, mirrored, or attached to the wrong landform. |
| `V` | `reject_not_visible` | A positive label has no convincing visible landslide morphology, or the label is too unclear to trust. |
| `E` | `reject_engineered_landform` | The apparent feature is mainly a road cut, quarry, embankment, excavation, or other artificial terrain. |
| `D` | `reject_bad_dem` | DEM artifacts, voids, striping, interpolation errors, or missing coverage make the patch unreliable. |
| `U` | Clear / unreviewed | Remove the current decision and leave the patch unreviewed. |

Navigation:

| Key | Action |
|---|---|
| `Right`, `Space`, `Page Down` | Next patch |
| `Left`, `Backspace`, `Page Up` | Previous patch |
| `Ctrl+F` | Search by patch ID |
| `Ctrl+S` | Save |
| `Q` | Save and quit |

Every decision is saved to:

```text
dataset_pilot\patches_qc.csv
```

---

## 3. Accept criteria

### 3.1 Positive-interior patches

Use `accept` when:

- The red mask covers terrain with recognizable landslide morphology.
- The mask and terrain are spatially aligned.
- The patch has sufficient ground coverage.
- There are no major DEM artifacts.
- The label is not primarily an engineered landform.
- Most of the positive area plausibly belongs to the same mapped landslide body.

Useful visual indicators include:

- Head scarps or arcuate breaks in slope
- Hummocky or irregular deposits
- Disrupted drainage
- Bulging toes
- Benches or stepped terrain
- Displaced or rotated blocks
- Rough morphology distinct from nearby stable hillslope

A patch does not need to show every indicator.

### 3.2 Positive-boundary patches

Use `accept` when:

- The mask edge follows a clear geomorphic transition.
- Both positive and negative portions are plausible.
- The boundary looks accurate enough for pixel-level training.

Use `accept_approximate_boundary` when:

- The landslide is clearly present.
- The SLIDO boundary is close but visibly generalized.
- The polygon slightly overextends or underextends the apparent feature.
- Terrain near the polygon edge is ambiguous, but the sample remains useful.

Boundary patches are valuable. Do not reject them merely because only a small part of the mask is present.

### 3.3 Negative patches

Use `accept` when:

- The patch contains no SLIDO mask.
- The terrain is valid and well reconstructed.
- There is no obvious unmapped landslide.
- The sample is a useful stable-terrain or hard-negative example.

Keep difficult negatives such as:

- Road cuts
- Stream banks
- Gullies
- Steep intact slopes
- Terraces
- Quarries outside mapped polygons
- Rough terrain that is not clearly a landslide

Hard negatives are important because they reduce false positives.

---

## 4. Reject criteria

### 4.1 `reject_misaligned`

Use when there is a clear spatial-labeling failure:

- The red polygon is shifted away from the visible landform.
- The mask is on the wrong side of a road, ridge, or channel.
- A consistent translation, rotation, or mirroring is visible.
- The polygon covers unrelated terrain while the likely landslide is elsewhere.
- Tile or CRS alignment appears wrong.

Do not use this status for a slightly coarse boundary. Use `accept_approximate_boundary` instead.

### 4.2 `reject_not_visible`

Use for positive-labeled patches when:

- No convincing landslide morphology is visible.
- The terrain looks stable despite the positive label.
- The mapped feature is too subtle or degraded to support reliable visual training.
- The patch contains too little of the feature to interpret.
- Heavy smoothing or low relief makes the label unverifiable.

For a negative patch that appears to contain an obvious unmapped landslide, use this status temporarily and add:

```text
possible_unmapped_landslide
```

to the notes.

### 4.3 `reject_engineered_landform`

Use when the labeled or apparent feature is dominated by:

- Highway or railway cuts
- Embankments
- Quarries
- Mines
- Construction grading
- Waste piles
- Dams or levees
- Artificial terraces

Do not reject a real landslide only because a road crosses it. Reject only when the landform itself appears mainly engineered.

### 4.4 `reject_bad_dem`

Use when terrain quality prevents reliable interpretation:

- Large NoData regions
- Severe flight-line striping
- Repeating interpolation patterns
- Strong edge artifacts
- Corrupted or incomplete tiles
- Very low ground-point coverage
- Obvious vertical spikes or pits
- Terrain derivatives disagree because of data failure

Minor texture or normal LiDAR noise is not enough for rejection.

---

## 5. Special cases

### Small positive fraction

A small red corner can still be valid if:

- The visible boundary is correctly aligned.
- The labeled terrain is interpretable.
- The patch contributes useful boundary information.

Reject only when the positive fragment is too small or unclear to evaluate.

### Nearly full positive mask

A nearly full mask is acceptable when:

- The entire patch plausibly belongs to the mapped landslide.
- Terrain morphology supports the label.
- The patch is not simply a large generalized polygon over unrelated terrain.

### Roads crossing landslides

Roads can cut through real landslides. Accept when the surrounding morphology supports a landslide and the mask is plausible.

### Vegetation

LiDAR bare-earth terrain should reduce vegetation effects. Do not reject only because the original area is forested.

### Old or low-relief landslides

Old landslides may be visually subtle. Use `accept_approximate_boundary` when the feature is still plausible but the boundary is uncertain. Use `reject_not_visible` only when the label cannot be supported visually.

### Multiple polygons in one patch

Accept when each polygon corresponds to plausible terrain. Reject or note the sample if one or more polygons are clearly wrong.

---

## 6. Consistency rules

Apply the same standard across train, validation, and test.

Do not:

- Be stricter on test patches than train patches.
- Accept obvious positives while rejecting all subtle positives.
- Reject difficult negatives just because they resemble landslides.
- Use slope alone as evidence.
- Assume every SLIDO boundary is exact.
- Correct the mask manually during this QC stage.
- Reject based only on low or high positive fraction.

When uncertain between `accept` and rejection, prefer:

```text
accept_approximate_boundary
```

and add a short note.

---

## 7. Recommended notes

Notes should be short and specific.

Examples:

```text
clear head scarp and hummocky deposit
boundary slightly overextends downslope
road crosses feature but morphology remains clear
possible unmapped landslide in negative patch
mask shifted approximately 20 m east
quarry face, not natural landslide morphology
severe striping in lower-right quadrant
very low relief; label not visually supportable
```

Avoid vague notes such as:

```text
bad
weird
not sure
looks okay
```

---

## 8. Recommended review order

For a small pilot dataset, review every patch.

For a larger dataset:

1. Review all validation patches.
2. Review all test patches.
3. Review all positive-interior training patches.
4. Review all positive-boundary training patches.
5. Review all hard negatives.
6. Review a random sample of remaining negatives.

Resume unfinished work with:

```powershell
python .\diagnostics\qc_patch_viewer.py `
  --dataset-dir .\dataset_pilot `
  --only-unreviewed
```

Review one split at a time:

```powershell
python .\diagnostics\qc_patch_viewer.py `
  --dataset-dir .\dataset_pilot `
  --split validation
```

Review one category at a time:

```powershell
python .\diagnostics\qc_patch_viewer.py `
  --dataset-dir .\dataset_pilot `
  --category positive_boundary
```

---

## 9. Minimum QC completion checks

After review, inspect QC counts:

```powershell
python -c "import pandas as pd; d=pd.read_csv(r'.\dataset_pilot\patches_qc.csv'); print(pd.crosstab(d['split'], d['qc_status'], margins=True))"
```

Check accepted samples by split and category:

```powershell
python -c "import pandas as pd; d=pd.read_csv(r'.\dataset_pilot\patches_qc.csv'); a=d[d.qc_status.isin(['accept','accept_approximate_boundary'])]; print(pd.crosstab([a['split']], a['category'], margins=True))"
```

Before training, confirm:

- Train, validation, and test all retain positive samples.
- Validation and test retain more than one positive patch.
- No split contains only easy negatives.
- Hard negatives remain represented.
- Accepted positive patches come from more than one tile where possible.
- Rejection reasons are not dominated by a systematic pipeline failure.

A large number of `reject_misaligned` or `reject_bad_dem` decisions indicates a pipeline issue that should be fixed before training.

---

## 10. Train using QC-approved patches

After QC:

```powershell
python .\train_baseline.py `
  --dataset-dir .\dataset_pilot `
  --outdir .\training_output_pilot_qc `
  --epochs 20 `
  --require-qc
```

The accepted statuses are:

```text
accept
accept_approximate_boundary
```

Rejected and unreviewed patches should not be used when `--require-qc` is enabled.

---

## 11. Quick decision flow

```text
Is the DEM usable?
├─ No  → reject_bad_dem
└─ Yes
   │
   Is the patch labeled positive?
   ├─ Yes
   │  │
   │  Is landslide morphology visible?
   │  ├─ No  → reject_not_visible
   │  └─ Yes
   │     │
   │     Is the mask spatially aligned?
   │     ├─ No  → reject_misaligned
   │     └─ Yes
   │        │
   │        Is the feature mainly engineered?
   │        ├─ Yes → reject_engineered_landform
   │        └─ No
   │           │
   │           Is the boundary reasonably exact?
   │           ├─ Yes → accept
   │           └─ No  → accept_approximate_boundary
   │
   └─ No
      │
      Is there an obvious unmapped landslide?
      ├─ Yes → reject_not_visible + note possible_unmapped_landslide
      └─ No
         │
         Is the terrain a useful valid negative?
         ├─ Yes → accept
         └─ No  → use the most specific rejection reason
```

---

## 12. QC objective

QC is not intended to make every label perfect. Its purpose is to:

- Remove clearly wrong spatial labels.
- Remove unusable terrain products.
- Exclude labels unsupported by visible morphology.
- Preserve representative difficult cases.
- Document uncertainty consistently.
- Produce trustworthy validation and test subsets.

The preferred result is a diverse, auditable dataset—not only visually obvious landslides and flat negatives.
