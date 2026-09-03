# -*- coding: utf-8 -*-
"""Visualize a single patch: all 7 feature channels + landslide mask."""

import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATASET_DIR = Path("dataset_tillamook_probe_15m")

# --- Load channel names ---
channels = json.loads((DATASET_DIR / "channels.json").read_text())["feature_names"]

# --- Find a patch WITH landslide pixels (positive patch) for more interesting visual ---
manifest_path = DATASET_DIR / "patches_qc.csv"
if not manifest_path.exists():
    manifest_path = DATASET_DIR / "patches.csv"

with manifest_path.open("r", newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# Try to find a positive patch (has landslide)
positive_patch_path = None
negative_patch_path = None

for row in rows:
    p = DATASET_DIR / row["patch_path"]
    if not p.exists():
        continue
    with np.load(p) as data:
        mask = data["mask"]
    landslide_frac = float(np.mean(mask == 1))
    if landslide_frac > 0.01:  # at least 1% landslide coverage
        positive_patch_path = p
        print(
            f"Found positive patch: {p.name} "
            f"({landslide_frac:.1%} landslide)"
        )
        break

# Fallback to first available patch
if positive_patch_path is None:
    for row in rows:
        p = DATASET_DIR / row["patch_path"]
        if p.exists():
            positive_patch_path = p
            print(f"No positive patch found, using: {p.name}")
            break

if positive_patch_path is None:
    print("ERROR: No patch files found!")
    exit(1)

# --- Load the patch ---
with np.load(positive_patch_path) as data:
    features = data["features"]  # (C, H, W)
    mask = data["mask"]           # (H, W)

positive_pixels = int(np.sum(mask == 1))
ignore_pixels = int(np.sum(mask == 255))
positive_fraction = float(np.mean(mask == 1))
ignore_fraction = float(np.mean(mask == 255))

print(f"\nPatch: {positive_patch_path.name}")
print(f"  features shape: {features.shape}  (channels, height, width)")
print(f"  mask shape:     {mask.shape}")
print(
    f"  landslide pixels: {positive_pixels} / {mask.size} "
    f"({positive_fraction:.2%})"
)
print(
    f"  ignore pixels:    {ignore_pixels} / {mask.size} "
    f"({ignore_fraction:.2%})"
)
print()

for i, name in enumerate(channels):
    ch = features[i]
    print(f"  ch[{i}] {name:30s}  min={ch.min():.3f}  max={ch.max():.3f}  mean={ch.mean():.3f}")

# --- Plot ---
n_channels = len(channels)
n_total = n_channels + 1  # +1 for the mask
ncols = 4
nrows = (n_total + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
fig.suptitle(f"Single Patch: {positive_patch_path.name}\nShape per channel: {features.shape[1]}×{features.shape[2]} px",
             fontsize=13, fontweight="bold", y=1.02)

axes = axes.flatten()

# Colormaps per channel type
cmaps = {
    "local_relief": "terrain",
    "slope_degrees": "YlOrRd",
    "aspect_sin": "coolwarm",
    "aspect_cos": "coolwarm",
    "curvature": "PuOr",
    "multidirectional_hillshade": "gray",
    "tri": "inferno",
}

for i, name in enumerate(channels):
    ax = axes[i]
    cmap = cmaps.get(name, "viridis")
    im = ax.imshow(features[i], cmap=cmap, interpolation="nearest")
    ax.set_title(f"ch[{i}]: {name}", fontsize=10, fontweight="bold")
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Mask
ax = axes[n_channels]

display_mask = np.zeros(mask.shape, dtype=np.uint8)
display_mask[mask == 1] = 1
display_mask[mask == 255] = 2

im = ax.imshow(
    display_mask,
    cmap=plt.matplotlib.colors.ListedColormap([
        "white",
        "red",
        "gray",
    ]),
    interpolation="nearest",
    vmin=0,
    vmax=2,
)

positive_fraction = float(np.mean(mask == 1))
ignore_fraction = float(np.mean(mask == 255))

ax.set_title(
    f"MASK\n"
    f"{positive_fraction:.1%} landslide, "
    f"{ignore_fraction:.1%} ignore",
    fontsize=10,
    fontweight="bold",
    color="red",
)

ax.set_xticks([])
ax.set_yticks([])
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# Hide unused subplots
for j in range(n_channels + 1, len(axes)):
    axes[j].set_visible(False)

plt.tight_layout()
output_path = "patch_visualization.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
print(f"\nSaved to {output_path}")
plt.show()
