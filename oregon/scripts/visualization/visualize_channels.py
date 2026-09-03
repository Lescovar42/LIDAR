from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


DATASET_DIR = Path("dataset_tillamook_probe_15m")
MANIFEST = DATASET_DIR / "patches_boundary_aware.csv"

CHANNELS = [
    "local_relief",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "curvature",
    "multidirectional_hillshade",
    "tri",
]


# Ambil satu positive training patch
with MANIFEST.open("r", encoding="utf-8", newline="") as f:
    rows = list(csv.DictReader(f))

row = next(
    r for r in rows
    if r["split"] == "train"
    and float(r.get("positive_fraction", 0) or 0) > 0
)

patch_path = DATASET_DIR / row["patch_path"]

with np.load(patch_path) as data:
    features = data["features"].astype(np.float32)
    mask = data["mask"]

print("Patch:", patch_path)
print("Feature shape:", features.shape)
print("Mask shape:", mask.shape)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i, name in enumerate(CHANNELS):
    ax = axes.flat[i]

    image = features[i]

    # Robust display range so extreme outliers don't ruin contrast
    vmin, vmax = np.nanpercentile(image, [2, 98])

    im = ax.imshow(
        image,
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_title(name)
    ax.axis("off")

    fig.colorbar(
        im,
        ax=ax,
        fraction=0.046,
        pad=0.04,
    )

# Last panel = SLIDO mask for context
ax = axes.flat[7]

display_mask = np.ma.masked_where(mask == 255, mask)

im = ax.imshow(
    display_mask,
    cmap="gray",
    vmin=0,
    vmax=1,
)

ax.set_title("SLIDO mask")
ax.axis("off")

fig.suptitle(
    "7 LiDAR-derived terrain channels used by Mini U-Net",
    fontsize=16,
)

plt.tight_layout()

output = Path("tillamook_7_input_channels.png")
plt.savefig(
    output,
    dpi=200,
    bbox_inches="tight",
)

print("Saved:", output.resolve())

plt.show()