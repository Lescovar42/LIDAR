"""
06c_diagnose_striping.py
=========================
Checks whether the diagonal lines visible in slope/hillshade outputs are
real lidar flight-line seams, by inspecting the raw LAZ point attributes
BEFORE any DEM gridding or smoothing hides the signal.

Two independent checks:
  1. If laspy exposes point_source_id (it usually does for USGS 3DEP LPC --
     this field records which flight line/swath each point came from), we
     can directly visualize swath boundaries and see if they line up with
     the diagonal lines in the DEM-derived slope image.
  2. Even without point_source_id, we can grid RAW POINT DENSITY (not
     elevation) at the same 1m resolution as the DEM. Flight-line overlap
     zones have measurably different point density than swath interiors --
     if the density map shows the same diagonal banding as the slope map,
     that's strong independent evidence for flight-line seams rather than
     e.g. real linear ground features or a bug in the gridding code.

RUN THIS LOCALLY -- needs the same lidar_tiles/ directory as the main
pipeline.

Usage:
    python 06c_diagnose_striping.py lidar_tiles/USGS_LPC_OR_OLCMetro_2019_A19_w2051n2779.laz
"""
import sys
import numpy as np
import matplotlib.pyplot as plt


def diagnose(laz_path, cell_size=1.0):
    import laspy
    print(f"Reading {laz_path} ...")
    las = laspy.read(laz_path)

    has_source_id = hasattr(las, "point_source_id")
    print(f"point_source_id available: {has_source_id}")

    ground_mask = las.classification == 2
    gx = np.array(las.x[ground_mask], dtype=np.float64)
    gy = np.array(las.y[ground_mask], dtype=np.float64)

    xmin, xmax = float(las.x.min()), float(las.x.max())
    ymin, ymax = float(las.y.min()), float(las.y.max())
    ncols = int(np.ceil((xmax - xmin) / cell_size))
    nrows = int(np.ceil((ymax - ymin) / cell_size))

    col_idx = np.clip(((gx - xmin) / cell_size).astype(int), 0, ncols - 1)
    row_idx = np.clip(((ymax - gy) / cell_size).astype(int), 0, nrows - 1)

    # --- Check 1: point_source_id spatial pattern, if available ---
    if has_source_id:
        source_ids = np.array(las.point_source_id[ground_mask])
        unique_sources = np.unique(source_ids)
        print(f"\nDistinct flight lines (point_source_id values): {len(unique_sources)}")
        print(f"  IDs: {unique_sources[:20]}{'...' if len(unique_sources) > 20 else ''}")

        if len(unique_sources) > 1:
            # Grid the DOMINANT source id per cell -- reveals swath boundaries directly
            source_grid = np.full((nrows, ncols), -1, dtype=np.int32)
            # simple approach: last-write-wins per cell (fine for boundary visualization)
            source_grid[row_idx, col_idx] = source_ids

            fig, ax = plt.subplots(figsize=(8, 8))
            im = ax.imshow(source_grid, cmap='tab20')
            ax.set_title(f"Flight line ID per grid cell\n({len(unique_sources)} distinct swaths)")
            plt.colorbar(im, ax=ax, label="point_source_id")
            plt.tight_layout()
            plt.savefig("diagnostic_flightlines.png", dpi=150)
            print("Saved diagnostic_flightlines.png -- compare its boundaries")
            print("against the diagonal lines in your slope/hillshade output.")
        else:
            print("Only one flight line ID present in this tile -- striping is")
            print("NOT a swath-boundary effect for this specific tile. Look")
            print("elsewhere (gridding/smoothing bug, or a real linear feature).")
    else:
        print("\nlaspy build doesn't expose point_source_id on this file --")
        print("falling back to density-based check only.")

    # --- Check 2: raw point density grid (works regardless of source_id) ---
    density = np.zeros((nrows, ncols), dtype=np.int32)
    np.add.at(density, (row_idx, col_idx), 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    im1 = axes[0].imshow(density, cmap='viridis', vmax=np.percentile(density, 95))
    axes[0].set_title("Ground point density per 1m cell")
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0], fraction=0.046)

    # Same-scale zoom on density gradient to make banding easier to see
    density_smooth_diff = density.astype(float) - \
        np.pad(density[:, :-1], ((0, 0), (1, 0)), mode='edge').astype(float)
    im2 = axes[1].imshow(density_smooth_diff, cmap='RdBu_r',
                          vmin=-10, vmax=10)
    axes[1].set_title("Density gradient (highlights swath edges)")
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1], fraction=0.046)

    plt.tight_layout()
    plt.savefig("diagnostic_density.png", dpi=150)
    print("\nSaved diagnostic_density.png")
    print("If this shows the SAME diagonal banding as your slope/hillshade")
    print("output, that's strong evidence for flight-line seams (overlapping")
    print("swaths have different point density, which affects the DEM's")
    print("per-cell average differently across the seam). If density is")
    print("uniform but slope still stripes, the cause is elsewhere --")
    print("worth re-checking the gridding/smoothing code instead.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 06c_diagnose_striping.py <path_to_laz>")
        sys.exit(1)
    diagnose(sys.argv[1])
