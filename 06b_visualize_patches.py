"""
06b_visualize_patches.py
========================
Visualizes the extracted terrain derivative patches and their corresponding 
landslide masks from the MVP pipeline.
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

# Import functions directly from our pipeline script
import importlib.util
spec = importlib.util.spec_from_file_location("mvp", "06_mvp_training_pipeline.py")
mvp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mvp)

def main():
    # Tile 3 had positive patches in our MVP run
    target_tile = "lidar_tiles/USGS_LPC_OR_OLCMetro_2019_A19_w2051n2779.laz"
    
    if not os.path.exists(target_tile):
        print(f"Error: Could not find {target_tile}")
        return
        
    print(f"Loading and processing {target_tile}...")
    dem, geo_transform = mvp.laz_to_dem(target_tile)
    slope = mvp.compute_slope(dem)
    aspect = mvp.compute_aspect(dem)
    curvature = mvp.compute_curvature(dem)
    hillshade = mvp.compute_hillshade(dem)

    # Match the main pipeline's exact stack order (dem, slope, aspect,
    # curvature, hillshade) instead of building a different local stack.
    # The previous version padded hillshade into slots 2/3/4 to hit shape
    # (5,H,W) -- that happened to work because this script only ever read
    # its own stack, but it's a latent bug: if this script is later pointed
    # at features produced by the main pipeline, feat_patch[2] silently
    # becomes aspect instead of hillshade, with no error thrown. Using the
    # real stack (and named indices below) removes that failure mode.
    features = np.stack([dem, slope, aspect, curvature, hillshade])
    
    print("Rasterizing masks...")
    mask = mvp.rasterize_landslides(mvp.DEPOSITS_PATH, geo_transform, dem.shape)
    
    print("Extracting patches...")
    patches, pos_patches, neg_patches = mvp.extract_patches(features, mask)
    
    if len(pos_patches) == 0:
        print("No positive patches found in this tile!")
        return
        
    print(f"Found {len(pos_patches)} positive patches. Generating visualization...")
    
    # Select 3 positive patches to visualize
    num_samples = min(3, len(pos_patches))
    samples = pos_patches[:num_samples]
    
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    fig.suptitle("LiDAR Terrain Derivatives vs Landslide Ground Truth Masks", fontsize=16, y=0.98)
    
    # Custom colormap for masks (transparent background, red foreground)
    mask_cmap = ListedColormap(['none', 'red'])
    
    # Named channel indices matching the stack order above -- avoids the
    # silent-wrong-channel failure mode described in the comment above.
    CH_DEM, CH_SLOPE, CH_ASPECT, CH_CURVATURE, CH_HILLSHADE = range(5)

    for i in range(num_samples):
        feat_patch, mask_patch = samples[i]
        
        hillshade_patch = feat_patch[CH_HILLSHADE]
        slope_patch = feat_patch[CH_SLOPE]
        
        ax_hs, ax_sl, ax_mk = axes[i] if num_samples > 1 else axes
        
        # 1. Hillshade
        im1 = ax_hs.imshow(hillshade_patch, cmap='gray')
        ax_hs.set_title(f"Patch {i+1}: Hillshade")
        ax_hs.axis('off')
        
        # 2. Slope
        im2 = ax_sl.imshow(slope_patch, cmap='magma')
        ax_sl.set_title(f"Patch {i+1}: Slope (Degrees)")
        ax_sl.axis('off')
        plt.colorbar(im2, ax=ax_sl, fraction=0.046, pad=0.04)
        
        # 3. Mask Overlay
        ax_mk.imshow(hillshade_patch, cmap='gray')  # Background
        ax_mk.imshow(mask_patch, cmap=mask_cmap, alpha=0.6)  # Red mask overlay
        ax_mk.set_title(f"Patch {i+1}: Ground Truth Overlay")
        ax_mk.axis('off')
        
        # Add a border to positive mask pixels
        ax_mk.contour(mask_patch, levels=[0.5], colors='red', linewidths=1.5)

    plt.tight_layout()
    out_file = "patch_visualization.png"
    plt.savefig(out_file, dpi=200, bbox_inches='tight')
    print(f"Visualization saved successfully to {out_file}")

if __name__ == "__main__":
    main()
