# -*- coding: utf-8 -*-
"""
06_mvp_training_pipeline.py
===========================
Proof of concept: demonstrates that 3DEP LiDAR point clouds (LAZ) and
SLIDO landslide-deposit polygons (GeoJSON) can be ingested, co-registered,
and used to train a binary segmentation CNN end-to-end.

Pipeline steps
--------------
  1. LAZ → bare-earth DEM  (ground-point gridding + gap-fill)
  2. DEM → terrain features  (slope, aspect, curvature, hillshade, TRI)
  3. SLIDO polygons → binary raster mask  (WGS84 → tile CRS reproject)
  4. Sliding-window patch extraction  (256 × 256 px, 50 % overlap)
  5. Balanced DataLoader  (pos_weight to handle landslide rarity)
  6. MiniSegNet (U-Net-style encoder-decoder) training loop

Processes up to 3 tiles to keep runtime short (~2–5 min on CPU).
Output: console log of patch counts and training loss per epoch.

Dependencies: laspy, lazrs, scipy, pyproj, numpy, Pillow, torch
"""

import json
import time
import glob
import os
import sys
import numpy as np

# ── 0. Configuration ────────────────────────────────────────────────────────
TILE_DIR       = "lidar_tiles"
DEPOSITS_PATH  = "slido_deposits_oregon_city.geojson"
CELL_SIZE      = 1.0        # 1 m DEM resolution
PATCH_SIZE     = 256        # 256 x 256 pixel patches
PATCH_STRIDE   = 128        # 50% overlap
EPOCHS         = 10
BATCH_SIZE     = 8
LEARNING_RATE  = 1e-3
LAZ_CRS_EPSG   = 6350       # NAD83(2011) / Conus Albers (from VLR WKT)

# ── 1. Read LAZ & build bare-earth DEM ───────────────────────────────────────
def laz_to_dem(laz_path, cell_size=CELL_SIZE):
    """Read a LAZ file, filter ground points (class 2), grid to DEM."""
    import laspy
    print(f"  Reading {os.path.basename(laz_path)} ...")
    las = laspy.read(laz_path)
    
    # Filter ground points (classification == 2)
    ground_mask = las.classification == 2
    gx = np.array(las.x[ground_mask], dtype=np.float64)
    gy = np.array(las.y[ground_mask], dtype=np.float64)
    gz = np.array(las.z[ground_mask], dtype=np.float64)
    
    print(f"  {ground_mask.sum():,} ground points out of {len(las.points):,} total")
    
    # Tile bounding box (from all points, not just ground)
    xmin, xmax = float(las.x.min()), float(las.x.max())
    ymin, ymax = float(las.y.min()), float(las.y.max())
    
    ncols = int(np.ceil((xmax - xmin) / cell_size))
    nrows = int(np.ceil((ymax - ymin) / cell_size))
    
    # Bin ground points into grid cells (row, col)
    col_idx = np.clip(((gx - xmin) / cell_size).astype(int), 0, ncols - 1)
    row_idx = np.clip(((ymax - gy) / cell_size).astype(int), 0, nrows - 1)  # y-axis flipped
    
    # Average Z per cell
    from scipy.ndimage import uniform_filter
    dem = np.zeros((nrows, ncols), dtype=np.float64)
    count = np.zeros((nrows, ncols), dtype=np.int32)
    
    # Accumulate Z values (init to 0, not NaN, so add.at works)
    np.add.at(dem,  (row_idx, col_idx), gz)
    np.add.at(count, (row_idx, col_idx), 1)
    
    # Average where we have data, mark empty cells as NaN
    valid = count > 0
    dem[valid] /= count[valid]
    dem[~valid] = np.nan
    
    # Fill small gaps via nearest-neighbor interpolation
    from scipy.ndimage import distance_transform_edt
    nan_mask = ~valid
    if nan_mask.any():
        _, nearest_idx = distance_transform_edt(nan_mask, return_distances=True, return_indices=True)
        dem = dem[tuple(nearest_idx)]
    
    dem = dem.astype(np.float32)
    
    # Light smoothing to reduce gridding artifacts
    dem = uniform_filter(dem, size=3)
    
    print(f"  DEM shape: {dem.shape} ({nrows}x{ncols}), "
          f"Z range: {np.nanmin(dem):.1f} - {np.nanmax(dem):.1f} m")
    
    geo_transform = (xmin, cell_size, ymax, -cell_size)  # (x_origin, dx, y_origin, dy)
    return dem, geo_transform


# ── 2. Compute terrain derivatives ──────────────────────────────────────────
def compute_slope(dem, cell_size=CELL_SIZE):
    """Slope in degrees from finite differences (Horn's method)."""
    dy, dx = np.gradient(dem, cell_size)
    slope_rad = np.arctan(np.sqrt(dx**2 + dy**2))
    return np.degrees(slope_rad).astype(np.float32)

def compute_aspect(dem, cell_size=CELL_SIZE):
    """Aspect in degrees (0=N, 90=E, 180=S, 270=W)."""
    dy, dx = np.gradient(dem, cell_size)
    aspect = np.degrees(np.arctan2(-dx, dy))
    aspect[aspect < 0] += 360
    return aspect.astype(np.float32)

def compute_curvature(dem, cell_size=CELL_SIZE):
    """Total (Laplacian) curvature."""
    dy, dx = np.gradient(dem, cell_size)
    dyy, _ = np.gradient(dy, cell_size)
    _, dxx = np.gradient(dx, cell_size)
    return (dxx + dyy).astype(np.float32)

def compute_tri(dem):
    """Topographic Ruggedness Index (Riley et al. 1999) — captures local roughness."""
    from scipy.ndimage import generic_filter
    def _tri_kernel(window):
        center = window[4]  # 3x3 window, center is index 4
        return np.sqrt(np.mean((window - center)**2))
    return generic_filter(dem, _tri_kernel, size=3).astype(np.float32)

def compute_hillshade(dem, cell_size=CELL_SIZE, azimuth=315, altitude=45):
    """Standard hillshade."""
    dy, dx = np.gradient(dem, cell_size)
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dx, dy)
    
    az_rad = np.radians(azimuth)
    alt_rad = np.radians(altitude)
    
    hs = (np.sin(alt_rad) * np.cos(slope) +
          np.cos(alt_rad) * np.sin(slope) * np.cos(az_rad - aspect))
    return np.clip(hs * 255, 0, 255).astype(np.float32)


# ── 3. Rasterize SLIDO polygons into binary mask ────────────────────────────
def rasterize_landslides(geojson_path, geo_transform, dem_shape):
    """
    Project SLIDO WGS84 polygons into the tile's CRS and rasterize
    into a binary mask aligned to the DEM grid.
    """
    from pyproj import Transformer
    from PIL import Image, ImageDraw
    
    x_origin, dx, y_origin, dy = geo_transform
    nrows, ncols = dem_shape
    
    # WGS84 → tile CRS (EPSG:6350 Conus Albers)
    transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{LAZ_CRS_EPSG}", always_xy=True)
    
    with open(geojson_path) as f:
        deposits = json.load(f)
    
    mask_img = Image.new("L", (ncols, nrows), 0)
    draw = ImageDraw.Draw(mask_img)
    
    n_drawn = 0
    for feat in deposits["features"]:
        geom = feat["geometry"]
        geom_type = geom["type"]
        
        if geom_type == "Polygon":
            rings = [geom["coordinates"][0]]  # outer ring only
        elif geom_type == "MultiPolygon":
            rings = [poly[0] for poly in geom["coordinates"]]
        else:
            continue
        
        for ring in rings:
            # Transform lon/lat → projected coordinates
            lons = [pt[0] for pt in ring]
            lats = [pt[1] for pt in ring]
            px, py = transformer.transform(lons, lats)
            
            # Projected coords → pixel coords
            pixel_coords = []
            in_tile = False
            for xi, yi in zip(px, py):
                col = (xi - x_origin) / dx
                row = (y_origin - yi) / (-dy)   # dy is negative
                pixel_coords.append((col, row))
                if 0 <= col < ncols and 0 <= row < nrows:
                    in_tile = True
            
            if in_tile and len(pixel_coords) >= 3:
                draw.polygon(pixel_coords, fill=1)
                n_drawn += 1
    
    mask = np.array(mask_img, dtype=np.float32)
    landslide_pixels = int(mask.sum())
    total_pixels = nrows * ncols
    print(f"  Rasterized {n_drawn} polygon ring(s) into mask")
    print(f"  Landslide pixels: {landslide_pixels:,} / {total_pixels:,} "
          f"({100*landslide_pixels/total_pixels:.2f}%)")
    return mask


# ── 4. Extract patches ──────────────────────────────────────────────────────
def extract_patches(features, mask, patch_size=PATCH_SIZE, stride=PATCH_STRIDE,
                    min_landslide_frac=0.0):
    """
    Slide a window across the feature stack and mask.
    features: (C, H, W) numpy array
    mask: (H, W) numpy array
    Returns list of (feature_patch, mask_patch) tuples.
    """
    C, H, W = features.shape
    patches = []
    
    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            f_patch = features[:, y:y+patch_size, x:x+patch_size]
            m_patch = mask[y:y+patch_size, x:x+patch_size]
            
            # Skip patches that are all NaN (shouldn't happen after gap-fill, but safe)
            if np.isnan(f_patch).any():
                continue
            
            patches.append((f_patch, m_patch))
    
    # Separate into positive (contains landslide) and negative patches
    pos = [p for p in patches if p[1].sum() > 0]
    neg = [p for p in patches if p[1].sum() == 0]
    
    return patches, pos, neg


# ── 5. PyTorch Dataset & simple U-Net-lite model ────────────────────────────
def build_dataloader(patches, batch_size=BATCH_SIZE):
    import torch
    from torch.utils.data import TensorDataset, DataLoader
    
    X = np.stack([p[0] for p in patches])       # (N, C, H, W)
    Y = np.stack([p[1] for p in patches])[:, np.newaxis, :, :]  # (N, 1, H, W)
    
    # Normalize each channel to zero-mean, unit-variance
    for c in range(X.shape[1]):
        mu = X[:, c].mean()
        sd = X[:, c].std() + 1e-8
        X[:, c] = (X[:, c] - mu) / sd
    
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(Y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def make_model(in_channels):
    """Tiny encoder-decoder CNN for binary segmentation."""
    import torch
    import torch.nn as nn
    
    class MiniSegNet(nn.Module):
        def __init__(self, C):
            super().__init__()
            # Encoder
            self.enc1 = nn.Sequential(
                nn.Conv2d(C, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
            self.pool1 = nn.MaxPool2d(2)
            
            self.enc2 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
            self.pool2 = nn.MaxPool2d(2)
            
            # Bottleneck
            self.bottleneck = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU())
            
            # Decoder
            self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
            self.dec2 = nn.Sequential(
                nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
                nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU())
            
            self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
            self.dec1 = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
                nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
            
            self.head = nn.Conv2d(32, 1, 1)
        
        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            b  = self.bottleneck(self.pool2(e2))
            
            d2 = self.up2(b)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            
            return self.head(d1)   # raw logits, use BCEWithLogitsLoss
    
    return MiniSegNet(in_channels)


# ── 6. Training loop ────────────────────────────────────────────────────────
def train(model, loader, epochs=EPOCHS, lr=LEARNING_RATE):
    import torch
    import torch.nn as nn
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Weighted loss to handle class imbalance (landslides are rare)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([10.0]).to(device))
    
    print("\n" + "="*60)
    print("  Training on %s | %s parameters" % (device, f"{sum(p.numel() for p in model.parameters()):,}"))
    print("="*60)
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0
        
        for X_batch, Y_batch in loader:
            X_batch, Y_batch = X_batch.to(device), Y_batch.to(device)
            
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, Y_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            n_batches += 1
        
        avg_loss = running_loss / max(n_batches, 1)
        print(f"  Epoch {epoch:2d}/{epochs}  loss = {avg_loss:.4f}")
    
    print("="*60)
    return model


# ── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()
    
    # Pick tiles that are most likely to contain landslides.
    # We'll process up to 3 tiles to get enough positive patches.
    laz_files = sorted(glob.glob(os.path.join(TILE_DIR, "*.laz")))
    if not laz_files:
        print("ERROR: No LAZ files found in", TILE_DIR)
        sys.exit(1)
    
    print(f"Found {len(laz_files)} LAZ tile(s) in {TILE_DIR}/")
    
    # Process a small number of tiles for the MVP
    max_tiles = min(3, len(laz_files))
    all_patches = []
    
    for i, laz_path in enumerate(laz_files[:max_tiles]):
        tile_name = os.path.basename(laz_path)
        print("\n" + "-"*60)
        print("TILE %d/%d: %s" % (i+1, max_tiles, tile_name))
        print("-"*60)
        
        # Step 1: LAZ -> DEM
        print("\n[1/4] Building bare-earth DEM ...")
        dem, geo_transform = laz_to_dem(laz_path)
        
        # Step 2: Terrain derivatives
        print("\n[2/4] Computing terrain derivatives ...")
        slope     = compute_slope(dem)
        aspect    = compute_aspect(dem)
        curvature = compute_curvature(dem)
        hillshade = compute_hillshade(dem)
        tri       = compute_tri(dem)
        print("  Slope range: %.1f deg - %.1f deg" % (slope.min(), slope.max()))

        # 6-channel feature stack: elevation, slope, aspect, curvature, hillshade, TRI
        features = np.stack([dem, slope, aspect, curvature, hillshade, tri])  # (6, H, W)
        print(f"  Feature stack shape: {features.shape} (channels, H, W)")
        
        # Step 3: Rasterize landslide mask
        print("\n[3/4] Rasterizing landslide mask ...")
        mask = rasterize_landslides(DEPOSITS_PATH, geo_transform, dem.shape)
        
        # Step 4: Extract patches
        print("\n[4/4] Extracting patches ...")
        patches, pos, neg = extract_patches(features, mask)
        print(f"  Total patches: {len(patches)}")
        print(f"  Positive (has landslide): {len(pos)}")
        print(f"  Negative (no landslide):  {len(neg)}")
        
        all_patches.extend(patches)
        
        # If we already have enough positive patches, stop early
        total_pos = sum(1 for p in all_patches if p[1].sum() > 0)
        if total_pos >= 10:
            print(f"\n  Got {total_pos} positive patches, enough for MVP demo.")
            break
    
    total_pos = sum(1 for p in all_patches if p[1].sum() > 0)
    total_neg = len(all_patches) - total_pos
    print("\n" + "="*60)
    print("  COMBINED: %d patches (%d positive, %d negative)" % (len(all_patches), total_pos, total_neg))
    print("="*60)
    
    if total_pos == 0:
        print("\nWARNING: Zero positive patches! The landslides may not overlap")
        print("these specific tiles. Try processing more tiles or check the")
        print("spatial coverage map.")
        # Still train on negative patches to prove the pipeline works
    
    # Balance the dataset: keep all positive, sample equal negatives
    pos_patches = [p for p in all_patches if p[1].sum() > 0]
    neg_patches = [p for p in all_patches if p[1].sum() == 0]
    
    if pos_patches:
        n_neg_keep = min(len(neg_patches), max(len(pos_patches) * 3, 20))
        rng = np.random.default_rng(42)
        neg_sample = [neg_patches[i] for i in rng.choice(len(neg_patches), n_neg_keep, replace=False)]
        train_patches = pos_patches + neg_sample
        rng.shuffle(train_patches)
    else:
        # No positives — just use a random sample to prove training works
        n_sample = min(len(all_patches), 50)
        train_patches = all_patches[:n_sample]
    
    print(f"  Training set: {len(train_patches)} patches")
    
    # Build DataLoader
    print("\n[5/5] Building DataLoader & model ...")
    loader = build_dataloader(train_patches)
    
    in_channels = train_patches[0][0].shape[0]
    model = make_model(in_channels)
    
    # Train
    trained_model = train(model, loader)
    
    elapsed = time.time() - t0
    print("\n[OK] MVP pipeline completed in %.1f minutes" % (elapsed/60))
    print("[OK] This proves the LiDAR + SLIDO data can be combined and trained.")
    print("  Next steps: more tiles, ResU-Net architecture, proper train/val/test split.")


if __name__ == "__main__":
    main()
