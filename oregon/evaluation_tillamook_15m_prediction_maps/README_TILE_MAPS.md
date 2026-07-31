# Stitched Validation Prediction Maps

Each PNG compares the ground-truth mosaic with the predicted mosaic for validation patches from one LiDAR tile.

These are cropped validation-patch mosaics, not predictions for every pixel in the original LAZ tile. Gray gaps were not represented by a selected validation patch.

- `USGS_LPC_OR_WesternWildfires_A22_s04710w14520.laz`: Dice `0.236`, precision `0.162`, recall `0.431`, patches `33` — `stitched_tile_maps\USGS_LPC_OR_WesternWildfires_A22_s04710w14520.laz__predicted_vs_gt.png`
- `USGS_LPC_OR_WesternWildfires_A22_s04410w14130.laz`: Dice `0.092`, precision `0.050`, recall `0.689`, patches `11` — `stitched_tile_maps\USGS_LPC_OR_WesternWildfires_A22_s04410w14130.laz__predicted_vs_gt.png`
- `USGS_LPC_OR_WesternWildfires_A22_s04380w14100.laz`: Dice `0.000`, precision `0.000`, recall `0.000`, patches `10` — `stitched_tile_maps\USGS_LPC_OR_WesternWildfires_A22_s04380w14100.laz__predicted_vs_gt.png`
