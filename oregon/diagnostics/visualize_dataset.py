#!/usr/bin/env python3
"""Create diverse visual-QC pages from a built Oregon patch dataset.

Unlike the old diagnostic, this script is not tied to one tile or the first
three positive patches. It samples across tiles, splits, and patch categories,
then writes a review CSV for manual accept/reject decisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def select_diverse(rows: list[dict[str, str]], count: int, seed: int) -> list[dict[str, str]]:
    rng = random.Random(seed)
    buckets: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row.get("split", ""), row.get("category", ""), row.get("tile_name", ""))
        buckets[key].append(row)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    selected: list[dict[str, str]] = []
    keys = list(buckets)
    rng.shuffle(keys)
    while len(selected) < count and keys:
        next_keys: list[tuple[str, str, str]] = []
        for key in keys:
            bucket = buckets[key]
            if bucket and len(selected) < count:
                selected.append(bucket.pop())
            if bucket:
                next_keys.append(key)
        keys = next_keys
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Render stratified QC pages for dataset patches.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_pilot"))
    parser.add_argument("--outdir", type=Path, default=None)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--per-page", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    outdir = (args.outdir or (dataset_dir / "qc")).resolve()
    manifest_path = dataset_dir / "patches.csv"
    channels_path = dataset_dir / "channels.json"
    if not manifest_path.exists():
        parser.error(f"Missing {manifest_path}")
    if not channels_path.exists():
        parser.error(f"Missing {channels_path}")
    if args.samples <= 0 or args.per_page <= 0:
        parser.error("--samples and --per-page must be positive")

    rows = read_csv(manifest_path)
    if not rows:
        parser.error("patches.csv contains no rows")
    channels = json.loads(channels_path.read_text(encoding="utf-8")).get("feature_names", [])
    try:
        slope_index = channels.index("slope_degrees")
        hillshade_index = channels.index("multidirectional_hillshade")
    except ValueError as exc:
        parser.error(f"Expected slope/hillshade channels, found: {channels}")
        raise exc

    selected = select_diverse(rows, min(args.samples, len(rows)), args.seed)
    outdir.mkdir(parents=True, exist_ok=True)

    review_rows: list[dict[str, Any]] = []
    pages = int(math.ceil(len(selected) / args.per_page))
    for page_index in range(pages):
        page_rows = selected[page_index * args.per_page : (page_index + 1) * args.per_page]
        figure, axes = plt.subplots(len(page_rows), 3, figsize=(13, 4 * len(page_rows)), squeeze=False)
        figure.suptitle(
            f"Oregon LiDAR / SLIDO visual QC — page {page_index + 1}/{pages}",
            fontsize=16,
            y=0.995,
        )

        for row_index, row in enumerate(page_rows):
            patch_path = dataset_dir / row["patch_path"]
            with np.load(patch_path) as data:
                features = data["features"].astype(np.float32)
                mask = data["mask"].astype(np.uint8)
            slope = features[slope_index]
            hillshade = features[hillshade_index]

            ax_hillshade, ax_slope, ax_overlay = axes[row_index]
            ax_hillshade.imshow(hillshade, cmap="gray")
            ax_hillshade.set_title(
                f"{row['patch_id']}\n{row['tile_name']} | {row['split']} | {row['category']}",
                fontsize=8,
            )
            ax_hillshade.axis("off")

            slope_image = ax_slope.imshow(slope, cmap="magma", vmin=0, vmax=max(45, float(np.percentile(slope, 99))))
            ax_slope.set_title(
                f"Slope | mean={float(row['mean_slope_degrees']):.1f}° | label={100*float(row['positive_fraction']):.1f}%",
                fontsize=9,
            )
            ax_slope.axis("off")
            figure.colorbar(slope_image, ax=ax_slope, fraction=0.046, pad=0.04)

            ax_overlay.imshow(hillshade, cmap="gray")
            overlay = np.ma.masked_where(mask == 0, mask)
            ax_overlay.imshow(overlay, cmap="Reds", alpha=0.55, vmin=0, vmax=1)
            if mask.min() != mask.max():
                ax_overlay.contour(mask, levels=[0.5], colors="red", linewidths=1.0)
            ax_overlay.set_title("SLIDO ground-truth overlay", fontsize=9)
            ax_overlay.axis("off")

            review_rows.append(
                {
                    **row,
                    "qc_page": page_index + 1,
                    "qc_status": "",
                    "qc_notes": "",
                }
            )

        figure.tight_layout(rect=(0, 0, 1, 0.985))
        page_path = outdir / f"qc_page_{page_index + 1:03d}.png"
        figure.savefig(page_path, dpi=args.dpi, bbox_inches="tight")
        plt.close(figure)
        print(f"Wrote {page_path}")

    review_fields = list(review_rows[0].keys())
    review_path = outdir / "qc_review.csv"
    write_csv(review_path, review_rows, review_fields)
    print(f"Wrote {review_path}")
    print("Fill qc_status with values such as accept, accept_approximate_boundary, reject_misaligned, reject_not_visible, reject_engineered_landform, or reject_bad_dem.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
