#!/usr/bin/env python3
"""
Phase 3 QC for the frozen Tillamook 64/16/20 binary dataset.

Checks:
- exact patch/split/tile counts
- unique patch IDs
- tile isolation across splits
- strict binary masks {0,1}
- feature tensor shape and channel count
- channel metadata
- no cross-split landslide-ID overlap
- split category / hard-negative / positive-pixel composition
- no test data used as train/validation
- frozen Phase 1 verification still says 0 spatial violations

This script only READS the dataset. It does not modify patches or manifests.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

EXPECTED_CHANNELS = [
    "local_relief",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "curvature",
    "multidirectional_hillshade",
    "tri",
]

EXPECTED_PATCH_COUNTS = {
    "train": 3950,
    "validation": 891,
    "test": 1210,
}

EXPECTED_TILE_COUNTS = {
    "train": 63,
    "validation": 16,
    "test": 19,
}

VALID_SPLITS = ("train", "validation", "test")


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def parse_bool(value: str) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def parse_ids(value: str) -> set[str]:
    return {x.strip() for x in str(value or "").split(";") if x.strip()}


def finite_float(value: str, default=float("nan")):
    try:
        x = float(value)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def pct(n, d):
    return 100.0 * n / d if d else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset_tillamook_100_binary_15m_split641620"),
    )
    ap.add_argument(
        "--report-prefix",
        default="phase3_qc",
    )
    args = ap.parse_args()

    d = args.dataset_dir.resolve()
    manifest = d / "patches.csv"
    channels_path = d / "channels.json"
    verification_path = d / "split_verification.json"
    lock_path = d / "TEST_SET_LOCKED.txt"

    required = [manifest, channels_path, verification_path, lock_path]
    for p in required:
        if not p.exists():
            raise SystemExit(f"FAIL: required file missing: {p}")

    rows = read_csv(manifest)
    print(f"Manifest rows: {len(rows)}")

    if len(rows) != 6051:
        raise SystemExit(f"FAIL: expected 6051 rows, found {len(rows)}")

    patch_ids = [r["patch_id"] for r in rows]
    if len(set(patch_ids)) != len(patch_ids):
        dupes = [k for k, v in Counter(patch_ids).items() if v > 1]
        raise SystemExit(f"FAIL: duplicate patch IDs: {dupes[:20]}")

    split_counts = Counter(r["split"] for r in rows)
    if dict(split_counts) != EXPECTED_PATCH_COUNTS:
        raise SystemExit(
            f"FAIL: split counts mismatch. expected={EXPECTED_PATCH_COUNTS}, got={dict(split_counts)}"
        )

    unknown_splits = set(split_counts) - set(VALID_SPLITS)
    if unknown_splits:
        raise SystemExit(f"FAIL: unknown splits: {sorted(unknown_splits)}")

    # Tile isolation.
    tile_splits = defaultdict(set)
    split_tiles = {s: set() for s in VALID_SPLITS}
    for r in rows:
        tile_splits[r["tile_name"]].add(r["split"])
        split_tiles[r["split"]].add(r["tile_name"])

    leaked_tiles = {t: sorted(v) for t, v in tile_splits.items() if len(v) > 1}
    if leaked_tiles:
        raise SystemExit(f"FAIL: tiles appear in multiple splits: {leaked_tiles}")

    tile_counts = {s: len(split_tiles[s]) for s in VALID_SPLITS}
    if tile_counts != EXPECTED_TILE_COUNTS:
        raise SystemExit(
            f"FAIL: tile counts mismatch. expected={EXPECTED_TILE_COUNTS}, got={tile_counts}"
        )

    # Channel metadata.
    channels_obj = json.loads(channels_path.read_text(encoding="utf-8"))
    channels = channels_obj.get("feature_names", channels_obj)
    if channels != EXPECTED_CHANNELS:
        raise SystemExit(
            "FAIL: channel metadata mismatch.\n"
            f"Expected: {EXPECTED_CHANNELS}\n"
            f"Found:    {channels}"
        )

    # Phase 1/2 spatial verification must remain clean.
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    violation_count = int(verification.get("violation_count", -1))
    if violation_count != 0:
        raise SystemExit(
            f"FAIL: split verification reports violation_count={violation_count}"
        )
    if float(verification.get("buffer_m", 0.0)) < 500.0:
        raise SystemExit(
            f"FAIL: split verification buffer is only {verification.get('buffer_m')} m"
        )

    # Cross-split landslide-ID check + manifest composition.
    split_polygon_ids = {s: set() for s in VALID_SPLITS}
    categories = {s: Counter() for s in VALID_SPLITS}
    hard_negative_counts = Counter()
    positive_manifest_rows = Counter()
    ground_fraction_values = defaultdict(list)
    positive_fraction_values = defaultdict(list)

    for r in rows:
        s = r["split"]
        split_polygon_ids[s].update(parse_ids(r.get("landslide_ids_in_tile", "")))
        categories[s][r.get("category", "")] += 1
        if parse_bool(r.get("is_hard_negative", "")):
            hard_negative_counts[s] += 1

        pf = finite_float(r.get("positive_fraction", ""))
        gf = finite_float(r.get("ground_fraction", ""))
        if math.isfinite(pf):
            positive_fraction_values[s].append(pf)
            if pf > 0:
                positive_manifest_rows[s] += 1
        if math.isfinite(gf):
            ground_fraction_values[s].append(gf)

    polygon_overlap = {}
    for i, a in enumerate(VALID_SPLITS):
        for b in VALID_SPLITS[i + 1:]:
            shared = split_polygon_ids[a] & split_polygon_ids[b]
            polygon_overlap[f"{a}__{b}"] = sorted(shared)

    if any(polygon_overlap.values()):
        raise SystemExit(
            "FAIL: cross-split landslide-ID overlap: "
            + str({k: len(v) for k, v in polygon_overlap.items()})
        )

    # Full NPZ audit.
    npz_seen = set()
    mask_values = set()
    feature_shapes = Counter()
    mask_shapes = Counter()
    positive_pixels = Counter()
    negative_pixels = Counter()
    missing_npz = []
    bad_npz = []

    print("Auditing all NPZ files...")
    for i, r in enumerate(rows, start=1):
        p = d / r["patch_path"]
        if not p.exists():
            missing_npz.append(str(p))
            continue

        resolved = p.resolve()
        npz_seen.add(resolved)

        try:
            with np.load(p) as data:
                if "features" not in data or "mask" not in data:
                    bad_npz.append((r["patch_id"], "missing features or mask"))
                    continue

                features = data["features"]
                mask = data["mask"]

                feature_shapes[tuple(features.shape)] += 1
                mask_shapes[tuple(mask.shape)] += 1

                vals = set(np.unique(mask).tolist())
                mask_values.update(vals)

                if not vals <= {0, 1}:
                    bad_npz.append(
                        (r["patch_id"], f"non-binary mask values {sorted(vals)}")
                    )

                if features.shape != (7, 256, 256):
                    bad_npz.append(
                        (r["patch_id"], f"feature shape {features.shape}")
                    )

                if mask.shape != (256, 256):
                    bad_npz.append(
                        (r["patch_id"], f"mask shape {mask.shape}")
                    )

                s = r["split"]
                positive_pixels[s] += int((mask == 1).sum())
                negative_pixels[s] += int((mask == 0).sum())

        except Exception as exc:
            bad_npz.append((r["patch_id"], f"{type(exc).__name__}: {exc}"))

        if i % 500 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)}")

    if missing_npz:
        raise SystemExit(
            f"FAIL: {len(missing_npz)} NPZ files missing. First: {missing_npz[:5]}"
        )

    if bad_npz:
        raise SystemExit(
            f"FAIL: {len(bad_npz)} invalid NPZ files. First: {bad_npz[:10]}"
        )

    if mask_values != {0, 1}:
        raise SystemExit(
            f"FAIL: expected dataset-wide mask values [0,1], got {sorted(mask_values)}"
        )

    # Ensure there are no obvious duplicate patch_path targets.
    paths = [str(Path(r["patch_path"])).casefold() for r in rows]
    if len(paths) != len(set(paths)):
        raise SystemExit("FAIL: duplicate patch_path entries in final manifest")

    # Primary validation/test must contain positives.
    for s in ("validation", "test"):
        if positive_pixels[s] <= 0:
            raise SystemExit(f"FAIL: {s} has zero positive pixels")
        if positive_manifest_rows[s] <= 0:
            raise SystemExit(f"FAIL: {s} has zero positive manifest patches")

    report = {
        "status": "PASS",
        "dataset_dir": str(d),
        "ground_truth_policy": "strict_binary_0_1",
        "total_patches": len(rows),
        "split_patch_counts": dict(split_counts),
        "split_tile_counts": tile_counts,
        "feature_shapes": {str(k): v for k, v in feature_shapes.items()},
        "mask_shapes": {str(k): v for k, v in mask_shapes.items()},
        "mask_values": sorted(mask_values),
        "spatial_buffer_m": verification["buffer_m"],
        "spatial_violation_count": violation_count,
        "cross_split_polygon_overlap_count": {
            k: len(v) for k, v in polygon_overlap.items()
        },
        "split_unique_polygon_ids": {
            s: len(split_polygon_ids[s]) for s in VALID_SPLITS
        },
        "split_categories": {
            s: dict(categories[s]) for s in VALID_SPLITS
        },
        "split_hard_negative_patch_counts": dict(hard_negative_counts),
        "split_positive_manifest_patch_counts": dict(positive_manifest_rows),
        "split_pixel_counts": {
            s: {
                "positive": positive_pixels[s],
                "negative": negative_pixels[s],
                "positive_fraction": (
                    positive_pixels[s] /
                    max(1, positive_pixels[s] + negative_pixels[s])
                ),
            }
            for s in VALID_SPLITS
        },
        "split_ground_fraction": {
            s: {
                "mean": float(np.mean(ground_fraction_values[s])) if ground_fraction_values[s] else None,
                "min": float(np.min(ground_fraction_values[s])) if ground_fraction_values[s] else None,
                "max": float(np.max(ground_fraction_values[s])) if ground_fraction_values[s] else None,
            }
            for s in VALID_SPLITS
        },
        "channels": channels,
        "test_lock_present": lock_path.exists(),
        "test_policy": (
            "Test is locked and must not be used for normalization, class weighting, "
            "checkpoint selection, threshold selection, architecture selection, "
            "feature selection, or hyperparameter tuning."
        ),
    }

    json_path = d / f"{args.report_prefix}.json"
    md_path = d / f"{args.report_prefix}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Phase 3 Tillamook Dataset QC",
        "",
        "**Status: PASS**",
        "",
        "| Split | Tiles | Patches | Positive-mask patches | Positive pixels | GT positive % | Polygon IDs | Hard negatives |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in VALID_SPLITS:
        pix = positive_pixels[s] + negative_pixels[s]
        lines.append(
            f"| {s} | {tile_counts[s]} | {split_counts[s]} | "
            f"{positive_manifest_rows[s]} | {positive_pixels[s]} | "
            f"{pct(positive_pixels[s], pix):.3f}% | "
            f"{len(split_polygon_ids[s])} | {hard_negative_counts[s]} |"
        )

    lines += [
        "",
        "## Hard gates",
        "",
        "- Strict binary mask values: **PASS — [0, 1]**",
        "- Feature shape: **PASS — 7 × 256 × 256**",
        "- Spatial split buffer: **PASS — 500 m, 0 violations**",
        "- Cross-split tile overlap: **PASS — 0**",
        "- Cross-split landslide-ID overlap: **PASS — 0**",
        "- Internal test lock file: **PASS**",
        "",
        "## Channels",
        "",
        *[f"- `{x}`" for x in channels],
        "",
        "The test split remains locked until architecture, feature set, epoch/checkpoint, and validation threshold are frozen.",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print()
    print("PHASE 3 QC: PASS")
    print(f"Spatial violations: {violation_count}")
    print("Cross-split polygon overlap: 0")
    print(f"Mask values: {sorted(mask_values)}")
    print(f"Feature shapes: {dict(feature_shapes)}")
    for s in VALID_SPLITS:
        total_pix = positive_pixels[s] + negative_pixels[s]
        print(
            f"{s:10s}: tiles={tile_counts[s]:2d} patches={split_counts[s]:4d} "
            f"positive_patches={positive_manifest_rows[s]:4d} "
            f"GT_positive={pct(positive_pixels[s], total_pix):6.3f}% "
            f"polygons={len(split_polygon_ids[s]):3d}"
        )
    print(f"Report: {json_path}")
    print()
    print("NEXT: patch/run the binary 2x2 feature-depth trainer. Do not evaluate test.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
