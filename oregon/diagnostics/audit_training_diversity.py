#!/usr/bin/env python3
"""Audit source-region training/validation diversity without touching external tests."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from phase_common import (
    as_bool,
    as_float,
    potential_redundancy,
    read_csv,
    split_tokens,
    summarize_manifest,
    write_csv,
    write_json,
)

NUMERIC_MANIFEST_FIELDS = (
    "positive_fraction",
    "ignore_fraction",
    "ground_fraction",
    "mean_slope_degrees",
    "distance_to_positive_m",
    "boundary_pixel_fraction",
    "boundary_of_positive_fraction",
)


def describe(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray([value for value in values if np.isfinite(value)], dtype=np.float64)
    if not array.size:
        return {"count": 0, "min": math.nan, "p10": math.nan, "p25": math.nan, "median": math.nan, "p75": math.nan, "p90": math.nan, "max": math.nan, "mean": math.nan, "std": math.nan}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "p10": float(np.percentile(array, 10)),
        "p25": float(np.percentile(array, 25)),
        "median": float(np.median(array)),
        "p75": float(np.percentile(array, 75)),
        "p90": float(np.percentile(array, 90)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def group_summary(rows: list[dict[str, str]], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    grouping_specs = (
        ("split",),
        ("split", "category"),
        ("split", "coverage_class"),
        ("split", "is_hard_negative"),
    )
    for group_fields in grouping_specs:
        groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
        for row in rows:
            groups[tuple(row.get(field, "") for field in group_fields)].append(row)
        for key, subset in sorted(groups.items()):
            prefix = {field: value for field, value in zip(group_fields, key)}
            for field in fields:
                stats = describe(as_float(row.get(field)) for row in subset)
                output.append({"group_by": "+".join(group_fields), **prefix, "metric": field, **stats})
    return output


def tile_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("split", ""), row.get("tile_name", ""))].append(row)
    output: list[dict[str, Any]] = []
    for (split, tile), subset in sorted(groups.items()):
        positives = [row for row in subset if as_float(row.get("positive_fraction"), 0.0) > 0]
        negatives = [row for row in subset if as_float(row.get("positive_fraction"), 0.0) <= 0]
        polygons = {key for row in positives for key in split_tokens(row.get("patch_polygon_keys"))}
        output.append(
            {
                "split": split,
                "tile_name": tile,
                "patches": len(subset),
                "positive_patches": len(positives),
                "negative_patches": len(negatives),
                "positive_patch_fraction": len(positives) / len(subset) if subset else 0.0,
                "unique_positive_polygon_keys": len(polygons),
                "hard_negatives": sum(as_bool(row.get("is_hard_negative")) for row in negatives),
                "boundary_patches": sum(as_bool(row.get("contains_positive_boundary")) for row in subset),
                "mean_positive_fraction": float(np.mean([as_float(row.get("positive_fraction"), 0.0) for row in subset])),
                "mean_ignore_fraction": float(np.mean([as_float(row.get("ignore_fraction"), 0.0) for row in subset])),
                "mean_slope_degrees": float(np.mean([as_float(row.get("mean_slope_degrees"), 0.0) for row in subset])),
            }
        )
    return output


def polygon_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    memberships: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        for key in split_tokens(row.get("patch_polygon_keys")):
            memberships[(row.get("split", ""), key)].append(row)
    output: list[dict[str, Any]] = []
    for (split, polygon), subset in sorted(memberships.items()):
        output.append(
            {
                "split": split,
                "polygon_key": polygon,
                "patch_memberships": len(subset),
                "unique_tiles": len({row.get("tile_name", "") for row in subset}),
                "boundary_patch_memberships": sum(as_bool(row.get("contains_positive_boundary")) for row in subset),
                "coverage_classes": ";".join(sorted({row.get("coverage_class", "") for row in subset})),
                "positive_fraction_min": min(as_float(row.get("positive_fraction"), 0.0) for row in subset),
                "positive_fraction_median": float(np.median([as_float(row.get("positive_fraction"), 0.0) for row in subset])),
                "positive_fraction_max": max(as_float(row.get("positive_fraction"), 0.0) for row in subset),
            }
        )
    return output


def spatial_grid_summary(rows: list[dict[str, str]], bins: int) -> list[dict[str, Any]]:
    valid_rows = []
    for row in rows:
        x = (as_float(row.get("x_min")) + as_float(row.get("x_max"))) / 2
        y = (as_float(row.get("y_min")) + as_float(row.get("y_max"))) / 2
        if np.isfinite(x) and np.isfinite(y):
            valid_rows.append((row, x, y))
    if not valid_rows:
        return []
    x_values = np.asarray([x for _, x, _ in valid_rows])
    y_values = np.asarray([y for _, _, y in valid_rows])
    x_edges = np.linspace(x_values.min(), x_values.max() + 1e-9, bins + 1)
    y_edges = np.linspace(y_values.min(), y_values.max() + 1e-9, bins + 1)
    groups: dict[tuple[str, int, int], list[dict[str, str]]] = defaultdict(list)
    for row, x, y in valid_rows:
        x_bin = min(bins - 1, max(0, int(np.searchsorted(x_edges, x, side="right") - 1)))
        y_bin = min(bins - 1, max(0, int(np.searchsorted(y_edges, y, side="right") - 1)))
        groups[(row.get("split", ""), x_bin, y_bin)].append(row)
    output: list[dict[str, Any]] = []
    for (split, x_bin, y_bin), subset in sorted(groups.items()):
        output.append(
            {
                "split": split,
                "x_bin": x_bin,
                "y_bin": y_bin,
                "x_min": float(x_edges[x_bin]),
                "x_max": float(x_edges[x_bin + 1]),
                "y_min": float(y_edges[y_bin]),
                "y_max": float(y_edges[y_bin + 1]),
                "patches": len(subset),
                "positive_patches": sum(as_float(row.get("positive_fraction"), 0.0) > 0 for row in subset),
                "unique_tiles": len({row.get("tile_name", "") for row in subset}),
                "unique_polygon_keys": len({key for row in subset for key in split_tokens(row.get("patch_polygon_keys"))}),
            }
        )
    return output


def provenance_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    fields = (
        "region_id",
        "region_role",
        "lidar_project",
        "lidar_year",
        "lidar_year_source",
        "lidar_acquisition_year",
        "lidar_acquisition_start_year",
        "lidar_acquisition_end_year",
        "lidar_acquisition_source",
        "lidar_acquisition_evidence",
        "lidar_acquisition_verified",
        "lidar_file_creation_year",
        "lidar_inferred_year_hint",
        "lidar_inferred_year_hint_source",
        "label_quality",
    )
    output: list[dict[str, Any]] = []
    for field in fields:
        counts = Counter(str(row.get(field, "")).strip() for row in rows)
        for value, count in counts.most_common():
            output.append({"field": field, "value": value, "rows": count})
    return output


def support_gap(rows: list[dict[str, str]], numeric_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    train = [row for row in rows if row.get("split") == "train"]
    validation = [row for row in rows if row.get("split") == "validation"]
    output: list[dict[str, Any]] = []
    for field in numeric_fields:
        train_values = np.asarray([as_float(row.get(field)) for row in train], dtype=np.float64)
        val_values = np.asarray([as_float(row.get(field)) for row in validation], dtype=np.float64)
        train_values = train_values[np.isfinite(train_values)]
        val_values = val_values[np.isfinite(val_values)]
        if not train_values.size or not val_values.size:
            continue
        p01, p99 = np.percentile(train_values, [1, 99])
        outside = (val_values < p01) | (val_values > p99)
        output.append(
            {
                "metric": field,
                "train_min": float(train_values.min()),
                "train_p01": float(p01),
                "train_median": float(np.median(train_values)),
                "train_p99": float(p99),
                "train_max": float(train_values.max()),
                "validation_min": float(val_values.min()),
                "validation_median": float(np.median(val_values)),
                "validation_max": float(val_values.max()),
                "validation_outside_train_p01_p99_count": int(outside.sum()),
                "validation_outside_train_p01_p99_fraction": float(outside.mean()),
                "interpretation": "screening indicator, not proof of terrain-domain mismatch",
            }
        )
    return output


def feature_patch_statistics(dataset_dir: Path, rows: list[dict[str, str]], channels: list[str]) -> tuple[list[dict[str, Any]], list[str]]:
    output: list[dict[str, Any]] = []
    missing: list[str] = []
    for index, row in enumerate(rows, start=1):
        path = dataset_dir / Path(row.get("patch_path", ""))
        if not path.exists():
            missing.append(str(path))
            continue
        with np.load(path) as data:
            features = data["features"].astype(np.float32)
        if features.shape[0] != len(channels):
            raise ValueError(f"Channel mismatch in {path}: {features.shape[0]} vs {len(channels)}")
        for channel_index, channel in enumerate(channels):
            values = features[channel_index]
            values = values[np.isfinite(values)]
            stats = describe(values.tolist())
            output.append(
                {
                    "patch_id": row.get("patch_id", ""),
                    "split": row.get("split", ""),
                    "category": row.get("category", ""),
                    "coverage_class": row.get("coverage_class", ""),
                    "tile_name": row.get("tile_name", ""),
                    "channel": channel,
                    **stats,
                }
            )
        if index % 100 == 0 or index == len(rows):
            print(f"Feature audit: {index}/{len(rows)} manifest rows")
    return output, missing


def aggregate_feature_patch_statistics(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["split"]), str(row["category"]), str(row["channel"]))].append(row)
    output: list[dict[str, Any]] = []
    for (split, category, channel), subset in sorted(groups.items()):
        for patch_stat in ("mean", "std", "p10", "median", "p90"):
            stats = describe(float(row[patch_stat]) for row in subset)
            output.append(
                {
                    "split": split,
                    "category": category,
                    "channel": channel,
                    "patch_statistic": patch_stat,
                    **stats,
                }
            )
    return output


def stratified_hard_negative_review(rows: list[dict[str, str]], per_stratum: int = 5) -> list[dict[str, Any]]:
    candidates = [
        row for row in rows
        if row.get("split") == "train"
        and as_float(row.get("positive_fraction"), 0.0) <= 0
        and as_bool(row.get("is_hard_negative"))
    ]
    if not candidates:
        return []
    slopes = np.asarray([as_float(row.get("mean_slope_degrees"), 0.0) for row in candidates], dtype=np.float64)
    distances = np.asarray([as_float(row.get("distance_to_positive_m"), 0.0) for row in candidates], dtype=np.float64)
    for array in (slopes, distances):
        finite = array[np.isfinite(array)]
        replacement = float(finite.max()) if finite.size else 0.0
        array[~np.isfinite(array)] = replacement
    slope_edges = np.percentile(slopes, [0, 33.3, 66.7, 100])
    distance_edges = np.percentile(distances, [0, 33.3, 66.7, 100])
    groups: dict[tuple[int, int], list[dict[str, str]]] = defaultdict(list)
    for row in candidates:
        slope = as_float(row.get("mean_slope_degrees"), 0.0)
        distance = as_float(row.get("distance_to_positive_m"), 0.0)
        slope_bin = min(2, int(np.searchsorted(slope_edges, slope, side="right") - 1))
        distance_bin = min(2, int(np.searchsorted(distance_edges, distance, side="right") - 1))
        groups[(max(0, slope_bin), max(0, distance_bin))].append(row)
    output: list[dict[str, Any]] = []
    for (slope_bin, distance_bin), subset in sorted(groups.items()):
        # Spread selection over tiles rather than taking repeated windows from one tile.
        subset = sorted(subset, key=lambda row: (row.get("tile_name", ""), row.get("patch_id", "")))
        selected: list[dict[str, str]] = []
        used_tiles: set[str] = set()
        for row in subset:
            tile = row.get("tile_name", "")
            if tile not in used_tiles:
                selected.append(row)
                used_tiles.add(tile)
            if len(selected) >= per_stratum:
                break
        for row in subset:
            if len(selected) >= per_stratum:
                break
            if row not in selected:
                selected.append(row)
        for row in selected:
            output.append(
                {
                    "patch_id": row.get("patch_id", ""),
                    "patch_path": row.get("patch_path", ""),
                    "tile_name": row.get("tile_name", ""),
                    "slope_stratum": slope_bin,
                    "distance_stratum": distance_bin,
                    "mean_slope_degrees": row.get("mean_slope_degrees", ""),
                    "distance_to_positive_m": row.get("distance_to_positive_m", ""),
                    "manual_terrain_category": "unreviewed",
                    "allowed_categories": (
                        "steep_non_landslide_slope;drainage_or_valley_side;ridge_or_convex_terrain;"
                        "rough_natural_terrain;road_or_engineered_cut;forest_management_disturbance;other"
                    ),
                    "review_notes": "",
                    "evidence_status": "requires_visual_or_ancillary_review",
                }
            )
    return output


def markdown_report(
    summary: dict[str, Any],
    tiles: list[dict[str, Any]],
    polygons: list[dict[str, Any]],
    redundancy: dict[str, Any],
    support: list[dict[str, Any]],
    feature_status: str,
) -> str:
    train = summary.get("train", {})
    validation = summary.get("validation", {})
    train_tiles = sorted((row for row in tiles if row["split"] == "train"), key=lambda row: row["positive_patches"], reverse=True)
    total_train_positive = max(1, int(train.get("positive_patches", 0)))
    top4_positive = sum(int(row["positive_patches"]) for row in train_tiles[:4])
    train_polygons = sorted((row for row in polygons if row["split"] == "train"), key=lambda row: row["patch_memberships"], reverse=True)
    total_memberships = max(1, sum(int(row["patch_memberships"]) for row in train_polygons))
    top4_memberships = sum(int(row["patch_memberships"]) for row in train_polygons[:4])
    train_polygon_set = {row["polygon_key"] for row in train_polygons}
    val_polygon_set = {row["polygon_key"] for row in polygons if row["split"] == "validation"}

    lines = [
        "# Tillamook training-data diversity audit",
        "",
        "## Scope",
        "",
        "This report audits only the source-region training and validation manifest. It does not use Buxton/Vernonia or Oregon City data and does not change labels, ignore masks, normalization, or model architecture.",
        "",
        "## Measured findings",
        "",
        f"- Train: **{train.get('patches', 0)} patches**, **{train.get('unique_tiles', 0)} tiles**, **{train.get('positive_patches', 0)} positive patches**, and **{train.get('unique_positive_polygon_keys', 0)} unique positive polygon keys**.",
        f"- Validation: **{validation.get('patches', 0)} patches**, **{validation.get('unique_tiles', 0)} tiles**, **{validation.get('positive_patches', 0)} positive patches**, and **{validation.get('unique_positive_polygon_keys', 0)} unique positive polygon keys**.",
        f"- Train/validation positive polygon-key overlap: **{len(train_polygon_set & val_polygon_set)}**.",
        f"- The four training tiles with the most positive patches contain **{top4_positive / total_train_positive:.2%}** of all training positive patches.",
        f"- The four most repeated training polygon keys account for **{top4_memberships / total_memberships:.2%}** of positive polygon-patch memberships.",
        f"- Potential overlapping-window indicator: **{redundancy['same_polygon_pairs_overlap_fraction_ge_threshold']}** same-polygon pairs have at least {redundancy['overlap_threshold_of_smaller_patch']:.0%} overlap of the smaller patch. This indicates possible redundancy, not automatically invalid data.",
        f"- Feature-file audit status: **{feature_status}**.",
        "",
        "### Coverage composition",
        "",
        f"- Train coverage classes: `{json.dumps(train.get('coverage_counts', {}), sort_keys=True)}`.",
        f"- Validation coverage classes: `{json.dumps(validation.get('coverage_counts', {}), sort_keys=True)}`.",
        "",
        "### Manifest-level train/validation support screening",
        "",
    ]
    if support:
        lines.append("| Metric | Validation outside train 1st–99th percentile |")
        lines.append("|---|---:|")
        for row in support:
            lines.append(f"| {row['metric']} | {row['validation_outside_train_p01_p99_fraction']:.2%} |")
    else:
        lines.append("No comparable numeric fields were available.")

    lines.extend(
        [
            "",
            "## What is not measured automatically",
            "",
            "The manifest does not identify roads, engineered cuts, drainage forms, ridges, forest-management disturbance, or uncertain/incomplete inventory labels. Counts of hard negatives therefore measure sampling status, **not semantic hard-negative diversity**. Those categories require visual review of the exported candidates and, where needed, ancillary imagery or vector layers.",
            "",
            "## Hypotheses to test with the ablation and visual review",
            "",
            "- Repeated overlapping windows around a small number of polygons may reduce effective morphological diversity.",
            "- Numerically abundant hard negatives may still be too homogeneous, allowing systematic false positives on underrepresented terrain forms.",
            "- Validation estimates may be unstable because the positive validation subset contains few polygons and lacks some positive-coverage classes present in training.",
            "",
            "## Expansion priorities supported by this audit",
            "",
            "1. Select spatially distinct tiles and new polygon groups before adding more windows around already dominant polygons.",
            "2. Add visually verified hard negatives across drainage/valley sides, ridges/convex slopes, rough natural terrain, roads/cuts, and forest-management disturbance.",
            "3. Add underrepresented landslide sizes and coverage patterns, especially classes missing from validation, while keeping validation frozen for the current ablation.",
            "4. Track per-tile and per-polygon caps so nominal patch count is not confused with independent terrain or morphology diversity.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("patches_boundary_aware.csv"))
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--spatial-bins", type=int, default=5)
    parser.add_argument("--skip-npz", action="store_true", help="Manifest-only audit; skip terrain-channel files")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    manifest_path = args.manifest.resolve() if args.manifest.is_absolute() else (dataset_dir / args.manifest).resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        parser.error(f"Manifest not found: {manifest_path}")

    rows = read_csv(manifest_path)
    source_rows = [row for row in rows if row.get("split") in {"train", "validation"}]
    if not source_rows:
        parser.error("No train/validation rows in manifest")
    roles = sorted({row.get("region_role", "") for row in source_rows})
    if roles != ["train_val"]:
        parser.error(f"Audit is source-only; expected region_role=train_val, found {roles}")

    summary = summarize_manifest(source_rows)
    tiles = tile_summary(source_rows)
    polygons = polygon_summary(source_rows)
    numeric = group_summary(source_rows, NUMERIC_MANIFEST_FIELDS)
    spatial = spatial_grid_summary(source_rows, args.spatial_bins)
    support = support_gap(source_rows, NUMERIC_MANIFEST_FIELDS)
    provenance = provenance_summary(source_rows)
    redundancy = potential_redundancy([row for row in source_rows if row.get("split") == "train"], 0.5)
    hard_negative_review = stratified_hard_negative_review(source_rows)

    write_json(outdir / "manifest_summary.json", summary)
    write_csv(outdir / "tile_summary.csv", tiles)
    write_csv(outdir / "polygon_summary.csv", polygons)
    write_csv(outdir / "manifest_numeric_distributions.csv", numeric)
    write_csv(outdir / "spatial_grid_summary.csv", spatial)
    write_csv(outdir / "train_validation_support_screen.csv", support)
    write_csv(outdir / "provenance_summary.csv", provenance)
    write_json(outdir / "potential_redundancy.json", redundancy)
    write_csv(outdir / "hard_negative_review_candidates.csv", hard_negative_review)

    feature_status = "skipped by --skip-npz"
    missing_feature_paths: list[str] = []
    if not args.skip_npz:
        channels_path = dataset_dir / "channels.json"
        if not channels_path.exists():
            feature_status = "channels.json missing; manifest outputs completed"
        else:
            channels = list(json.loads(channels_path.read_text(encoding="utf-8"))["feature_names"])
            feature_rows, missing_feature_paths = feature_patch_statistics(dataset_dir, source_rows, channels)
            write_csv(outdir / "per_patch_feature_statistics.csv", feature_rows)
            write_csv(outdir / "feature_distribution_by_split_category.csv", aggregate_feature_patch_statistics(feature_rows))
            feature_status = f"processed {len(feature_rows)} patch-channel records; missing patch files={len(missing_feature_paths)}"
            if missing_feature_paths:
                (outdir / "missing_patch_files.txt").write_text("\n".join(missing_feature_paths), encoding="utf-8")

    report = markdown_report(summary, tiles, polygons, redundancy, support, feature_status)
    (outdir / "training_data_diversity_report.md").write_text(report, encoding="utf-8")
    write_json(
        outdir / "audit_provenance.json",
        {
            "dataset_dir": str(dataset_dir),
            "manifest": str(manifest_path),
            "outdir": str(outdir),
            "source_splits": ["train", "validation"],
            "external_test_regions_used": False,
            "ignore_policy_changed": False,
            "feature_status": feature_status,
            "missing_patch_files": len(missing_feature_paths),
        },
    )
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
