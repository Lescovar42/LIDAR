#!/usr/bin/env python3
"""Shared utilities for the class-weight ablation and diversity-audit phase."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

IGNORE_INDEX = 255


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def stable_values_sha256(values: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for value in sorted(str(item) for item in values):
        digest.update(value.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def as_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def as_bool(value: Any) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def split_tokens(value: Any) -> list[str]:
    return [token.strip() for token in str(value or "").split(";") if token.strip()]


def threshold_values(start: float, stop: float, step: float) -> list[float]:
    if not (0.0 <= start <= 1.0 and 0.0 <= stop <= 1.0):
        raise ValueError("Threshold start/stop must be inside [0, 1]")
    if step <= 0:
        raise ValueError("Threshold step must be positive")
    if stop < start:
        raise ValueError("Threshold stop must be >= start")
    count = int(round((stop - start) / step))
    values = [round(start + index * step, 10) for index in range(count + 1)]
    if values[-1] < stop - 1e-9:
        values.append(round(stop, 10))
    return [value for value in values if value <= stop + 1e-9]


def confusion_from_arrays(probability: np.ndarray, target: np.ndarray, threshold: float) -> dict[str, int]:
    if probability.shape != target.shape:
        raise ValueError(f"Shape mismatch: probability={probability.shape}, target={target.shape}")
    valid = target != IGNORE_INDEX
    prediction = probability >= threshold
    truth = target == 1
    return {
        "tp": int(np.count_nonzero(valid & prediction & truth)),
        "fp": int(np.count_nonzero(valid & prediction & ~truth)),
        "fn": int(np.count_nonzero(valid & ~prediction & truth)),
        "tn": int(np.count_nonzero(valid & ~prediction & ~truth)),
        "ignored": int(np.count_nonzero(~valid)),
        "total": int(target.size),
    }


def add_counts(destination: dict[str, int], source: Mapping[str, int]) -> None:
    for key in ("tp", "fp", "fn", "tn", "ignored", "total"):
        destination[key] = int(destination.get(key, 0)) + int(source.get(key, 0))


def metrics_from_counts(counts: Mapping[str, int]) -> dict[str, float | int]:
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    tn = int(counts.get("tn", 0))
    ignored = int(counts.get("ignored", 0))
    total = int(counts.get("total", tp + fp + fn + tn + ignored))
    valid = tp + fp + fn + tn

    def ratio(numerator: float, denominator: float) -> float:
        return float(numerator / denominator) if denominator else 0.0

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "ignored": ignored,
        "valid_pixels": valid,
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "gt_positive_fraction": ratio(tp + fn, valid),
        "predicted_positive_fraction": ratio(tp + fp, valid),
        "ignore_fraction": ratio(ignored, total),
    }


def choose_best_threshold(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("No threshold rows")
    # Primary model-selection metric remains Dice. Secondary keys make ties deterministic
    # without silently privileging either class weight.
    return dict(
        max(
            rows,
            key=lambda row: (
                as_float(row.get("dice"), -math.inf),
                as_float(row.get("iou"), -math.inf),
                as_float(row.get("precision"), -math.inf),
                as_float(row.get("recall"), -math.inf),
                -abs(as_float(row.get("threshold"), 0.5) - 0.5),
            ),
        )
    )


def dataset_fingerprint(manifest_path: Path, rows: Sequence[Mapping[str, str]], split: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("split") == split]
    return {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": sha256_file(manifest_path),
        "split": split,
        "row_count": len(selected),
        "patch_ids_sha256": stable_values_sha256(row.get("patch_id", "") for row in selected),
        "patch_paths_sha256": stable_values_sha256(row.get("patch_path", "") for row in selected),
        "region_ids": sorted({row.get("region_id", "") for row in selected}),
        "region_roles": sorted({row.get("region_role", "") for row in selected}),
    }


def summarize_manifest(rows: Sequence[Mapping[str, str]]) -> dict[str, Any]:
    by_split: dict[str, dict[str, Any]] = {}
    for split in sorted({row.get("split", "") for row in rows}):
        subset = [row for row in rows if row.get("split") == split]
        positives = [row for row in subset if as_float(row.get("positive_fraction"), 0.0) > 0]
        negatives = [row for row in subset if as_float(row.get("positive_fraction"), 0.0) <= 0]
        polygon_keys = {key for row in positives for key in split_tokens(row.get("patch_polygon_keys"))}
        tile_names = {row.get("tile_name", "") for row in subset}
        by_split[split] = {
            "patches": len(subset),
            "positive_patches": len(positives),
            "negative_patches": len(negatives),
            "unique_tiles": len(tile_names),
            "unique_positive_polygon_keys": len(polygon_keys),
            "boundary_patches": sum(as_bool(row.get("contains_positive_boundary")) for row in subset),
            "hard_negative_patches": sum(as_bool(row.get("is_hard_negative")) for row in negatives),
            "category_counts": dict(Counter(row.get("category", "") for row in subset)),
            "coverage_counts": dict(Counter(row.get("coverage_class", "") for row in subset)),
        }
    return by_split


def bbox_overlap_fraction(a: Mapping[str, str], b: Mapping[str, str]) -> float:
    ax1, ay1 = as_float(a.get("x_min")), as_float(a.get("y_min"))
    ax2, ay2 = as_float(a.get("x_max")), as_float(a.get("y_max"))
    bx1, by1 = as_float(b.get("x_min")), as_float(b.get("y_min"))
    bx2, by2 = as_float(b.get("x_max")), as_float(b.get("y_max"))
    if any(math.isnan(v) for v in (ax1, ay1, ax2, ay2, bx1, by1, bx2, by2)):
        return 0.0
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denominator = min(area_a, area_b)
    return float(intersection / denominator) if denominator else 0.0


def potential_redundancy(rows: Sequence[Mapping[str, str]], overlap_threshold: float = 0.5) -> dict[str, Any]:
    positives = [row for row in rows if as_float(row.get("positive_fraction"), 0.0) > 0]
    by_tile: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in positives:
        by_tile[row.get("tile_name", "")].append(row)
    pair_count = 0
    overlap_pairs = 0
    same_polygon_overlap_pairs = 0
    for tile_rows in by_tile.values():
        for i, first in enumerate(tile_rows):
            first_polygons = set(split_tokens(first.get("patch_polygon_keys")))
            for second in tile_rows[i + 1 :]:
                pair_count += 1
                overlap = bbox_overlap_fraction(first, second)
                if overlap >= overlap_threshold:
                    overlap_pairs += 1
                    if first_polygons & set(split_tokens(second.get("patch_polygon_keys"))):
                        same_polygon_overlap_pairs += 1
    return {
        "positive_patch_pairs_within_tile": pair_count,
        "pairs_overlap_fraction_ge_threshold": overlap_pairs,
        "same_polygon_pairs_overlap_fraction_ge_threshold": same_polygon_overlap_pairs,
        "overlap_threshold_of_smaller_patch": overlap_threshold,
        "note": "Potential redundancy indicator only; overlapping windows are not automatically invalid samples.",
    }
