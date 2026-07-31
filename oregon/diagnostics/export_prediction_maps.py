#!/usr/bin/env python3
"""Export direct ground-truth versus predicted segmentation maps.

The output is intentionally simpler than an error-inspection overlay:

- Ground Truth mask
- Predicted binary mask
- Difference/error map

It also stitches overlapping validation patches into one cropped mosaic per
LiDAR tile. A tile mosaic contains only areas represented by the selected
manifest rows; gaps are marked as unevaluated.

This script does not rebuild data, alter labels, retrain the model, or evaluate
the held-out test split unless explicitly requested.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import inspect_visual_errors as ive  # noqa: E402

IGNORE_INDEX = ive.IGNORE_INDEX


@dataclass
class TileMosaic:
    tile_name: str
    min_row: int
    min_col: int
    probability: np.ndarray
    target: np.ndarray
    source_count: np.ndarray
    conflict_mask: np.ndarray
    row_count: int
    patch_count: int
    cell_size: float | None
    x_origin: float | None
    y_origin_top: float | None
    crs: str


def write_csv(
    path: Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fieldnames),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "unnamed"


def compute_patch_record(
    result: ive.PatchResult,
    threshold: float,
) -> dict[str, Any]:
    record = ive.record_for_threshold(result, threshold)
    record.pop("counts", None)
    record["row_offset"] = int(result.row.get("row_offset", "0"))
    record["col_offset"] = int(result.row.get("col_offset", "0"))
    record["patch_path"] = result.row.get("patch_path", "")
    record["dominant_error"] = ive.dominant_error(record)
    return record


def mask_rgb(target: np.ndarray) -> np.ndarray:
    """Black background, white positive, mid-gray ignore."""
    rgb = np.zeros((*target.shape, 3), dtype=np.float32)
    rgb[target == 1] = 1.0
    rgb[target == IGNORE_INDEX] = 0.45
    return rgb


def prediction_rgb(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Black negative, white predicted positive, mid-gray unevaluated."""
    rgb = np.zeros((*target.shape, 3), dtype=np.float32)
    rgb[probability >= threshold] = 1.0
    rgb[target == IGNORE_INDEX] = 0.45
    return rgb


def error_rgb(
    probability: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Dark TN, yellow TP, red FP, green FN, gray ignore."""
    valid = target != IGNORE_INDEX
    truth = target == 1
    prediction = probability >= threshold

    rgb = np.zeros((*target.shape, 3), dtype=np.float32)
    rgb[valid & ~truth & ~prediction] = (0.10, 0.10, 0.10)
    rgb[valid & truth & prediction] = (1.00, 1.00, 0.00)
    rgb[valid & ~truth & prediction] = (1.00, 0.00, 0.00)
    rgb[valid & truth & ~prediction] = (0.00, 1.00, 0.00)
    rgb[~valid] = (0.45, 0.45, 0.45)
    return rgb


def probability_rgb(
    probability: np.ndarray,
    target: np.ndarray,
) -> np.ma.MaskedArray:
    return np.ma.masked_where(target == IGNORE_INDEX, probability)


def save_single_patch_outputs(
    *,
    output_dir: Path,
    result: ive.PatchResult,
    record: Mapping[str, Any],
    threshold: float,
) -> dict[str, str]:
    import matplotlib.pyplot as plt

    patch_id = str(record["patch_id"])
    stem = sanitize_filename(patch_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_path = output_dir / f"{stem}__gt.png"
    pred_path = output_dir / f"{stem}__pred_t{threshold:.2f}.png"
    error_path = output_dir / f"{stem}__error_t{threshold:.2f}.png"
    comparison_path = output_dir / f"{stem}__predicted_vs_gt.png"

    plt.imsave(gt_path, mask_rgb(result.target))
    plt.imsave(
        pred_path,
        prediction_rgb(result.probabilities, result.target, threshold),
    )
    plt.imsave(
        error_path,
        error_rgb(result.probabilities, result.target, threshold),
    )

    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    axes[0].imshow(mask_rgb(result.target))
    axes[0].set_title("Ground Truth")
    axes[1].imshow(
        prediction_rgb(result.probabilities, result.target, threshold)
    )
    axes[1].set_title(f"Predicted Mask @ {threshold:.2f}")
    probability_image = axes[2].imshow(
        probability_rgb(result.probabilities, result.target),
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Predicted Probability")
    fig.colorbar(probability_image, ax=axes[2], fraction=0.046, pad=0.04)
    axes[3].imshow(
        error_rgb(result.probabilities, result.target, threshold)
    )
    axes[3].set_title("Difference\nYellow TP | Red FP | Green FN")

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(
        f"{patch_id}\n"
        f"Dice={float(record['dice']):.3f} | "
        f"Precision={float(record['precision']):.3f} | "
        f"Recall={float(record['recall']):.3f} | "
        f"Ignore={100*float(record['ignore_fraction']):.1f}%"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(comparison_path, dpi=180)
    plt.close(fig)

    return {
        "gt_png": str(gt_path),
        "pred_png": str(pred_path),
        "error_png": str(error_path),
        "comparison_png": str(comparison_path),
    }


def choose_professor_cases(
    records: Sequence[Mapping[str, Any]],
    *,
    max_ignore_fraction: float,
    max_cases: int,
) -> tuple[list[dict[str, Any]], float]:
    selected, used_limit = ive.select_representative_records(
        records,
        max_ignore_fraction=max_ignore_fraction,
    )

    preferred_order = [
        "representative_good",
        "representative_median",
        "true_boundary_typical",
        "false_positive_dominant",
        "false_negative_dominant",
    ]
    ordered: list[dict[str, Any]] = []
    used: set[str] = set()

    for label in preferred_order:
        for record in selected:
            if (
                record.get("case_label") == label
                and str(record.get("patch_id")) not in used
            ):
                ordered.append(dict(record))
                used.add(str(record.get("patch_id")))
                break

    eligible = [
        dict(record)
        for record in records
        if float(record.get("ignore_fraction", 1.0)) <= used_limit
        and str(record.get("patch_id")) not in used
    ]
    eligible.sort(
        key=lambda record: (
            -abs(
                float(record.get("predicted_positive_fraction_valid", 0.0))
                - float(record.get("gt_positive_fraction_valid", 0.0))
            ),
            float(record.get("ignore_fraction", 1.0)),
            str(record.get("patch_id", "")),
        )
    )

    for record in eligible:
        if len(ordered) >= max_cases:
            break
        record["case_label"] = "additional_large_disagreement"
        ordered.append(record)
        used.add(str(record.get("patch_id")))

    return ordered[:max_cases], used_limit


def save_professor_contact_sheet(
    *,
    path: Path,
    cases: Sequence[Mapping[str, Any]],
    result_by_patch: Mapping[str, ive.PatchResult],
    threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    if not cases:
        return

    fig, axes = plt.subplots(
        len(cases),
        3,
        figsize=(10, 3.3 * len(cases)),
        squeeze=False,
    )

    for row_index, record in enumerate(cases):
        result = result_by_patch[str(record["patch_id"])]
        axes[row_index, 0].imshow(mask_rgb(result.target))
        axes[row_index, 1].imshow(
            prediction_rgb(
                result.probabilities,
                result.target,
                threshold,
            )
        )
        axes[row_index, 2].imshow(
            error_rgb(
                result.probabilities,
                result.target,
                threshold,
            )
        )

        if row_index == 0:
            axes[row_index, 0].set_title("Ground Truth")
            axes[row_index, 1].set_title(f"Predicted Mask @ {threshold:.2f}")
            axes[row_index, 2].set_title(
                "Difference\nYellow TP | Red FP | Green FN"
            )

        axes[row_index, 0].set_ylabel(
            str(record.get("case_label", "")).replace("_", " ")
            + "\n"
            + f"Dice={float(record.get('dice', 0)):.3f}, "
            + f"P={float(record.get('precision', 0)):.3f}, "
            + f"R={float(record.get('recall', 0)):.3f}",
            fontsize=9,
        )

        for column in range(3):
            axes[row_index, column].set_xticks([])
            axes[row_index, column].set_yticks([])

    fig.suptitle(
        "Validation Output: Ground Truth vs Predicted Segmentation\n"
        "Gray = ignored/unevaluated pixels",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def infer_cell_size(rows: Sequence[Mapping[str, str]]) -> float | None:
    values: list[float] = []
    for row in rows:
        try:
            width = float(row["x_max"]) - float(row["x_min"])
            patch_size = None
            # The NPZ shape is used later; this is only a metadata estimate.
            if width > 0:
                values.append(width / 256.0)
        except (KeyError, TypeError, ValueError):
            continue
    if not values:
        return None
    return float(np.median(values))


def build_tile_mosaic(
    tile_name: str,
    results: Sequence[ive.PatchResult],
    *,
    max_mosaic_pixels: int,
) -> TileMosaic:
    if not results:
        raise ValueError("Cannot mosaic an empty result collection")

    offsets: list[tuple[int, int, int, int]] = []
    for result in results:
        row_offset = int(result.row["row_offset"])
        col_offset = int(result.row["col_offset"])
        height, width = result.target.shape
        offsets.append((row_offset, col_offset, height, width))

    min_row = min(value[0] for value in offsets)
    min_col = min(value[1] for value in offsets)
    max_row = max(row + height for row, _, height, _ in offsets)
    max_col = max(col + width for _, col, _, width in offsets)
    mosaic_height = max_row - min_row
    mosaic_width = max_col - min_col
    pixel_count = mosaic_height * mosaic_width

    if pixel_count > max_mosaic_pixels:
        raise MemoryError(
            f"{tile_name}: mosaic would contain {pixel_count:,} pixels "
            f"(limit {max_mosaic_pixels:,})"
        )

    probability_sum = np.zeros(
        (mosaic_height, mosaic_width),
        dtype=np.float32,
    )
    source_count = np.zeros(
        (mosaic_height, mosaic_width),
        dtype=np.uint16,
    )
    target_valid = np.zeros(
        (mosaic_height, mosaic_width),
        dtype=bool,
    )
    target_positive = np.zeros(
        (mosaic_height, mosaic_width),
        dtype=bool,
    )
    conflict_mask = np.zeros(
        (mosaic_height, mosaic_width),
        dtype=bool,
    )

    for result in results:
        row_offset = int(result.row["row_offset"]) - min_row
        col_offset = int(result.row["col_offset"]) - min_col
        height, width = result.target.shape
        row_slice = slice(row_offset, row_offset + height)
        col_slice = slice(col_offset, col_offset + width)

        valid = result.target != IGNORE_INDEX
        positive = result.target == 1

        existing_valid = target_valid[row_slice, col_slice]
        existing_positive = target_positive[row_slice, col_slice]
        conflict_mask[row_slice, col_slice] |= (
            existing_valid & valid & (existing_positive != positive)
        )

        target_valid[row_slice, col_slice] |= valid
        target_positive[row_slice, col_slice] |= valid & positive
        probability_sum[row_slice, col_slice][valid] += (
            result.probabilities[valid]
        )
        source_count[row_slice, col_slice][valid] += 1

    probability = np.full(
        (mosaic_height, mosaic_width),
        np.nan,
        dtype=np.float32,
    )
    observed = source_count > 0
    probability[observed] = (
        probability_sum[observed] / source_count[observed]
    )

    target = np.full(
        (mosaic_height, mosaic_width),
        IGNORE_INDEX,
        dtype=np.uint8,
    )
    target[target_valid & ~target_positive] = 0
    target[target_valid & target_positive] = 1

    rows = [result.row for result in results]
    cell_size = infer_cell_size(rows)

    x_origin_candidates: list[float] = []
    y_origin_candidates: list[float] = []
    if cell_size is not None:
        for row in rows:
            try:
                x_origin_candidates.append(
                    float(row["x_min"])
                    - int(row["col_offset"]) * cell_size
                )
                y_origin_candidates.append(
                    float(row["y_max"])
                    + int(row["row_offset"]) * cell_size
                )
            except (KeyError, TypeError, ValueError):
                continue

    x_origin = (
        float(np.median(x_origin_candidates))
        if x_origin_candidates
        else None
    )
    y_origin_top = (
        float(np.median(y_origin_candidates))
        if y_origin_candidates
        else None
    )

    return TileMosaic(
        tile_name=tile_name,
        min_row=min_row,
        min_col=min_col,
        probability=probability,
        target=target,
        source_count=source_count,
        conflict_mask=conflict_mask,
        row_count=len(rows),
        patch_count=len(results),
        cell_size=cell_size,
        x_origin=x_origin,
        y_origin_top=y_origin_top,
        crs=rows[0].get("crs", ""),
    )


def mosaic_metrics(
    mosaic: TileMosaic,
    threshold: float,
) -> dict[str, Any]:
    counts = ive.confusion_counts(
        np.nan_to_num(mosaic.probability, nan=0.0),
        mosaic.target,
        threshold,
    )
    return {
        "tile_name": mosaic.tile_name,
        "patches": mosaic.patch_count,
        "mosaic_height": mosaic.target.shape[0],
        "mosaic_width": mosaic.target.shape[1],
        "observed_unique_pixels": int((mosaic.target != IGNORE_INDEX).sum()),
        "overlap_pixels": int((mosaic.source_count > 1).sum()),
        "label_conflict_pixels": int(mosaic.conflict_mask.sum()),
        "threshold": threshold,
        **counts,
        **ive.metrics_from_counts(counts),
    }


def save_tile_mosaic_png(
    *,
    path: Path,
    mosaic: TileMosaic,
    metrics: Mapping[str, Any],
    threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    axes[0].imshow(mask_rgb(mosaic.target))
    axes[0].set_title("Ground Truth Mosaic")

    axes[1].imshow(
        prediction_rgb(
            np.nan_to_num(mosaic.probability, nan=0.0),
            mosaic.target,
            threshold,
        )
    )
    axes[1].set_title(f"Predicted Mosaic @ {threshold:.2f}")

    probability_image = axes[2].imshow(
        probability_rgb(
            np.nan_to_num(mosaic.probability, nan=0.0),
            mosaic.target,
        ),
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Mean Predicted Probability")
    fig.colorbar(probability_image, ax=axes[2], fraction=0.046, pad=0.04)

    axes[3].imshow(
        error_rgb(
            np.nan_to_num(mosaic.probability, nan=0.0),
            mosaic.target,
            threshold,
        )
    )
    axes[3].set_title(
        "Difference\nYellow TP | Red FP | Green FN"
    )

    for axis in axes:
        axis.set_xticks([])
        axis.set_yticks([])

    fig.suptitle(
        f"{mosaic.tile_name} — validation-patch mosaic\n"
        f"Dice={float(metrics['dice']):.3f} | "
        f"IoU={float(metrics['iou']):.3f} | "
        f"Precision={float(metrics['precision']):.3f} | "
        f"Recall={float(metrics['recall']):.3f} | "
        f"Patches={int(metrics['patches'])}\n"
        "Gray areas are outside the evaluated validation-patch coverage"
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=190)
    plt.close(fig)


def write_geotiffs(
    *,
    output_dir: Path,
    mosaic: TileMosaic,
    threshold: float,
) -> dict[str, str]:
    if (
        mosaic.cell_size is None
        or mosaic.x_origin is None
        or mosaic.y_origin_top is None
        or not mosaic.crs
    ):
        return {}

    try:
        import rasterio
        from rasterio.transform import from_origin
    except ImportError:
        return {}

    top_left_x = mosaic.x_origin + mosaic.min_col * mosaic.cell_size
    top_left_y = mosaic.y_origin_top - mosaic.min_row * mosaic.cell_size
    transform = from_origin(
        top_left_x,
        top_left_y,
        mosaic.cell_size,
        mosaic.cell_size,
    )
    height, width = mosaic.target.shape
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = sanitize_filename(mosaic.tile_name)
    gt_path = output_dir / f"{stem}__gt.tif"
    pred_path = output_dir / f"{stem}__pred_t{threshold:.2f}.tif"
    probability_path = output_dir / f"{stem}__probability.tif"
    error_path = output_dir / f"{stem}__error_t{threshold:.2f}.tif"

    probability = mosaic.probability.copy()
    valid = mosaic.target != IGNORE_INDEX
    predicted = np.full(
        mosaic.target.shape,
        IGNORE_INDEX,
        dtype=np.uint8,
    )
    predicted[valid] = (
        np.nan_to_num(probability[valid], nan=0.0) >= threshold
    ).astype(np.uint8)

    error = np.full(
        mosaic.target.shape,
        IGNORE_INDEX,
        dtype=np.uint8,
    )
    truth = mosaic.target == 1
    prediction = predicted == 1
    error[valid & ~truth & ~prediction] = 0  # TN
    error[valid & truth & prediction] = 1   # TP
    error[valid & ~truth & prediction] = 2  # FP
    error[valid & truth & ~prediction] = 3  # FN

    common = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": 1,
        "crs": mosaic.crs,
        "transform": transform,
        "compress": "deflate",
        "tiled": True,
    }

    with rasterio.open(
        gt_path,
        "w",
        dtype="uint8",
        nodata=IGNORE_INDEX,
        **common,
    ) as dst:
        dst.write(mosaic.target, 1)

    with rasterio.open(
        pred_path,
        "w",
        dtype="uint8",
        nodata=IGNORE_INDEX,
        **common,
    ) as dst:
        dst.write(predicted, 1)

    probability_to_write = np.where(
        valid,
        np.nan_to_num(probability, nan=-9999.0),
        -9999.0,
    ).astype(np.float32)
    with rasterio.open(
        probability_path,
        "w",
        dtype="float32",
        nodata=-9999.0,
        **common,
    ) as dst:
        dst.write(probability_to_write, 1)

    with rasterio.open(
        error_path,
        "w",
        dtype="uint8",
        nodata=IGNORE_INDEX,
        **common,
    ) as dst:
        dst.write(error, 1)

    return {
        "gt_tif": str(gt_path),
        "pred_tif": str(pred_path),
        "probability_tif": str(probability_path),
        "error_tif": str(error_path),
    }


def save_tile_index(
    output_dir: Path,
    tile_rows: Sequence[Mapping[str, Any]],
) -> None:
    lines = [
        "# Stitched Validation Prediction Maps",
        "",
        (
            "Each PNG compares the ground-truth mosaic with the predicted mosaic "
            "for validation patches from one LiDAR tile."
        ),
        "",
        (
            "These are cropped validation-patch mosaics, not predictions for every "
            "pixel in the original LAZ tile. Gray gaps were not represented by a "
            "selected validation patch."
        ),
        "",
    ]
    for row in tile_rows:
        lines.append(
            f"- `{row['tile_name']}`: Dice `{float(row['dice']):.3f}`, "
            f"precision `{float(row['precision']):.3f}`, "
            f"recall `{float(row['recall']):.3f}`, "
            f"patches `{int(row['patches'])}` — "
            f"`{row.get('comparison_png', '')}`"
        )
    (output_dir / "README_TILE_MAPS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Export direct predicted segmentation maps versus ground-truth masks."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--region", action="append", default=[])
    parser.add_argument("--require-qc", action="store_true")
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--max-ignore-fraction", type=float, default=0.30)
    parser.add_argument("--professor-case-count", type=int, default=5)
    parser.add_argument(
        "--max-mosaic-pixels",
        type=int,
        default=25_000_000,
    )
    parser.add_argument(
        "--skip-all-patch-exports",
        action="store_true",
        help="Only export professor cases and stitched tile maps.",
    )
    parser.add_argument(
        "--skip-geotiff",
        action="store_true",
        help="Do not write georeferenced tile-mosaic GeoTIFFs.",
    )
    args = parser.parse_args()

    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1")
    if not 0 <= args.max_ignore_fraction <= 1:
        parser.error("--max-ignore-fraction must be between 0 and 1")
    if args.professor_case_count < 1:
        parser.error("--professor-case-count must be at least 1")
    if args.max_mosaic_pixels < 1:
        parser.error("--max-mosaic-pixels must be positive")

    dataset_dir = args.dataset_dir.resolve()
    manifest_path = ive.resolve_manifest(dataset_dir, args.manifest)
    checkpoint_path = args.checkpoint.resolve()
    normalization_path = args.normalization.resolve()
    outdir = args.outdir.resolve()
    channels_path = dataset_dir / "channels.json"

    for path, label in (
        (manifest_path, "manifest"),
        (checkpoint_path, "checkpoint"),
        (normalization_path, "normalization"),
        (channels_path, "channels.json"),
    ):
        if not path.is_file():
            parser.error(f"Required {label} does not exist: {path}")

    rows = ive.read_csv(manifest_path)
    selected_rows = ive.filter_rows(
        rows,
        split=args.split,
        regions=args.region,
        require_qc=args.require_qc,
    )
    if not selected_rows:
        parser.error("No manifest rows match the requested filters")

    channels = json.loads(
        channels_path.read_text(encoding="utf-8")
    )["feature_names"]

    print(f"Manifest: {manifest_path}")
    print(f"Rows selected: {len(selected_rows)}")
    print(f"Threshold: {args.threshold:.2f}")

    results, resolved_device = ive.run_inference(
        dataset_dir=dataset_dir,
        rows=selected_rows,
        channels=channels,
        checkpoint_path=checkpoint_path,
        normalization_path=normalization_path,
        device_name=args.device,
        selected_threshold=args.threshold,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    patch_records = [
        compute_patch_record(result, args.threshold)
        for result in results
    ]
    result_by_patch = {
        result.row["patch_id"]: result
        for result in results
    }

    professor_cases, ignore_limit_used = choose_professor_cases(
        patch_records,
        max_ignore_fraction=args.max_ignore_fraction,
        max_cases=args.professor_case_count,
    )
    save_professor_contact_sheet(
        path=outdir / "professor_predicted_vs_gt.png",
        cases=professor_cases,
        result_by_patch=result_by_patch,
        threshold=args.threshold,
    )

    professor_dir = outdir / "professor_cases"
    for record in professor_cases:
        result = result_by_patch[str(record["patch_id"])]
        paths = save_single_patch_outputs(
            output_dir=professor_dir,
            result=result,
            record=record,
            threshold=args.threshold,
        )
        record.update(
            {
                key: str(Path(value).relative_to(outdir))
                for key, value in paths.items()
            }
        )

    if not args.skip_all_patch_exports:
        all_patch_dir = outdir / "all_patch_maps"
        for index, (result, record) in enumerate(
            zip(results, patch_records),
            start=1,
        ):
            save_single_patch_outputs(
                output_dir=all_patch_dir,
                result=result,
                record=record,
                threshold=args.threshold,
            )
            if index % 10 == 0 or index == len(results):
                print(
                    f"  patch-map export: {index}/{len(results)} patches"
                )

    tile_groups: dict[str, list[ive.PatchResult]] = defaultdict(list)
    for result in results:
        tile_groups[result.row["tile_name"]].append(result)

    tile_rows: list[dict[str, Any]] = []
    skipped_tiles: list[dict[str, str]] = []
    tile_dir = outdir / "stitched_tile_maps"
    geotiff_dir = outdir / "stitched_tile_geotiffs"

    for tile_name in sorted(tile_groups, key=str.casefold):
        try:
            mosaic = build_tile_mosaic(
                tile_name,
                tile_groups[tile_name],
                max_mosaic_pixels=args.max_mosaic_pixels,
            )
        except MemoryError as exc:
            skipped_tiles.append(
                {"tile_name": tile_name, "reason": str(exc)}
            )
            print(f"  skipped mosaic: {exc}")
            continue

        metrics = mosaic_metrics(mosaic, args.threshold)
        comparison_path = (
            tile_dir
            / f"{sanitize_filename(tile_name)}__predicted_vs_gt.png"
        )
        save_tile_mosaic_png(
            path=comparison_path,
            mosaic=mosaic,
            metrics=metrics,
            threshold=args.threshold,
        )

        metrics["comparison_png"] = str(
            comparison_path.relative_to(outdir)
        )
        if not args.skip_geotiff:
            geotiff_paths = write_geotiffs(
                output_dir=geotiff_dir,
                mosaic=mosaic,
                threshold=args.threshold,
            )
            metrics.update(
                {
                    key: str(Path(value).relative_to(outdir))
                    for key, value in geotiff_paths.items()
                }
            )
        tile_rows.append(metrics)

    tile_rows.sort(
        key=lambda row: (
            -float(row.get("observed_unique_pixels", 0)),
            str(row.get("tile_name", "")),
        )
    )
    tile_fields = [
        "tile_name",
        "patches",
        "mosaic_height",
        "mosaic_width",
        "observed_unique_pixels",
        "overlap_pixels",
        "label_conflict_pixels",
        "threshold",
        "tp",
        "fp",
        "fn",
        "tn",
        "ignored",
        "total",
        "valid_pixels",
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "ignore_fraction",
        "gt_positive_fraction_valid",
        "predicted_positive_fraction_valid",
        "false_positive_fraction_valid",
        "false_negative_fraction_valid",
        "comparison_png",
        "gt_tif",
        "pred_tif",
        "probability_tif",
        "error_tif",
    ]
    write_csv(
        outdir / "stitched_tile_metrics.csv",
        tile_rows,
        tile_fields,
    )
    if skipped_tiles:
        write_csv(
            outdir / "skipped_tile_mosaics.csv",
            skipped_tiles,
            ["tile_name", "reason"],
        )
    save_tile_index(outdir, tile_rows)

    patch_fields = [
        "case_label",
        "patch_id",
        "region_id",
        "tile_name",
        "split",
        "category",
        "coverage_class",
        "contains_positive_boundary",
        "threshold",
        "tp",
        "fp",
        "fn",
        "tn",
        "ignored",
        "total",
        "valid_pixels",
        "dice",
        "iou",
        "precision",
        "recall",
        "specificity",
        "ignore_fraction",
        "gt_positive_fraction_valid",
        "predicted_positive_fraction_valid",
        "false_positive_fraction_valid",
        "false_negative_fraction_valid",
        "dominant_error",
        "row_offset",
        "col_offset",
        "patch_path",
        "gt_png",
        "pred_png",
        "error_png",
        "comparison_png",
    ]
    write_csv(
        outdir / "professor_cases.csv",
        professor_cases,
        patch_fields,
    )

    overall_counts = ive.aggregate_counts(patch_records)
    overall_metrics = ive.metrics_from_counts(overall_counts)
    summary = {
        "version": 1,
        "dataset_dir": str(dataset_dir),
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "normalization": str(normalization_path),
        "split": args.split,
        "threshold": args.threshold,
        "device": resolved_device,
        "patch_count": len(results),
        "tile_mosaic_count": len(tile_rows),
        "skipped_tile_mosaic_count": len(skipped_tiles),
        "professor_ignore_limit_used": ignore_limit_used,
        "aggregate_patch_evaluation": {
            **overall_counts,
            **overall_metrics,
        },
        "professor_cases": professor_cases,
        "outputs": {
            "professor_contact_sheet": "professor_predicted_vs_gt.png",
            "professor_cases_csv": "professor_cases.csv",
            "stitched_tile_metrics": "stitched_tile_metrics.csv",
            "tile_map_index": "README_TILE_MAPS.md",
            "all_patch_maps": (
                None
                if args.skip_all_patch_exports
                else "all_patch_maps"
            ),
        },
        "interpretation_guardrail": (
            "The outputs use the validation split. The threshold and examples "
            "are validation-selected and are not independent test results."
        ),
    }
    (outdir / "prediction_map_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nPrediction-map export complete.")
    print(
        f"Professor output: {outdir / 'professor_predicted_vs_gt.png'}"
    )
    print(f"Professor cases:  {professor_dir}")
    print(f"Tile mosaics:     {tile_dir}")
    print(
        f"Tile metrics:     {outdir / 'stitched_tile_metrics.csv'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
