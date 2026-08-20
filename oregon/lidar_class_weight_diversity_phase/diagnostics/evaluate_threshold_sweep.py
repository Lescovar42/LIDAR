#!/usr/bin/env python3
"""Independent validation threshold sweep with direct GT-versus-prediction maps.

This tool never changes training, normalization, labels, or the ignore mask. It
loads normalization from the checkpoint and restricts evaluation to one explicit
manifest split and region.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
OREGON_DIR = SCRIPT_DIR.parent
if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))

from train_baseline import EXPECTED_CHANNELS, MiniUNet, validate_channels  # noqa: E402
from phase_common import (  # noqa: E402
    IGNORE_INDEX,
    add_counts,
    choose_best_threshold,
    confusion_from_arrays,
    dataset_fingerprint,
    metrics_from_counts,
    read_csv,
    threshold_values,
    write_csv,
    write_json,
)


def resolve_manifest(dataset_dir: Path, manifest: Path) -> Path:
    return manifest.resolve() if manifest.is_absolute() else (dataset_dir / manifest).resolve()


def infer_region(checkpoint: dict[str, Any], rows: list[dict[str, str]], split: str, requested: str | None) -> str:
    if requested:
        return requested
    training_regions = checkpoint.get("training_regions") or {}
    if len(training_regions) == 1:
        return next(iter(training_regions))
    candidates = sorted({row.get("region_id", "") for row in rows if row.get("split") == split})
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError(f"Could not infer one region for split={split!r}; pass --region. Candidates: {candidates}")


def select_rows(rows: list[dict[str, str]], split: str, region: str) -> list[dict[str, str]]:
    selected = [
        row
        for row in rows
        if row.get("split") == split and str(row.get("region_id", "")).strip() == region
    ]
    if not selected:
        raise ValueError(f"No rows for split={split!r}, region={region!r}")
    roles = sorted({row.get("region_role", "") for row in selected})
    if split == "validation" and roles != ["train_val"]:
        raise ValueError(f"Validation rows must remain source-region train_val rows; found roles={roles}")
    return selected


def load_patch(dataset_dir: Path, row: dict[str, str]) -> tuple[np.ndarray, np.ndarray]:
    path = dataset_dir / Path(row["patch_path"])
    with np.load(path) as data:
        features = data["features"].astype(np.float32)
        mask = data["mask"].astype(np.uint8)
    return features, mask


def predict_rows(
    model: torch.nn.Module,
    dataset_dir: Path,
    rows: list[dict[str, str]],
    mean: np.ndarray,
    std: np.ndarray,
    device: torch.device,
) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    mean_3d = mean[:, None, None]
    std_3d = std[:, None, None]
    model.eval()
    with torch.no_grad():
        for index, row in enumerate(rows, start=1):
            features, mask = load_patch(dataset_dir, row)
            normalized = (features - mean_3d) / std_3d
            tensor = torch.from_numpy(normalized[None]).to(device)
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy().astype(np.float32)
            outputs.append({"row": row, "features": features, "mask": mask, "probability": probability})
            if index % 25 == 0 or index == len(rows):
                print(f"Predicted {index}/{len(rows)} patches")
    return outputs


def sweep_predictions(predictions: list[dict[str, Any]], thresholds: list[float]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for threshold in thresholds:
        aggregate = {key: 0 for key in ("tp", "fp", "fn", "tn", "ignored", "total")}
        for item in predictions:
            add_counts(aggregate, confusion_from_arrays(item["probability"], item["mask"], threshold))
        result.append({"threshold": threshold, **metrics_from_counts(aggregate)})
    return result


def per_patch_metrics(predictions: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in predictions:
        source = item["row"]
        metrics = metrics_from_counts(confusion_from_arrays(item["probability"], item["mask"], threshold))
        rows.append(
            {
                "patch_id": source.get("patch_id", ""),
                "patch_path": source.get("patch_path", ""),
                "tile_name": source.get("tile_name", ""),
                "category": source.get("category", ""),
                "coverage_class": source.get("coverage_class", ""),
                "positive_fraction_manifest": source.get("positive_fraction", ""),
                "ignore_fraction_manifest": source.get("ignore_fraction", ""),
                "mean_slope_degrees": source.get("mean_slope_degrees", ""),
                "distance_to_positive_m": source.get("distance_to_positive_m", ""),
                "is_hard_negative": source.get("is_hard_negative", ""),
                "patch_polygon_keys": source.get("patch_polygon_keys", ""),
                "threshold": threshold,
                **metrics,
            }
        )
    return rows


def error_rgb(mask: np.ndarray, probability: np.ndarray, threshold: float) -> np.ndarray:
    valid = mask != IGNORE_INDEX
    truth = mask == 1
    prediction = probability >= threshold
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[valid & ~truth & ~prediction] = (30, 30, 30)       # TN
    image[valid & truth & prediction] = (255, 255, 255)       # TP
    image[valid & ~truth & prediction] = (230, 60, 60)        # FP
    image[valid & truth & ~prediction] = (60, 120, 230)       # FN
    image[~valid] = (140, 140, 140)                            # unchanged ignore display
    return image


def binary_display(mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    result[valid & mask] = (255, 255, 255)
    result[valid & ~mask] = (0, 0, 0)
    result[~valid] = (140, 140, 140)
    return result


def save_comparison(item: dict[str, Any], threshold: float, output_path: Path, channel_names: list[str]) -> None:
    features = item["features"]
    mask = item["mask"]
    probability = item["probability"]
    valid = mask != IGNORE_INDEX
    prediction = probability >= threshold
    try:
        backdrop_index = channel_names.index("multidirectional_hillshade")
    except ValueError:
        backdrop_index = 0
    backdrop = features[backdrop_index]
    finite = np.isfinite(backdrop)
    if finite.any():
        lower, upper = np.nanpercentile(backdrop[finite], [1, 99])
    else:
        lower, upper = 0.0, 1.0

    counts = confusion_from_arrays(probability, mask, threshold)
    metrics = metrics_from_counts(counts)
    row = item["row"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5))
    axes[0].imshow(backdrop, cmap="gray", vmin=lower, vmax=upper)
    axes[0].set_title("Terrain context")
    axes[1].imshow(binary_display(mask == 1, valid))
    axes[1].set_title("Ground truth")
    axes[2].imshow(binary_display(prediction, valid))
    axes[2].set_title(f"Prediction ≥ {threshold:.2f}")
    axes[3].imshow(error_rgb(mask, probability, threshold))
    axes[3].set_title("Error: FP red, FN blue")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(
        f"{row.get('patch_id', '')}\nDice={metrics['dice']:.3f}  Precision={metrics['precision']:.3f}  "
        f"Recall={metrics['recall']:.3f}  FP={metrics['fp']}  FN={metrics['fn']}",
        fontsize=10,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def select_review_candidates(per_patch: list[dict[str, Any]], count_each: int) -> list[tuple[str, dict[str, Any]]]:
    negatives = [row for row in per_patch if int(row["tp"]) + int(row["fn"]) == 0]
    positives = [row for row in per_patch if int(row["tp"]) + int(row["fn"]) > 0]
    selections: list[tuple[str, dict[str, Any]]] = []
    selections.extend(("negative_high_fp", row) for row in sorted(negatives, key=lambda r: int(r["fp"]), reverse=True)[:count_each])
    selections.extend(("positive_high_fp", row) for row in sorted(positives, key=lambda r: int(r["fp"]), reverse=True)[:count_each])
    selections.extend(("positive_high_fn", row) for row in sorted(positives, key=lambda r: int(r["fn"]), reverse=True)[:count_each])
    selections.extend(("positive_best_dice", row) for row in sorted(positives, key=lambda r: float(r["dice"]), reverse=True)[:count_each])
    seen: set[str] = set()
    unique: list[tuple[str, dict[str, Any]]] = []
    for reason, row in selections:
        patch_id = str(row["patch_id"])
        if patch_id not in seen:
            seen.add(patch_id)
            unique.append((reason, row))
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("patches_boundary_aware.csv"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--region")
    parser.add_argument("--threshold-start", type=float, default=0.30)
    parser.add_argument("--threshold-stop", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--review-count-each", type=int, default=6)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    manifest_path = resolve_manifest(dataset_dir, args.manifest)
    checkpoint_path = args.checkpoint.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if not manifest_path.exists():
        parser.error(f"Manifest not found: {manifest_path}")
    if not checkpoint_path.exists():
        parser.error(f"Checkpoint not found: {checkpoint_path}")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested but CUDA is unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    rows = read_csv(manifest_path)
    try:
        region = infer_region(checkpoint, rows, args.split, args.region)
        selected = select_rows(rows, args.split, region)
    except ValueError as exc:
        parser.error(str(exc))

    channels = list(checkpoint["channels"])
    validate_channels(channels)
    mean = np.asarray(checkpoint["mean"], dtype=np.float32)
    std = np.asarray(checkpoint["std"], dtype=np.float32)
    if len(mean) != len(EXPECTED_CHANNELS) or len(std) != len(EXPECTED_CHANNELS):
        parser.error("Checkpoint normalization does not match seven expected channels")
    if np.any(std <= 0):
        parser.error("Checkpoint contains non-positive normalization standard deviation")

    model = MiniUNet(len(channels)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    predictions = predict_rows(model, dataset_dir, selected, mean, std, device)
    thresholds = threshold_values(args.threshold_start, args.threshold_stop, args.threshold_step)
    sweep = sweep_predictions(predictions, thresholds)
    best = choose_best_threshold(sweep)
    best_threshold = float(best["threshold"])
    patch_rows = per_patch_metrics(predictions, best_threshold)

    write_csv(outdir / "threshold_sweep.csv", sweep)
    write_json(
        outdir / "best_threshold.json",
        {
            "checkpoint": str(checkpoint_path),
            "split": args.split,
            "region": region,
            "selection_metric": "dice",
            "best": best,
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_pos_weight_mode": checkpoint.get("pos_weight_mode", "not_recorded"),
            "checkpoint_pos_weight_used": checkpoint.get("pos_weight_used", "not_recorded"),
            "checkpoint_auto_pos_weight": checkpoint.get("auto_pos_weight", "not_recorded"),
        },
    )
    write_csv(outdir / "per_patch_metrics.csv", patch_rows)
    write_json(outdir / "dataset_fingerprint.json", dataset_fingerprint(manifest_path, rows, args.split))

    by_patch = {item["row"].get("patch_id", ""): item for item in predictions}
    candidates = select_review_candidates(patch_rows, args.review_count_each)
    review_rows: list[dict[str, Any]] = []
    for rank, (reason, row) in enumerate(candidates, start=1):
        patch_id = str(row["patch_id"])
        safe_name = "".join(char if char.isalnum() or char in "-_" else "_" for char in patch_id)
        relative_image = Path("comparison_images") / f"{rank:02d}_{reason}_{safe_name}.png"
        save_comparison(by_patch[patch_id], best_threshold, outdir / relative_image, channels)
        review_rows.append(
            {
                **row,
                "selection_reason": reason,
                "comparison_image": str(relative_image),
                "manual_error_category": "unreviewed",
                "allowed_categories": (
                    "steep_non_landslide_slope;drainage_or_valley_side;ridge_or_convex_terrain;"
                    "rough_natural_terrain;road_or_engineered_cut;forest_management_disturbance;"
                    "boundary_mismatch;incomplete_or_uncertain_inventory_label;other"
                ),
                "review_notes": "",
                "evidence_status": "requires_visual_review",
            }
        )
    write_csv(outdir / "error_review_candidates.csv", review_rows)
    write_json(
        outdir / "evaluation_provenance.json",
        {
            "dataset_dir": str(dataset_dir),
            "manifest": str(manifest_path),
            "checkpoint": str(checkpoint_path),
            "outdir": str(outdir),
            "split": args.split,
            "region": region,
            "thresholds": thresholds,
            "device": str(device),
            "rows": len(selected),
            "ignore_policy_changed": False,
            "normalization_source": "checkpoint_train_only_statistics",
        },
    )
    print(json.dumps({"best": best, "output": str(outdir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
