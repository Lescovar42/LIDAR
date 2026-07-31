#!/usr/bin/env python3
"""Inspect segmentation errors with representative, ignore-aware comparisons.

This diagnostic does not rebuild data or retrain the model. It:

- evaluates every selected patch over a threshold sweep;
- uses the exact training normalization and checkpoint;
- avoids selecting misleading examples dominated by ignore pixels;
- selects representative good, median, false-positive, false-negative, and
  true-boundary cases;
- produces professor-friendly ground-truth/prediction comparisons;
- aggregates errors by tile, coverage class, and true in-patch boundary status;
- summarizes raw terrain-channel signatures for TP/FP/FN/TN pixels.

Run from the ``oregon`` directory, for example:

    python diagnostics/inspect_visual_errors.py ^
      --dataset-dir dataset_tillamook_probe_15m ^
      --manifest patches_boundary_aware.csv ^
      --checkpoint training_output_tillamook_15m_boundary/best_model.pt ^
      --normalization training_output_tillamook_15m_boundary/normalization.json ^
      --outdir evaluation_tillamook_15m_boundary_visual_errors ^
      --split validation ^
      --selected-threshold 0.65 ^
      --comparison-threshold 0.50 ^
      --comparison-threshold 0.60 ^
      --comparison-threshold 0.65 ^
      --device auto
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

IGNORE_INDEX = 255
ACCEPTED_QC = {"accept", "accept_approximate_boundary"}
COUNT_KEYS = ("tp", "fp", "fn", "tn", "ignored", "total")


@dataclass
class PatchResult:
    index: int
    row: dict[str, str]
    target: np.ndarray
    probabilities: np.ndarray
    raw_features: np.ndarray
    selected_counts: dict[str, int]
    selected_metrics: dict[str, float]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any, default: float = 0.0) -> float:
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError):
        return default
    return parsed if math.isfinite(parsed) else default


def safe_bool(value: Any) -> bool:
    return str(value or "").strip().casefold() in {"1", "true", "yes", "y"}


def ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def confusion_counts(
    probabilities: np.ndarray,
    target: np.ndarray,
    threshold: float,
) -> dict[str, int]:
    target_2d = np.squeeze(target)
    probabilities_2d = np.squeeze(probabilities)
    valid = target_2d != IGNORE_INDEX
    truth = target_2d == 1
    prediction = probabilities_2d >= threshold
    return {
        "tp": int((valid & prediction & truth).sum()),
        "fp": int((valid & prediction & ~truth).sum()),
        "fn": int((valid & ~prediction & truth).sum()),
        "tn": int((valid & ~prediction & ~truth).sum()),
        "ignored": int((~valid).sum()),
        "total": int(target_2d.size),
    }


def metrics_from_counts(counts: Mapping[str, int]) -> dict[str, float]:
    tp = int(counts.get("tp", 0))
    fp = int(counts.get("fp", 0))
    fn = int(counts.get("fn", 0))
    tn = int(counts.get("tn", 0))
    ignored = int(counts.get("ignored", 0))
    total = int(counts.get("total", tp + fp + fn + tn + ignored))
    valid = tp + fp + fn + tn
    return {
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ignore_fraction": ratio(ignored, total),
        "gt_positive_fraction_valid": ratio(tp + fn, valid),
        "predicted_positive_fraction_valid": ratio(tp + fp, valid),
        "false_positive_fraction_valid": ratio(fp, valid),
        "false_negative_fraction_valid": ratio(fn, valid),
        "valid_pixels": float(valid),
    }


def aggregate_counts(records: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    result = {key: 0 for key in COUNT_KEYS}
    for record in records:
        counts = record.get("counts", record)
        for key in COUNT_KEYS:
            result[key] += int(counts.get(key, 0))
    return result


def resolve_manifest(dataset_dir: Path, manifest_arg: Path | None) -> Path:
    if manifest_arg is not None:
        if manifest_arg.is_absolute():
            return manifest_arg.resolve()
        inside_dataset = (dataset_dir / manifest_arg).resolve()
        if inside_dataset.exists():
            return inside_dataset
        return manifest_arg.resolve()

    for name in ("patches_boundary_aware.csv", "patches_qc.csv", "patches.csv"):
        candidate = dataset_dir / name
        if candidate.exists():
            return candidate.resolve()
    return (dataset_dir / "patches.csv").resolve()


def filter_rows(
    rows: list[dict[str, str]],
    *,
    split: str,
    regions: Sequence[str],
    require_qc: bool,
) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("split", "").strip() == split]
    if regions:
        wanted = {region.strip().casefold() for region in regions}
        selected = [
            row
            for row in selected
            if row.get("region_id", "").strip().casefold() in wanted
        ]
    if require_qc:
        selected = [
            row
            for row in selected
            if row.get("qc_status", "").strip().casefold() in ACCEPTED_QC
        ]
    return selected


def load_normalization(path: Path, channels: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    mean = np.asarray([payload[name]["mean"] for name in channels], dtype=np.float32)
    std = np.asarray([payload[name]["std"] for name in channels], dtype=np.float32)
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("normalization.json contains invalid mean/std values")
    return mean, std


def _import_training_components(oregon_dir: Path):
    if str(oregon_dir) not in sys.path:
        sys.path.insert(0, str(oregon_dir))
    import torch  # type: ignore
    from train_baseline import EXPECTED_CHANNELS, MiniUNet  # type: ignore

    return torch, EXPECTED_CHANNELS, MiniUNet


def run_inference(
    *,
    dataset_dir: Path,
    rows: list[dict[str, str]],
    channels: list[str],
    checkpoint_path: Path,
    normalization_path: Path,
    device_name: str,
    selected_threshold: float,
) -> tuple[list[PatchResult], str]:
    oregon_dir = Path(__file__).resolve().parents[1]
    torch, expected_channels, mini_unet = _import_training_components(oregon_dir)

    if tuple(channels) != tuple(expected_channels):
        raise ValueError(
            "Dataset channels do not match the canonical training channels: "
            + ", ".join(expected_channels)
        )

    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)

    mean, std = load_normalization(normalization_path, channels)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    checkpoint_channels = checkpoint.get("channels")
    if checkpoint_channels != channels:
        raise ValueError("Checkpoint channels do not exactly match dataset channels")

    checkpoint_mean = np.asarray(checkpoint.get("mean"), dtype=np.float32)
    checkpoint_std = np.asarray(checkpoint.get("std"), dtype=np.float32)
    if checkpoint_mean.shape != mean.shape or checkpoint_std.shape != std.shape:
        raise ValueError("Checkpoint normalization fingerprint has the wrong shape")
    if not np.allclose(checkpoint_mean, mean, rtol=1e-6, atol=1e-7):
        raise ValueError("Checkpoint mean does not match normalization.json")
    if not np.allclose(checkpoint_std, std, rtol=1e-6, atol=1e-7):
        raise ValueError("Checkpoint std does not match normalization.json")

    model = mini_unet(len(channels)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean_3d = mean[:, None, None]
    std_3d = std[:, None, None]
    results: list[PatchResult] = []

    with torch.inference_mode():
        for index, row in enumerate(rows):
            patch_path = dataset_dir / row["patch_path"]
            with np.load(patch_path) as data:
                raw_features = data["features"].astype(np.float32)
                target = data["mask"].astype(np.uint8)

            normalized = (raw_features - mean_3d) / std_3d
            tensor = torch.from_numpy(normalized).unsqueeze(0).to(device)
            logits = model(tensor)
            probabilities = torch.sigmoid(logits).cpu().numpy()[0, 0]

            counts = confusion_counts(probabilities, target, selected_threshold)
            results.append(
                PatchResult(
                    index=index,
                    row=row,
                    target=target,
                    probabilities=probabilities,
                    raw_features=raw_features,
                    selected_counts=counts,
                    selected_metrics=metrics_from_counts(counts),
                )
            )

            if (index + 1) % 10 == 0 or index + 1 == len(rows):
                print(f"  inference: {index + 1}/{len(rows)} patches")

    return results, device_name


def record_for_threshold(result: PatchResult, threshold: float) -> dict[str, Any]:
    counts = confusion_counts(result.probabilities, result.target, threshold)
    metrics = metrics_from_counts(counts)
    row = result.row
    return {
        "index": result.index,
        "patch_id": row.get("patch_id", f"patch_{result.index:06d}"),
        "region_id": row.get("region_id", ""),
        "tile_name": row.get("tile_name", ""),
        "split": row.get("split", ""),
        "category": row.get("category", ""),
        "coverage_class": row.get("coverage_class", ""),
        "contains_positive_boundary": row.get("contains_positive_boundary", ""),
        "threshold": threshold,
        "counts": counts,
        **counts,
        **metrics,
    }


def aggregate_threshold_metrics(
    results: Sequence[PatchResult],
    thresholds: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        patch_records = [record_for_threshold(result, threshold) for result in results]
        counts = aggregate_counts(patch_records)
        rows.append({"threshold": threshold, **counts, **metrics_from_counts(counts)})
    return rows


def dominant_error(record: Mapping[str, Any]) -> str:
    fp = int(record.get("fp", 0))
    fn = int(record.get("fn", 0))
    tp = int(record.get("tp", 0))
    truth_support = tp + fn
    if fp == 0 and fn == 0:
        return "near_perfect"
    if truth_support == 0:
        return "false_positive_on_negative"
    if fp >= 2 * max(1, fn):
        return "false_positive_dominant"
    if fn >= 2 * max(1, fp):
        return "false_negative_dominant"
    return "mixed_error"


def _unique_by_patch(records: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: set[str] = set()
    output: list[Mapping[str, Any]] = []
    for record in records:
        patch_id = str(record.get("patch_id", ""))
        if patch_id and patch_id not in seen:
            seen.add(patch_id)
            output.append(record)
    return output


def _choose_closest(
    records: Sequence[Mapping[str, Any]],
    field: str,
    target: float,
    *,
    used_patches: set[str],
    used_tiles: set[str],
) -> Mapping[str, Any] | None:
    candidates = [record for record in records if str(record.get("patch_id", "")) not in used_patches]
    if not candidates:
        return None
    candidates.sort(
        key=lambda record: (
            str(record.get("tile_name", "")) in used_tiles,
            abs(float(record.get(field, 0.0)) - target),
            float(record.get("ignore_fraction", 0.0)),
            str(record.get("patch_id", "")),
        )
    )
    return candidates[0]


def select_representative_records(
    records: Sequence[Mapping[str, Any]],
    *,
    max_ignore_fraction: float,
) -> tuple[list[dict[str, Any]], float]:
    """Select representative records while avoiding ignore-dominated patches.

    Returns selected records and the ignore-fraction limit actually used.
    """

    limits = [max_ignore_fraction, max(0.50, max_ignore_fraction), 0.75, 1.0]
    eligible: list[Mapping[str, Any]] = []
    used_limit = 1.0
    for limit in limits:
        eligible = [
            record
            for record in records
            if float(record.get("ignore_fraction", 1.0)) <= limit
            and int(record.get("valid_pixels", 0)) > 0
        ]
        positive = [
            record
            for record in eligible
            if float(record.get("gt_positive_fraction_valid", 0.0)) >= 0.01
        ]
        if len(eligible) >= 4 and len(positive) >= 3:
            used_limit = limit
            break

    positive = [
        record
        for record in eligible
        if float(record.get("gt_positive_fraction_valid", 0.0)) >= 0.01
    ]
    negatives = [
        record
        for record in eligible
        if float(record.get("gt_positive_fraction_valid", 0.0)) < 0.005
    ]
    boundaries = [
        record
        for record in positive
        if safe_bool(record.get("contains_positive_boundary"))
    ]

    used_patches: set[str] = set()
    used_tiles: set[str] = set()
    selections: list[dict[str, Any]] = []

    def add(label: str, record: Mapping[str, Any] | None) -> None:
        if record is None:
            return
        patch_id = str(record.get("patch_id", ""))
        if patch_id in used_patches:
            return
        used_patches.add(patch_id)
        used_tiles.add(str(record.get("tile_name", "")))
        selections.append({"case_label": label, **dict(record)})

    if positive:
        dice_values = np.asarray([float(record.get("dice", 0.0)) for record in positive])
        q75 = float(np.quantile(dice_values, 0.75))
        median = float(np.median(dice_values))
        add(
            "representative_good",
            _choose_closest(
                positive, "dice", q75, used_patches=used_patches, used_tiles=used_tiles
            ),
        )
        add(
            "representative_median",
            _choose_closest(
                positive, "dice", median, used_patches=used_patches, used_tiles=used_tiles
            ),
        )

    if boundaries:
        boundary_median = float(
            np.median([float(record.get("dice", 0.0)) for record in boundaries])
        )
        add(
            "true_boundary_typical",
            _choose_closest(
                boundaries,
                "dice",
                boundary_median,
                used_patches=used_patches,
                used_tiles=used_tiles,
            ),
        )

    fp_candidates = negatives if negatives else eligible
    fp_candidates = sorted(
        fp_candidates,
        key=lambda record: (
            -float(record.get("false_positive_fraction_valid", 0.0)),
            float(record.get("ignore_fraction", 0.0)),
            str(record.get("patch_id", "")),
        ),
    )
    add(
        "false_positive_dominant",
        _choose_closest(
            fp_candidates,
            "false_positive_fraction_valid",
            float(fp_candidates[0].get("false_positive_fraction_valid", 0.0))
            if fp_candidates
            else 0.0,
            used_patches=used_patches,
            used_tiles=used_tiles,
        ),
    )

    fn_candidates = sorted(
        positive,
        key=lambda record: (
            -float(record.get("false_negative_fraction_valid", 0.0)),
            float(record.get("ignore_fraction", 0.0)),
            str(record.get("patch_id", "")),
        ),
    )
    add(
        "false_negative_dominant",
        _choose_closest(
            fn_candidates,
            "false_negative_fraction_valid",
            float(fn_candidates[0].get("false_negative_fraction_valid", 0.0))
            if fn_candidates
            else 0.0,
            used_patches=used_patches,
            used_tiles=used_tiles,
        ),
    )

    return selections, used_limit


def _normalize_background(feature: np.ndarray) -> np.ndarray:
    finite = np.isfinite(feature)
    if not finite.any():
        return np.zeros(feature.shape, dtype=np.float32)
    low, high = np.percentile(feature[finite], (2, 98))
    if high <= low:
        return np.zeros(feature.shape, dtype=np.float32)
    return np.nan_to_num(np.clip((feature - low) / (high - low), 0.0, 1.0))


def _rgba_mask(mask: np.ndarray, rgb: tuple[float, float, float], alpha: float) -> np.ndarray:
    rgba = np.zeros((*mask.shape, 4), dtype=np.float32)
    rgba[..., 0] = rgb[0]
    rgba[..., 1] = rgb[1]
    rgba[..., 2] = rgb[2]
    rgba[..., 3] = mask.astype(np.float32) * alpha
    return rgba


def _draw_hillshade(ax: Any, hillshade: np.ndarray) -> None:
    ax.imshow(_normalize_background(hillshade), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_ignore(ax: Any, target: np.ndarray, *, alpha: float = 0.10) -> None:
    ignored = target == IGNORE_INDEX
    if ignored.any():
        ax.imshow(_rgba_mask(ignored, (0.0, 0.35, 1.0), alpha))
        try:
            ax.contour(
                ignored.astype(np.uint8),
                levels=[0.5],
                colors=["#1769ff"],
                linewidths=0.55,
                linestyles="dashed",
            )
        except ValueError:
            pass


def _draw_truth(ax: Any, hillshade: np.ndarray, target: np.ndarray) -> None:
    _draw_hillshade(ax, hillshade)
    truth = target == 1
    ax.imshow(_rgba_mask(truth, (0.0, 1.0, 0.0), 0.34))
    if truth.any():
        try:
            ax.contour(
                truth.astype(np.uint8),
                levels=[0.5],
                colors=["#00ff00"],
                linewidths=1.2,
            )
        except ValueError:
            pass
    _draw_ignore(ax, target)


def _draw_prediction(
    ax: Any,
    hillshade: np.ndarray,
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    _draw_hillshade(ax, hillshade)
    prediction = probabilities >= threshold
    ax.imshow(_rgba_mask(prediction, (1.0, 0.0, 0.0), 0.34))
    if prediction.any():
        try:
            ax.contour(
                prediction.astype(np.uint8),
                levels=[0.5],
                colors=["#ff2b2b"],
                linewidths=1.0,
            )
        except ValueError:
            pass
    _draw_ignore(ax, target)


def _draw_error(
    ax: Any,
    hillshade: np.ndarray,
    target: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    _draw_hillshade(ax, hillshade)
    valid = target != IGNORE_INDEX
    truth = target == 1
    prediction = probabilities >= threshold
    tp = valid & truth & prediction
    fp = valid & ~truth & prediction
    fn = valid & truth & ~prediction
    ax.imshow(_rgba_mask(tp, (1.0, 1.0, 0.0), 0.78))
    ax.imshow(_rgba_mask(fp, (1.0, 0.0, 0.0), 0.64))
    ax.imshow(_rgba_mask(fn, (0.0, 1.0, 0.0), 0.68))
    _draw_ignore(ax, target, alpha=0.08)


def load_naip_rgb(dataset_dir: Path, patch_id: str) -> np.ndarray | None:
    manifest_path = dataset_dir / "naip" / "naip_manifest.csv"
    if not manifest_path.exists():
        return None
    rows = read_csv(manifest_path)
    record = next(
        (
            row
            for row in rows
            if row.get("patch_id") == patch_id
            and row.get("status", "").strip().casefold() in {"ok", "cached"}
            and row.get("naip_path", "").strip()
        ),
        None,
    )
    if record is None:
        return None
    path = dataset_dir / "naip" / record["naip_path"]
    if not path.exists():
        return None
    try:
        with np.load(path) as data:
            bands = data["bands"].astype(np.float32)
            valid = data["valid_mask"].astype(bool)
        if bands.shape[0] < 3:
            return None
        rgb = np.transpose(bands[:3], (1, 2, 0)) / 255.0
        rgb[~valid] = 0.0
        return np.clip(rgb, 0.0, 1.0)
    except Exception:
        return None


def make_detailed_figure(
    *,
    output_path: Path,
    result: PatchResult,
    record: Mapping[str, Any],
    channels: Sequence[str],
    comparison_thresholds: Sequence[float],
    selected_threshold: float,
    dataset_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    hillshade_index = channels.index("multidirectional_hillshade")
    hillshade = result.raw_features[hillshade_index]
    target = result.target
    probabilities = result.probabilities
    patch_id = str(record["patch_id"])
    naip_rgb = load_naip_rgb(dataset_dir, patch_id)

    thresholds = list(comparison_thresholds)[:3]
    while len(thresholds) < 3:
        thresholds.append(selected_threshold)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    _draw_hillshade(axes[0, 0], hillshade)
    axes[0, 0].set_title("LiDAR hillshade")

    _draw_truth(axes[0, 1], hillshade, target)
    axes[0, 1].set_title("Ground truth\nGreen=landslide; blue dashed=ignore")

    masked_probability = np.ma.masked_where(target == IGNORE_INDEX, probabilities)
    _draw_hillshade(axes[0, 2], hillshade)
    image = axes[0, 2].imshow(masked_probability, cmap="viridis", vmin=0, vmax=1, alpha=0.78)
    axes[0, 2].set_title("Predicted probability")
    axes[0, 2].set_xticks([])
    axes[0, 2].set_yticks([])
    fig.colorbar(image, ax=axes[0, 2], fraction=0.046, pad=0.04)

    if naip_rgb is not None:
        axes[0, 3].imshow(naip_rgb)
        axes[0, 3].set_title("Cached NAIP context")
        axes[0, 3].set_xticks([])
        axes[0, 3].set_yticks([])
    else:
        axes[0, 3].axis("off")
        text = (
            f"Case: {record.get('case_label', '')}\n"
            f"Dice @ {selected_threshold:.2f}: {float(record.get('dice', 0)):.3f}\n"
            f"Precision: {float(record.get('precision', 0)):.3f}\n"
            f"Recall: {float(record.get('recall', 0)):.3f}\n"
            f"GT positive: {100*float(record.get('gt_positive_fraction_valid', 0)):.1f}%\n"
            f"Pred positive: {100*float(record.get('predicted_positive_fraction_valid', 0)):.1f}%\n"
            f"Ignore: {100*float(record.get('ignore_fraction', 0)):.1f}%"
        )
        axes[0, 3].text(0.03, 0.97, text, va="top", ha="left", fontsize=11)

    for column, threshold in enumerate(thresholds):
        _draw_prediction(
            axes[1, column], hillshade, target, probabilities, threshold
        )
        axes[1, column].set_title(f"Prediction threshold {threshold:.2f}")

    _draw_error(
        axes[1, 3], hillshade, target, probabilities, selected_threshold
    )
    axes[1, 3].set_title(
        f"Error @ {selected_threshold:.2f}\n"
        "Yellow=TP, red=FP, green=FN, blue dashed=ignore"
    )

    fig.suptitle(
        f"{record.get('case_label', '')}: {patch_id}\n"
        f"Tile: {record.get('tile_name', '')}",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def make_professor_overview(
    *,
    output_path: Path,
    selected: Sequence[Mapping[str, Any]],
    result_by_patch: Mapping[str, PatchResult],
    channels: Sequence[str],
    selected_threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    overview_cases = [
        record
        for label in (
            "representative_good",
            "false_positive_dominant",
            "false_negative_dominant",
        )
        for record in selected
        if record.get("case_label") == label
    ]
    if not overview_cases:
        return

    hillshade_index = channels.index("multidirectional_hillshade")
    fig, axes = plt.subplots(
        len(overview_cases),
        5,
        figsize=(17, 4.4 * len(overview_cases)),
        squeeze=False,
    )

    for row_index, record in enumerate(overview_cases):
        result = result_by_patch[str(record["patch_id"])]
        hillshade = result.raw_features[hillshade_index]
        target = result.target
        probabilities = result.probabilities

        _draw_hillshade(axes[row_index, 0], hillshade)
        _draw_truth(axes[row_index, 1], hillshade, target)
        _draw_prediction(axes[row_index, 2], hillshade, target, probabilities, 0.50)
        _draw_prediction(
            axes[row_index, 3],
            hillshade,
            target,
            probabilities,
            selected_threshold,
        )
        _draw_error(
            axes[row_index, 4],
            hillshade,
            target,
            probabilities,
            selected_threshold,
        )

        if row_index == 0:
            axes[row_index, 0].set_title("LiDAR hillshade")
            axes[row_index, 1].set_title("Ground truth")
            axes[row_index, 2].set_title("Prediction @ 0.50")
            axes[row_index, 3].set_title(f"Prediction @ {selected_threshold:.2f}")
            axes[row_index, 4].set_title(
                "Error map\nYellow TP | red FP | green FN"
            )

        axes[row_index, 0].set_ylabel(
            str(record.get("case_label", "")).replace("_", " ")
            + "\n"
            + f"Dice={float(record.get('dice', 0)):.3f}",
            fontsize=10,
        )

    fig.suptitle(
        "Representative validation error comparison\n"
        "Ignore pixels are shown only as faint blue dashed boundaries",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_threshold_plot(
    output_path: Path,
    threshold_rows: Sequence[Mapping[str, Any]],
    selected_threshold: float,
) -> None:
    import matplotlib.pyplot as plt

    thresholds = [float(row["threshold"]) for row in threshold_rows]
    fig, ax = plt.subplots(figsize=(9, 6))
    for metric in ("dice", "iou", "precision", "recall", "specificity"):
        ax.plot(
            thresholds,
            [float(row[metric]) for row in threshold_rows],
            marker="o",
            label=metric.capitalize(),
        )
    ax.axvline(selected_threshold, linestyle="--", label=f"Selected={selected_threshold:.2f}")
    ax.set_xlabel("Prediction threshold")
    ax.set_ylabel("Metric value")
    ax.set_ylim(0, 1)
    ax.set_title("Validation metrics across prediction thresholds")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def group_metric_rows(
    records: Sequence[Mapping[str, Any]],
    field: str,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        value = str(record.get(field, "")).strip() or "<missing>"
        grouped[value].append(record)
    output: list[dict[str, Any]] = []
    for value in sorted(grouped, key=str.casefold):
        members = grouped[value]
        counts = aggregate_counts(members)
        output.append(
            {
                field: value,
                "patches": len(members),
                **counts,
                **metrics_from_counts(counts),
            }
        )
    return output


def tile_metric_rows(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows = group_metric_rows(records, "tile_name")
    return sorted(
        rows,
        key=lambda row: (
            -float(row.get("false_positive_fraction_valid", 0.0)),
            str(row.get("tile_name", "")),
        ),
    )


def feature_signature_rows(
    results: Sequence[PatchResult],
    channels: Sequence[str],
    selected_threshold: float,
) -> list[dict[str, Any]]:
    accumulators: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "sum": 0.0, "sum_sq": 0.0}
    )
    for result in results:
        target = result.target
        valid = target != IGNORE_INDEX
        truth = target == 1
        prediction = result.probabilities >= selected_threshold
        masks = {
            "tp": valid & truth & prediction,
            "fp": valid & ~truth & prediction,
            "fn": valid & truth & ~prediction,
            "tn": valid & ~truth & ~prediction,
        }
        for channel_index, channel in enumerate(channels):
            values = result.raw_features[channel_index]
            for error_class, mask in masks.items():
                if not mask.any():
                    continue
                selected_values = values[mask].astype(np.float64)
                acc = accumulators[(channel, error_class)]
                acc["count"] += float(selected_values.size)
                acc["sum"] += float(selected_values.sum())
                acc["sum_sq"] += float(np.square(selected_values).sum())

    rows: list[dict[str, Any]] = []
    for channel in channels:
        for error_class in ("tp", "fp", "fn", "tn"):
            acc = accumulators[(channel, error_class)]
            count = int(acc["count"])
            mean = acc["sum"] / count if count else 0.0
            variance = max(0.0, acc["sum_sq"] / count - mean * mean) if count else 0.0
            rows.append(
                {
                    "channel": channel,
                    "error_class": error_class,
                    "pixel_count": count,
                    "mean": mean,
                    "std": math.sqrt(variance),
                }
            )
    return rows


def write_markdown_summary(
    *,
    path: Path,
    manifest_path: Path,
    checkpoint_path: Path,
    split: str,
    selected_threshold: float,
    threshold_rows: Sequence[Mapping[str, Any]],
    selected_cases: Sequence[Mapping[str, Any]],
    ignore_limit_used: float,
    tile_rows: Sequence[Mapping[str, Any]],
) -> None:
    selected_row = min(
        threshold_rows,
        key=lambda row: abs(float(row["threshold"]) - selected_threshold),
    )
    default_row = min(
        threshold_rows,
        key=lambda row: abs(float(row["threshold"]) - 0.50),
    )
    fp = int(selected_row["fp"])
    fn = int(selected_row["fn"])
    precision = float(selected_row["precision"])
    recall = float(selected_row["recall"])

    if fp >= 2 * max(1, fn):
        diagnosis = "False-positive/over-segmentation is the dominant aggregate error."
    elif fn >= 2 * max(1, fp):
        diagnosis = "False-negative/under-segmentation is the dominant aggregate error."
    else:
        diagnosis = "False-positive and false-negative errors are both material."

    lines = [
        "# Visual Error Inspection Summary",
        "",
        "## Run identity",
        "",
        f"- Manifest: `{manifest_path}`",
        f"- Checkpoint: `{checkpoint_path}`",
        f"- Split: `{split}`",
        f"- Selected operating threshold: `{selected_threshold:.2f}`",
        "",
        "## Aggregate result",
        "",
        f"- Dice: **{float(selected_row['dice']):.4f}**",
        f"- IoU: **{float(selected_row['iou']):.4f}**",
        f"- Precision: **{precision:.4f}**",
        f"- Recall: **{recall:.4f}**",
        f"- Specificity: **{float(selected_row['specificity']):.4f}**",
        f"- TP / FP / FN / TN: `{int(selected_row['tp'])} / {fp} / {fn} / {int(selected_row['tn'])}`",
        f"- GT positive fraction among valid pixels: **{100*float(selected_row['gt_positive_fraction_valid']):.2f}%**",
        f"- Predicted positive fraction among valid pixels: **{100*float(selected_row['predicted_positive_fraction_valid']):.2f}%**",
        "",
        f"**Diagnostic:** {diagnosis}",
        "",
        "## Threshold comparison",
        "",
        f"- Dice at 0.50: `{float(default_row['dice']):.4f}`",
        f"- Dice at {selected_threshold:.2f}: `{float(selected_row['dice']):.4f}`",
        f"- FP at 0.50: `{int(default_row['fp'])}`",
        f"- FP at {selected_threshold:.2f}: `{fp}`",
        "",
        "## Representative selection",
        "",
        (
            "Examples were selected from patches with ignore fraction at or below "
            f"`{ignore_limit_used:.2f}` whenever the validation set allowed it."
        ),
    ]
    for case in selected_cases:
        lines.append(
            f"- `{case.get('case_label')}`: `{case.get('patch_id')}` "
            f"(tile `{case.get('tile_name')}`, Dice `{float(case.get('dice', 0)):.3f}`, "
            f"ignore `{100*float(case.get('ignore_fraction', 0)):.1f}%`)"
        )

    lines.extend(
        [
            "",
            "## Highest false-positive tiles",
            "",
        ]
    )
    for row in list(tile_rows)[:5]:
        lines.append(
            f"- `{row.get('tile_name')}`: FP fraction "
            f"`{100*float(row.get('false_positive_fraction_valid', 0)):.2f}%`, "
            f"Dice `{float(row.get('dice', 0)):.3f}`, patches `{int(row.get('patches', 0))}`"
        )

    lines.extend(
        [
            "",
            "## Interpretation guardrail",
            "",
            (
                "These are validation diagnostics. The threshold and representative cases "
                "were selected using the validation split and are not independent test results."
            ),
            (
                "The visual figures keep ignored pixels faint and outlined rather than painting "
                "them solid blue, so model errors remain visible."
            ),
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create ignore-aware representative visual error comparisons."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Default: patches_boundary_aware.csv, then patches_qc.csv, then patches.csv. "
            "Relative paths are first resolved inside --dataset-dir."
        ),
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--normalization", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--region", action="append", default=[])
    parser.add_argument("--require-qc", action="store_true")
    parser.add_argument(
        "--selected-threshold",
        type=float,
        default=0.65,
        help="Operating threshold used for representative error analysis.",
    )
    parser.add_argument(
        "--comparison-threshold",
        action="append",
        type=float,
        default=[],
        help="Threshold shown in detailed figures; repeat up to three times.",
    )
    parser.add_argument("--sweep-start", type=float, default=0.30)
    parser.add_argument("--sweep-stop", type=float, default=0.90)
    parser.add_argument("--sweep-step", type=float, default=0.05)
    parser.add_argument("--max-ignore-fraction", type=float, default=0.30)
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    if not 0 < args.selected_threshold < 1:
        parser.error("--selected-threshold must be between 0 and 1")
    if args.sweep_step <= 0 or args.sweep_start <= 0 or args.sweep_stop >= 1:
        parser.error("Threshold sweep must stay within (0, 1) with positive step")
    if args.sweep_stop < args.sweep_start:
        parser.error("--sweep-stop must be >= --sweep-start")
    if not 0 <= args.max_ignore_fraction <= 1:
        parser.error("--max-ignore-fraction must be between 0 and 1")

    dataset_dir = args.dataset_dir.resolve()
    manifest_path = resolve_manifest(dataset_dir, args.manifest)
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

    rows = read_csv(manifest_path)
    selected_rows = filter_rows(
        rows,
        split=args.split,
        regions=args.region,
        require_qc=args.require_qc,
    )
    if not selected_rows:
        parser.error("No rows match the requested split/region/QC filters")

    channels = json.loads(channels_path.read_text(encoding="utf-8"))["feature_names"]
    comparison_thresholds = args.comparison_threshold or [0.50, 0.60, args.selected_threshold]
    comparison_thresholds = list(dict.fromkeys(round(value, 6) for value in comparison_thresholds))
    for threshold in comparison_thresholds:
        if not 0 < threshold < 1:
            parser.error("--comparison-threshold values must be between 0 and 1")

    thresholds = [
        round(float(value), 6)
        for value in np.arange(
            args.sweep_start,
            args.sweep_stop + args.sweep_step / 2,
            args.sweep_step,
        )
        if 0 < value < 1
    ]
    thresholds = sorted(set(thresholds + [round(args.selected_threshold, 6), 0.50]))

    print(f"Manifest: {manifest_path}")
    print(f"Rows selected: {len(selected_rows)}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Selected threshold: {args.selected_threshold:.2f}")

    results, resolved_device = run_inference(
        dataset_dir=dataset_dir,
        rows=selected_rows,
        channels=channels,
        checkpoint_path=checkpoint_path,
        normalization_path=normalization_path,
        device_name=args.device,
        selected_threshold=args.selected_threshold,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    threshold_rows = aggregate_threshold_metrics(results, thresholds)
    threshold_fields = [
        "threshold",
        *COUNT_KEYS,
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
    ]
    write_csv(outdir / "threshold_metrics.csv", threshold_rows, threshold_fields)
    make_threshold_plot(
        outdir / "threshold_metric_comparison.png",
        threshold_rows,
        args.selected_threshold,
    )

    selected_records = [
        record_for_threshold(result, args.selected_threshold) for result in results
    ]
    for record in selected_records:
        record["dominant_error"] = dominant_error(record)
        record["sampling_reason"] = selected_rows[int(record["index"])].get(
            "sampling_reason", ""
        )
        record["patch_path"] = selected_rows[int(record["index"])].get("patch_path", "")
        record["qc_status"] = selected_rows[int(record["index"])].get("qc_status", "")

    patch_fields = [
        "index",
        "patch_id",
        "region_id",
        "tile_name",
        "split",
        "category",
        "coverage_class",
        "contains_positive_boundary",
        "sampling_reason",
        "qc_status",
        "threshold",
        *COUNT_KEYS,
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
        "patch_path",
    ]
    write_csv(outdir / "patch_metrics_selected_threshold.csv", selected_records, patch_fields)

    all_threshold_patch_rows: list[dict[str, Any]] = []
    for result in results:
        for threshold in thresholds:
            row = record_for_threshold(result, threshold)
            row.pop("counts", None)
            row["dominant_error"] = dominant_error(row)
            all_threshold_patch_rows.append(row)
    write_csv(
        outdir / "patch_metrics_all_thresholds.csv",
        all_threshold_patch_rows,
        [
            "index",
            "patch_id",
            "region_id",
            "tile_name",
            "split",
            "category",
            "coverage_class",
            "contains_positive_boundary",
            "threshold",
            *COUNT_KEYS,
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
        ],
    )

    coverage_rows = group_metric_rows(selected_records, "coverage_class")
    boundary_rows = group_metric_rows(selected_records, "contains_positive_boundary")
    error_rows = group_metric_rows(selected_records, "dominant_error")
    tile_rows = tile_metric_rows(selected_records)
    group_fields_base = [
        "patches",
        *COUNT_KEYS,
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
    ]
    write_csv(
        outdir / "metrics_by_coverage_class.csv",
        coverage_rows,
        ["coverage_class", *group_fields_base],
    )
    write_csv(
        outdir / "metrics_by_true_boundary.csv",
        boundary_rows,
        ["contains_positive_boundary", *group_fields_base],
    )
    write_csv(
        outdir / "metrics_by_error_type.csv",
        error_rows,
        ["dominant_error", *group_fields_base],
    )
    write_csv(
        outdir / "metrics_by_tile.csv",
        tile_rows,
        ["tile_name", *group_fields_base],
    )

    signature_rows = feature_signature_rows(
        results, channels, args.selected_threshold
    )
    write_csv(
        outdir / "terrain_feature_signature_by_error_class.csv",
        signature_rows,
        ["channel", "error_class", "pixel_count", "mean", "std"],
    )

    representative, ignore_limit_used = select_representative_records(
        selected_records,
        max_ignore_fraction=args.max_ignore_fraction,
    )
    representative_fields = [
        "case_label",
        *patch_fields,
    ]
    write_csv(
        outdir / "representative_cases.csv",
        representative,
        representative_fields,
    )

    result_by_patch = {
        result.row.get("patch_id", f"patch_{result.index:06d}"): result
        for result in results
    }
    case_dir = outdir / "representative_cases"
    for order, record in enumerate(representative, start=1):
        patch_id = str(record["patch_id"])
        result = result_by_patch[patch_id]
        safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", patch_id)
        output_path = case_dir / f"{order:02d}_{record['case_label']}_{safe_id}.png"
        make_detailed_figure(
            output_path=output_path,
            result=result,
            record=record,
            channels=channels,
            comparison_thresholds=comparison_thresholds,
            selected_threshold=args.selected_threshold,
            dataset_dir=dataset_dir,
        )
        record["figure"] = str(output_path.relative_to(outdir))

    write_csv(
        outdir / "representative_cases.csv",
        representative,
        [*representative_fields, "figure"],
    )

    make_professor_overview(
        output_path=outdir / "professor_comparison.png",
        selected=representative,
        result_by_patch=result_by_patch,
        channels=channels,
        selected_threshold=args.selected_threshold,
    )

    write_markdown_summary(
        path=outdir / "error_summary.md",
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        split=args.split,
        selected_threshold=args.selected_threshold,
        threshold_rows=threshold_rows,
        selected_cases=representative,
        ignore_limit_used=ignore_limit_used,
        tile_rows=tile_rows,
    )

    selected_threshold_row = min(
        threshold_rows,
        key=lambda row: abs(float(row["threshold"]) - args.selected_threshold),
    )
    summary = {
        "version": 1,
        "dataset_dir": str(dataset_dir),
        "manifest": str(manifest_path),
        "checkpoint": str(checkpoint_path),
        "normalization": str(normalization_path),
        "split": args.split,
        "region_filter": args.region,
        "require_qc": args.require_qc,
        "device": resolved_device,
        "selected_threshold": args.selected_threshold,
        "comparison_thresholds": comparison_thresholds,
        "patch_count": len(results),
        "aggregate": selected_threshold_row,
        "representative_ignore_limit_used": ignore_limit_used,
        "representative_cases": representative,
        "outputs": {
            "professor_comparison": "professor_comparison.png",
            "threshold_plot": "threshold_metric_comparison.png",
            "summary_markdown": "error_summary.md",
            "patch_metrics": "patch_metrics_selected_threshold.csv",
            "coverage_metrics": "metrics_by_coverage_class.csv",
            "boundary_metrics": "metrics_by_true_boundary.csv",
            "tile_metrics": "metrics_by_tile.csv",
            "feature_signatures": "terrain_feature_signature_by_error_class.csv",
        },
    }
    (outdir / "error_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print("\nVisual error inspection complete.")
    print(f"Professor comparison: {outdir / 'professor_comparison.png'}")
    print(f"Detailed cases:       {case_dir}")
    print(f"Summary:              {outdir / 'error_summary.md'}")
    print(f"Threshold metrics:    {outdir / 'threshold_metrics.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
