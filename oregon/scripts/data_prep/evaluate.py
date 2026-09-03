#!/usr/bin/env python3
"""Evaluate a trained baseline by split and region without retraining."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import torch
from PIL import Image

try:
    from .train_baseline import (
        ACCEPTED_QC,
        EXPECTED_CHANNELS,
        MiniUNet,
        PatchDataset,
        confusion_counts,
        filter_rows,
        metrics_from_counts,
        read_manifest,
        segmentation_loss,
        validate_channels,
    )
except ImportError:  # Direct execution: python oregon/evaluate.py
    from train_baseline import (  # type: ignore[no-redef]
        ACCEPTED_QC,
        EXPECTED_CHANNELS,
        MiniUNet,
        PatchDataset,
        confusion_counts,
        filter_rows,
        metrics_from_counts,
        read_manifest,
        segmentation_loss,
        validate_channels,
    )

COUNT_KEYS = ("tp", "fp", "fn", "tn", "ignored", "total")


def valid_pixel_error_rate(record: Mapping[str, Any]) -> float:
    """Return (FP + FN) / valid pixels from a patch's confusion counts."""
    valid_pixels = sum(int(record.get(key, 0)) for key in ("tp", "fp", "fn", "tn"))
    if valid_pixels <= 0:
        return 0.0
    return (int(record.get("fp", 0)) + int(record.get("fn", 0))) / valid_pixels


def select_overlay_records(
    records: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    """Select deterministic best/worst overlay records without mutating input.

    Worst is the highest valid-pixel error rate, then highest loss. Best Dice is
    selected from patches with positive ground-truth support when any exist; for
    an all-negative collection, lowest error and loss make true negatives useful.
    Patch ID is the final ascending tie-break in every case.
    """
    candidates = list(records)
    if not candidates:
        raise ValueError("At least one record is required for overlay selection")

    def patch_id(record: Mapping[str, Any]) -> str:
        return str(record.get("patch_id", ""))

    def loss(record: Mapping[str, Any]) -> float:
        value = float(record.get("loss", 0.0))
        return value if np.isfinite(value) else float("inf")

    def dice(record: Mapping[str, Any]) -> float:
        value = float(record.get("dice", 0.0))
        return value if np.isfinite(value) else float("-inf")

    positive_support = [
        record
        for record in candidates
        if int(record.get("tp", 0)) + int(record.get("fn", 0)) > 0
    ]
    if positive_support:
        best = min(
            positive_support,
            key=lambda record: (
                -dice(record),
                valid_pixel_error_rate(record),
                loss(record),
                patch_id(record),
            ),
        )
    else:
        best = min(
            candidates,
            key=lambda record: (
                valid_pixel_error_rate(record),
                loss(record),
                patch_id(record),
            ),
        )

    worst = min(
        candidates,
        key=lambda record: (
            -valid_pixel_error_rate(record),
            -loss(record),
            patch_id(record),
        ),
    )
    return best, worst


def load_normalization(path: Path, channels: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Load persisted training normalization; evaluation never recomputes it."""
    if not path.is_file():
        raise FileNotFoundError(f"Normalization file does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    try:
        mean = np.asarray([payload[name]["mean"] for name in channels], dtype=np.float32)
        std = np.asarray([payload[name]["std"] for name in channels], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("normalization.json does not contain mean/std for every channel") from exc
    if not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("normalization.json contains non-finite values or non-positive standard deviations")
    return mean, std


def validate_checkpoint_normalization(
    checkpoint: Mapping[str, Any], mean: np.ndarray, std: np.ndarray
) -> None:
    """Ensure the required normalization file belongs to this training run."""
    try:
        checkpoint_mean = np.asarray(checkpoint["mean"], dtype=np.float32)
        checkpoint_std = np.asarray(checkpoint["std"], dtype=np.float32)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint is missing its training normalization fingerprint") from exc
    if checkpoint_mean.shape != mean.shape or checkpoint_std.shape != std.shape:
        raise ValueError("Checkpoint and normalization.json have different channel counts")
    if not np.allclose(checkpoint_mean, mean, rtol=1e-6, atol=1e-7) or not np.allclose(
        checkpoint_std, std, rtol=1e-6, atol=1e-7
    ):
        raise ValueError("normalization.json does not match the supplied checkpoint")


def aggregate_region_metrics(records: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Aggregate patch confusion counts into exact pixel-level region metrics."""
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        region_id = str(record.get("region_id", "")).strip()
        if not region_id:
            raise ValueError("Every evaluated record must have a region_id")
        grouped[region_id].append(record)

    result: dict[str, dict[str, float | int]] = {}
    for region_id in sorted(grouped):
        region_records = grouped[region_id]
        counts = {key: sum(int(record.get(key, 0)) for record in region_records) for key in COUNT_KEYS}
        valid_pixels = counts["total"] - counts["ignored"]
        weighted_loss = sum(
            float(record.get("loss", 0.0))
            * (int(record.get("total", 0)) - int(record.get("ignored", 0)))
            for record in region_records
        )
        result[region_id] = {
            "patches": len(region_records),
            "valid_pixels": valid_pixels,
            **counts,
            "loss": weighted_loss / valid_pixels if valid_pixels else 0.0,
            **metrics_from_counts(counts),
        }
    return result


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "region"


def _background_rgb(feature: np.ndarray) -> np.ndarray:
    finite = np.isfinite(feature)
    if not finite.any():
        gray = np.zeros(feature.shape, dtype=np.uint8)
    else:
        low, high = np.percentile(feature[finite], (2, 98))
        if high <= low:
            scaled = np.zeros(feature.shape, dtype=np.float32)
        else:
            scaled = np.clip((feature - low) / (high - low), 0.0, 1.0)
        gray = np.nan_to_num(scaled * 255.0, nan=0.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def save_prediction_overlay(
    path: Path,
    background: np.ndarray,
    target: np.ndarray,
    prediction: np.ndarray,
) -> None:
    """Save a deterministic overlay: truth green, prediction red, overlap yellow, ignore blue."""
    rgb = _background_rgb(background)
    target_2d = np.squeeze(target)
    prediction_2d = np.squeeze(prediction).astype(bool)
    ignored = target_2d == 255
    truth = target_2d == 1
    rgb[truth & ~prediction_2d] = (0, 255, 0)
    rgb[prediction_2d & ~truth & ~ignored] = (255, 0, 0)
    rgb[truth & prediction_2d] = (255, 255, 0)
    rgb[ignored] = (0, 96, 255)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb, mode="RGB").save(path, format="PNG", optimize=False)


def _predict_patch(
    model: torch.nn.Module,
    dataset: PatchDataset,
    index: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, dict[str, int]]:
    features, target = dataset[index]
    with torch.no_grad():
        logits = model(features.unsqueeze(0).to(device))
        target_batch = target.unsqueeze(0).to(device)
        loss = float(segmentation_loss(logits, target_batch).item())
        counts = confusion_counts(logits, target_batch)
        prediction = (torch.sigmoid(logits) >= 0.5).cpu().numpy()[0, 0]
    return features.numpy(), target.numpy()[0], prediction, loss, counts


def _select_rows(
    rows: list[dict[str, str]],
    split: str,
    regions: list[str],
    require_qc: bool,
) -> list[dict[str, str]]:
    selected = filter_rows(rows, split, require_qc)
    if regions:
        wanted = {region.casefold() for region in regions}
        selected = [row for row in selected if row.get("region_id", "").strip().casefold() in wanted]
    missing_region = [row.get("patch_id", "<unknown>") for row in selected if not row.get("region_id", "").strip()]
    if missing_region:
        raise ValueError(f"Selected rows are missing region_id, including {missing_region[0]}")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate best_model.pt with persisted training normalization by split and region."
    )
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Required best_model.pt path.")
    parser.add_argument("--normalization", type=Path, required=True, help="Required training normalization.json path.")
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--split", required=True, help="Manifest split to evaluate, e.g. validation or test_rural.")
    parser.add_argument("--region", action="append", default=[], help="Region ID to include; repeat as needed. Default: all in split.")
    parser.add_argument("--require-qc", action="store_true")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda was requested but torch.cuda.is_available() is False")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else (
        "cpu" if args.device == "auto" else args.device
    ))

    dataset_dir = args.dataset_dir.resolve()
    checkpoint_path = args.checkpoint.resolve()
    normalization_path = args.normalization.resolve()
    manifest_path = dataset_dir / "patches_qc.csv"
    if not manifest_path.exists():
        manifest_path = dataset_dir / "patches.csv"
    channels_path = dataset_dir / "channels.json"
    for required_path, label in (
        (manifest_path, "patch manifest"),
        (channels_path, "channels.json"),
        (checkpoint_path, "best_model.pt"),
        (normalization_path, "normalization.json"),
    ):
        if not required_path.is_file():
            parser.error(f"Required {label} does not exist: {required_path}")

    channels = json.loads(channels_path.read_text(encoding="utf-8"))["feature_names"]
    try:
        validate_channels(channels)
        mean, std = load_normalization(normalization_path, channels)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_channels = checkpoint.get("channels")
    if checkpoint_channels != channels:
        parser.error("Checkpoint channels do not exactly match dataset channels")
    try:
        validate_checkpoint_normalization(checkpoint, mean, std)
    except ValueError as exc:
        parser.error(str(exc))
    model = MiniUNet(len(EXPECTED_CHANNELS)).to(device)
    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except (KeyError, RuntimeError) as exc:
        parser.error(f"Could not load model_state_dict from checkpoint: {exc}")
    model.eval()

    rows = read_manifest(manifest_path)
    try:
        selected = _select_rows(rows, args.split, args.region, args.require_qc)
    except ValueError as exc:
        parser.error(str(exc))
    if not selected:
        parser.error("No rows match the requested split, region, and QC filters")

    dataset = PatchDataset(dataset_dir, selected, mean, std)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        _, _, _, loss, counts = _predict_patch(model, dataset, index, device)
        patch_metrics = metrics_from_counts(counts)
        records.append({
            "index": index,
            "patch_id": row.get("patch_id", f"patch_{index:06d}"),
            "region_id": row["region_id"].strip(),
            "loss": loss,
            "valid_pixel_error_rate": valid_pixel_error_rate(counts),
            **counts,
            **patch_metrics,
        })

    per_region = aggregate_region_metrics(records)
    output_dir = args.outdir.resolve()
    overlay_dir = output_dir / "overlays"
    output_dir.mkdir(parents=True, exist_ok=True)

    for region_id in sorted(per_region):
        region_records = [record for record in records if record["region_id"] == region_id]
        best, worst = select_overlay_records(region_records)
        for label, record in (("best", best), ("worst", worst)):
            features, target, prediction, _, _ = _predict_patch(
                model, dataset, int(record["index"]), device
            )
            overlay_path = overlay_dir / f"{_safe_name(region_id)}_{label}.png"
            hillshade_index = channels.index("multidirectional_hillshade")
            save_prediction_overlay(overlay_path, features[hillshade_index], target, prediction)
            per_region[region_id][f"{label}_patch_id"] = str(record["patch_id"])
            per_region[region_id][f"{label}_patch_dice"] = float(record["dice"])
            per_region[region_id][f"{label}_patch_error_rate"] = float(
                record["valid_pixel_error_rate"]
            )
            per_region[region_id][f"{label}_patch_loss"] = float(record["loss"])
            per_region[region_id][f"{label}_overlay"] = str(overlay_path.relative_to(output_dir))

    summary = {
        "dataset_dir": str(dataset_dir),
        "checkpoint": str(checkpoint_path),
        "normalization": str(normalization_path),
        "split": args.split,
        "region_filter": args.region,
        "require_qc": args.require_qc,
        "regions": per_region,
    }
    (output_dir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metric_columns = [
        "region_id", "patches", "valid_pixels", "ignored", "ignore_fraction", "loss",
        "dice", "iou", "precision", "recall", "specificity",
        "best_patch_id", "best_patch_dice", "best_patch_error_rate", "best_patch_loss", "best_overlay",
        "worst_patch_id", "worst_patch_dice", "worst_patch_error_rate", "worst_patch_loss", "worst_overlay",
    ]
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=metric_columns, extrasaction="ignore")
        writer.writeheader()
        for region_id in sorted(per_region):
            writer.writerow({"region_id": region_id, **per_region[region_id]})

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
