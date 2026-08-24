#!/usr/bin/env python3
"""
Strict-binary Tillamook feature/depth ablation trainer.

Development protocol:
- train split: fitting, normalization, automatic pos_weight
- validation split: checkpoint selection + threshold selection
- test split: NEVER loaded/opened by this script

Ground truth:
- 0 = background/non-landslide by dataset policy
- 1 = landslide
- no ignore index

Controlled factors:
- architecture label: shallow | deep (shared EfficientNetV2-S U-Net)
- features: 3ch
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


CANONICAL_CHANNELS = [
    "local_relief",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "curvature",
    "multidirectional_hillshade",
    "tri",
]

FEATURE_SETS = {
    "7ch": CANONICAL_CHANNELS,
    "3ch": ["slope_degrees", "aspect_sin", "aspect_cos"],
}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def hash_ids(rows: list[dict[str, str]]) -> str:
    payload = "\n".join(sorted(r["patch_id"] for r in rows)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_channels(dataset_dir: Path) -> list[str]:
    obj = json.loads((dataset_dir / "channels.json").read_text(encoding="utf-8"))
    channels = obj["feature_names"] if isinstance(obj, dict) else obj
    if channels != CANONICAL_CHANNELS:
        raise ValueError(
            "Dataset channel order differs from canonical seven-channel order.\n"
            f"Expected: {CANONICAL_CHANNELS}\nFound: {channels}"
        )
    return channels


def select_rows(rows: list[dict[str, str]]):
    train = [r for r in rows if r.get("split") == "train"]
    val = [r for r in rows if r.get("split") == "validation"]
    test = [r for r in rows if r.get("split") == "test"]
    unknown = sorted({r.get("split", "") for r in rows} - {"train", "validation", "test"})
    if unknown:
        raise ValueError(f"Unknown split labels: {unknown}")
    if not train or not val or not test:
        raise ValueError(
            f"Expected all three frozen splits; got train={len(train)}, "
            f"validation={len(val)}, test={len(test)}"
        )
    return train, val, test


def assert_split_isolation(train, val, test):
    ids = {
        "train": {r["patch_id"] for r in train},
        "validation": {r["patch_id"] for r in val},
        "test": {r["patch_id"] for r in test},
    }
    tiles = {
        "train": {r["tile_name"] for r in train},
        "validation": {r["tile_name"] for r in val},
        "test": {r["tile_name"] for r in test},
    }
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        if ids[a] & ids[b]:
            raise ValueError(f"Patch leakage between {a} and {b}")
        if tiles[a] & tiles[b]:
            raise ValueError(f"Tile leakage between {a} and {b}")


def compute_channel_statistics(
    dataset_dir: Path,
    rows: list[dict[str, str]],
    indices: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(len(indices), dtype=np.float64)
    sums_sq = np.zeros(len(indices), dtype=np.float64)
    count = 0

    for i, row in enumerate(rows, 1):
        with np.load(dataset_dir / row["patch_path"]) as data:
            features = data["features"][indices].astype(np.float64)

        flat = features.reshape(len(indices), -1)
        sums += flat.sum(axis=1)
        sums_sq += np.square(flat).sum(axis=1)
        count += flat.shape[1]

        if i % 250 == 0 or i == len(rows):
            print(f"  normalization: {i}/{len(rows)} train patches")

    mean = sums / count
    var = np.maximum(sums_sq / count - np.square(mean), 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


def compute_pos_weight(dataset_dir: Path, rows: list[dict[str, str]], cap: float = 50.0):
    positive = 0
    negative = 0
    for i, row in enumerate(rows, 1):
        with np.load(dataset_dir / row["patch_path"]) as data:
            mask = data["mask"]
        values = set(np.unique(mask).tolist())
        if not values <= {0, 1}:
            raise ValueError(
                f"Non-binary train mask in {row['patch_id']}: {sorted(values)}"
            )
        positive += int((mask == 1).sum())
        negative += int((mask == 0).sum())
        if i % 500 == 0 or i == len(rows):
            print(f"  class weight: {i}/{len(rows)} train patches")

    if positive == 0:
        raise ValueError("Training split has zero positive pixels")

    raw = negative / positive
    return float(min(cap, max(1.0, raw))), positive, negative


class PatchDataset(Dataset):
    def __init__(
        self,
        dataset_dir: Path,
        rows: list[dict[str, str]],
        channel_indices: list[int],
        mean: np.ndarray,
        std: np.ndarray,
    ):
        self.dataset_dir = dataset_dir
        self.rows = rows
        self.channel_indices = channel_indices
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with np.load(self.dataset_dir / row["patch_path"]) as data:
            features = data["features"][self.channel_indices].astype(np.float32)
            mask = data["mask"].astype(np.float32)

        # Fail closed if an old ignore-mask NPZ somehow enters the experiment.
        values = np.unique(mask)
        if not np.all(np.isin(values, [0.0, 1.0])):
            raise RuntimeError(
                f"Strict-binary violation in {row['patch_id']}: {values.tolist()}"
            )

        features = (features - self.mean) / self.std
        return torch.from_numpy(features), torch.from_numpy(mask[None, :, :])


def conv_block(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ShallowUNet(nn.Module):
    """Original two-pooling Mini U-Net topology."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.bottleneck = conv_block(64, 128)
        self.pool = nn.MaxPool2d(2)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


class DeepUNet(nn.Module):
    """Four-pooling U-Net requested for the model-depth comparison."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)

        self.enc1 = conv_block(in_channels, 32)
        self.enc2 = conv_block(32, 64)
        self.enc3 = conv_block(64, 128)
        self.enc4 = conv_block(128, 256)
        self.bottleneck = conv_block(256, 512)

        self.up4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec4 = conv_block(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.head(d1)


def build_model(architecture: str, in_channels: int):
    """Build the shared EfficientNetV2-S U-Net for either legacy label."""
    if architecture not in {"shallow", "deep"}:
        raise ValueError(architecture)
    if in_channels != 3:
        raise ValueError("EfficientNetV2 U-Net requires exactly 3 input channels")
    return smp.Unet(
        encoder_name="tu-efficientnetv2_s",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )


def binary_loss(logits, target, pos_weight):
    bce = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight
    )
    probs = torch.sigmoid(logits)
    intersection = (probs * target).sum(dim=(1, 2, 3))
    denominator = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    return bce + (1.0 - dice.mean())


def metrics_from_counts(tp, fp, fn, tn):
    def ratio(a, b):
        return float(a / b) if b else 0.0
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "dice": ratio(2 * tp, 2 * tp + fp + fn),
        "iou": ratio(tp, tp + fp + fn),
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "gt_positive_fraction": ratio(tp + fn, tp + fp + fn + tn),
        "predicted_positive_fraction": ratio(tp + fp, tp + fp + fn + tn),
    }


@torch.no_grad()
def evaluate_at_threshold(model, loader, device, threshold, pos_weight_tensor):
    model.eval()
    tp = fp = fn = tn = 0
    total_loss = 0.0
    batches = 0

    for features, target in loader:
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        logits = model(features)
        total_loss += float(binary_loss(logits, target, pos_weight_tensor).item())
        batches += 1

        pred = torch.sigmoid(logits) >= threshold
        truth = target == 1

        tp += int((pred & truth).sum().item())
        fp += int((pred & ~truth).sum().item())
        fn += int((~pred & truth).sum().item())
        tn += int((~pred & ~truth).sum().item())

    return {
        "loss": total_loss / max(1, batches),
        "threshold": float(threshold),
        **metrics_from_counts(tp, fp, fn, tn),
    }


@torch.no_grad()
def threshold_sweep(model, loader, device, thresholds):
    model.eval()
    counts = {
        float(t): {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
        for t in thresholds
    }

    for batch_index, (features, target) in enumerate(loader, 1):
        features = features.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        probs = torch.sigmoid(model(features))
        truth = target == 1

        for threshold in thresholds:
            t = float(threshold)
            pred = probs >= t
            c = counts[t]
            c["tp"] += int((pred & truth).sum().item())
            c["fp"] += int((pred & ~truth).sum().item())
            c["fn"] += int((~pred & truth).sum().item())
            c["tn"] += int((~pred & ~truth).sum().item())

        if batch_index % 25 == 0 or batch_index == len(loader):
            print(f"  threshold inference: {batch_index}/{len(loader)} batches")

    results = []
    for threshold in thresholds:
        t = float(threshold)
        c = counts[t]
        results.append({
            "threshold": t,
            **metrics_from_counts(c["tp"], c["fp"], c["fn"], c["tn"]),
        })
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--architecture", choices=("shallow", "deep"), required=True)
    ap.add_argument("--features", choices=("3ch",), required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--pos-weight", default="auto")
    ap.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--threshold-min", type=float, default=0.30)
    ap.add_argument("--threshold-max", type=float, default=0.90)
    ap.add_argument("--threshold-step", type=float, default=0.05)
    args = ap.parse_args()

    if args.epochs <= 0 or args.batch_size <= 0:
        ap.error("--epochs and --batch-size must be positive")
    if args.threshold_step <= 0:
        ap.error("--threshold-step must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if args.device == "cuda" and not torch.cuda.is_available():
        ap.error("--device cuda requested but CUDA is unavailable")

    device = torch.device(
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    dataset_dir = args.dataset_dir.resolve()
    manifest_path = dataset_dir / "patches.csv"
    if not manifest_path.exists():
        ap.error(f"Missing manifest: {manifest_path}")

    rows = read_manifest(manifest_path)
    train_rows, val_rows, test_rows = select_rows(rows)
    assert_split_isolation(train_rows, val_rows, test_rows)

    # HARD TEST LOCK: this script intentionally never constructs a Dataset,
    # DataLoader, normalization pass, class-weight pass, or evaluation pass
    # from test_rows.
    print(
        f"Frozen rows: train={len(train_rows)}, validation={len(val_rows)}, "
        f"test={len(test_rows)} [LOCKED / NOT LOADED]"
    )

    if (len(train_rows), len(val_rows), len(test_rows)) != (3950, 891, 1210):
        ap.error(
            "Frozen split counts differ from Phase 3 QC "
            f"({len(train_rows)}, {len(val_rows)}, {len(test_rows)})"
        )

    channels = load_channels(dataset_dir)
    selected_names = FEATURE_SETS[args.features]
    indices = [channels.index(name) for name in selected_names]

    print(f"Architecture: {args.architecture}")
    print(f"Feature set: {args.features} -> {selected_names}")

    print("Computing TRAIN-ONLY normalization...")
    mean, std = compute_channel_statistics(dataset_dir, train_rows, indices)

    print("Computing TRAIN-ONLY class weight...")
    auto_weight, positive_pixels, negative_pixels = compute_pos_weight(
        dataset_dir, train_rows
    )

    if args.pos_weight.casefold() == "auto":
        pos_weight = auto_weight
        pos_weight_mode = "auto"
    else:
        try:
            pos_weight = float(args.pos_weight)
        except ValueError:
            ap.error("--pos-weight must be 'auto' or a positive number")
        if not np.isfinite(pos_weight) or pos_weight <= 0:
            ap.error("--pos-weight must be > 0")
        pos_weight_mode = "fixed"

    print(
        f"pos_weight: used={pos_weight:.6f} auto={auto_weight:.6f} "
        f"train positive pixels={positive_pixels} negative pixels={negative_pixels}"
    )

    train_ds = PatchDataset(dataset_dir, train_rows, indices, mean, std)
    val_ds = PatchDataset(dataset_dir, val_rows, indices, mean, std)

    generator = torch.Generator().manual_seed(args.seed)
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
        pin_memory=pin,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
    )

    model = build_model(args.architecture, 3).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    pos_weight_tensor = torch.tensor(pos_weight, device=device)

    # AMP is identical across all four runs when enabled.
    use_amp = bool(args.amp and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    run_config = {
        "dataset_dir": str(dataset_dir),
        "manifest_path": str(manifest_path),
        "ground_truth_policy": "strict_binary_0_1",
        "test_policy": "LOCKED_NOT_LOADED",
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "test_rows_locked": len(test_rows),
        "train_patch_id_sha256": hash_ids(train_rows),
        "validation_patch_id_sha256": hash_ids(val_rows),
        "test_patch_id_sha256_metadata_only": hash_ids(test_rows),
        "architecture": args.architecture,
        "feature_set": args.features,
        "channels": selected_names,
        "channel_indices": indices,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "optimizer": "AdamW",
        "loss": "BCEWithLogits(pos_weight) + soft Dice",
        "pos_weight_mode": pos_weight_mode,
        "pos_weight_used": pos_weight,
        "auto_pos_weight": auto_weight,
        "normalization_source": "train_only",
        "mean": mean.tolist(),
        "std": std.tolist(),
        "amp": use_amp,
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu"
        ),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }
    (outdir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    print(
        f"Training on {run_config['device_name']} | "
        f"parameters={parameter_count:,} | AMP={use_amp}"
    )

    history = []
    best_dice = -1.0
    best_path = outdir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        batches = 0

        for features, target in train_loader:
            features = features.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(features)
                    loss = binary_loss(logits, target, pos_weight_tensor)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(features)
                loss = binary_loss(logits, target, pos_weight_tensor)
                loss.backward()
                optimizer.step()

            loss_sum += float(loss.item())
            batches += 1

        train_loss = loss_sum / max(1, batches)
        val_metrics = evaluate_at_threshold(
            model, val_loader, device, 0.50, pos_weight_tensor
        )

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"validation_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"val Dice@0.50 {val_metrics['dice']:.4f} | "
            f"IoU {val_metrics['iou']:.4f} | "
            f"P {val_metrics['precision']:.4f} | "
            f"R {val_metrics['recall']:.4f}"
        )

        if val_metrics["dice"] > best_dice:
            best_dice = val_metrics["dice"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "architecture": args.architecture,
                    "feature_set": args.features,
                    "channels": selected_names,
                    "channel_indices": indices,
                    "mean": mean,
                    "std": std,
                    "epoch": epoch,
                    "validation_metrics_at_0_5": val_metrics,
                    "parameter_count": parameter_count,
                    "pos_weight_used": pos_weight,
                    "run_config": run_config,
                },
                best_path,
            )

    with (outdir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])

    thresholds = np.arange(
        args.threshold_min,
        args.threshold_max + args.threshold_step / 2.0,
        args.threshold_step,
    )
    thresholds = [round(float(t), 6) for t in thresholds]

    print(
        f"Best epoch={checkpoint['epoch']} at threshold 0.50. "
        "Sweeping VALIDATION thresholds only..."
    )
    sweep = threshold_sweep(model, val_loader, device, thresholds)

    best_threshold_result = max(
        sweep,
        key=lambda x: (
            x["dice"],
            x["iou"],
            x["precision"],
            -abs(x["threshold"] - 0.5),
        ),
    )

    with (outdir / "validation_threshold_sweep.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(sweep[0].keys()))
        writer.writeheader()
        writer.writerows(sweep)

    metrics = {
        "architecture": args.architecture,
        "feature_set": args.features,
        "channels": selected_names,
        "parameter_count": parameter_count,
        "best_epoch": checkpoint["epoch"],
        "validation_at_0_5": checkpoint["validation_metrics_at_0_5"],
        "validation_best_threshold": best_threshold_result,
        "threshold_grid": thresholds,
        "train_rows": len(train_rows),
        "validation_rows": len(val_rows),
        "test_rows_locked_not_evaluated": len(test_rows),
        "pos_weight_used": pos_weight,
        "auto_pos_weight": auto_weight,
        "ground_truth_policy": "strict_binary_0_1",
        "test": None,
    }
    (outdir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print()
    print("RUN COMPLETE — TEST REMAINS LOCKED")
    print(json.dumps(metrics, indent=2))
    print(f"Checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
