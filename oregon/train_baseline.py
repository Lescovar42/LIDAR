#!/usr/bin/env python3
"""Train a small segmentation baseline from a prebuilt patch dataset.

The dataset must first be created by ``build_dataset.py``. Splits are read from
``patches.csv`` and are therefore tile-level rather than random patch-level.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

ACCEPTED_QC = {"accept", "accept_approximate_boundary"}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def filter_rows(rows: list[dict[str, str]], split: str, require_qc: bool) -> list[dict[str, str]]:
    selected = [row for row in rows if row.get("split") == split]
    if require_qc:
        selected = [row for row in selected if row.get("qc_status", "").strip().casefold() in ACCEPTED_QC]
    return selected


def compute_channel_statistics(dataset_dir: Path, rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        raise ValueError("Cannot compute normalization statistics from zero training patches")
    channel_sum: np.ndarray | None = None
    channel_sum_sq: np.ndarray | None = None
    pixel_count = 0
    for index, row in enumerate(rows, start=1):
        with np.load(dataset_dir / row["patch_path"]) as data:
            features = data["features"].astype(np.float64)
        if channel_sum is None:
            channel_sum = np.zeros(features.shape[0], dtype=np.float64)
            channel_sum_sq = np.zeros(features.shape[0], dtype=np.float64)
        flattened = features.reshape(features.shape[0], -1)
        channel_sum += flattened.sum(axis=1)
        channel_sum_sq += np.square(flattened).sum(axis=1)
        pixel_count += flattened.shape[1]
        if index % 100 == 0:
            print(f"  normalization stats: {index}/{len(rows)} patches")

    assert channel_sum is not None and channel_sum_sq is not None
    mean = channel_sum / pixel_count
    variance = np.maximum(channel_sum_sq / pixel_count - np.square(mean), 1e-8)
    std = np.sqrt(variance)
    return mean.astype(np.float32), std.astype(np.float32)


def compute_pos_weight(dataset_dir: Path, rows: list[dict[str, str]], cap: float = 50.0) -> float:
    positive = 0
    total = 0
    for row in rows:
        with np.load(dataset_dir / row["patch_path"]) as data:
            mask = data["mask"]
        positive += int(mask.sum())
        total += int(mask.size)
    negative = max(0, total - positive)
    if positive == 0:
        raise ValueError("Training split contains zero positive pixels")
    return float(min(cap, max(1.0, negative / positive)))


class PatchDataset(Dataset):
    def __init__(self, dataset_dir: Path, rows: list[dict[str, str]], mean: np.ndarray, std: np.ndarray):
        self.dataset_dir = dataset_dir
        self.rows = rows
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with np.load(self.dataset_dir / row["patch_path"]) as data:
            features = data["features"].astype(np.float32)
            mask = data["mask"].astype(np.float32)[None, :, :]
        features = (features - self.mean) / self.std
        return torch.from_numpy(features), torch.from_numpy(mask)


class MiniUNet(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()

        def block(input_channels: int, output_channels: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_channels, output_channels, 3, padding=1),
                nn.BatchNorm2d(output_channels),
                nn.ReLU(inplace=True),
            )

        self.enc1 = block(in_channels, 32)
        self.enc2 = block(32, 64)
        self.bottleneck = block(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = block(64, 32)
        self.head = nn.Conv2d(32, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        bottleneck = self.bottleneck(self.pool(enc2))
        dec2 = self.up2(bottleneck)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))
        dec1 = self.up1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))
        return self.head(dec1)


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1.0) -> torch.Tensor:
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * target).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    tp = fp = fn = tn = 0
    total_loss = 0.0
    batches = 0
    bce = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for features, target in loader:
            features = features.to(device)
            target = target.to(device)
            logits = model(features)
            loss = bce(logits, target) + soft_dice_loss(logits, target)
            total_loss += float(loss.item())
            batches += 1
            prediction = torch.sigmoid(logits) >= 0.5
            truth = target >= 0.5
            tp += int((prediction & truth).sum().item())
            fp += int((prediction & ~truth).sum().item())
            fn += int((~prediction & truth).sum().item())
            tn += int((~prediction & ~truth).sum().item())

    epsilon = 1e-9
    return {
        "loss": total_loss / max(1, batches),
        "dice": (2 * tp) / max(epsilon, 2 * tp + fp + fn),
        "iou": tp / max(epsilon, tp + fp + fn),
        "precision": tp / max(epsilon, tp + fp),
        "recall": tp / max(epsilon, tp + fn),
        "specificity": tn / max(epsilon, tn + fp),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a tile-split Oregon landslide segmentation baseline.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_pilot"))
    parser.add_argument("--outdir", type=Path, default=Path("training_output"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require-qc",
        action="store_true",
        help="Use only rows with qc_status accept/accept_approximate_boundary.",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_dir = args.dataset_dir.resolve()
    qc_manifest_path = dataset_dir / "patches_qc.csv"
    manifest_path = qc_manifest_path if qc_manifest_path.exists() else dataset_dir / "patches.csv"
    channels_path = dataset_dir / "channels.json"
    if not manifest_path.exists() or not channels_path.exists():
        parser.error("Dataset is missing patches.csv/patches_qc.csv or channels.json; run build_dataset.py first")

    rows = read_manifest(manifest_path)
    train_rows = filter_rows(rows, "train", args.require_qc)
    validation_rows = filter_rows(rows, "validation", args.require_qc)
    test_rows = filter_rows(rows, "test", args.require_qc)
    if not train_rows:
        parser.error("No training rows after filtering")

    channels = json.loads(channels_path.read_text(encoding="utf-8"))["feature_names"]
    print(f"Rows: train={len(train_rows)}, validation={len(validation_rows)}, test={len(test_rows)}")
    print(f"Channels ({len(channels)}): {channels}")

    output_dir = args.outdir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Computing train-only normalization statistics...")
    mean, std = compute_channel_statistics(dataset_dir, train_rows)
    normalization = {name: {"mean": float(mu), "std": float(sd)} for name, mu, sd in zip(channels, mean, std)}
    (output_dir / "normalization.json").write_text(json.dumps(normalization, indent=2), encoding="utf-8")

    pos_weight = compute_pos_weight(dataset_dir, train_rows)
    print(f"Positive-class weight: {pos_weight:.3f}")

    train_dataset = PatchDataset(dataset_dir, train_rows, mean, std)
    validation_dataset = PatchDataset(dataset_dir, validation_rows, mean, std)
    test_dataset = PatchDataset(dataset_dir, test_rows, mean, std)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        generator=generator,
    )
    validation_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MiniUNet(len(channels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    history: list[dict[str, Any]] = []
    best_score = -1.0
    best_path = output_dir / "best_model.pt"
    print(f"Training on {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        batches = 0
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = bce(logits, target) + soft_dice_loss(logits, target)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            batches += 1

        train_loss = running_loss / max(1, batches)
        if validation_rows:
            validation_metrics = evaluate(model, validation_loader, device)
            selection_score = validation_metrics["dice"]
        else:
            validation_metrics = {"loss": float("nan"), "dice": float("nan"), "iou": float("nan"), "precision": float("nan"), "recall": float("nan"), "specificity": float("nan")}
            selection_score = -train_loss

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        history.append(record)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | train loss {train_loss:.4f} | "
            f"val Dice {validation_metrics['dice']:.4f} | val IoU {validation_metrics['iou']:.4f}"
        )

        if selection_score > best_score:
            best_score = selection_score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "channels": channels,
                    "mean": mean,
                    "std": std,
                    "epoch": epoch,
                    "validation_metrics": validation_metrics,
                },
                best_path,
            )

    with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    checkpoint = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    final_metrics = {
        "best_epoch": checkpoint["epoch"],
        "validation": evaluate(model, validation_loader, device) if validation_rows else None,
        "test": evaluate(model, test_loader, device) if test_rows else None,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "test_rows": len(test_rows),
        "require_qc": args.require_qc,
    }
    (output_dir / "metrics.json").write_text(json.dumps(final_metrics, indent=2), encoding="utf-8")
    print(json.dumps(final_metrics, indent=2))
    print(f"Saved best checkpoint: {best_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
