#!/usr/bin/env python3
"""
Independent reproduction of the official Landslide4Sense 2022 U-Net baseline.

This is intentionally NOT the Tillamook trainer.

Method reproduced from the official baseline:
- 14 input channels
- fixed official per-channel mean/std
- 128x128 patches
- U-Net: base width 64, 4 downsampling stages, bilinear decoder
- 2 output classes
- CrossEntropyLoss(ignore_index=255)
- Adam, lr=1e-3, weight_decay=5e-4
- batch size 32
- 5000 optimizer steps
- validation every 500 steps
- landslide-class precision / recall / F1 from globally pooled pixel counts
- no AMP, no class weighting, no Dice loss, no threshold sweep

Dataset layouts supported:
1) IBM/NASA Hugging Face mirror:
   root/
     images/train/image_1.h5
     annotations/train/mask_1.h5
     images/validation/image_1.h5
     annotations/validation/mask_1.h5

2) Original competition-style layout:
   root/
     TrainData/img/image_1.h5
     TrainData/mask/mask_1.h5
     ValidData/img/image_1.h5
     ValidData/mask/mask_1.h5   # only if labels are available locally

For the published validation sanity check, use the IBM/NASA mirror because it
contains validation masks.

Important protocol note:
The current official Train.py source constructs its evaluation loader using
the training list, despite defining a test_list argument. That source-code
quirk cannot reproduce the published validation metric. This script therefore
defaults to --protocol reported-validation, which keeps the official model and
training hyperparameters but evaluates on the validation annotations.

Use --protocol literal-source to deliberately mirror the Train.py evaluation
quirk and evaluate the training set every 500 iterations.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


OFFICIAL_MEAN = np.asarray([
    -0.4914, -0.3074, -0.1277, -0.0625,  0.0439,  0.0803,  0.0644,
     0.0802,  0.3000,  0.4082,  0.0823,  0.0516,  0.3338,  0.7819
], dtype=np.float32)

OFFICIAL_STD = np.asarray([
    0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354,
    0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913
], dtype=np.float32)

REPORTED = {
    "precision": 0.5175,
    "recall": 0.6550,
    "f1": 0.5782,
}


def numeric_suffix(path: Path) -> tuple[int, str]:
    stem = path.stem
    digits = "".join(ch for ch in stem if ch.isdigit())
    return (int(digits) if digits else 10**12, path.name)


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    name: str


def first_existing(paths: Iterable[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def discover_split(root: Path, split: str) -> list[Sample]:
    split = split.lower()

    if split == "train":
        image_dir = first_existing([
            root / "images" / "train",
            root / "TrainData" / "img",
        ])
        mask_dir = first_existing([
            root / "annotations" / "train",
            root / "TrainData" / "mask",
        ])
    elif split in {"validation", "valid", "val"}:
        image_dir = first_existing([
            root / "images" / "validation",
            root / "ValidData" / "img",
        ])
        mask_dir = first_existing([
            root / "annotations" / "validation",
            root / "ValidData" / "mask",
        ])
    else:
        raise ValueError(f"Unsupported labeled split: {split}")

    if image_dir is None:
        raise FileNotFoundError(f"Could not find image directory for split={split} under {root}")
    if mask_dir is None:
        raise FileNotFoundError(
            f"Could not find mask directory for split={split} under {root}. "
            "For validation, use a mirror that includes validation annotations."
        )

    images = sorted(image_dir.glob("image_*.h5"), key=numeric_suffix)
    if not images:
        raise FileNotFoundError(f"No image_*.h5 files in {image_dir}")

    samples: list[Sample] = []
    missing: list[Path] = []

    for image in images:
        mask_name = image.name.replace("image_", "mask_", 1)
        mask = mask_dir / mask_name
        if not mask.exists():
            # Some mirrors may store label files with odd naming. Do not silently guess
            # beyond the canonical image_N -> mask_N mapping.
            missing.append(mask)
            continue
        samples.append(Sample(image=image, mask=mask, name=image.name))

    if missing:
        preview = "\n".join(str(p) for p in missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} masks are missing for split={split}. First missing:\n{preview}"
        )

    return samples


def read_h5_array(path: Path, preferred_key: str) -> np.ndarray:
    with h5py.File(path, "r") as hf:
        if preferred_key in hf:
            return hf[preferred_key][:]
        keys = list(hf.keys())
        if len(keys) == 1:
            return hf[keys[0]][:]
        raise KeyError(
            f"{path}: expected HDF5 key '{preferred_key}', available keys={keys}"
        )


class L4SDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        repeat_to_at_least: int | None = None,
        literal_official_repeat_quirk: bool = False,
    ):
        self.base_samples = samples
        self.samples = list(samples)

        if repeat_to_at_least is not None:
            n = len(samples)
            if literal_official_repeat_quirk:
                # Mirrors the expression in the official loader. It can create more
                # than the nominal requested number because the final slice can be negative.
                n_repeat = int(np.ceil(repeat_to_at_least / n))
                extra_stop = repeat_to_at_least - n_repeat * n
                self.samples = samples * n_repeat + samples[:extra_stop]
            else:
                n_repeat = int(math.ceil(repeat_to_at_least / n))
                self.samples = (samples * n_repeat)[:repeat_to_at_least]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]
        image = read_h5_array(sample.image, "img").astype(np.float32)
        mask = read_h5_array(sample.mask, "mask").astype(np.float32)

        # Official files are H x W x 14.
        if image.ndim != 3:
            raise RuntimeError(f"{sample.image}: expected 3D image, got {image.shape}")

        if image.shape[-1] == 14:
            image = image.transpose(2, 0, 1)
        elif image.shape[0] == 14:
            pass
        else:
            raise RuntimeError(f"{sample.image}: cannot identify 14-channel axis: {image.shape}")

        if image.shape[1:] != (128, 128):
            raise RuntimeError(f"{sample.image}: expected 128x128, got {image.shape}")

        if mask.ndim == 3 and 1 in mask.shape:
            mask = np.squeeze(mask)
        if mask.shape != (128, 128):
            raise RuntimeError(f"{sample.mask}: expected 128x128 mask, got {mask.shape}")

        # Keep 255 valid as the official CE ignore value, although standard L4S masks are 0/1.
        vals = np.unique(mask)
        if not np.all(np.isin(vals, [0, 1, 255])):
            raise RuntimeError(f"{sample.mask}: unexpected label values {vals.tolist()}")

        image = (image - OFFICIAL_MEAN[:, None, None]) / OFFICIAL_STD[:, None, None]

        return (
            torch.from_numpy(image.copy()),
            torch.from_numpy(mask.astype(np.int64, copy=False)),
            sample.name,
        )


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int | None = None):
        super().__init__()
        mid = out_channels if mid_channels is None else mid_channels
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, bilinear: bool = True):
        super().__init__()
        self.bilinear = bilinear

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, mid_channels=in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, decoder_x, skip_x):
        decoder_x = self.up(decoder_x)

        diff_y = skip_x.size(2) - decoder_x.size(2)
        diff_x = skip_x.size(3) - decoder_x.size(3)
        decoder_x = F.pad(
            decoder_x,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2,
            ],
        )
        x = torch.cat([skip_x, decoder_x], dim=1)
        return self.conv(x)


class OfficialUNet(nn.Module):
    def __init__(self, n_classes: int = 2, n_channels: int = 14, bilinear: bool = True):
        super().__init__()
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)

        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def pooled_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
    eps = 1e-14
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2.0 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + fp + fn + tn + eps)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "iou": float(iou),
        "overall_accuracy": float(oa),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_predictions: Path | None = None,
):
    model.eval()
    tp = fp = fn = tn = 0

    if save_predictions is not None:
        save_predictions.mkdir(parents=True, exist_ok=True)

    for batch_idx, (images, labels, names) in enumerate(loader, 1):
        images = images.float().to(device, non_blocking=True)
        labels_np = labels.numpy()

        logits = model(images)
        # Official Train.py applies a 128x128 bilinear interpolation before
        # argmax. U-Net already returns 128x128, but we preserve the operation.
        logits = F.interpolate(
            logits, size=(128, 128), mode="bilinear", align_corners=False
        )
        preds = torch.argmax(torch.softmax(logits, dim=1), dim=1).cpu().numpy()

        valid = (labels_np >= 0) & (labels_np < 2)

        # Landslide class = 1; globally pool counts over all validation pixels.
        truth1 = labels_np == 1
        pred1 = preds == 1

        tp += int(np.logical_and(pred1, truth1 & valid).sum())
        fp += int(np.logical_and(pred1, (~truth1) & valid).sum())
        fn += int(np.logical_and(~pred1, truth1 & valid).sum())
        tn += int(np.logical_and(~pred1, (~truth1) & valid).sum())

        if save_predictions is not None:
            for j, name in enumerate(names):
                mask_name = Path(name).name.replace("image_", "mask_", 1)
                with h5py.File(save_predictions / mask_name, "w") as hf:
                    hf.create_dataset("mask", data=preds[j].astype(np.uint8))

        if batch_idx % 50 == 0 or batch_idx == len(loader):
            print(f"  eval {batch_idx}/{len(loader)}")

    return pooled_metrics(tp, fp, fn, tn)


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--outdir", type=Path, default=Path("./l4s_reproduction"))
    p.add_argument(
        "--protocol",
        choices=("reported-validation", "literal-source"),
        default="reported-validation",
        help=(
            "reported-validation: evaluate true validation annotations; "
            "literal-source: mirror current official Train.py quirk and evaluate training set"
        ),
    )
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Official source does not set a seed. Omit for closest source behavior.",
    )
    p.add_argument(
        "--literal-repeat-quirk",
        action="store_true",
        help="Mirror the official dataset repeat expression exactly.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.steps <= 0 or args.batch_size <= 0 or args.eval_every <= 0:
        raise SystemExit("steps, batch-size and eval-every must be positive")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    set_seed(args.seed)

    # Official training enables cuDNN and benchmarking.
    if device.type == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    root = args.data_root.resolve()
    outdir = args.outdir.resolve()
    ckpt_dir = outdir / "checkpoints"
    pred_root = outdir / "validation_predictions"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_samples = discover_split(root, "train")
    val_samples = discover_split(root, "validation")

    if len(train_samples) != 3799:
        print(f"WARNING: official train count is 3799, found {len(train_samples)}")
    if len(val_samples) != 245:
        print(f"WARNING: official validation count is 245, found {len(val_samples)}")

    required_samples = args.steps * args.batch_size
    train_ds = L4SDataset(
        train_samples,
        repeat_to_at_least=required_samples,
        literal_official_repeat_quirk=args.literal_repeat_quirk,
    )

    eval_samples = train_samples if args.protocol == "literal-source" else val_samples
    eval_ds = L4SDataset(eval_samples)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = OfficialUNet(n_classes=2, n_channels=14, bilinear=True).to(device)
    params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    run_config = {
        "implementation": "independent_official_baseline_reproduction",
        "protocol": args.protocol,
        "data_root": str(root),
        "train_count": len(train_samples),
        "validation_count": len(val_samples),
        "input_channels": 14,
        "input_size": [128, 128],
        "classes": 2,
        "model": "official_style_unet_bilinear",
        "parameter_count": params,
        "official_mean": OFFICIAL_MEAN.tolist(),
        "official_std": OFFICIAL_STD.tolist(),
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss": "CrossEntropyLoss(ignore_index=255)",
        "batch_size": args.batch_size,
        "steps": args.steps,
        "eval_every": args.eval_every,
        "seed": args.seed,
        "amp": False,
        "class_weighting": None,
        "threshold_tuning": None,
        "reported_validation_reference": REPORTED,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "literal_repeat_quirk": bool(args.literal_repeat_quirk),
    }
    (outdir / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Protocol: {args.protocol}")
    print(f"Device: {run_config['device_name']}")
    print(f"Parameters: {params:,}")
    print(
        f"Official hyperparameters: batch={args.batch_size}, Adam lr={args.learning_rate}, "
        f"weight_decay={args.weight_decay}, CE, steps={args.steps}"
    )

    history: list[dict] = []
    best_f1 = 0.5  # same initial threshold used by official Train.py
    best_checkpoint: Path | None = None

    model.train()
    for step_idx, (images, labels, _) in enumerate(train_loader, start=1):
        if step_idx > args.steps:
            break

        t0 = time.time()
        model.train()
        optimizer.zero_grad(set_to_none=True)

        images = images.float().to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        logits = model(images)
        logits_interp = F.interpolate(
            logits, size=(128, 128), mode="bilinear", align_corners=False
        )

        loss = criterion(logits_interp, labels)

        with torch.no_grad():
            predicted = torch.argmax(logits_interp, dim=1)
            batch_oa = float((predicted == labels).float().mean().item())

        loss.backward()
        optimizer.step()

        elapsed = time.time() - t0

        row = {
            "step": step_idx,
            "train_loss": float(loss.item()),
            "train_batch_oa": batch_oa,
            "step_seconds": elapsed,
        }

        if step_idx % 10 == 0:
            print(
                f"Iter {step_idx:04d}/{args.steps} | "
                f"loss={loss.item():.4f} | batch_OA={batch_oa*100:.2f}% | "
                f"{elapsed:.2f}s"
            )

        if step_idx % args.eval_every == 0:
            print(f"\nEvaluation at step {step_idx}...")
            metrics = evaluate(model, eval_loader, device)
            row.update({f"eval_{k}": v for k, v in metrics.items()})

            print(
                f"  Landslide P={metrics['precision']*100:.2f}% "
                f"R={metrics['recall']*100:.2f}% "
                f"F1={metrics['f1']*100:.2f}% "
                f"IoU={metrics['iou']*100:.2f}%"
            )

            if metrics["f1"] > best_f1:
                best_f1 = float(metrics["f1"])
                best_checkpoint = ckpt_dir / f"batch{step_idx}_F1_{int(best_f1*10000)}.pth"
                torch.save(model.state_dict(), best_checkpoint)
                print(f"  Saved new best: {best_checkpoint}")

            model.train()

        history.append(row)

    # Write step history; eval fields are sparse.
    fieldnames = sorted({k for row in history for k in row.keys()})
    with (outdir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    if best_checkpoint is None:
        # Preserve a useful final model even if F1 never exceeds the official source's
        # initial 0.5 best threshold.
        best_checkpoint = ckpt_dir / "final_model_no_F1_above_0_5.pth"
        torch.save(model.state_dict(), best_checkpoint)
        print("WARNING: no evaluation exceeded F1=0.5; saved final model instead.")

    print(f"\nLoading selected checkpoint: {best_checkpoint}")
    state = torch.load(best_checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state)

    # Always save final predictions for the scientifically useful validation split,
    # even in literal-source training-evaluation mode.
    val_loader = DataLoader(
        L4SDataset(val_samples),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    print("\nFinal VALIDATION evaluation and prediction export...")
    final_metrics = evaluate(
        model,
        val_loader,
        device,
        save_predictions=pred_root,
    )

    result = {
        "selected_checkpoint": str(best_checkpoint),
        "validation": final_metrics,
        "reported_reference": REPORTED,
        "absolute_difference_from_reported": {
            "precision": abs(final_metrics["precision"] - REPORTED["precision"]),
            "recall": abs(final_metrics["recall"] - REPORTED["recall"]),
            "f1": abs(final_metrics["f1"] - REPORTED["f1"]),
        },
        "prediction_dir": str(pred_root),
    }

    (outdir / "metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    print("\nREPRODUCTION COMPLETE")
    print(json.dumps(result, indent=2))
    print(f"\nPredictions: {pred_root}")
    print(f"Metrics:     {outdir / 'metrics.json'}")
    print(f"Config:      {outdir / 'run_config.json'}")


if __name__ == "__main__":
    main()
