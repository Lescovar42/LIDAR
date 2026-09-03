#!/usr/bin/env python3
"""
Landslide4Sense EfficientNetV2-S experiment.

Purpose
-------
Test the professor's hypothesis that replacing the plain U-Net encoder with a
strong pretrained EfficientNetV2-S encoder improves Landslide4Sense segmentation.

This is NOT a reproduction-to-match experiment. It keeps the public dataset,
normalization, labels, loss family, optimizer family, training budget, and
evaluation definition close to the Landslide4Sense baseline, while deliberately
changing the model to:

    U-Net decoder + ImageNet-pretrained EfficientNetV2-S encoder

Important:
- Landslide4Sense input remains all 14 channels.
- Pretrained first-convolution weights are adapted from 3 -> 14 channels by timm/SMP.
- Output remains two-class semantic segmentation.
- Validation metric is globally pooled landslide-class Precision / Recall / F1.
- The script never uses Landslide4Sense test labels for checkpoint selection.
- `literal-source` mode mirrors the official Train.py checkpoint-selection quirk
  (evaluate training set every 500 optimizer updates), making comparison against
  the earlier literal-source U-Net reproduction cleaner.
- `reported-validation` selects checkpoints on validation F1 instead.

Recommended first run:
    --protocol literal-source

That makes the main comparison:
    plain official-style U-Net literal reproduction F1 = ~0.5960
    vs
    EfficientNetV2-S U-Net literal protocol F1 = ???

Because architecture AND encoder pretraining change together, this experiment tests
whether this EfficientNet-based model improves performance, not whether EfficientNet
architecture alone is causally responsible.
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

try:
    import segmentation_models_pytorch as smp
except ImportError as e:
    raise SystemExit(
        "segmentation_models_pytorch is required.\n"
        "Install with:\n"
        "  python -m pip install -U segmentation-models-pytorch timm"
    ) from e


OFFICIAL_MEAN = np.asarray([
    -0.4914, -0.3074, -0.1277, -0.0625,  0.0439,  0.0803,  0.0644,
     0.0802,  0.3000,  0.4082,  0.0823,  0.0516,  0.3338,  0.7819
], dtype=np.float32)

OFFICIAL_STD = np.asarray([
    0.9325, 0.8775, 0.8860, 0.8869, 0.8857, 0.8418, 0.8354,
    0.8491, 0.9061, 1.6072, 0.8848, 0.9232, 0.9018, 1.2913
], dtype=np.float32)

REPORTED_REFERENCE = {
    "precision": 0.5175,
    "recall": 0.6550,
    "f1": 0.5782,
}

# Earlier independent plain-U-Net result from this project.
PROJECT_LITERAL_UNET_REFERENCE = {
    "precision": 0.5449144372137864,
    "recall": 0.6577496545203287,
    "f1": 0.5960388861426874,
    "iou": 0.424540879558343,
}

DEFAULT_ENCODER = "tu-tf_efficientnetv2_s.in21k_ft_in1k"


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    name: str


def numeric_suffix(path: Path):
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return (int(digits) if digits else 10**12, path.name)


def first_existing(paths: Iterable[Path]):
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
        raise ValueError(split)

    if image_dir is None:
        raise FileNotFoundError(f"Could not find {split} image directory under {root}")
    if mask_dir is None:
        raise FileNotFoundError(f"Could not find {split} mask directory under {root}")

    images = sorted(image_dir.glob("image_*.h5"), key=numeric_suffix)
    samples = []
    for image in images:
        mask = mask_dir / image.name.replace("image_", "mask_", 1)
        if not mask.exists():
            raise FileNotFoundError(f"Missing mask for {image}: {mask}")
        samples.append(Sample(image, mask, image.name))

    if not samples:
        raise FileNotFoundError(f"No samples found for {split}")

    return samples


def read_h5(path: Path, preferred_key: str):
    with h5py.File(path, "r") as hf:
        if preferred_key in hf:
            return hf[preferred_key][:]
        keys = list(hf.keys())
        if len(keys) == 1:
            return hf[keys[0]][:]
        raise KeyError(f"{path}: expected key {preferred_key!r}; keys={keys}")


class L4SDataset(Dataset):
    def __init__(
        self,
        samples: list[Sample],
        repeat_to_at_least: int | None = None,
        literal_repeat_quirk: bool = False,
    ):
        self.base_samples = samples
        self.samples = list(samples)

        if repeat_to_at_least is not None:
            n = len(samples)

            if literal_repeat_quirk:
                # Mirrors the original source expression exactly.
                n_repeat = int(np.ceil(repeat_to_at_least / n))
                extra_stop = repeat_to_at_least - n_repeat * n
                self.samples = samples * n_repeat + samples[:extra_stop]
            else:
                n_repeat = int(math.ceil(repeat_to_at_least / n))
                self.samples = (samples * n_repeat)[:repeat_to_at_least]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]

        x = read_h5(s.image, "img").astype(np.float32)
        y = read_h5(s.mask, "mask")

        if x.ndim != 3:
            raise RuntimeError(f"{s.image}: expected 3D input, got {x.shape}")

        if x.shape[-1] == 14:
            x = x.transpose(2, 0, 1)
        elif x.shape[0] == 14:
            pass
        else:
            raise RuntimeError(f"{s.image}: cannot find 14-channel axis: {x.shape}")

        y = np.squeeze(y)
        if x.shape[1:] != (128, 128):
            raise RuntimeError(f"{s.image}: expected 128x128, got {x.shape}")
        if y.shape != (128, 128):
            raise RuntimeError(f"{s.mask}: expected 128x128, got {y.shape}")

        vals = np.unique(y)
        if not np.all(np.isin(vals, [0, 1, 255])):
            raise RuntimeError(f"{s.mask}: unexpected mask values {vals.tolist()}")

        # Keep the same fixed Landslide4Sense normalization used by the official loader.
        x = (x - OFFICIAL_MEAN[:, None, None]) / OFFICIAL_STD[:, None, None]

        return (
            torch.from_numpy(x.copy()),
            torch.from_numpy(y.astype(np.int64, copy=False)),
            s.name,
        )


def build_model(encoder_name: str, pretrained: bool):
    """
    U-Net with EfficientNetV2-S encoder.

    For SMP `tu-` encoders:
      encoder_weights=True -> use the pretrained variant specified in encoder_name
      encoder_weights=None -> random initialization
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=True if pretrained else None,
        in_channels=14,
        classes=2,
        activation=None,
    )


def pooled_metrics(tp, fp, fn, tn):
    eps = 1e-14
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
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
def evaluate(model, loader, device, save_predictions=None, use_amp=False):
    model.eval()

    tp = fp = fn = tn = 0

    if save_predictions is not None:
        save_predictions.mkdir(parents=True, exist_ok=True)

    for batch_idx, (images, labels, names) in enumerate(loader, 1):
        images = images.float().to(device, non_blocking=True)
        labels_np = labels.numpy()

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
        else:
            logits = model(images)

        # SMP U-Net should already output the original spatial size.
        if logits.shape[-2:] != (128, 128):
            logits = F.interpolate(
                logits,
                size=(128, 128),
                mode="bilinear",
                align_corners=False,
            )

        pred = torch.argmax(logits, dim=1).cpu().numpy()

        valid = (labels_np >= 0) & (labels_np < 2)
        truth1 = labels_np == 1
        pred1 = pred == 1

        tp += int(np.logical_and(pred1, truth1 & valid).sum())
        fp += int(np.logical_and(pred1, (~truth1) & valid).sum())
        fn += int(np.logical_and(~pred1, truth1 & valid).sum())
        tn += int(np.logical_and(~pred1, (~truth1) & valid).sum())

        if save_predictions is not None:
            for j, name in enumerate(names):
                mask_name = Path(name).name.replace("image_", "mask_", 1)
                with h5py.File(save_predictions / mask_name, "w") as hf:
                    hf.create_dataset("mask", data=pred[j].astype(np.uint8))

        if batch_idx % 50 == 0 or batch_idx == len(loader):
            print(f"  eval {batch_idx}/{len(loader)}")

    return pooled_metrics(tp, fp, fn, tn)


def set_seed(seed):
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
    p.add_argument("--outdir", type=Path, required=True)

    p.add_argument(
        "--protocol",
        choices=("literal-source", "reported-validation"),
        default="literal-source",
        help=(
            "literal-source: checkpoint selection on training-set F1, matching "
            "the earlier literal U-Net sanity run. "
            "reported-validation: checkpoint selection on validation F1."
        ),
    )

    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument(
        "--random-init",
        action="store_true",
        help="Disable ImageNet pretrained encoder weights.",
    )

    # 5000 optimizer updates preserves the baseline training budget.
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--eval-every", type=int, default=500)

    # Microbatch + accumulation permits EfficientNetV2-S on smaller GPUs while
    # keeping an effective batch size of 32 by default.
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulation-steps", type=int, default=4)

    p.add_argument("--learning-rate", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=5e-4)

    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=None)

    p.add_argument(
        "--literal-repeat-quirk",
        action="store_true",
        help="Mirror the original Landslide4Sense repeat expression.",
    )

    return p.parse_args()


def main():
    args = parse_args()

    if args.steps <= 0:
        raise SystemExit("--steps must be > 0")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be > 0")
    if args.accumulation_steps <= 0:
        raise SystemExit("--accumulation-steps must be > 0")
    if args.eval_every <= 0:
        raise SystemExit("--eval-every must be > 0")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )

    use_amp = bool(args.amp and device.type == "cuda")

    set_seed(args.seed)

    if device.type == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    root = args.data_root.resolve()
    outdir = args.outdir.resolve()
    ckpt_dir = outdir / "checkpoints"
    pred_dir = outdir / "validation_predictions"

    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_samples = discover_split(root, "train")
    val_samples = discover_split(root, "validation")

    print(f"Train samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")

    if len(train_samples) != 3799:
        print(f"WARNING: expected 3799 official training samples, found {len(train_samples)}")
    if len(val_samples) != 245:
        print(f"WARNING: expected 245 official validation samples, found {len(val_samples)}")

    effective_batch_size = args.batch_size * args.accumulation_steps
    required_samples = args.steps * effective_batch_size

    train_ds = L4SDataset(
        train_samples,
        repeat_to_at_least=required_samples,
        literal_repeat_quirk=args.literal_repeat_quirk,
    )

    # Checkpoint selection loader.
    select_samples = train_samples if args.protocol == "literal-source" else val_samples
    select_ds = L4SDataset(select_samples)

    pin = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    select_loader = DataLoader(
        select_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    val_loader = DataLoader(
        L4SDataset(val_samples),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    pretrained = not args.random_init

    print("Building model...")
    print("Encoder:", args.encoder)
    print("Pretrained:", pretrained)
    print("Input channels: 14")

    model = build_model(args.encoder, pretrained).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Parameters: {parameter_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_config = {
        "experiment": "landslide4sense_efficientnetv2_s_unet",
        "goal": "test whether EfficientNet-based U-Net improves segmentation metrics",
        "data_root": str(root),
        "protocol": args.protocol,
        "train_count": len(train_samples),
        "validation_count": len(val_samples),
        "input_channels": 14,
        "input_size": [128, 128],
        "normalization": "official_fixed_landslide4sense_mean_std",
        "architecture": "smp.Unet",
        "encoder_name": args.encoder,
        "encoder_pretrained": pretrained,
        "encoder_pretraining": (
            "ImageNet-21k pretraining + ImageNet-1k fine-tuning"
            if pretrained and "in21k_ft_in1k" in args.encoder
            else "pretrained variant encoded in encoder_name"
            if pretrained
            else "none_random_init"
        ),
        "classes": 2,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_count,
        "optimizer": "Adam",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "loss": "CrossEntropyLoss(ignore_index=255)",
        "optimizer_steps": args.steps,
        "eval_every_optimizer_steps": args.eval_every,
        "micro_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "amp": use_amp,
        "seed": args.seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "reported_reference": REPORTED_REFERENCE,
        "project_literal_plain_unet_reference": PROJECT_LITERAL_UNET_REFERENCE,
        "comparison_note": (
            "Architecture and encoder pretraining change together. "
            "This is a practical model-improvement test, not an isolated causal ablation."
        ),
    }

    (outdir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Device: {run_config['device_name']}")
    print(f"AMP: {use_amp}")
    print(
        f"Microbatch={args.batch_size}, accumulation={args.accumulation_steps}, "
        f"effective batch={effective_batch_size}"
    )
    print(f"Optimizer updates: {args.steps}")
    print(f"Checkpoint-selection protocol: {args.protocol}")

    history = []
    best_f1 = -1.0
    best_checkpoint = None

    model.train()
    optimizer.zero_grad(set_to_none=True)

    optimizer_step = 0
    micro_since_step = 0
    running_loss = 0.0
    running_micro = 0
    t_start = time.time()

    for micro_idx, (images, labels, _) in enumerate(train_loader, start=1):
        images = images.float().to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                if logits.shape[-2:] != labels.shape[-2:]:
                    logits = F.interpolate(
                        logits,
                        size=labels.shape[-2:],
                        mode="bilinear",
                        align_corners=False,
                    )
                loss_raw = criterion(logits, labels)
                loss = loss_raw / args.accumulation_steps

            scaler.scale(loss).backward()
        else:
            logits = model(images)
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            loss_raw = criterion(logits, labels)
            loss = loss_raw / args.accumulation_steps
            loss.backward()

        running_loss += float(loss_raw.item())
        running_micro += 1
        micro_since_step += 1

        if micro_since_step < args.accumulation_steps:
            continue

        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        optimizer.zero_grad(set_to_none=True)

        optimizer_step += 1
        micro_since_step = 0

        mean_recent_loss = running_loss / max(1, running_micro)

        record = {
            "optimizer_step": optimizer_step,
            "train_loss_recent": mean_recent_loss,
            "elapsed_seconds": time.time() - t_start,
        }

        if optimizer_step % 10 == 0:
            print(
                f"Step {optimizer_step:04d}/{args.steps} | "
                f"loss={mean_recent_loss:.4f} | "
                f"elapsed={(time.time()-t_start)/3600:.2f} h"
            )

        running_loss = 0.0
        running_micro = 0

        if optimizer_step % args.eval_every == 0:
            print(f"\nCheckpoint-selection evaluation at optimizer step {optimizer_step}...")
            select_metrics = evaluate(
                model,
                select_loader,
                device,
                save_predictions=None,
                use_amp=use_amp,
            )

            for k, v in select_metrics.items():
                record[f"select_{k}"] = v

            print(
                f"  P={select_metrics['precision']:.4f} "
                f"R={select_metrics['recall']:.4f} "
                f"F1={select_metrics['f1']:.4f} "
                f"IoU={select_metrics['iou']:.4f}"
            )

            if select_metrics["f1"] > best_f1:
                best_f1 = float(select_metrics["f1"])
                best_checkpoint = (
                    ckpt_dir
                    / f"step{optimizer_step}_selectF1_{int(best_f1 * 10000):04d}.pt"
                )

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_step": optimizer_step,
                        "selection_metrics": select_metrics,
                        "run_config": run_config,
                    },
                    best_checkpoint,
                )

                print("  New best:", best_checkpoint)

            model.train()

        history.append(record)

        if optimizer_step >= args.steps:
            break

    if not history:
        raise RuntimeError("No optimizer steps were completed")

    fieldnames = sorted({k for row in history for k in row})
    with (outdir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)

    if best_checkpoint is None:
        best_checkpoint = ckpt_dir / "final_model.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_step": optimizer_step,
                "selection_metrics": None,
                "run_config": run_config,
            },
            best_checkpoint,
        )

    print("\nLoading selected checkpoint:", best_checkpoint)
    ckpt = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\nFinal evaluation on TRUE validation labels...")
    val_metrics = evaluate(
        model,
        val_loader,
        device,
        save_predictions=pred_dir,
        use_amp=use_amp,
    )

    plain = PROJECT_LITERAL_UNET_REFERENCE

    result = {
        "selected_checkpoint": str(best_checkpoint),
        "selected_optimizer_step": ckpt.get("optimizer_step"),
        "checkpoint_selection_protocol": args.protocol,
        "validation": val_metrics,
        "reported_official_reference": REPORTED_REFERENCE,
        "project_plain_unet_literal_reference": plain,
        "difference_vs_project_plain_unet": {
            "precision": val_metrics["precision"] - plain["precision"],
            "recall": val_metrics["recall"] - plain["recall"],
            "f1": val_metrics["f1"] - plain["f1"],
            "iou": val_metrics["iou"] - plain["iou"],
        },
        "relative_f1_change_vs_project_plain_unet_percent": (
            100.0 * (val_metrics["f1"] - plain["f1"]) / plain["f1"]
        ),
        "efficientnet_improves_project_plain_unet_f1": (
            val_metrics["f1"] > plain["f1"]
        ),
        "prediction_dir": str(pred_dir),
    }

    (outdir / "metrics.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    print("\nEFFICIENTNET LANDSLIDE4SENSE RUN COMPLETE")
    print(json.dumps(result, indent=2))
    print()
    print("Main question:")
    if result["efficientnet_improves_project_plain_unet_f1"]:
        print(
            f"  YES — F1 improved from {plain['f1']:.4f} "
            f"to {val_metrics['f1']:.4f} "
            f"(Δ={result['difference_vs_project_plain_unet']['f1']:+.4f})."
        )
    else:
        print(
            f"  NO — F1 changed from {plain['f1']:.4f} "
            f"to {val_metrics['f1']:.4f} "
            f"(Δ={result['difference_vs_project_plain_unet']['f1']:+.4f})."
        )

    print("\nOutputs:")
    print("  Config:", outdir / "run_config.json")
    print("  History:", outdir / "history.csv")
    print("  Metrics:", outdir / "metrics.json")
    print("  Predictions:", pred_dir)


if __name__ == "__main__":
    main()
