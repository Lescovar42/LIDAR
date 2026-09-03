#!/usr/bin/env python3
"""
Landslide4Sense SAM2-UNet Training and Evaluation.

Purpose
-------
Transfer and evaluate the official SAM2-UNet architecture (SAM2 Hiera-Large +
prompt adapters + Receptive Field Block (RFB) + U-Net decoder) on the
Landslide4Sense 2022 public benchmark.

Key Features & Method
---------------------
1. Multi-Spectral Input Adapter:
   - Landslide4Sense contains 14 input channels (Sentinel-2 multispectral bands,
     ALOS PALSAR SAR, SRTM DEM, and slope).
   - SAM2 backbone requires 3-channel input.
   - A learnable 14 -> 3 channel 1x1 conv adapter projects all 14 channels into
     a 3-channel feature space.
   - By default, it is initialized with Sentinel-2 natural optical RGB
     (B4=Red -> ch 0, B3=Green -> ch 1, B2=Blue -> ch 2), providing an immediate
     physical inductive bias for the ImageNet/SA-1B pretrained SAM2 backbone,
     while allowing gradients to backpropagate into all 14 channels.

2. Architecture:
   - Official WZH0120 / SAM2-UNet (SAM2 Hiera-Large backbone).
   - Frozen Hiera-L vision trunk (~212M parameters).
   - Trainable prompt adapters inside attention blocks (~1.71M parameters).
   - Trainable RFB multiscale modules, U-Net decoder, and side supervision heads
     (~2.67M parameters).
   - Total trainable parameters: ~4.38M (parameter-efficient transfer).

3. Loss Function:
   - Multi-scale deep supervision across 3 heads: final output + 2 side heads.
   - Structure loss combining boundary-aware weighted Binary Cross Entropy (wBCE)
     and weighted Intersection-over-Union (wIoU).
   - Handles void / nodata pixels (class index 255) by masking them out of loss
     and metrics.

4. Optimization & Hyperparameters:
   - Two-tier learning rate:
       * task_lr (input adapter + decoder + RFB): 3e-4
       * sam_adapter_lr (SAM prompt adapters): 1e-4
   - AdamW optimizer with weight decay 1e-4.
   - Linear warmup (default 300 steps) followed by cosine learning rate decay.
   - Gradient clipping norm 1.0.
   - Automatic Mixed Precision (AMP fp16) enabled by default (uses ~2.2 GB VRAM
     at batch size 8 on 128x128 patches).
   - Default microbatch 8 with accumulation 4 -> effective batch size 32
     (exact match to Landslide4Sense baseline batch size).

5. Evaluation & Protocols:
   - Globally pooled landslide-class Precision, Recall, F1, IoU, and OA.
   - Evaluates at default decision boundary (threshold 0.50 / logit 0.0) for
     clean benchmark comparison, and sweeps validation thresholds [0.30 .. 0.70].
   - Supports both checkpoint selection protocols:
       * `--protocol reported-validation`: selects checkpoints on validation F1
         (recommended for research).
       * `--protocol literal-source`: selects checkpoints on training set F1
         (mirrors the official Train.py evaluation quirk).
   - Saves prediction masks as HDF5 files matching official Landslide4Sense format
     (`mask_*.h5` with key `mask`) for use with `compare_landslide4sense_reproduction.py`.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Fixed Landslide4Sense official per-channel normalization constants (14 bands).
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

PROJECT_LITERAL_UNET_REFERENCE = {
    "precision": 0.5449144372137864,
    "recall": 0.6577496545203287,
    "f1": 0.5960388861426874,
    "iou": 0.424540879558343,
}

PROJECT_EFFICIENTNETV2_REFERENCE = {
    "architecture": "U-Net + ImageNet Pretrained EfficientNetV2-S",
    "note": "Reference point from train_landslide4sense_efficientnetv2.py",
}

def _find_sam2_repo() -> Path:
    for candidate in [
        Path(__file__).resolve().parents[2] / "third_party" / "SAM2-UNet",
        Path(__file__).resolve().parents[1] / "third_party" / "SAM2-UNet",
        Path(__file__).resolve().parent / "third_party" / "SAM2-UNet",
    ]:
        if candidate.exists():
            return candidate
    return Path(__file__).resolve().parents[2] / "third_party" / "SAM2-UNet"


DEFAULT_SAM2_REPO = _find_sam2_repo()
DEFAULT_HIERA_PATH = DEFAULT_SAM2_REPO / "sam2_hiera_large.pt"


@dataclass(frozen=True)
class Sample:
    image: Path
    mask: Path
    name: str


def numeric_suffix(path: Path) -> tuple[int, str]:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return (int(digits) if digits else 10**12, path.name)


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
        raise ValueError(f"Unknown split: {split}")

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
        raise FileNotFoundError(f"No samples found for {split} under {root}")

    return samples


def read_h5(path: Path, preferred_key: str) -> np.ndarray:
    with h5py.File(path, "r") as hf:
        if preferred_key in hf:
            return hf[preferred_key][:]
        keys = list(hf.keys())
        if len(keys) == 1:
            return hf[keys[0]][:]
        raise KeyError(f"{path}: expected key {preferred_key!r}; available keys={keys}")


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
                n_repeat = int(np.ceil(repeat_to_at_least / n))
                extra_stop = repeat_to_at_least - n_repeat * n
                self.samples = samples * n_repeat + samples[:extra_stop]
            else:
                n_repeat = int(math.ceil(repeat_to_at_least / n))
                self.samples = (samples * n_repeat)[:repeat_to_at_least]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
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

        x = (x - OFFICIAL_MEAN[:, None, None]) / OFFICIAL_STD[:, None, None]

        return (
            torch.from_numpy(x.copy()),
            torch.from_numpy(y.astype(np.int64, copy=False)),
            s.name,
        )


def import_official_sam2unet(repo_dir: Path):
    repo_dir = repo_dir.resolve()
    if not (repo_dir / "SAM2UNet.py").exists():
        raise FileNotFoundError(f"Missing SAM2UNet.py in {repo_dir}")
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    module = importlib.import_module("SAM2UNet")
    return module.SAM2UNet


class Landslide4SenseInputAdapter(nn.Module):
    """
    Projects 14-channel Landslide4Sense multispectral input to 3 channels for SAM2.

    Initialization Modes:
    - rgb: Maps Sentinel-2 B4 (Red, idx 3), B3 (Green, idx 2), B2 (Blue, idx 1)
           to the 3 output channels with weight 1.0, initializing other channels to 0.
           Provides instant visual features for the pretrained SAM2 backbone.
    - uniform: Initializes each output channel with uniform weights (1 / in_channels).
    - kaiming: Kaiming normal initialization.
    """
    def __init__(self, in_channels: int = 14, init_mode: str = "rgb"):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        self.init_mode = init_mode

        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            if init_mode == "rgb":
                # Sentinel-2: B4=Red (idx 3), B3=Green (idx 2), B2=Blue (idx 1)
                self.proj.weight[0, 3, 0, 0] = 1.0
                self.proj.weight[1, 2, 0, 0] = 1.0
                self.proj.weight[2, 1, 0, 0] = 1.0
            elif init_mode == "uniform":
                self.proj.weight.fill_(1.0 / in_channels)
            elif init_mode == "kaiming":
                nn.init.kaiming_normal_(self.proj.weight, mode="fan_out", nonlinearity="relu")
            else:
                raise ValueError(f"Unknown init_mode: {init_mode}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


class Landslide4SenseSAM2UNet(nn.Module):
    """
    Full model wrapper:
    Landslide4Sense 14-channel input -> Input Adapter -> SAM2-UNet -> 3-scale deep supervision outputs.
    """
    def __init__(
        self,
        sam2unet_class,
        hiera_checkpoint: Path,
        in_channels: int = 14,
        init_adapter: str = "rgb",
    ):
        super().__init__()
        self.input_adapter = Landslide4SenseInputAdapter(in_channels=in_channels, init_mode=init_adapter)
        self.sam2unet = sam2unet_class(str(hiera_checkpoint))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x3 = self.input_adapter(x)
        return self.sam2unet(x3)


def final_logits(outputs: torch.Tensor | Sequence[torch.Tensor]) -> torch.Tensor:
    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


def structure_loss(pred: torch.Tensor, mask: torch.Tensor, valid_mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Official SAM2-UNet structure loss (boundary-weighted BCE + weighted IoU).
    Adapted to support an optional valid pixel mask (to ignore void label 255).
    """
    if valid_mask is not None:
        target = mask.clone()
        target[~valid_mask] = 0.0
    else:
        target = mask
        valid_mask = torch.ones_like(mask, dtype=torch.bool)

    weight = 1 + 5 * torch.abs(F.avg_pool2d(target, 31, stride=1, padding=15) - target)
    weight = weight * valid_mask.float()

    wbce = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    weight_sum = weight.sum(dim=(2, 3)) + 1e-8
    wbce = (weight * wbce).sum(dim=(2, 3)) / weight_sum

    prob = torch.sigmoid(pred) * valid_mask.float()
    inter = ((prob * target) * weight).sum(dim=(2, 3))
    union = ((prob + target) * weight).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1 + 1e-8)
    return (wbce + wiou).mean()


def deep_supervision_loss(
    outputs: Sequence[torch.Tensor],
    target: torch.Tensor,
    valid_mask: torch.Tensor | None = None,
    loss_family: str = "structure",
) -> torch.Tensor:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise RuntimeError(f"Expected official SAM2-UNet to return 3 outputs, got {type(outputs)}")

    if loss_family == "structure":
        return sum(structure_loss(out, target, valid_mask) for out in outputs)
    elif loss_family == "bce":
        losses = []
        for out in outputs:
            bce = F.binary_cross_entropy_with_logits(out, target, reduction="none")
            if valid_mask is not None:
                losses.append((bce * valid_mask.float()).sum() / (valid_mask.float().sum() + 1e-8))
            else:
                losses.append(bce.mean())
        return sum(losses)
    else:
        raise ValueError(f"Unknown loss family: {loss_family}")


def pooled_metrics(tp: int, fp: int, fn: int, tn: int) -> dict[str, float | int]:
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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_predictions: Path | None = None,
    use_amp: bool = False,
    threshold: float = 0.5,
) -> dict[str, float | int]:
    model.eval()
    tp = fp = fn = tn = 0

    if save_predictions is not None:
        save_predictions.mkdir(parents=True, exist_ok=True)

    for batch_idx, (images, labels, names) in enumerate(loader, start=1):
        images = images.float().to(device, non_blocking=True)
        labels_np = labels.numpy()

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                logits = final_logits(outputs)
        else:
            outputs = model(images)
            logits = final_logits(outputs)

        if logits.shape[-2:] != (128, 128):
            logits = F.interpolate(logits, size=(128, 128), mode="bilinear", align_corners=False)

        probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()
        pred1 = probs >= threshold

        valid = (labels_np >= 0) & (labels_np < 2)
        truth1 = labels_np == 1

        tp += int(np.logical_and(pred1, truth1 & valid).sum())
        fp += int(np.logical_and(pred1, (~truth1) & valid).sum())
        fn += int(np.logical_and(~pred1, truth1 & valid).sum())
        tn += int(np.logical_and(~pred1, (~truth1) & valid).sum())

        if save_predictions is not None:
            for j, name in enumerate(names):
                mask_name = Path(name).name.replace("image_", "mask_", 1)
                with h5py.File(save_predictions / mask_name, "w") as hf:
                    hf.create_dataset("mask", data=pred1[j].astype(np.uint8))

        if batch_idx % 50 == 0 or batch_idx == len(loader):
            print(f"  eval {batch_idx}/{len(loader)}")

    metrics = pooled_metrics(tp, fp, fn, tn)
    metrics["threshold"] = float(threshold)
    return metrics


@torch.no_grad()
def run_threshold_sweep(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    thresholds: Sequence[float],
    use_amp: bool = False,
) -> list[dict]:
    model.eval()
    counts = {float(t): {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for t in thresholds}

    for images, labels, _ in loader:
        images = images.float().to(device, non_blocking=True)
        labels_np = labels.numpy()
        valid = (labels_np >= 0) & (labels_np < 2)
        truth1 = labels_np == 1

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = final_logits(model(images))
        else:
            logits = final_logits(model(images))

        probs = torch.sigmoid(logits.squeeze(1)).cpu().numpy()

        for t in thresholds:
            tf = float(t)
            pred1 = probs >= tf
            c = counts[tf]
            c["tp"] += int(np.logical_and(pred1, truth1 & valid).sum())
            c["fp"] += int(np.logical_and(pred1, (~truth1) & valid).sum())
            c["fn"] += int(np.logical_and(~pred1, truth1 & valid).sum())
            c["tn"] += int(np.logical_and(~pred1, (~truth1) & valid).sum())

    results = []
    for t in thresholds:
        tf = float(t)
        m = pooled_metrics(**counts[tf])
        m["threshold"] = tf
        results.append(m)
    return results


def make_optimizer(
    model: nn.Module,
    sam_adapter_lr: float,
    task_lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    sam_params = []
    task_params = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "sam2unet.encoder.blocks" in name and "prompt_learn" in name:
            sam_params.append(p)
        else:
            task_params.append(p)

    if not sam_params:
        raise RuntimeError("No trainable SAM prompt-adapter params found")
    if not task_params:
        raise RuntimeError("No task-specific trainable params found")

    optim = torch.optim.AdamW([
        {"params": sam_params, "lr": sam_adapter_lr, "name": "sam_prompt_adapters"},
        {"params": task_params, "lr": task_lr, "name": "input_adapter_and_decoder"},
    ], weight_decay=weight_decay)

    return optim


def make_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float = 0.01,
):
    def lr_lambda(current_step: int) -> float:
        step = current_step + 1
        if warmup_steps > 0 and step <= warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def set_seed(seed: int | None):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args():
    p = argparse.ArgumentParser(
        description="Landslide4Sense SAM2-UNet training and evaluation script."
    )

    p.add_argument(
        "--data-root",
        type=Path,
        default=Path(r"F:\LIDAR\public_benchmarks\Landslide4sense"),
        help="Path to Landslide4Sense dataset directory.",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for checkpoints, metrics, and prediction maps.",
    )
    p.add_argument(
        "--sam2unet-repo",
        type=Path,
        default=DEFAULT_SAM2_REPO,
        help="Path to third_party/SAM2-UNet repository directory.",
    )
    p.add_argument(
        "--hiera-path",
        type=Path,
        default=DEFAULT_HIERA_PATH,
        help="Path to pretrained sam2_hiera_large.pt checkpoint.",
    )
    p.add_argument(
        "--protocol",
        choices=("reported-validation", "literal-source"),
        default="reported-validation",
        help=(
            "reported-validation: checkpoint selection on true validation F1 (recommended). "
            "literal-source: checkpoint selection on training set F1 (mirrors official Train.py quirk)."
        ),
    )
    p.add_argument(
        "--init-adapter",
        choices=("rgb", "uniform", "kaiming"),
        default="rgb",
        help="Initialization mode for 14->3 channel input adapter (default: rgb).",
    )
    p.add_argument(
        "--loss",
        choices=("structure", "bce"),
        default="structure",
        help="Loss family: 'structure' (official weighted BCE + weighted IoU) or 'bce'.",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=5000,
        help="Total optimizer update steps (default: 5000, baseline training budget).",
    )
    p.add_argument(
        "--eval-every",
        type=int,
        default=500,
        help="Evaluation interval in optimizer steps (default: 500).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Micro-batch size per forward step (default: 8).",
    )
    p.add_argument(
        "--accumulation-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4 -> effective batch size 32).",
    )
    p.add_argument(
        "--task-lr",
        type=float,
        default=3e-4,
        help="Learning rate for input adapter and U-Net decoder (default: 3e-4).",
    )
    p.add_argument(
        "--sam-adapter-lr",
        type=float,
        default=1e-4,
        help="Learning rate for SAM prompt adapters (default: 1e-4).",
    )
    p.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay (default: 1e-4).",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=300,
        help="Linear warmup steps for learning rate scheduler (default: 300).",
    )
    p.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.01,
        help="Minimum LR ratio for cosine decay (default: 0.01).",
    )
    p.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping maximum norm (default: 1.0).",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.50,
        help="Primary evaluation threshold for binary prediction (default: 0.50).",
    )
    p.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers (default: 4).",
    )
    p.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Hardware device to use (default: auto).",
    )
    p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable Automatic Mixed Precision (AMP) on CUDA.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for repeatability (default: 42).",
    )
    p.add_argument(
        "--literal-repeat-quirk",
        action="store_true",
        help="Mirror the original Landslide4Sense repeat indexing expression.",
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

    use_amp = bool((not args.no_amp) and device.type == "cuda")
    set_seed(args.seed)

    if device.type == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    root = args.data_root.resolve()
    outdir = args.outdir.resolve()
    sam_repo = args.sam2unet_repo.resolve()
    hiera_path = args.hiera_path.resolve()

    if not root.exists():
        raise FileNotFoundError(f"Data root does not exist: {root}")
    if not hiera_path.exists():
        raise FileNotFoundError(f"SAM2 Hiera checkpoint not found: {hiera_path}")

    ckpt_dir = outdir / "checkpoints"
    pred_dir = outdir / "validation_predictions"
    outdir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" LANDSLIDE4SENSE SAM2-UNET EXPERIMENT")
    print("=" * 70)
    print(f"Data root:        {root}")
    print(f"Output directory: {outdir}")
    print(f"SAM2-UNet repo:   {sam_repo}")
    print(f"Hiera checkpoint: {hiera_path}")
    print(f"Protocol:         {args.protocol}")
    print(f"Device:           {device} ({torch.cuda.get_device_name(device) if device.type == 'cuda' else 'CPU'})")
    print(f"AMP enabled:      {use_amp}")
    print(f"Input adapter:    14 -> 3 channels (init mode: {args.init_adapter})")
    print(f"Loss function:    3-scale deep supervision {args.loss} loss")
    print(f"Optimizer budget: {args.steps} steps (eval every {args.eval_every} steps)")
    effective_batch_size = args.batch_size * args.accumulation_steps
    print(f"Batch config:     micro={args.batch_size}, accum={args.accumulation_steps} -> effective={effective_batch_size}")
    print(f"Learning rates:   task={args.task_lr:.2e}, sam_adapter={args.sam_adapter_lr:.2e}")
    print("=" * 70)

    train_samples = discover_split(root, "train")
    val_samples = discover_split(root, "validation")
    print(f"Discovered train samples: {len(train_samples)}")
    print(f"Discovered validation samples: {len(val_samples)}")

    if len(train_samples) != 3799:
        print(f"WARNING: expected 3799 official training samples, found {len(train_samples)}")
    if len(val_samples) != 245:
        print(f"WARNING: expected 245 official validation samples, found {len(val_samples)}")

    required_samples = args.steps * effective_batch_size
    train_ds = L4SDataset(
        train_samples,
        repeat_to_at_least=required_samples,
        literal_repeat_quirk=args.literal_repeat_quirk,
    )

    select_samples = train_samples if args.protocol == "literal-source" else val_samples
    select_ds = L4SDataset(select_samples)
    val_ds = L4SDataset(val_samples)

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
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin,
        persistent_workers=args.num_workers > 0,
    )

    print("\nLoading SAM2-UNet module and checkpoint...")
    SAM2UNetClass = import_official_sam2unet(sam_repo)
    model = Landslide4SenseSAM2UNet(
        sam2unet_class=SAM2UNetClass,
        hiera_checkpoint=hiera_path,
        in_channels=14,
        init_adapter=args.init_adapter,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sam_prompt_params = sum(
        p.numel() for n, p in model.named_parameters()
        if p.requires_grad and "sam2unet.encoder.blocks" in n and "prompt_learn" in n
    )
    task_params = trainable_params - sam_prompt_params

    print(f"Total parameters:      {total_params:,}")
    print(f"Trainable parameters:  {trainable_params:,} ({100.0 * trainable_params / total_params:.2f}%)")
    print(f"  SAM prompt adapters: {sam_prompt_params:,}")
    print(f"  Adapter & decoder:   {task_params:,}")

    optimizer = make_optimizer(
        model=model,
        sam_adapter_lr=args.sam_adapter_lr,
        task_lr=args.task_lr,
        weight_decay=args.weight_decay,
    )

    scheduler = make_scheduler(
        optimizer=optimizer,
        total_steps=args.steps,
        warmup_steps=args.warmup_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    run_config = {
        "experiment": "landslide4sense_sam2unet",
        "goal": "evaluate official SAM2-UNet transfer performance on Landslide4Sense 14-band benchmark",
        "data_root": str(root),
        "protocol": args.protocol,
        "train_count": len(train_samples),
        "validation_count": len(val_samples),
        "input_channels": 14,
        "input_size": [128, 128],
        "normalization": "official_fixed_landslide4sense_mean_std",
        "sam2unet_repo": str(sam_repo),
        "hiera_checkpoint": str(hiera_path),
        "architecture": "Official SAM2-UNet Hiera-L + Learnable 14->3ch Input Adapter",
        "input_adapter_init": args.init_adapter,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "sam_prompt_parameters": sam_prompt_params,
        "task_adapter_decoder_parameters": task_params,
        "loss": f"3-scale deep supervision {args.loss} loss",
        "optimizer": "AdamW",
        "task_lr": args.task_lr,
        "sam_adapter_lr": args.sam_adapter_lr,
        "weight_decay": args.weight_decay,
        "scheduler": f"linear warmup ({args.warmup_steps} steps) + cosine decay (min_ratio={args.min_lr_ratio})",
        "grad_clip_norm": args.grad_clip,
        "optimizer_steps": args.steps,
        "eval_every_optimizer_steps": args.eval_every,
        "micro_batch_size": args.batch_size,
        "gradient_accumulation_steps": args.accumulation_steps,
        "effective_batch_size": effective_batch_size,
        "default_threshold": args.threshold,
        "amp": use_amp,
        "seed": args.seed,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "reported_official_reference": REPORTED_REFERENCE,
        "project_plain_unet_literal_reference": PROJECT_LITERAL_UNET_REFERENCE,
    }

    (outdir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

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

    print("\nBeginning training...")
    for micro_idx, (images, labels, _) in enumerate(train_loader, start=1):
        images = images.float().to(device, non_blocking=True)
        # Binary target tensor of shape (B, 1, H, W)
        target = (labels == 1).unsqueeze(1).float().to(device, non_blocking=True)
        valid_mask = ((labels >= 0) & (labels < 2)).unsqueeze(1).to(device, non_blocking=True)

        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss_raw = deep_supervision_loss(
                    outputs=outputs,
                    target=target,
                    valid_mask=valid_mask,
                    loss_family=args.loss,
                )
                loss = loss_raw / args.accumulation_steps
            scaler.scale(loss).backward()
        else:
            outputs = model(images)
            loss_raw = deep_supervision_loss(
                outputs=outputs,
                target=target,
                valid_mask=valid_mask,
                loss_family=args.loss,
            )
            loss = loss_raw / args.accumulation_steps
            loss.backward()

        running_loss += float(loss_raw.item())
        running_micro += 1
        micro_since_step += 1

        if micro_since_step < args.accumulation_steps:
            continue

        if use_amp:
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip)
            optimizer.step()

        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        optimizer_step += 1
        micro_since_step = 0

        mean_recent_loss = running_loss / max(1, running_micro)
        current_lrs = [group["lr"] for group in optimizer.param_groups]

        record = {
            "optimizer_step": optimizer_step,
            "train_loss_recent": mean_recent_loss,
            "lr_sam_adapter": current_lrs[0] if len(current_lrs) > 0 else None,
            "lr_task": current_lrs[1] if len(current_lrs) > 1 else current_lrs[0],
            "elapsed_seconds": time.time() - t_start,
        }

        if optimizer_step % 10 == 0:
            print(
                f"Step {optimizer_step:04d}/{args.steps} | "
                f"loss={mean_recent_loss:.4f} | "
                f"lr_task={record['lr_task']:.2e} | "
                f"lr_sam={record['lr_sam_adapter']:.2e} | "
                f"elapsed={(time.time() - t_start) / 3600:.2f} h"
            )

        running_loss = 0.0
        running_micro = 0

        if optimizer_step % args.eval_every == 0:
            print(f"\nCheckpoint selection evaluation at step {optimizer_step} ({args.protocol})...")
            select_metrics = evaluate(
                model=model,
                loader=select_loader,
                device=device,
                save_predictions=None,
                use_amp=use_amp,
                threshold=args.threshold,
            )

            for k, v in select_metrics.items():
                record[f"select_{k}"] = v

            print(
                f"  Select Metrics (th={args.threshold:.2f}): "
                f"P={select_metrics['precision']:.4f} | "
                f"R={select_metrics['recall']:.4f} | "
                f"F1={select_metrics['f1']:.4f} | "
                f"IoU={select_metrics['iou']:.4f}"
            )

            if select_metrics["f1"] > best_f1:
                best_f1 = float(select_metrics["f1"])
                best_checkpoint = ckpt_dir / f"step{optimizer_step}_selectF1_{int(best_f1 * 10000):04d}.pt"
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_step": optimizer_step,
                        "selection_metrics": select_metrics,
                        "run_config": run_config,
                    },
                    best_checkpoint,
                )
                print(f"  [*] New best checkpoint saved: {best_checkpoint.name} (F1={best_f1:.4f})")

            model.train()

        history.append(record)
        if optimizer_step >= args.steps:
            break

    if not history:
        raise RuntimeError("No optimizer steps were completed")

    # Write history CSV.
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

    print("\n" + "=" * 70)
    print(f"Loading best checkpoint: {best_checkpoint.name}")
    print("=" * 70)
    ckpt = torch.load(best_checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\nRunning final evaluation on TRUE validation annotations...")
    val_metrics_default = evaluate(
        model=model,
        loader=val_loader,
        device=device,
        save_predictions=pred_dir,
        use_amp=use_amp,
        threshold=args.threshold,
    )

    # Perform threshold sweep on validation split.
    thresholds = [round(t, 2) for t in np.arange(0.30, 0.75, 0.05)]
    sweep_results = run_threshold_sweep(
        model=model,
        loader=val_loader,
        device=device,
        thresholds=thresholds,
        use_amp=use_amp,
    )

    # Save threshold sweep table.
    with (outdir / "validation_threshold_sweep.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["threshold", "precision", "recall", "f1", "iou", "overall_accuracy", "tp", "fp", "fn", "tn"])
        writer.writeheader()
        writer.writerows(sweep_results)

    best_sweep = max(sweep_results, key=lambda r: r["f1"])
    print("\nValidation Threshold Sweep Summary:")
    for r in sweep_results:
        star = " [*]" if r["threshold"] == best_sweep["threshold"] else ""
        print(f"  th={r['threshold']:.2f} | P={r['precision']:.4f} | R={r['recall']:.4f} | F1={r['f1']:.4f} | IoU={r['iou']:.4f}{star}")

    plain = PROJECT_LITERAL_UNET_REFERENCE
    official = REPORTED_REFERENCE

    result = {
        "selected_checkpoint": str(best_checkpoint),
        "selected_optimizer_step": ckpt.get("optimizer_step"),
        "checkpoint_selection_protocol": args.protocol,
        "validation_at_default_threshold": val_metrics_default,
        "validation_at_best_threshold": best_sweep,
        "reported_official_reference": official,
        "project_plain_unet_literal_reference": plain,
        "difference_vs_project_plain_unet": {
            "precision": val_metrics_default["precision"] - plain["precision"],
            "recall": val_metrics_default["recall"] - plain["recall"],
            "f1": val_metrics_default["f1"] - plain["f1"],
            "iou": val_metrics_default["iou"] - plain["iou"],
        },
        "difference_vs_reported_official": {
            "precision": val_metrics_default["precision"] - official["precision"],
            "recall": val_metrics_default["recall"] - official["recall"],
            "f1": val_metrics_default["f1"] - official["f1"],
        },
        "sam2unet_improves_project_plain_unet_f1": val_metrics_default["f1"] > plain["f1"],
        "sam2unet_improves_reported_official_f1": val_metrics_default["f1"] > official["f1"],
        "prediction_dir": str(pred_dir),
    }

    (outdir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(" SAM2-UNET LANDSLIDE4SENSE BENCHMARK RESULTS")
    print("=" * 70)
    print(f"Official Baseline (reported):   P={official['precision']:.4f}  R={official['recall']:.4f}  F1={official['f1']:.4f}")
    print(f"Project Plain U-Net (repro):     P={plain['precision']:.4f}  R={plain['recall']:.4f}  F1={plain['f1']:.4f}  IoU={plain['iou']:.4f}")
    print(f"SAM2-UNet (threshold={args.threshold:.2f}):      P={val_metrics_default['precision']:.4f}  R={val_metrics_default['recall']:.4f}  F1={val_metrics_default['f1']:.4f}  IoU={val_metrics_default['iou']:.4f}")
    print(f"SAM2-UNet (best th={best_sweep['threshold']:.2f}):       P={best_sweep['precision']:.4f}  R={best_sweep['recall']:.4f}  F1={best_sweep['f1']:.4f}  IoU={best_sweep['iou']:.4f}")
    print("-" * 70)
    f1_delta = val_metrics_default["f1"] - plain["f1"]
    if f1_delta > 0:
        print(f"Verdict: YES -- SAM2-UNet improved F1 over plain U-Net from {plain['f1']:.4f} to {val_metrics_default['f1']:.4f} (delta={f1_delta:+.4f})")
    else:
        print(f"Verdict: NO -- SAM2-UNet F1 is {val_metrics_default['f1']:.4f} vs plain U-Net {plain['f1']:.4f} (delta={f1_delta:+.4f})")
    print("=" * 70)
    print("\nOutput Files:")
    print(f"  Config:           {outdir / 'run_config.json'}")
    print(f"  History:          {outdir / 'history.csv'}")
    print(f"  Metrics:          {outdir / 'metrics.json'}")
    print(f"  Threshold sweep:  {outdir / 'validation_threshold_sweep.csv'}")
    print(f"  Predictions:      {pred_dir}")


if __name__ == "__main__":
    main()
