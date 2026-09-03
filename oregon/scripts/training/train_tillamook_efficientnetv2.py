#!/usr/bin/env python3
"""
Tillamook strict-binary EfficientNetV2-S U-Net trainer.

Preserves the frozen Tillamook protocol:
- train split: fitting + train-only normalization
- validation split: checkpoint + threshold selection
- test split: LOCKED / NEVER LOADED

Model:
- smp.Unet with ImageNet-pretrained EfficientNetV2-S encoder
  (tu-tf_efficientnetv2_s.in21k_ft_in1k)
- native 7ch or 3ch terrain input (SMP adapts the pretrained first
  convolution from 3 -> N input channels)
- single-output weighted BCE + weighted IoU structure loss

Recommended settings for RTX 2060 SUPER (8GB) + 64GB RAM:
- 40 epochs
- microbatch 8, gradient accumulation 4 (effective batch 32, as in the
  Landslide4Sense EfficientNetV2-S experiment; NOTE this differs from the
  SAM2-UNet run which used effective batch 4)
- AdamW, lr 3e-4, weight decay 1e-4
- 3 epoch warmup + cosine decay
- grad clipping 1.0
- AMP + channels_last
- RAM-cached patches (whole dataset decodes to ~4.5GB float16)
- native 256x256 patches
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

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

DEFAULT_ENCODER = "tu-tf_efficientnetv2_s.in21k_ft_in1k"


def read_manifest(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def hash_ids(rows):
    payload = "\n".join(sorted(r["patch_id"] for r in rows)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_channels(dataset_dir: Path):
    obj = json.loads((dataset_dir / "channels.json").read_text(encoding="utf-8"))
    channels = obj["feature_names"] if isinstance(obj, dict) else obj
    if channels != CANONICAL_CHANNELS:
        raise ValueError(f"Channel order mismatch. Expected {CANONICAL_CHANNELS}, found {channels}")
    return channels


def select_rows(rows):
    train = [r for r in rows if r.get("split") == "train"]
    val = [r for r in rows if r.get("split") == "validation"]
    test = [r for r in rows if r.get("split") == "test"]
    if not train or not val or not test:
        raise ValueError(f"Expected all frozen splits; got train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def assert_split_isolation(train, val, test):
    ids = {k: {r["patch_id"] for r in v} for k, v in {"train":train, "val":val, "test":test}.items()}
    tiles = {k: {r["tile_name"] for r in v} for k, v in {"train":train, "val":val, "test":test}.items()}
    for a, b in (("train","val"), ("train","test"), ("val","test")):
        if ids[a] & ids[b]:
            raise ValueError(f"Patch leakage between {a} and {b}")
        if tiles[a] & tiles[b]:
            raise ValueError(f"Tile leakage between {a} and {b}")


def compute_channel_statistics(dataset_dir: Path, rows, indices):
    sums = np.zeros(len(indices), dtype=np.float64)
    sums_sq = np.zeros(len(indices), dtype=np.float64)
    count = 0
    for i, row in enumerate(rows, 1):
        with np.load(dataset_dir / row["patch_path"]) as data:
            x = data["features"][indices].astype(np.float64)
        flat = x.reshape(len(indices), -1)
        sums += flat.sum(axis=1)
        sums_sq += np.square(flat).sum(axis=1)
        count += flat.shape[1]
        if i % 250 == 0 or i == len(rows):
            print(f"  normalization: {i}/{len(rows)} train patches")
    mean = sums / count
    var = np.maximum(sums_sq / count - mean**2, 1e-8)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


class PatchDataset(Dataset):
    """Optionally preloads and validates all NPZ patches into RAM (float16)."""

    def __init__(self, dataset_dir, rows, channel_indices, mean, std, cache=True):
        self.dataset_dir = dataset_dir
        self.rows = rows
        self.channel_indices = channel_indices
        self.mean = mean[:, None, None]
        self.std = std[:, None, None]
        self._x_buf = np.empty((len(channel_indices), 256, 256), dtype=np.float32)
        self._y_buf = np.empty((1, 256, 256), dtype=np.float32)
        self._cache = None
        if cache:
            self._cache = [self._load_raw(i) for i in range(len(rows))]
            print(f"  RAM cache: {len(rows)} patches cached")

    def _load_raw(self, index):
        row = self.rows[index]
        with np.load(self.dataset_dir / row["patch_path"]) as data:
            x = data["features"][self.channel_indices]
            y = data["mask"]
        vals = np.unique(y)
        if not np.all(np.isin(vals, [0, 1])):
            raise RuntimeError(f"Strict-binary violation in {row['patch_id']}: {vals.tolist()}")
        return x, y[None]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if self._cache is not None:
            x, y = self._cache[index]
            np.copyto(self._x_buf, x)
            np.copyto(self._y_buf, y)
        else:
            x, y = self._load_raw(index)
            np.copyto(self._x_buf, x)
            np.copyto(self._y_buf, y)
        self._x_buf -= self.mean
        self._x_buf /= self.std
        return torch.from_numpy(self._x_buf), torch.from_numpy(self._y_buf)


def build_model(encoder_name: str, pretrained: bool, in_channels: int):
    """
    U-Net with EfficientNetV2-S encoder.

    For SMP `tu-` encoders:
      encoder_weights=True -> use the pretrained variant specified in encoder_name
      encoder_weights=None -> random initialization
    """
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=True if pretrained else None,
        in_channels=in_channels,
        classes=1,
        activation=None,
    )


def structure_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, 1, 15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weight * wbce).sum(dim=(2,3)) / weight.sum(dim=(2,3))
    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weight).sum(dim=(2,3))
    union = ((prob + mask) * weight).sum(dim=(2,3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def metrics_from_counts(tp, fp, fn, tn):
    def ratio(a, b): return float(a / b) if b else 0.0
    total = tp + fp + fn + tn
    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "dice": ratio(2*tp, 2*tp+fp+fn),
        "iou": ratio(tp, tp+fp+fn),
        "precision": ratio(tp, tp+fp),
        "recall": ratio(tp, tp+fn),
        "specificity": ratio(tn, tn+fp),
        "gt_positive_fraction": ratio(tp+fn, total),
        "predicted_positive_fraction": ratio(tp+fp, total),
    }


@torch.no_grad()
def evaluate_at_threshold(model, loader, device, threshold, use_amp):
    model.eval()
    tp = fp = fn = tn = 0
    loss_sum = 0.0
    batches = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(x)
                loss = structure_loss(logits, y)
        else:
            logits = model(x)
            loss = structure_loss(logits, y)
        loss_sum += float(loss.item()); batches += 1
        pred = torch.sigmoid(logits) >= threshold
        truth = y == 1
        tp += int((pred & truth).sum().item())
        fp += int((pred & ~truth).sum().item())
        fn += int((~pred & truth).sum().item())
        tn += int((~pred & ~truth).sum().item())
    return {"loss": loss_sum/max(1,batches), "threshold": float(threshold), **metrics_from_counts(tp,fp,fn,tn)}


@torch.no_grad()
def threshold_sweep(model, loader, device, thresholds, use_amp):
    model.eval()
    counts = {float(t): {"tp":0,"fp":0,"fn":0,"tn":0} for t in thresholds}
    for bi, (x, y) in enumerate(loader, 1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                probs = torch.sigmoid(model(x))
        else:
            probs = torch.sigmoid(model(x))
        truth = y == 1
        for t in thresholds:
            pred = probs >= float(t)
            c = counts[float(t)]
            c["tp"] += int((pred & truth).sum().item())
            c["fp"] += int((pred & ~truth).sum().item())
            c["fn"] += int((~pred & truth).sum().item())
            c["tn"] += int((~pred & ~truth).sum().item())
        if bi % 50 == 0 or bi == len(loader):
            print(f"  threshold inference: {bi}/{len(loader)}")
    return [{"threshold": float(t), **metrics_from_counts(**counts[float(t)])} for t in thresholds]


def make_optimizer(model, learning_rate, weight_decay):
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable params found")
    return torch.optim.AdamW(trainable, lr=learning_rate, weight_decay=weight_decay)


def make_scheduler(optimizer, epochs, warmup_epochs, min_lr_ratio):
    def fac(epoch_index):
        e = epoch_index + 1
        if warmup_epochs and e <= warmup_epochs:
            return e / warmup_epochs
        progress = (e - warmup_epochs) / max(1, epochs - warmup_epochs)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1-min_lr_ratio)*cosine
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=fac)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--features", choices=("7ch","3ch"), default="7ch")
    p.add_argument("--encoder", default=DEFAULT_ENCODER)
    p.add_argument("--random-init", action="store_true",
                   help="Disable ImageNet pretrained encoder weights.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--accumulation-steps", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=4)
    p.add_argument("--no-cache-ram", action="store_true",
                   help="Stream patches from disk instead of caching decoded arrays in RAM.")
    p.add_argument("--channels-last", action="store_true",
                   help="Use NHWC memory format for model and inputs.")
    p.add_argument("--compile", action="store_true",
                   help="torch.compile the model (experimental on Windows).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("auto","cuda","cpu"), default="auto")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--threshold-min", type=float, default=0.30)
    p.add_argument("--threshold-max", type=float, default=0.90)
    p.add_argument("--threshold-step", type=float, default=0.05)
    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    if device.type == "cuda":
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    use_amp = bool(args.amp and device.type == "cuda")

    dataset_dir = args.dataset_dir.resolve(); outdir = args.outdir.resolve()
    manifest_path = dataset_dir / "patches.csv"
    if not manifest_path.exists(): raise FileNotFoundError(manifest_path)

    rows = read_manifest(manifest_path)
    train_rows, val_rows, test_rows = select_rows(rows)
    assert_split_isolation(train_rows, val_rows, test_rows)
    print(f"Frozen rows: train={len(train_rows)}, validation={len(val_rows)}, test={len(test_rows)} [LOCKED / NOT LOADED]")
    if (len(train_rows), len(val_rows), len(test_rows)) != (3950, 891, 1210):
        raise RuntimeError("Frozen split counts differ from expected Phase 3 QC")

    channels = load_channels(dataset_dir)
    selected_names = FEATURE_SETS[args.features]
    indices = [channels.index(n) for n in selected_names]
    print("Feature set:", args.features, selected_names)
    print("Computing TRAIN-ONLY normalization...")
    mean, std = compute_channel_statistics(dataset_dir, train_rows, indices)

    train_ds = PatchDataset(dataset_dir, train_rows, indices, mean, std, cache=not args.no_cache_ram)
    val_ds = PatchDataset(dataset_dir, val_rows, indices, mean, std, cache=not args.no_cache_ram)
    pin = device.type == "cuda"
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": pin,
                     "prefetch_factor": args.prefetch_factor}
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    pretrained = not args.random_init
    print("Building model...")
    print("Encoder:", args.encoder)
    print("Pretrained:", pretrained)
    print(f"Input channels: {len(selected_names)}")
    model = build_model(args.encoder, pretrained, len(selected_names)).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    raw_model = model
    if args.compile:
        model = torch.compile(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = make_optimizer(model, args.learning_rate, args.weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr_ratio)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    outdir.mkdir(parents=True, exist_ok=True)
    best_path = outdir / "best_model.pt"
    run_config = {
        "experiment":"tillamook_efficientnetv2_s_unet",
        "dataset_dir":str(dataset_dir),
        "ground_truth_policy":"strict_binary_0_1",
        "test_policy":"LOCKED_NOT_LOADED",
        "train_rows":len(train_rows), "validation_rows":len(val_rows), "test_rows_locked":len(test_rows),
        "train_patch_id_sha256":hash_ids(train_rows), "validation_patch_id_sha256":hash_ids(val_rows),
        "architecture":"smp.Unet + ImageNet-pretrained EfficientNetV2-S encoder",
        "encoder_name":args.encoder,
        "encoder_pretrained":pretrained,
        "encoder_pretraining": (
            "ImageNet-21k pretraining + ImageNet-1k fine-tuning"
            if pretrained and "in21k_ft_in1k" in args.encoder
            else "pretrained variant encoded in encoder_name"
            if pretrained
            else "none_random_init"
        ),
        "feature_set":args.features, "channels":selected_names, "channel_indices":indices,
        "total_parameter_count":total_params, "trainable_parameter_count":trainable_params,
        "epochs":args.epochs, "micro_batch_size":args.batch_size,
        "gradient_accumulation_steps":args.accumulation_steps,
        "effective_batch_size":args.batch_size*args.accumulation_steps,
        "optimizer":"AdamW", "learning_rate":args.learning_rate,
        "weight_decay":args.weight_decay, "scheduler":"3-epoch warmup + cosine",
        "warmup_epochs":args.warmup_epochs, "gradient_clip_norm":args.grad_clip,
        "loss":"weighted BCE + weighted IoU structure loss (single output)",
        "global_pos_weight":None,
        "normalization_source":"train_only", "mean":mean.tolist(), "std":std.tolist(),
        "augmentation":"none for first run because terrain stack contains directional channels",
        "cache_ram":not args.no_cache_ram, "channels_last":args.channels_last, "torch_compile":args.compile,
        "seed":args.seed, "amp":use_amp, "device":str(device),
        "device_name":torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version":torch.__version__, "numpy_version":np.__version__,
    }
    (outdir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"Training on {run_config['device_name']} | total params={total_params:,} | trainable={trainable_params:,} | AMP={use_amp}")
    print(f"LR: {args.learning_rate:g} | microbatch={args.batch_size} x accum={args.accumulation_steps}")

    history=[]; best_dice=-1.0; best_epoch=None; t0=time.time()
    for epoch in range(1, args.epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        epoch_loss=0.0; micro_batches=0
        for bi, (x,y) in enumerate(train_loader, 1):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            if args.channels_last:
                x = x.to(memory_format=torch.channels_last)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits=model(x); raw_loss=structure_loss(logits,y); loss=raw_loss/args.accumulation_steps
                scaler.scale(loss).backward()
            else:
                logits=model(x); raw_loss=structure_loss(logits,y); loss=raw_loss/args.accumulation_steps; loss.backward()
            epoch_loss += float(raw_loss.item()); micro_batches += 1
            do_step = (bi % args.accumulation_steps == 0) or (bi == len(train_loader))
            if do_step:
                if use_amp: scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
                if use_amp:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            if bi % 250 == 0 or bi == len(train_loader):
                print(f"  epoch {epoch:02d} batch {bi}/{len(train_loader)} loss={raw_loss.item():.4f}")
        train_loss=epoch_loss/max(1,micro_batches)
        if device.type == "cuda": torch.cuda.empty_cache()
        val=evaluate_at_threshold(model,val_loader,device,0.50,use_amp)
        rec={"epoch":epoch,"train_loss":train_loss,"validation_loss":val["loss"],
             "validation_dice":val["dice"],"validation_iou":val["iou"],
             "validation_precision":val["precision"],"validation_recall":val["recall"],
             "validation_specificity":val["specificity"],
             "validation_gt_positive_fraction":val["gt_positive_fraction"],
             "validation_predicted_positive_fraction":val["predicted_positive_fraction"],
             "lr":optimizer.param_groups[0]["lr"],
             "elapsed_hours":(time.time()-t0)/3600.0}
        history.append(rec)
        print(f"Epoch {epoch:03d}/{args.epochs} | loss {train_loss:.4f} | val Dice@0.50 {val['dice']:.4f} | IoU {val['iou']:.4f} | P {val['precision']:.4f} | R {val['recall']:.4f} | Pred+ {100*val['predicted_positive_fraction']:.2f}% | {rec['elapsed_hours']:.2f}h")
        if val["dice"] > best_dice:
            best_dice=val["dice"]; best_epoch=epoch
            torch.save({"model_state_dict":raw_model.state_dict(),"epoch":epoch,"validation_metrics_at_0_5":val,
                        "run_config":run_config,"channels":selected_names,"channel_indices":indices,"mean":mean,"std":std}, best_path)
            print(f"  Saved new best checkpoint: Dice={best_dice:.4f}")
        scheduler.step()

    with (outdir/"history.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)

    ckpt=torch.load(best_path,map_location=device,weights_only=False); raw_model.load_state_dict(ckpt["model_state_dict"])
    thresholds=[round(float(t),6) for t in np.arange(args.threshold_min,args.threshold_max+args.threshold_step/2,args.threshold_step)]
    print(f"Best epoch={ckpt['epoch']}. Sweeping VALIDATION thresholds only...")
    sweep=threshold_sweep(model,val_loader,device,thresholds,use_amp)
    best_thr=max(sweep,key=lambda x:(x["dice"],x["iou"],x["precision"],-abs(x["threshold"]-0.5)))
    with (outdir/"validation_threshold_sweep.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(sweep[0].keys())); w.writeheader(); w.writerows(sweep)
    metrics={"architecture":"smp.Unet + EfficientNetV2-S encoder","encoder_name":args.encoder,
             "encoder_pretrained":pretrained,"feature_set":args.features,
             "channels":selected_names,"best_epoch":ckpt["epoch"],"validation_at_0_5":ckpt["validation_metrics_at_0_5"],
             "validation_best_threshold":best_thr,"threshold_grid":thresholds,"train_rows":len(train_rows),
             "validation_rows":len(val_rows),"test_rows_locked_not_evaluated":len(test_rows),"test":None}
    (outdir/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print("\nEFFICIENTNETV2 TILLAMOOK RUN COMPLETE — TEST REMAINS LOCKED")
    print(json.dumps(metrics,indent=2))


if __name__ == "__main__":
    main()
