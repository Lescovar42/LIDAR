#!/usr/bin/env python3
"""
Tillamook strict-binary SAM2-UNet trainer.

Preserves the frozen Tillamook protocol:
- train split: fitting + train-only normalization
- validation split: checkpoint + threshold selection
- test split: LOCKED / NEVER LOADED

Model:
- official WZH0120/SAM2-UNet (SAM2 Hiera-L + adapters + RFB + U decoder)
- learnable terrain input adapter: 7ch/3ch -> 3ch for pretrained SAM2
- deep supervision with the official weighted BCE + weighted IoU structure loss

Recommended Tillamook transfer settings (not upstream defaults):
- 40 epochs
- microbatch 1, gradient accumulation 4
- task/input-adapter/decoder LR 3e-4
- SAM prompt-adapter LR 1e-4
- AdamW, weight decay 1e-4
- 3 epoch warmup + cosine decay
- grad clipping 1.0
- AMP
- native 256x256 patches
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
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
    def __init__(self, dataset_dir, rows, channel_indices, mean, std):
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
            x = data["features"][self.channel_indices].astype(np.float32)
            y = data["mask"].astype(np.float32)
        vals = np.unique(y)
        if not np.all(np.isin(vals, [0.0, 1.0])):
            raise RuntimeError(f"Strict-binary violation in {row['patch_id']}: {vals.tolist()}")
        x = (x - self.mean) / self.std
        return torch.from_numpy(x), torch.from_numpy(y[None])


def import_official_sam2unet(repo_dir: Path):
    repo_dir = repo_dir.resolve()
    if not (repo_dir / "SAM2UNet.py").exists():
        raise FileNotFoundError(f"Missing {repo_dir / 'SAM2UNet.py'}")
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))
    module = importlib.import_module("SAM2UNet")
    return module.SAM2UNet


class TerrainInputAdapter(nn.Module):
    def __init__(self, selected_names):
        super().__init__()
        self.proj = nn.Conv2d(len(selected_names), 3, kernel_size=1, bias=True)
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            if len(selected_names) == 3:
                for c in range(3):
                    self.proj.weight[c, c, 0, 0] = 1.0
            else:
                for out_c, name in enumerate(["slope_degrees", "aspect_sin", "aspect_cos"]):
                    idx = selected_names.index(name)
                    self.proj.weight[out_c, idx, 0, 0] = 1.0

    def forward(self, x):
        return self.proj(x)


class TillamookSAM2UNet(nn.Module):
    def __init__(self, SAM2UNetClass, hiera_path: Path, selected_names):
        super().__init__()
        self.input_adapter = TerrainInputAdapter(selected_names)
        self.sam2unet = SAM2UNetClass(str(hiera_path))

    def forward(self, x):
        return self.sam2unet(self.input_adapter(x))


def structure_loss(pred, mask):
    weight = 1 + 5 * torch.abs(F.avg_pool2d(mask, 31, 1, 15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weight * wbce).sum(dim=(2,3)) / weight.sum(dim=(2,3))
    prob = torch.sigmoid(pred)
    inter = ((prob * mask) * weight).sum(dim=(2,3))
    union = ((prob + mask) * weight).sum(dim=(2,3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def deep_supervision_loss(outputs, target):
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise RuntimeError("Expected official SAM2-UNet to return 3 outputs")
    return sum(structure_loss(out, target) for out in outputs)


def final_logits(outputs):
    return outputs[0] if isinstance(outputs, (tuple, list)) else outputs


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
                outputs = model(x)
                loss = deep_supervision_loss(outputs, y)
                logits = final_logits(outputs)
        else:
            outputs = model(x)
            loss = deep_supervision_loss(outputs, y)
            logits = final_logits(outputs)
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
                logits = final_logits(model(x))
        else:
            logits = final_logits(model(x))
        probs = torch.sigmoid(logits)
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


def make_optimizer(model, sam_adapter_lr, task_lr, weight_decay):
    sam_params, task_params = [], []
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
        {"params": task_params, "lr": task_lr, "name": "input_adapter_decoder"},
    ], weight_decay=weight_decay)
    return optim


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
    p.add_argument("--sam2unet-repo", type=Path, required=True)
    p.add_argument("--hiera-path", type=Path, required=True)
    p.add_argument("--features", choices=("7ch","3ch"), default="7ch")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--accumulation-steps", type=int, default=4)
    p.add_argument("--task-lr", type=float, default=3e-4)
    p.add_argument("--sam-adapter-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
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
    use_amp = bool(args.amp and device.type == "cuda")

    dataset_dir = args.dataset_dir.resolve(); outdir = args.outdir.resolve()
    sam_repo = args.sam2unet_repo.resolve(); hiera_path = args.hiera_path.resolve()
    manifest_path = dataset_dir / "patches.csv"
    if not manifest_path.exists(): raise FileNotFoundError(manifest_path)
    if not hiera_path.exists(): raise FileNotFoundError(hiera_path)

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

    train_ds = PatchDataset(dataset_dir, train_rows, indices, mean, std)
    val_ds = PatchDataset(dataset_dir, val_rows, indices, mean, std)
    pin = device.type == "cuda"
    loader_kwargs = {"num_workers": args.num_workers, "pin_memory": pin}
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kwargs)

    SAM2UNetClass = import_official_sam2unet(sam_repo)
    old_cwd = Path.cwd()
    try:
        os.chdir(sam_repo)
        model = TillamookSAM2UNet(SAM2UNetClass, hiera_path, selected_names)
    finally:
        os.chdir(old_cwd)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = make_optimizer(model, args.sam_adapter_lr, args.task_lr, args.weight_decay)
    scheduler = make_scheduler(optimizer, args.epochs, args.warmup_epochs, args.min_lr_ratio)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    outdir.mkdir(parents=True, exist_ok=True)
    best_path = outdir / "best_model.pt"
    run_config = {
        "experiment":"tillamook_sam2unet",
        "dataset_dir":str(dataset_dir),
        "ground_truth_policy":"strict_binary_0_1",
        "test_policy":"LOCKED_NOT_LOADED",
        "train_rows":len(train_rows), "validation_rows":len(val_rows), "test_rows_locked":len(test_rows),
        "train_patch_id_sha256":hash_ids(train_rows), "validation_patch_id_sha256":hash_ids(val_rows),
        "architecture":"Official SAM2-UNet Hiera-L + learnable terrain input adapter",
        "sam2unet_repo":str(sam_repo), "hiera_checkpoint":str(hiera_path),
        "sam_version_required":"original SAM2, not SAM2.1",
        "feature_set":args.features, "channels":selected_names, "channel_indices":indices,
        "input_adapter":f"{len(selected_names)}ch->3ch 1x1 conv initialized from slope/aspect",
        "total_parameter_count":total_params, "trainable_parameter_count":trainable_params,
        "epochs":args.epochs, "micro_batch_size":args.batch_size,
        "gradient_accumulation_steps":args.accumulation_steps,
        "effective_batch_size":args.batch_size*args.accumulation_steps,
        "optimizer":"AdamW", "task_lr":args.task_lr, "sam_adapter_lr":args.sam_adapter_lr,
        "weight_decay":args.weight_decay, "scheduler":"3-epoch warmup + cosine",
        "warmup_epochs":args.warmup_epochs, "gradient_clip_norm":args.grad_clip,
        "loss":"SAM2-UNet weighted BCE + weighted IoU with 3-output deep supervision",
        "global_pos_weight":None,
        "normalization_source":"train_only", "mean":mean.tolist(), "std":std.tolist(),
        "augmentation":"none for first run because terrain stack contains directional channels",
        "seed":args.seed, "amp":use_amp, "device":str(device),
        "device_name":torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch_version":torch.__version__, "numpy_version":np.__version__,
    }
    (outdir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"Training on {run_config['device_name']} | total params={total_params:,} | trainable={trainable_params:,} | AMP={use_amp}")
    print(f"LR: task={args.task_lr:g}, SAM adapters={args.sam_adapter_lr:g} | microbatch={args.batch_size} x accum={args.accumulation_steps}")

    history=[]; best_dice=-1.0; best_epoch=None; t0=time.time()
    for epoch in range(1, args.epochs+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        epoch_loss=0.0; micro_batches=0
        for bi, (x,y) in enumerate(train_loader, 1):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    outputs=model(x); raw_loss=deep_supervision_loss(outputs,y); loss=raw_loss/args.accumulation_steps
                scaler.scale(loss).backward()
            else:
                outputs=model(x); raw_loss=deep_supervision_loss(outputs,y); loss=raw_loss/args.accumulation_steps; loss.backward()
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
             "lr_sam_adapter":optimizer.param_groups[0]["lr"],"lr_task":optimizer.param_groups[1]["lr"],
             "elapsed_hours":(time.time()-t0)/3600.0}
        history.append(rec)
        print(f"Epoch {epoch:03d}/{args.epochs} | loss {train_loss:.4f} | val Dice@0.50 {val['dice']:.4f} | IoU {val['iou']:.4f} | P {val['precision']:.4f} | R {val['recall']:.4f} | Pred+ {100*val['predicted_positive_fraction']:.2f}% | {rec['elapsed_hours']:.2f}h")
        if val["dice"] > best_dice:
            best_dice=val["dice"]; best_epoch=epoch
            torch.save({"model_state_dict":model.state_dict(),"epoch":epoch,"validation_metrics_at_0_5":val,
                        "run_config":run_config,"channels":selected_names,"channel_indices":indices,"mean":mean,"std":std}, best_path)
            print(f"  Saved new best checkpoint: Dice={best_dice:.4f}")
        scheduler.step()

    with (outdir/"history.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(history[0].keys())); w.writeheader(); w.writerows(history)

    ckpt=torch.load(best_path,map_location=device,weights_only=False); model.load_state_dict(ckpt["model_state_dict"])
    thresholds=[round(float(t),6) for t in np.arange(args.threshold_min,args.threshold_max+args.threshold_step/2,args.threshold_step)]
    print(f"Best epoch={ckpt['epoch']}. Sweeping VALIDATION thresholds only...")
    sweep=threshold_sweep(model,val_loader,device,thresholds,use_amp)
    best_thr=max(sweep,key=lambda x:(x["dice"],x["iou"],x["precision"],-abs(x["threshold"]-0.5)))
    with (outdir/"validation_threshold_sweep.csv").open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(sweep[0].keys())); w.writeheader(); w.writerows(sweep)
    metrics={"architecture":"SAM2-UNet Hiera-L + terrain input adapter","feature_set":args.features,
             "channels":selected_names,"best_epoch":ckpt["epoch"],"validation_at_0_5":ckpt["validation_metrics_at_0_5"],
             "validation_best_threshold":best_thr,"threshold_grid":thresholds,"train_rows":len(train_rows),
             "validation_rows":len(val_rows),"test_rows_locked_not_evaluated":len(test_rows),"test":None}
    (outdir/"metrics.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print("\nSAM2-UNET TILLAMOOK RUN COMPLETE — TEST REMAINS LOCKED")
    print(json.dumps(metrics,indent=2))


if __name__ == "__main__":
    main()
