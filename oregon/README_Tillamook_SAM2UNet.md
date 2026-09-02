# Tillamook SAM2-UNet experiment

This trainer keeps the frozen Tillamook train/validation/test split and never loads the internal test set.

## Setup

From `F:\LIDAR\oregon`:

```powershell
New-Item -ItemType Directory -Force .\third_party | Out-Null

git clone https://github.com/WZH0120/SAM2-UNet.git `
  .\third_party\SAM2-UNet

python -m pip install hydra-core==1.3.2
```

Do not blindly install the repository's full requirements into an already-working PyTorch environment because they pin a specific PyTorch generation.

Download the **original SAM2 Hiera-L** checkpoint, not SAM2.1:

```powershell
curl.exe -L `
  https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt `
  -o .\third_party\SAM2-UNet\sam2_hiera_large.pt
```

## Recommended first run

```powershell
python .\train_tillamook_sam2unet.py `
  --dataset-dir .\dataset_tillamook_100_binary_15m_split641620 `
  --outdir .\phase5_sam2unet_7ch `
  --sam2unet-repo .\third_party\SAM2-UNet `
  --hiera-path .\third_party\SAM2-UNet\sam2_hiera_large.pt `
  --features 7ch `
  --epochs 40 `
  --batch-size 1 `
  --accumulation-steps 4 `
  --task-lr 0.0003 `
  --sam-adapter-lr 0.0001 `
  --weight-decay 0.0001 `
  --warmup-epochs 3 `
  --min-lr-ratio 0.01 `
  --grad-clip 1.0 `
  --num-workers 2 `
  --seed 42 `
  --device cuda `
  --amp `
  --threshold-min 0.30 `
  --threshold-max 0.90 `
  --threshold-step 0.05
```

## Why these settings

The SAM2-UNet paper uses Hiera-L, AdamW, initial LR `1e-3`, cosine decay, batch 12 and 352x352 inputs. For Tillamook on an RTX 2060 SUPER, the first run instead uses native 256x256 patches, microbatch 1 with gradient accumulation, lower discriminative learning rates (`3e-4` task layers / `1e-4` SAM adapters), 3-epoch warmup, weight decay `1e-4`, gradient clipping and AMP.

The 7-channel terrain tensor is mapped to the 3-channel SAM2 input with a learnable 1x1 convolution initialized from slope/aspect. This lets the first forward pass behave like the already-tested slope/aspect representation while allowing the model to learn contributions from all seven terrain channels.

The first run deliberately avoids naive flips/rotations because the terrain stack includes directional variables such as aspect.

## Compare against current Tillamook development baseline

```text
deep_7ch
Dice = 0.4085
IoU = 0.2567
Precision = 0.2961
Recall = 0.6584
Predicted positive = 31.71%
GT positive = 14.26%
```

Watch especially whether SAM2-UNet reduces false positives and predicted-positive prevalence.

## If CUDA OOM occurs

```powershell
$env:PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
```

Keep batch size at 1. If Hiera-L still does not fit, stop rather than silently switching backbone: the official SAM2-UNet implementation hardcodes the Hiera-L feature dimensions, so a smaller Hiera variant would be a separate architecture adaptation.
