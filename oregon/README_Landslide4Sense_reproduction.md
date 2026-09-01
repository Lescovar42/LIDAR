# Landslide4Sense baseline reproduction sanity check

Purpose: verify that the local PyTorch segmentation environment and an independent
implementation can reproduce the behavior of the official Landslide4Sense U-Net
baseline before further Tillamook model tuning.

## What is matched to the official baseline

- 14 input bands
- fixed Landslide4Sense mean/std constants
- 128 x 128 patches
- official U-Net topology:
  - base width 64
  - four downsampling stages
  - bilinear upsampling
  - two output classes
- CrossEntropyLoss(ignore_index=255)
- Adam
- learning rate 0.001
- weight decay 0.0005
- batch size 32
- 5000 optimizer steps
- evaluation every 500 steps
- argmax over two-class softmax output
- globally pooled landslide-class Precision / Recall / F1
- no AMP
- no Dice loss
- no class weighting
- no threshold sweep

Published baseline validation reference:
- Precision 0.5175
- Recall 0.6550
- F1 0.5782

## Important source-code quirk

The current official `Train.py` creates its evaluation loader using `args.train_list`,
even though a separate `test_list` argument exists. That means a literal execution of
that source evaluates the training list during training.

For the published-validation sanity check, this package defaults to:

    --protocol reported-validation

which uses validation annotations for evaluation while keeping the official model and
training hyperparameters.

To mirror the source quirk:

    --protocol literal-source

## Dataset

Recommended local source:
IBM/NASA Landslide4Sense mirror, because it contains validation masks.

Expected mirror layout:

    Landslide4sense/
      images/
        train/
        validation/
        test/
      annotations/
        train/
        validation/
        test/

Install:

    python -m pip install torch numpy h5py huggingface_hub

Download:

    hf download ibm-nasa-geospatial/Landslide4sense ^
      --repo-type dataset ^
      --local-dir F:\LIDAR\public_benchmarks\Landslide4sense

PowerShell multiline equivalent:

    hf download ibm-nasa-geospatial/Landslide4sense `
      --repo-type dataset `
      --local-dir F:\LIDAR\public_benchmarks\Landslide4sense

## 1. Run our independent reproduction

From F:\LIDAR\oregon:

    python .\reproduce_landslide4sense_official_baseline.py `
      --data-root F:\LIDAR\public_benchmarks\Landslide4sense `
      --outdir .\public_benchmark_l4s\ours_reproduction `
      --protocol reported-validation `
      --device cuda

The official source does not set a random seed, so this script defaults to no explicit
seed. If you want repeatability for debugging, add:

    --seed 42

That is useful diagnostically but is no longer literal source behavior.

Outputs:

    public_benchmark_l4s/
      ours_reproduction/
        run_config.json
        history.csv
        metrics.json
        checkpoints/
        validation_predictions/

## 2. Generate official prediction maps

Clone the official repo separately:

    git clone https://github.com/iarai/Landslide4Sense-2022.git `
      F:\LIDAR\public_benchmarks\Landslide4Sense-2022-official

The original `Predict.py` expects the competition path/list-file arrangement.
You can either use the original-format dataset or create compatible list files.

If you have the official pretrained checkpoint, generate its validation maps with the
official `Predict.py`. Put those maps in a directory such as:

    F:\LIDAR\public_benchmarks\official_validation_map

The expected files are:

    mask_1.h5
    ...
    mask_245.h5

with HDF5 dataset key `mask`.

## 3. Evaluate ours only against the published reference

    python .\compare_landslide4sense_reproduction.py `
      --gt-dir F:\LIDAR\public_benchmarks\Landslide4sense\annotations\validation `
      --ours-pred-dir .\public_benchmark_l4s\ours_reproduction\validation_predictions `
      --outdir .\public_benchmark_l4s\comparison

## 4. Compare official prediction maps vs ours

    python .\compare_landslide4sense_reproduction.py `
      --gt-dir F:\LIDAR\public_benchmarks\Landslide4sense\annotations\validation `
      --official-pred-dir F:\LIDAR\public_benchmarks\official_validation_map `
      --ours-pred-dir .\public_benchmark_l4s\ours_reproduction\validation_predictions `
      --outdir .\public_benchmark_l4s\comparison

Outputs:

    comparison.json
    comparison.md
    per_image_metrics.csv

## How to interpret

Best order:

1. Evaluate official pretrained prediction maps on the mirror's validation masks.
2. Check whether those maps reproduce approximately P=0.5175, R=0.6550, F1=0.5782.
3. Then evaluate the independent reproduction.
4. If the official checkpoint matches the reported benchmark but ours is far away,
   audit our reproduction implementation/environment.
5. If both match approximately, the basic segmentation machinery is probably healthy.
6. Only then return to the Tillamook-specific pipeline.

Do NOT require pixel-identical output between separately trained models. Random
initialization and non-deterministic GPU kernels can change the exact prediction map.
The primary check is metric-level reproduction under the same protocol.
