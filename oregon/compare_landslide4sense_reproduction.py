#!/usr/bin/env python3
"""
Compare Landslide4Sense validation prediction maps.

Designed for:
- official baseline prediction maps produced by official Predict.py
- prediction maps produced by reproduce_landslide4sense_official_baseline.py

It computes the same landslide-class global pixel-count precision, recall and F1
used by the official baseline evaluation, plus IoU/OA for diagnostics.

It can compare:
A) official predictions vs ground truth
B) our reproduction predictions vs ground truth
C) official predictions vs our predictions (pixel agreement)

The reported official reference is:
Precision = 0.5175
Recall    = 0.6550
F1        = 0.5782
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np


REPORTED = {
    "precision": 0.5175,
    "recall": 0.6550,
    "f1": 0.5782,
}


def numeric_suffix(path: Path) -> tuple[int, str]:
    digits = "".join(ch for ch in path.stem if ch.isdigit())
    return (int(digits) if digits else 10**12, path.name)


def read_mask(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as hf:
        if "mask" in hf:
            arr = hf["mask"][:]
        else:
            keys = list(hf.keys())
            if len(keys) != 1:
                raise KeyError(f"{path}: expected key 'mask'; keys={keys}")
            arr = hf[keys[0]][:]

    arr = np.asarray(arr)
    if arr.ndim == 3 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.shape != (128, 128):
        raise RuntimeError(f"{path}: expected 128x128, got {arr.shape}")
    return arr


def score_prediction_dir(gt_dir: Path, pred_dir: Path) -> tuple[dict, list[dict]]:
    gt_files = sorted(gt_dir.glob("mask_*.h5"), key=numeric_suffix)
    if not gt_files:
        raise FileNotFoundError(f"No mask_*.h5 in ground-truth directory: {gt_dir}")

    tp = fp = fn = tn = 0
    per_image: list[dict] = []
    missing = []

    for gt_path in gt_files:
        pred_path = pred_dir / gt_path.name
        if not pred_path.exists():
            missing.append(pred_path)
            continue

        gt = read_mask(gt_path)
        pred = read_mask(pred_path)

        valid = (gt >= 0) & (gt < 2)
        truth1 = gt == 1
        pred1 = pred == 1

        i_tp = int(np.logical_and(pred1, truth1 & valid).sum())
        i_fp = int(np.logical_and(pred1, (~truth1) & valid).sum())
        i_fn = int(np.logical_and(~pred1, truth1 & valid).sum())
        i_tn = int(np.logical_and(~pred1, (~truth1) & valid).sum())

        tp += i_tp
        fp += i_fp
        fn += i_fn
        tn += i_tn

        eps = 1e-14
        p = i_tp / (i_tp + i_fp + eps)
        r = i_tp / (i_tp + i_fn + eps)
        f1 = 2 * p * r / (p + r + eps)

        per_image.append({
            "mask": gt_path.name,
            "tp": i_tp,
            "fp": i_fp,
            "fn": i_fn,
            "tn": i_tn,
            "precision": p,
            "recall": r,
            "f1": f1,
        })

    if missing:
        preview = "\n".join(str(p) for p in missing[:10])
        raise FileNotFoundError(
            f"{len(missing)} prediction masks missing in {pred_dir}. First missing:\n{preview}"
        )

    eps = 1e-14
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    oa = (tp + tn) / (tp + fp + fn + tn + eps)

    pooled = {
        "files": len(gt_files),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "overall_accuracy": oa,
    }
    return pooled, per_image


def compare_prediction_dirs(dir_a: Path, dir_b: Path) -> dict:
    files_a = {p.name: p for p in dir_a.glob("mask_*.h5")}
    files_b = {p.name: p for p in dir_b.glob("mask_*.h5")}
    common = sorted(set(files_a) & set(files_b), key=lambda n: numeric_suffix(Path(n)))

    if not common:
        raise FileNotFoundError("No common mask_*.h5 files between prediction directories")

    total = 0
    equal = 0
    both_pos = 0
    a_pos_b_neg = 0
    a_neg_b_pos = 0

    for name in common:
        a = read_mask(files_a[name])
        b = read_mask(files_b[name])

        total += a.size
        equal += int((a == b).sum())
        both_pos += int(((a == 1) & (b == 1)).sum())
        a_pos_b_neg += int(((a == 1) & (b == 0)).sum())
        a_neg_b_pos += int(((a == 0) & (b == 1)).sum())

    return {
        "common_files": len(common),
        "pixels": total,
        "exact_agreement_fraction": equal / total,
        "disagreement_fraction": 1.0 - equal / total,
        "both_positive_pixels": both_pos,
        "a_positive_b_negative": a_pos_b_neg,
        "a_negative_b_positive": a_neg_b_pos,
    }


def reference_delta(metrics: dict) -> dict:
    return {
        "precision_abs_delta": abs(metrics["precision"] - REPORTED["precision"]),
        "recall_abs_delta": abs(metrics["recall"] - REPORTED["recall"]),
        "f1_abs_delta": abs(metrics["f1"] - REPORTED["f1"]),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--gt-dir", type=Path, required=True)
    p.add_argument("--official-pred-dir", type=Path)
    p.add_argument("--ours-pred-dir", type=Path)
    p.add_argument("--outdir", type=Path, default=Path("./l4s_comparison"))
    p.add_argument(
        "--f1-tolerance",
        type=float,
        default=0.05,
        help=(
            "Diagnostic tolerance against the reported F1=0.5782. "
            "This is NOT a formal equivalence test."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.official_pred_dir is None and args.ours_pred_dir is None:
        raise SystemExit("Provide at least one of --official-pred-dir or --ours-pred-dir")

    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    gt_dir = args.gt_dir.resolve()

    result = {
        "reported_reference": REPORTED,
        "f1_tolerance_for_diagnostic": args.f1_tolerance,
        "official": None,
        "ours": None,
        "official_vs_ours": None,
    }

    per_image_sets = {}

    if args.official_pred_dir is not None:
        official_metrics, official_per_image = score_prediction_dir(
            gt_dir, args.official_pred_dir.resolve()
        )
        official_metrics["delta_from_reported"] = reference_delta(official_metrics)
        official_metrics["within_f1_tolerance_of_reported"] = (
            official_metrics["delta_from_reported"]["f1_abs_delta"] <= args.f1_tolerance
        )
        result["official"] = official_metrics
        per_image_sets["official"] = official_per_image

    if args.ours_pred_dir is not None:
        ours_metrics, ours_per_image = score_prediction_dir(
            gt_dir, args.ours_pred_dir.resolve()
        )
        ours_metrics["delta_from_reported"] = reference_delta(ours_metrics)
        ours_metrics["within_f1_tolerance_of_reported"] = (
            ours_metrics["delta_from_reported"]["f1_abs_delta"] <= args.f1_tolerance
        )
        result["ours"] = ours_metrics
        per_image_sets["ours"] = ours_per_image

    if args.official_pred_dir is not None and args.ours_pred_dir is not None:
        result["official_vs_ours"] = compare_prediction_dirs(
            args.official_pred_dir.resolve(),
            args.ours_pred_dir.resolve(),
        )

        # Metric-level deltas between implementations.
        result["metric_delta_official_vs_ours"] = {
            "precision": abs(result["official"]["precision"] - result["ours"]["precision"]),
            "recall": abs(result["official"]["recall"] - result["ours"]["recall"]),
            "f1": abs(result["official"]["f1"] - result["ours"]["f1"]),
            "iou": abs(result["official"]["iou"] - result["ours"]["iou"]),
        }

    (outdir / "comparison.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    # Wide per-image CSV where possible.
    names = sorted(
        {r["mask"] for rows in per_image_sets.values() for r in rows},
        key=lambda n: numeric_suffix(Path(n)),
    )
    lookup = {
        label: {r["mask"]: r for r in rows}
        for label, rows in per_image_sets.items()
    }

    with (outdir / "per_image_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        fields = ["mask"]
        for label in ("official", "ours"):
            if label in lookup:
                fields += [
                    f"{label}_precision",
                    f"{label}_recall",
                    f"{label}_f1",
                    f"{label}_tp",
                    f"{label}_fp",
                    f"{label}_fn",
                    f"{label}_tn",
                ]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for name in names:
            row = {"mask": name}
            for label in ("official", "ours"):
                rec = lookup.get(label, {}).get(name)
                if rec is not None:
                    for k in ("precision", "recall", "f1", "tp", "fp", "fn", "tn"):
                        row[f"{label}_{k}"] = rec[k]
            writer.writerow(row)

    def fmt(m):
        return (
            f"P={m['precision']:.4f}, R={m['recall']:.4f}, "
            f"F1={m['f1']:.4f}, IoU={m['iou']:.4f}"
        )

    lines = [
        "# Landslide4Sense reproduction comparison",
        "",
        "Official reported validation reference:",
        "",
        f"- Precision: {REPORTED['precision']:.4f}",
        f"- Recall: {REPORTED['recall']:.4f}",
        f"- F1: {REPORTED['f1']:.4f}",
        "",
    ]

    if result["official"] is not None:
        lines += [
            "## Official prediction maps",
            "",
            fmt(result["official"]),
            "",
            f"Absolute F1 delta from report: "
            f"{result['official']['delta_from_reported']['f1_abs_delta']:.4f}",
            "",
        ]

    if result["ours"] is not None:
        lines += [
            "## Our independent reproduction",
            "",
            fmt(result["ours"]),
            "",
            f"Absolute F1 delta from report: "
            f"{result['ours']['delta_from_reported']['f1_abs_delta']:.4f}",
            "",
        ]

    if result["official_vs_ours"] is not None:
        a = result["official_vs_ours"]
        d = result["metric_delta_official_vs_ours"]
        lines += [
            "## Official vs ours",
            "",
            f"- Exact pixel agreement: {a['exact_agreement_fraction']:.4%}",
            f"- Pixel disagreement: {a['disagreement_fraction']:.4%}",
            f"- F1 absolute difference: {d['f1']:.4f}",
            f"- Precision absolute difference: {d['precision']:.4f}",
            f"- Recall absolute difference: {d['recall']:.4f}",
            "",
            "Different independently trained checkpoints are not expected to produce "
            "pixel-identical predictions. Metric proximity is the primary sanity check.",
            "",
        ]

    lines += [
        "## Interpretation rule",
        "",
        "A large mismatch does not by itself prove the Tillamook pipeline is wrong. "
        "First determine whether the official checkpoint itself reproduces the reported "
        "validation metric on these validation labels. Then compare the independent "
        "reproduction under the same data, architecture, normalization, optimizer, loss, "
        "iteration budget, and metric definition.",
        "",
    ]

    (outdir / "comparison.md").write_text("\n".join(lines), encoding="utf-8")

    print(json.dumps(result, indent=2))
    print(f"\nSaved: {outdir / 'comparison.json'}")
    print(f"Saved: {outdir / 'comparison.md'}")
    print(f"Saved: {outdir / 'per_image_metrics.csv'}")


if __name__ == "__main__":
    main()
