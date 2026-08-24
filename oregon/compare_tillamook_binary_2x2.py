#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RUNS = [
    ("shallow_7ch", "Shallow", "7ch"),
    ("shallow_3ch", "Shallow", "3ch"),
    ("deep_7ch", "Deep", "7ch"),
    ("deep_3ch", "Deep", "3ch"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("phase4_tillamook_binary_2x2"))
    args = ap.parse_args()

    root = args.root.resolve()
    rows = []

    for dirname, arch_label, feat_label in RUNS:
        path = root / dirname / "metrics.json"
        if not path.exists():
            raise SystemExit(f"Missing run metrics: {path}")
        m = json.loads(path.read_text(encoding="utf-8"))
        b = m["validation_best_threshold"]
        rows.append({
            "run": dirname,
            "architecture": arch_label,
            "features": feat_label,
            "parameters": m["parameter_count"],
            "best_epoch": m["best_epoch"],
            "threshold": b["threshold"],
            "dice": b["dice"],
            "iou": b["iou"],
            "precision": b["precision"],
            "recall": b["recall"],
            "specificity": b["specificity"],
            "fp": b["fp"],
            "fn": b["fn"],
            "gt_positive_fraction": b["gt_positive_fraction"],
            "predicted_positive_fraction": b["predicted_positive_fraction"],
            "pos_weight": m["pos_weight_used"],
        })

    rows.sort(key=lambda r: r["dice"], reverse=True)
    winner = rows[0]

    csv_path = root / "comparison.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    md = [
        "# Tillamook strict-binary 2×2 feature/depth comparison",
        "",
        "All model selection and threshold selection below use the frozen Tillamook validation split only. "
        "The internal test split was not evaluated.",
        "",
        "| Run | Architecture | Features | Params | Epoch | Thr | Dice | IoU | Precision | Recall | Specificity | FP | FN | Pred + |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        md.append(
            f"| {r['run']} | {r['architecture']} | {r['features']} | "
            f"{r['parameters']:,} | {r['best_epoch']} | {r['threshold']:.2f} | "
            f"{r['dice']:.4f} | {r['iou']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['specificity']:.4f} | "
            f"{r['fp']:,} | {r['fn']:,} | {100*r['predicted_positive_fraction']:.2f}% |"
        )

    md += [
        "",
        f"**Validation winner:** `{winner['run']}` "
        f"(Dice={winner['dice']:.4f}, threshold={winner['threshold']:.2f}).",
        "",
        "**Do not evaluate the internal test until this configuration and threshold are explicitly frozen.**",
    ]

    (root / "comparison.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    print(f"\nCSV: {csv_path}")


if __name__ == "__main__":
    main()
