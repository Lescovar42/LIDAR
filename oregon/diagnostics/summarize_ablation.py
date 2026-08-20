#!/usr/bin/env python3
"""Compare independently swept auto-weight and pos_weight=1.0 validation results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from phase_common import as_float, write_csv, write_json

METRICS = (
    "threshold",
    "dice",
    "iou",
    "precision",
    "recall",
    "specificity",
    "fp",
    "fn",
    "gt_positive_fraction",
    "predicted_positive_fraction",
)


def load_best(directory: Path) -> dict[str, Any]:
    path = directory / "best_threshold.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {**payload, **payload["best"]}


def load_fingerprint(directory: Path) -> dict[str, Any]:
    path = directory / "dataset_fingerprint.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def recommendation(auto: dict[str, Any], fixed: dict[str, Any], max_recall_loss: float, dice_tolerance: float) -> tuple[str, str]:
    precision_gain = as_float(fixed["precision"]) - as_float(auto["precision"])
    recall_loss = as_float(auto["recall"]) - as_float(fixed["recall"])
    dice_delta = as_float(fixed["dice"]) - as_float(auto["dice"])
    fp_delta = int(fixed["fp"]) - int(auto["fp"])
    if precision_gain > 0 and fp_delta < 0 and recall_loss <= max_recall_loss and dice_delta >= -dice_tolerance:
        return "use_pos_weight_1.0", "Fixed weight reduced false positives and improved precision within the declared recall/Dice tolerances."
    if precision_gain > 0 and fp_delta < 0 and recall_loss > max_recall_loss:
        return "test_intermediate_fixed_weight", "Removing weighting reduced false positives, but recall loss exceeded the declared tolerance."
    if dice_delta < -dice_tolerance:
        return "retain_auto_then_improve_diversity", "Fixed weight degraded validation Dice beyond tolerance; retain auto as the current control while addressing data diversity."
    return "retain_weight_but_improve_diversity_first", "The weight-only ablation did not provide a decisive precision/FP improvement; prioritize measured diversity limitations before architecture changes."


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--auto-eval", type=Path, required=True)
    parser.add_argument("--posw1-eval", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--max-recall-loss", type=float, default=0.05)
    parser.add_argument("--dice-tolerance", type=float, default=0.01)
    args = parser.parse_args()

    auto_dir = args.auto_eval.resolve()
    fixed_dir = args.posw1_eval.resolve()
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    auto = load_best(auto_dir)
    fixed = load_best(fixed_dir)
    auto_fp = load_fingerprint(auto_dir)
    fixed_fp = load_fingerprint(fixed_dir)
    comparable_keys = ("manifest_sha256", "split", "row_count", "patch_ids_sha256", "patch_paths_sha256", "region_ids", "region_roles")
    mismatches = {key: [auto_fp.get(key), fixed_fp.get(key)] for key in comparable_keys if auto_fp.get(key) != fixed_fp.get(key)}
    if mismatches:
        raise SystemExit(f"Refusing comparison: validation dataset fingerprints differ: {json.dumps(mismatches, indent=2)}")

    table: list[dict[str, Any]] = []
    for label, payload in (("auto", auto), ("pos_weight_1.0", fixed)):
        table.append({"run": label, **{metric: payload.get(metric) for metric in METRICS}})
    delta = {metric: as_float(fixed.get(metric), 0) - as_float(auto.get(metric), 0) for metric in METRICS}
    delta["fp"] = int(fixed["fp"]) - int(auto["fp"])
    delta["fn"] = int(fixed["fn"]) - int(auto["fn"])
    table.append({"run": "delta_posw1_minus_auto", **delta})
    write_csv(outdir / "baseline_vs_posw1.csv", table)

    decision, rationale = recommendation(auto, fixed, args.max_recall_loss, args.dice_tolerance)
    payload = {
        "validation_fingerprints_match": True,
        "auto_evaluation": str(auto_dir),
        "posw1_evaluation": str(fixed_dir),
        "max_recall_loss": args.max_recall_loss,
        "dice_tolerance": args.dice_tolerance,
        "measured_deltas_posw1_minus_auto": delta,
        "decision_rule_result": decision,
        "decision_rule_rationale": rationale,
        "warning": "This is a source-validation ablation conclusion, not external-test or cross-region generalization evidence.",
    }
    write_json(outdir / "ablation_summary.json", payload)

    lines = [
        "# Positive-class-weight ablation summary",
        "",
        "The two evaluation fingerprints match, so the threshold sweeps use the same manifest rows.",
        "",
        "| Run | Best threshold | Dice | IoU | Precision | Recall | Specificity | FP | FN | Predicted positive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, item in (("Auto", auto), ("pos_weight=1.0", fixed)):
        lines.append(
            f"| {label} | {as_float(item['threshold']):.2f} | {as_float(item['dice']):.4f} | "
            f"{as_float(item['iou']):.4f} | {as_float(item['precision']):.4f} | {as_float(item['recall']):.4f} | "
            f"{as_float(item['specificity']):.4f} | {int(item['fp']):,} | {int(item['fn']):,} | "
            f"{as_float(item['predicted_positive_fraction']):.2%} |"
        )
    lines.extend(
        [
            "",
            f"**Rule-based phase recommendation:** `{decision}`.",
            "",
            rationale,
            "",
            "This conclusion is measured on frozen source-region validation only. It must not be described as cross-region generalization.",
        ]
    )
    (outdir / "baseline_vs_posw1.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
