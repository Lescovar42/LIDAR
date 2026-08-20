#!/usr/bin/env python3
"""Summarize visually reviewed validation errors without changing labels."""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from phase_common import as_float, read_csv, write_csv, write_json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    args = parser.parse_args()

    rows = read_csv(args.review_csv.resolve())
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        category = row.get("manual_error_category", "").strip() or "unreviewed"
        groups[category].append(row)

    summary: list[dict[str, Any]] = []
    for category, subset in sorted(groups.items()):
        summary.append(
            {
                "error_category": category,
                "patches": len(subset),
                "fp_pixels": sum(int(as_float(row.get("fp"), 0)) for row in subset),
                "fn_pixels": sum(int(as_float(row.get("fn"), 0)) for row in subset),
                "mean_precision": sum(as_float(row.get("precision"), 0) for row in subset) / len(subset),
                "mean_recall": sum(as_float(row.get("recall"), 0) for row in subset) / len(subset),
                "evidence_status": "measured_after_manual_visual_review" if category != "unreviewed" else "not_yet_reviewed",
            }
        )
    write_csv(outdir / "error_category_summary.csv", summary)
    write_json(outdir / "error_category_summary.json", summary)

    reviewed = sum(row["patches"] for row in summary if row["error_category"] != "unreviewed")
    total = sum(row["patches"] for row in summary)
    lines = [
        "# Validation error-category summary",
        "",
        f"Reviewed candidates: **{reviewed}/{total}**.",
        "",
        "| Category | Patches | FP pixels | FN pixels | Evidence |",
        "|---|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['error_category']} | {row['patches']} | {row['fp_pixels']} | "
            f"{row['fn_pixels']} | {row['evidence_status']} |"
        )
    if reviewed == 0:
        lines.extend(
            [
                "",
                "No semantic error category has been measured yet. Edit `manual_error_category` in the review CSV after inspecting the generated comparison images, then rerun this script. Do not infer road/cut/drainage categories from pixel metrics alone.",
            ]
        )
    (outdir / "error_category_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
