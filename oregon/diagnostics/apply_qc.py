#!/usr/bin/env python3
"""Merge manual decisions from qc_review.csv into the dataset manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    parser = argparse.ArgumentParser(description="Apply visual-QC decisions to patches.csv.")
    parser.add_argument("--dataset-dir", type=Path, default=Path("dataset_pilot"))
    parser.add_argument("--review", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    source_path = dataset_dir / "patches.csv"
    review_path = (args.review or (dataset_dir / "qc" / "qc_review.csv")).resolve()
    out_path = (args.out or (dataset_dir / "patches_qc.csv")).resolve()

    if not source_path.exists():
        parser.error(f"Missing {source_path}")
    if not review_path.exists():
        parser.error(f"Missing {review_path}")

    source_rows = read_rows(source_path)
    review_rows = read_rows(review_path)
    decisions = {
        row["patch_id"]: (row.get("qc_status", "").strip(), row.get("qc_notes", "").strip())
        for row in review_rows
        if row.get("patch_id") and row.get("qc_status", "").strip()
    }

    updated = 0
    for row in source_rows:
        decision = decisions.get(row.get("patch_id", ""))
        if decision is None:
            continue
        row["qc_status"], row["qc_notes"] = decision
        updated += 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(source_rows[0]) if source_rows else []
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(source_rows)

    print(f"Applied {updated} QC decision(s)")
    print(f"Wrote {out_path}")
    print("train_baseline.py automatically prefers patches_qc.csv when present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
