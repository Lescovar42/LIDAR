#!/usr/bin/env python3
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from prepare_boundary_aware_manifest import boundary_metrics, select_training_rows


def row(patch_id: str, split: str, coverage: str, fraction: float, key: str = "L1") -> dict[str, str]:
    return {
        "patch_id": patch_id,
        "split": split,
        "coverage_class": coverage,
        "positive_fraction": str(fraction),
        "patch_polygon_keys": key,
        "tile_name": "tile.laz",
        "ground_fraction": "1",
        "boundary_pixel_fraction": "0",
    }


def main() -> int:
    full = np.ones((16, 16), dtype=np.uint8)
    mixed = np.zeros((16, 16), dtype=np.uint8)
    mixed[:, :8] = 1
    assert boundary_metrics(full)[2] is False
    assert boundary_metrics(mixed)[2] is True

    rows = [row(f"full_{i}", "train", "full_positive", 1.0) for i in range(6)]
    rows += [row("mixed", "train", "mixed_positive", 0.5)]
    rows += [row("negative", "train", "negative", 0.0, "")]
    rows += [row("validation", "validation", "negative", 0.0, "")]
    selected, payload = select_training_rows(
        rows,
        max_near_full_per_group=4,
        max_full_per_group=2,
        seed=42,
        remove_cross_split_polygon_overlap=True,
    )
    ids = {item["patch_id"] for item in selected}
    assert "mixed" in ids and "negative" in ids and "validation" in ids
    assert len([item for item in selected if item["coverage_class"] == "full_positive"]) == 2
    assert payload["report"]["output_split_counts"]["validation"] == 1
    print("Boundary-aware self-test: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
