from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import box, mapping

OREGON_DIR = Path(__file__).resolve().parents[1]
DIAGNOSTICS = OREGON_DIR / "diagnostics"
if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, DIAGNOSTICS / filename)
    if spec is None or spec.loader is None:
        raise RuntimeError(filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


select_probe = load_module("select_a22_probe_tested", "select_a22_probe.py")
probe_labels = load_module("probe_a22_labels_tested", "probe_a22_labels.py")


class A22SelectionTests(unittest.TestCase):
    def test_spread_selection_prefers_distant_tiles(self):
        candidates = []
        for index, (x, y) in enumerate(((0, 0), (1, 0), (0, 1), (10, 10))):
            candidates.append(
                {
                    "id": f"tile-{index}",
                    "sizeInBytes": 100,
                    "geometry": mapping(box(x, y, x + 0.5, y + 0.5)),
                    "_positive_intersection_area": 1.0,
                }
            )
        selected, remaining = select_probe.spread_select(
            candidates,
            2,
            remaining_bytes=1_000,
            positive=True,
        )
        ids = {item["id"] for item in selected}
        self.assertEqual(len(ids), 2)
        self.assertIn("tile-3", ids)
        self.assertEqual(remaining, 800)

    def test_record_filename_uses_url_path(self):
        record = {
            "downloadURL": "https://example.test/path/a%20tile.laz",
            "title": "ignored",
        }
        self.assertEqual(select_probe.record_filename(record), "a tile.laz")


class A22PatchMetricTests(unittest.TestCase):
    def test_patch_counts_distinguish_positive_negative_and_ignore(self):
        valid = np.ones((512, 512), dtype=bool)
        labels = np.zeros((512, 512), dtype=np.uint8)
        labels[0:256, 0:256] = 1
        labels[256:512, 256:512] = 255

        summary, windows = probe_labels.summarize_patch_labels(
            valid,
            labels,
            patch_size=256,
            stride=256,
            min_patch_ground_fraction=0.5,
        )

        self.assertEqual(len(windows), 4)
        self.assertEqual(summary["accepted"], 4)
        self.assertEqual(summary["positive_total"], 1)
        self.assertEqual(summary["positive_interior"], 1)
        self.assertEqual(summary["negative"], 2)
        self.assertEqual(summary["ignore_only"], 1)

    def test_ground_threshold_rejects_sparse_patch(self):
        valid = np.ones((256, 512), dtype=bool)
        valid[:, 256:] = False
        labels = np.zeros((256, 512), dtype=np.uint8)

        summary, windows = probe_labels.summarize_patch_labels(
            valid,
            labels,
            patch_size=256,
            stride=256,
            min_patch_ground_fraction=0.5,
        )

        self.assertEqual(len(windows), 1)
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["accepted"], 1)
        self.assertEqual(summary["rejected_ground"], 1)
        self.assertEqual(summary["negative"], 1)


if __name__ == "__main__":
    unittest.main()
