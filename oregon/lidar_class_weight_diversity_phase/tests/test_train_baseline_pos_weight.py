"""Run after applying the patch inside the repository's oregon directory."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from train_baseline import resolve_pos_weight


class PosWeightResolutionTests(unittest.TestCase):
    def make_dataset(self, root: Path) -> list[dict[str, str]]:
        patches = root / "patches"
        patches.mkdir()
        mask = np.asarray([[1, 0, 0], [255, 0, 0]], dtype=np.uint8)
        np.savez_compressed(patches / "a.npz", mask=mask, features=np.zeros((7, 2, 3), dtype=np.float32))
        return [{"patch_path": "patches/a.npz"}]

    def test_auto_and_fixed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self.make_dataset(root)
            used, automatic, mode = resolve_pos_weight(root, rows, "auto")
            self.assertEqual((used, automatic, mode), (4.0, 4.0, "auto"))
            used, automatic, mode = resolve_pos_weight(root, rows, "1.0")
            self.assertEqual((used, automatic, mode), (1.0, 4.0, "fixed"))

    def test_invalid_fixed_weight(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self.make_dataset(root)
            with self.assertRaises(ValueError):
                resolve_pos_weight(root, rows, "0")
            with self.assertRaises(ValueError):
                resolve_pos_weight(root, rows, "not-a-number")


if __name__ == "__main__":
    unittest.main()
