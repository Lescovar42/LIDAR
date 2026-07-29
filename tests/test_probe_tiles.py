import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from pyproj import CRS
from rasterio.transform import Affine

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oregon"))
from terrain_utils import TerrainTile
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oregon" / "diagnostics"))
from probe_tiles import (
    dem_fragmentation,
    main,
    patch_ground_fractions,
    probe_projects,
    void_run_statistics,
)


class ProbeTests(unittest.TestCase):
    def test_metrics_and_injected_reader_do_not_write_patches(self):
        mask = np.array([[1,1,0,0],[1,1,0,1],[0,0,0,1],[1,1,1,1]], dtype=bool)
        tile = TerrainTile(np.ones((4,4), dtype=np.float32), Affine.identity(), CRS.from_epsg(4326), mask, 10, float(mask.mean()), Path("fake.laz"))
        calls = []
        def reader(path, **kwargs):
            calls.append((path, kwargs["cell_size"]))
            return tile
        result = probe_projects({"candidate": [Path("fake.laz")]}, cell_sizes=(1.0, 2.0), patch_size=2, stride=2, reader=reader)
        self.assertEqual(2, len(result["tiles"]))
        self.assertEqual(2, len(calls))
        self.assertEqual(4, result["tiles"][0]["patch_ground_fraction"]["count"])
        self.assertGreater(void_run_statistics(mask)["max_cells"], 0)
        self.assertEqual(2, dem_fragmentation(mask)["component_count"])
        self.assertEqual([1.0, 0.25, 0.5, 0.75], patch_ground_fractions(mask, patch_size=2, stride=2))
        summary = result["project_cell_size_summary"][0]
        self.assertIn("void_runs", summary)
        self.assertIn("dem_fragmentation", summary)
        self.assertIn("histogram", summary["patch_ground_fraction"])

    def test_pin_rejects_output_aliasing_registry_before_probe_or_write(self):
        with tempfile.TemporaryDirectory() as tmp:
            registry = Path(tmp) / "regions.json"
            original = '{"sentinel": "registry must survive"}\n'
            registry.write_text(original, encoding="utf-8")
            argv = [
                "probe_tiles.py",
                "--project", f"candidate={tmp}",
                "--output", str(registry),
                "--registry", str(registry),
                "--pin-region", "R1",
                "--pin-lidar-project", "candidate",
                "--pin-cell-size", "1.0",
                "--pin-reason", "measured decision",
            ]
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    main()
            self.assertEqual(original, registry.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
