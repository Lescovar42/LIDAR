import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oregon"))
from select_tiles import select_probe_tiles


def tile(name, project, bbox, size=100):
    return {
        "title": name,
        "project": project,
        "bbox": bbox,
        "sizeInBytes": size,
        "downloadURL": f"https://example/{name}.laz",
    }


class LargeProbeSelectionTests(unittest.TestCase):
    def test_more_than_one_thousand_unmatched_anchors_do_not_recurse_by_catalog_size(self):
        tiles = [
            tile(f"a{i:04d}", "a", [float(i), 0.0, float(i + 1), 1.0])
            for i in range(1200)
        ]
        tiles.extend(
            tile(f"b{i:04d}", "b", [float(i), 0.0, float(i + 1), 1.0])
            for i in range(1192, 1200)
        )

        result = select_probe_tiles(
            tiles,
            projects=["a", "b"],
            count=8,
            overlap_threshold=0.99,
        )

        self.assertEqual(8, len(result["a"]))
        self.assertEqual(8, len(result["b"]))
        self.assertEqual(
            [f"a{i:04d}" for i in range(1192, 1200)],
            [item["title"] for item in result["a"]],
        )
        self.assertEqual(
            [f"b{i:04d}" for i in range(1192, 1200)],
            [item["title"] for item in result["b"]],
        )


if __name__ == "__main__":
    unittest.main()
