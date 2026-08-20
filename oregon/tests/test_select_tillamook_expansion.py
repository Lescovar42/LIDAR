import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

from pyproj import Transformer
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))

from select_tillamook_expansion import (
    Candidate,
    PolygonRecord,
    build_candidates,
    greedy_select,
    laz_basename,
)


def tile(name, xmin, ymin, xmax, ymax, size=1000):
    return {
        "title": name[:-4].replace("_", " "),
        "downloadURL": f"https://example/{name}",
        "sizeInBytes": size,
        "boundingBox": {"minX": xmin, "minY": ymin, "maxX": xmax, "maxY": ymax},
    }


class ExpansionSelectorTests(unittest.TestCase):
    def setUp(self):
        self.to_metric = Transformer.from_crs("EPSG:4326", "EPSG:32610", always_xy=True)

    def poly(self, key, confidence, bounds):
        from shapely.ops import transform
        g = box(*bounds)
        return PolygonRecord(key, confidence, g, transform(self.to_metric.transform, g))

    def test_laz_basename_strips_aria2(self):
        self.assertEqual(laz_basename(r"F:\\x\\abc.laz.aria2"), "abc.laz")

    def test_validation_buffer_and_current_train_are_excluded(self):
        tiles = [
            tile("train.laz", -123.90, 45.50, -123.89, 45.51),
            tile("val.laz", -123.80, 45.50, -123.79, 45.51),
            tile("near_val.laz", -123.8005, 45.50, -123.7905, 45.51),
            tile("safe.laz", -123.70, 45.50, -123.69, 45.51),
        ]
        candidates, rejected, _ = build_candidates(
            tiles, [], {"train.laz"}, {"val.laz"}, set(), {}, {}, self.to_metric, 500.0
        )
        self.assertEqual([c.name for c in candidates], ["safe.laz"])
        self.assertIn("train.laz", rejected)
        self.assertIn("val.laz", rejected)
        self.assertIn("near_val.laz", rejected)

    def test_low_confidence_overlap_not_counted_as_clean_negative(self):
        tiles = [tile("low.laz", -123.70, 45.50, -123.69, 45.51)]
        polygons = [self.poly("low1", "low", (-123.699, 45.501, -123.695, 45.505))]
        candidates, rejected, _ = build_candidates(
            tiles, polygons, set(), set(), set(), {}, {}, self.to_metric, 0.0
        )
        self.assertEqual(candidates, [])
        self.assertEqual(rejected["low.laz"], "low_or_unknown_slido_overlap_only")

    def test_positive_new_polygon_priority(self):
        from shapely.ops import transform
        g1 = box(-123.70, 45.50, -123.695, 45.505)
        g2 = box(-123.60, 45.50, -123.595, 45.505)
        c1 = Candidate(
            "oldish.laz", {}, g1, transform(self.to_metric.transform, g1), transform(self.to_metric.transform, g1).centroid,
            category="positive_diversity", positive_keys={"a"}, new_positive_keys=set(), high_count=1,
        )
        c2 = Candidate(
            "new.laz", {}, g2, transform(self.to_metric.transform, g2), transform(self.to_metric.transform, g2).centroid,
            category="positive_diversity", positive_keys={"b"}, new_positive_keys={"b"}, high_count=1,
        )
        selected = greedy_select([c1, c2], 1, 0.0, None)
        self.assertEqual(selected[0].name, "new.laz")

    def test_hard_negative_quota(self):
        from shapely.ops import transform
        candidates = []
        for i in range(6):
            g = box(-123.9 + i * .02, 45.5, -123.89 + i * .02, 45.51)
            gm = transform(self.to_metric.transform, g)
            candidates.append(Candidate(
                f"p{i}.laz", {}, g, gm, gm.centroid,
                category="positive_diversity", positive_keys={f"p{i}"}, new_positive_keys={f"p{i}"}
            ))
        for i in range(6):
            g = box(-123.9 + i * .02, 45.6, -123.89 + i * .02, 45.61)
            gm = transform(self.to_metric.transform, g)
            candidates.append(Candidate(f"n{i}.laz", {}, g, gm, gm.centroid, category="hard_negative"))
        selected = greedy_select(candidates, 10, 0.4, None)
        self.assertEqual(len(selected), 10)
        self.assertEqual(sum(c.category == "hard_negative" for c in selected), 4)
        self.assertEqual(sum(c.category == "positive_diversity" for c in selected), 6)

    def test_downloaded_bonus_does_not_override_new_polygon_signal(self):
        from shapely.ops import transform
        g1 = box(-123.70, 45.50, -123.695, 45.505)
        g2 = box(-123.60, 45.50, -123.595, 45.505)
        gm1 = transform(self.to_metric.transform, g1)
        gm2 = transform(self.to_metric.transform, g2)
        downloaded_old = Candidate(
            "downloaded.laz", {}, g1, gm1, gm1.centroid, downloaded=True,
            category="positive_diversity", positive_keys={"old"}, new_positive_keys=set()
        )
        new_poly = Candidate(
            "newpoly.laz", {}, g2, gm2, gm2.centroid, downloaded=False,
            category="positive_diversity", positive_keys={"x","y"}, new_positive_keys={"x","y"}
        )
        selected = greedy_select([downloaded_old, new_poly], 1, 0.0, None)
        self.assertEqual(selected[0].name, "newpoly.laz")


if __name__ == "__main__":
    unittest.main()
