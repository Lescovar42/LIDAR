import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oregon"))
from select_tiles import deduplicate_footprints, select_probe_tiles, select_tiles


def tile(name, project, bbox, size=100):
    return {"title": name, "project": project, "bbox": bbox, "sizeInBytes": size, "downloadURL": f"https://example/{name}.laz"}


def polygon(bbox, confidence="High (>30)"):
    x1, y1, x2, y2 = bbox
    return {"type": "Feature", "properties": {"CONFIDENCE": confidence}, "geometry": {"type": "Polygon", "coordinates": [[[x1,y1],[x2,y1],[x2,y2],[x1,y2],[x1,y1]]]}}


class SelectionTests(unittest.TestCase):
    def test_cross_project_overlap_is_deduplicated(self):
        tiles = [tile("a", "old", [0,0,1,1]), tile("b", "new", [0,0,1,1]), tile("c", "new", [2,0,3,1])]
        self.assertEqual(2, len(deduplicate_footprints(tiles)))

    def test_byte_budget_is_never_exceeded(self):
        tiles = [tile(str(i), "p", [i,0,i+1,1], 400) for i in range(4)]
        selected = select_tiles(tiles, [], project="p", max_tiles=4, max_total_gb=0.0000009, negative_quota=1)
        self.assertEqual(2, len(selected))
        self.assertLessEqual(sum(item["sizeInBytes"] for item in selected), 900)

    def test_default_negative_quota_reserves_one_of_four(self):
        tiles = [tile(str(i), "p", [i,0,i+1,1]) for i in range(5)]
        polys = [polygon([0,0,0.5,1]), polygon([1,0,1.5,1]), polygon([2,0,2.5,1]), polygon([3,0,3.5,1])]
        selected = select_tiles(tiles, polys, project="p", max_tiles=4)
        self.assertEqual(4, len(selected))
        self.assertEqual(1, sum(item["_is_hard_negative"] for item in selected))
        self.assertGreater(selected[0]["_positive_intersection_area"], 0)

    def test_low_confidence_overlap_is_not_a_true_negative(self):
        tiles = [tile("ignored", "p", [0,0,1,1]), tile("negative", "p", [2,0,3,1])]
        selected = select_tiles(tiles, [polygon([0,0,0.5,1], "Low")], project="p", negative_quota=1)
        self.assertEqual(["negative"], [item["title"] for item in selected])

    def test_adjacent_tiles_with_slight_envelope_overlap_are_preserved(self):
        tiles = [tile("a", "p", [0,0,1,1]), tile("b", "p", [0.75,0,1.75,1])]
        self.assertEqual(2, len(deduplicate_footprints(tiles)))

    def test_exact_geometry_is_preferred_for_cross_project_dedup(self):
        geometry = polygon([0, 0, 1, 1])["geometry"]
        left = tile("a", "old", [0, 0, 1.2, 1]) | {"geometry": geometry}
        right = tile("b", "new", [-0.2, 0, 1, 1]) | {"geometry": geometry}
        self.assertEqual(1, len(deduplicate_footprints([left, right])))

    def test_real_tnm_record_matches_registry_candidate_without_using_source_id(self):
        record = {
            "title": "USGS Lidar Point Cloud OR_OLCMetro_2019_A19 w2051n2776",
            "sourceId": "639d2f03d34e0de3a1f25974",
            "vendorMetaUrl": (
                "https://prd-tnm.s3.amazonaws.com/index.html?prefix="
                "StagedProducts/Elevation/metadata/OR_OLCMetro_2019_A19/OR_OLCMetro_2019"
            ),
            "downloadURL": (
                "https://rockyweb.usgs.gov/vdelivery/Datasets/Staged/Elevation/LPC/Projects/"
                "OR_OLCMetro_2019_A19/OR_OLCMetro_2019/LAZ/"
                "USGS_LPC_OR_OLCMetro_2019_A19_w2051n2776.laz"
            ),
            "sizeInBytes": 167885409,
            "boundingBox": {
                "minX": -122.54091628731729,
                "maxX": -122.52511901558759,
                "minY": 45.37796153850478,
                "maxY": 45.38908672861234,
            },
        }
        selected = select_tiles(
            [record], [], project="USGS_LPC_OR_OLCMetro_2019", max_tiles=1,
            negative_quota=1,
        )
        self.assertEqual([record["sourceId"]], [item["sourceId"] for item in selected])

    def test_overlap_dedup_keeps_higher_ranked_tile(self):
        tiles = [tile("a", "p", [0,0,1,1]), tile("b", "p", [0.5,0,1.5,1])]
        selected = select_tiles(tiles, [polygon([1,0,1.5,1])], project="p", max_tiles=1, negative_quota=0)
        self.assertEqual("b", selected[0]["title"])

    def test_feature_collection_properties_are_used_for_metadata(self):
        feature = {
            "type": "Feature",
            "properties": {"title": "nested", "project": "p", "sizeInBytes": 100},
            "geometry": polygon([0,0,1,1])["geometry"],
        }
        selected = select_tiles([feature], [], project="p", max_total_gb=0.000001, negative_quota=1)
        self.assertEqual(1, len(selected))

    def test_unknown_size_is_rejected_when_budget_is_active(self):
        value = tile("unknown", "p", [0,0,1,1])
        value.pop("sizeInBytes")
        with self.assertRaisesRegex(ValueError, "missing size"):
            select_tiles([value], [], project="p", max_total_gb=1)

    def test_probe_backtracks_to_find_complete_colocation(self):
        tiles = [
            tile("a1", "a", [0,0,1,1]), tile("a2", "a", [0.2,0,1.2,1]),
            tile("b1", "b", [0.05,0,1.05,1]), tile("b2", "b", [-0.2,0,0.8,1]),
        ]
        result = select_probe_tiles(tiles, projects=["a", "b"], count=2, overlap_threshold=0.6)
        self.assertEqual(["b2", "b1"], [item["title"] for item in result["b"]])

    def test_probe_returns_matching_footprints_for_every_project(self):
        tiles = [
            tile("a1", "a", [0,0,1,1]), tile("a2", "a", [2,0,3,1]),
            tile("b1", "b", [0.02,0,1.02,1]), tile("b2", "b", [2.02,0,3.02,1]),
            tile("b3", "b", [9,9,10,10]),
        ]
        result = select_probe_tiles(tiles, projects=["a", "b"], count=2)
        self.assertEqual({"a": 2, "b": 2}, {key: len(value) for key, value in result.items()})


if __name__ == "__main__":
    unittest.main()
