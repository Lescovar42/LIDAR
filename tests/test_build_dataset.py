import copy
import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import numpy as np
from pyproj import CRS
from rasterio.transform import Affine
from shapely.geometry import box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oregon"))
from build_dataset import (
    RegionInput,
    SpatialTile,
    _build_parser,
    _region_summaries,
    assign_spatial_splits,
    determine_lidar_year,
    parse_lidar_year,
    resolve_build_cell_size,
    resolve_region_inputs,
    tile_matches_project,
)
from region_registry import load_registry
from terrain_utils import TerrainTile, rasterize_slido_mask


def feature(bounds, confidence, event_date=None, description="Landslide"):
    properties = {"DESCRIPTION": description, "CONFIDENCE": confidence}
    if event_date is not None:
        properties["EVENT_DATE"] = event_date
    return {"type": "Feature", "properties": properties, "geometry": mapping(box(*bounds))}


def tile(width=4, height=4):
    return TerrainTile(
        dem=np.zeros((height, width), dtype=np.float32),
        transform=Affine(1, 0, 0, 0, -1, height),
        crs=CRS.from_epsg(4326),
        valid_ground_mask=np.ones((height, width), dtype=bool),
        ground_point_count=width * height,
        ground_cell_fraction=1.0,
        source_path=Path("synthetic.laz"),
    )


class LabelRasterizationTests(unittest.TestCase):
    def write_geojson(self, directory, features):
        path = Path(directory) / "labels.geojson"
        path.write_text(json.dumps({"type": "FeatureCollection", "features": features}), encoding="utf-8")
        return path

    def test_all_confidences_and_pre_post_lidar_dates_have_exact_codes(self):
        features = [
            feature((0, 3, 1, 4), "High (>30)", "2000-01-01"),
            feature((1, 3, 2, 4), "Moderate", None),
            feature((2, 3, 3, 4), "Low", "2000"),
            feature((3, 3, 4, 4), "Unmapped", "2000"),
            feature((0, 2, 1, 3), "High", "2021-01-01"),
            feature((1, 2, 2, 3), "Moderate", "2020"),
            feature((2, 2, 3, 3), "High", "1990", description="Fan"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            mask, records = rasterize_slido_mask(
                self.write_geojson(tmp, features), tile(), lidar_year=2020, positive_buffer_m=0
            )
        expected = np.zeros((4, 4), dtype=np.uint8)
        expected[0] = [1, 1, 255, 255]
        expected[1, :2] = [255, 1]
        np.testing.assert_array_equal(expected, mask)
        self.assertEqual(6, len(records))
        self.assertEqual(1, sum(bool(record["temporal_excluded"]) for record in records))

    def test_temporal_exclusion_overrides_positive_overlap(self):
        features = [
            feature((1, 1, 2, 2), "High", "2000"),
            feature((1, 1, 2, 2), "Moderate", "2022"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            mask, _ = rasterize_slido_mask(self.write_geojson(tmp, features), tile(), lidar_year=2020)
        self.assertEqual(255, int(mask[2, 1]))

    def test_positive_buffer_is_ignore_ring_but_not_positive_interior(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_geojson(tmp, [feature((2, 2, 3, 3), "High", None)])
            mask, _ = rasterize_slido_mask(path, tile(5, 5), lidar_year=2020, positive_buffer_m=1)
        self.assertEqual(1, int(mask[2, 2]))
        self.assertEqual(255, int(mask[2, 1]))
        self.assertEqual(255, int(mask[1, 2]))
        self.assertEqual(0, int(mask[1, 1]))


class RegistryBuildInputTests(unittest.TestCase):
    def write_registry(self, directory, mutate=None):
        data = copy.deepcopy(load_registry())
        data.pop("_registry_path", None)
        if mutate is not None:
            mutate(data)
        path = Path(directory) / "regions.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def test_registry_build_refuses_unpinned_rural_region_without_diagnostic_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_registry(tmp)
            parser = _build_parser()
            args = parser.parse_args(["--region", "R1", "--registry", str(path)])
            with self.assertRaises(SystemExit):
                resolve_region_inputs(args, parser)

    def test_registry_build_consumes_pinned_project_and_cell_size(self):
        def pin_r1(data):
            entry = data["regions"][0]
            entry["lidar_project"] = entry["candidate_projects"][1]
            entry["cell_size"] = 1.5
            entry["selection_decision"] = {"reason": "measured diagnostic result"}

        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_registry(tmp, pin_r1)
            parser = _build_parser()
            args = parser.parse_args(["--region", "R1", "--registry", str(path)])
            regions = resolve_region_inputs(args, parser)
            self.assertEqual(load_registry(path)["regions"][0]["lidar_project"], regions[0].lidar_project_hint)
            self.assertTrue(regions[0].lidar_project_pinned)
            self.assertEqual(1.5, regions[0].cell_size)
            self.assertEqual(1.5, resolve_build_cell_size(args.cell_size, regions, parser))
            self.assertTrue(tile_matches_project(
                Path(f"USGS_LPC_{regions[0].lidar_project_hint}_w1234n5678.laz"),
                regions[0].lidar_project_hint,
            ))
            self.assertFalse(tile_matches_project(
                Path("USGS_LPC_DIFFERENT_PROJECT_w1234n5678.laz"),
                regions[0].lidar_project_hint,
            ))

    def test_registry_build_rejects_incompatible_pinned_cell_sizes(self):
        parser = _build_parser()
        regions = [
            RegionInput("R1", "train_val", Path("r1"), Path("r1.json"), "p1", 1.0),
            RegionInput("R2", "test_rural", Path("r2"), Path("r2.json"), "p2", 2.0),
        ]
        with self.assertRaises(SystemExit):
            resolve_build_cell_size(None, regions, parser)

    def test_legacy_cell_size_default_remains_one_meter(self):
        parser = _build_parser()
        legacy = [RegionInput("legacy", "train_val", Path("tiles"), Path("labels.json"))]
        self.assertEqual(1.0, resolve_build_cell_size(None, legacy, parser))


class SpatialSplitTests(unittest.TestCase):
    def test_whole_blocks_share_assignment_and_roles_are_forced(self):
        tiles = [
            SpatialTile("a", "R1", "train_val", box(10, 10, 20, 20)),
            SpatialTile("b", "R1", "train_val", box(30, 30, 40, 40)),
            SpatialTile("c", "R1", "train_val", box(110, 10, 120, 20)),
            SpatialTile("rural", "R2", "test_rural", box(0, 1000, 10, 1010)),
            SpatialTile("urban", "R3", "test_urban_ood", box(0, 2000, 10, 2010)),
        ]
        result = assign_spatial_splits(tiles, block_size_m=100, split_buffer_m=0, seed=7)
        self.assertEqual(result.assignments["a"], result.assignments["b"])
        self.assertEqual("test_rural", result.assignments["rural"])
        self.assertEqual("test_urban_ood", result.assignments["urban"])
        self.assertEqual({"train", "validation"}, {result.assignments["a"], result.assignments["c"]})

    def test_tiles_near_differently_assigned_block_are_dropped(self):
        tiles = [
            SpatialTile("far", "R1", "train_val", box(10, 10, 20, 20)),
            SpatialTile("left_edge", "R1", "train_val", box(95, 10, 99, 20)),
            SpatialTile("right_edge", "R1", "train_val", box(101, 10, 105, 20)),
        ]
        result = assign_spatial_splits(tiles, block_size_m=100, split_buffer_m=5, seed=1)
        self.assertIn("far", result.assignments)
        self.assertIn("left_edge", result.dropped)
        self.assertIn("right_edge", result.dropped)

    def test_region_summary_keeps_failed_region_and_deduplicates_temporal_polygons(self):
        regions = [
            RegionInput("R1", "train_val", Path("r1"), Path("r1.geojson")),
            RegionInput("R2", "test_rural", Path("r2"), Path("r2.geojson")),
        ]
        tile_summaries = [
            {"region_id": "R1", "mask_pixels": 100, "ignore_pixels": 10, "temporally_excluded_polygon_keys": ["slide-1"]},
            {"region_id": "R1", "mask_pixels": 100, "ignore_pixels": 20, "temporally_excluded_polygon_keys": ["slide-1"]},
        ]
        summaries = _region_summaries(
            regions,
            tile_summaries,
            [{"region_id": "R1", "split": "train", "category": "negative"}],
            [{"region_id": "R2", "tile_name": "", "error": "no tiles"}],
        )
        self.assertEqual(1, summaries["R1"]["temporally_excluded_polygon_count"])
        self.assertAlmostEqual(0.15, summaries["R1"]["ignore_pixel_fraction"])
        self.assertEqual("incomplete", summaries["R2"]["status"])
        self.assertEqual(1, summaries["R2"]["failed_or_dropped_tiles"])

    def test_lidar_year_fallback_parses_project_and_tnm_tokens(self):
        self.assertEqual(2019, parse_lidar_year("USGS_LPC_OR_OLCMetro_2019"))
        self.assertEqual(2022, parse_lidar_year("OR_WesternWildfires_A22_w123n456"))
        self.assertIsNone(parse_lidar_year("project_without_year"))

    def test_trustworthy_header_year_precedes_project_and_source_is_recorded(self):
        self.assertEqual(
            (2018, "las_header.creation_date"),
            determine_lidar_year(date(2018, 4, 2), "project_2019", "tile_2020.laz"),
        )
        self.assertEqual(
            (2019, "project_name"),
            determine_lidar_year(None, "project_2019", "tile_2020.laz"),
        )


if __name__ == "__main__":
    unittest.main()
