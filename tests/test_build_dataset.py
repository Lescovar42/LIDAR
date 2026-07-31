import copy
import json
import sys
import tempfile
import types
import unittest
from datetime import date
from pathlib import Path
from unittest import mock

import numpy as np
from pyproj import CRS
from rasterio.transform import Affine
from shapely.geometry import box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oregon"))
from build_dataset import (
    PATCH_FIELDS,
    RegionInput,
    SpatialTile,
    TileMetadata,
    _build_parser,
    _region_summaries,
    acquisition_scalar_errors,
    assign_spatial_splits,
    cli_acquisition,
    missing_acquisition_regions,
    parse_lidar_year,
    rasterize_tile_mask,
    read_tile_metadata,
    resolve_build_cell_size,
    resolve_region_inputs,
    tile_matches_project,
)
from lidar_vintage import (
    CLI_ORIGIN,
    REGISTRY_ORIGIN,
    AcquisitionMetadataError,
    LidarVintage,
    acquisition_from_cli,
    file_metadata_from_header,
    infer_year_hint,
    parse_acquisition,
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
            # Pin the project R1's acquisition metadata is declared for; pinning a
            # different project is rejected on purpose.
            entry["lidar_project"] = entry["lidar_acquisition"]["lidar_project"]
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

    def test_lidar_year_hint_parser_is_non_authoritative_but_unchanged(self):
        self.assertEqual(2019, parse_lidar_year("USGS_LPC_OR_OLCMetro_2019"))
        self.assertEqual(2022, parse_lidar_year("OR_WesternWildfires_A22_w123n456"))
        self.assertIsNone(parse_lidar_year("project_without_year"))


TILLAMOOK_PROJECT = "OR_WesternWildfires_A22"
A22_TILE_NAME = "USGS_LPC_OR_WesternWildfires_A22_w2051n2776.laz"


def registry_acquisition(**overrides):
    payload = {
        "start_year": 2020,
        "end_year": 2020,
        "nominal_year": 2020,
        "source": "USGS project/acquisition metadata",
        "evidence": "regions/tillamook_lidar_acquisition.md",
        "verified": True,
        "lidar_project": TILLAMOOK_PROJECT,
    }
    payload.update(overrides)
    return parse_acquisition(payload, origin=REGISTRY_ORIGIN)


def stub_laspy(creation_date, *, crs="EPSG:32610"):
    """A lightweight LAS header stub; no LAZ file or laspy install required."""

    class Header:
        mins = (450_000.0, 5_030_000.0, 0.0)
        maxs = (451_000.0, 5_031_000.0, 100.0)

        def __init__(self):
            self.creation_date = creation_date

        def parse_crs(self):
            return crs

    class Reader:
        def __init__(self):
            self.header = Header()

        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

    module = types.ModuleType("laspy")
    module.open = lambda *args, **kwargs: Reader()
    return module


class TileVintageTests(unittest.TestCase):
    def region(self, acquisition):
        return RegionInput(
            region_id="R1",
            region_role="train_val",
            laz_dir=Path("tiles"),
            slido_path=Path("slido.geojson"),
            lidar_project_hint=TILLAMOOK_PROJECT,
            cell_size=1.5,
            lidar_project_pinned=True,
            lidar_acquisition=acquisition,
        )

    def read(self, acquisition, *, creation_date=date(2024, 3, 14), tile=A22_TILE_NAME):
        with mock.patch.dict(sys.modules, {"laspy": stub_laspy(creation_date)}):
            return read_tile_metadata(Path(tile), self.region(acquisition))

    def test_header_2024_and_a22_hint_cannot_override_acquisition_2020(self):
        item = self.read(registry_acquisition())

        self.assertEqual(2020, item.vintage.acquisition_nominal_year)
        self.assertEqual(2020, item.lidar_year)
        self.assertEqual(REGISTRY_ORIGIN, item.lidar_year_source)
        self.assertEqual(2024, item.vintage.file_metadata.creation_year)
        self.assertEqual(2022, item.vintage.hint.year)
        self.assertEqual("project_name", item.vintage.hint.source)
        self.assertEqual(2020, item.slido_lidar_year())

    def test_header_alone_leaves_acquisition_unknown(self):
        item = self.read(None)

        self.assertIsNone(item.vintage.acquisition_nominal_year)
        self.assertIsNone(item.lidar_year)
        self.assertEqual("unknown", item.lidar_year_source)
        self.assertEqual(2024, item.vintage.file_metadata.creation_year)
        self.assertEqual(2022, item.vintage.hint.year)
        self.assertIsNone(item.slido_lidar_year())

    def test_acquisition_declared_for_another_project_is_rejected(self):
        acquisition = registry_acquisition(lidar_project="OR_TILLAMOOK_ODF_2007")
        with self.assertRaisesRegex(RuntimeError, "does not belong to it"):
            self.read(acquisition, tile=A22_TILE_NAME)

    def test_patch_row_and_tile_summary_carry_full_provenance(self):
        item = self.read(registry_acquisition())
        row_fields = item.vintage.as_row_fields()

        self.assertEqual(2020, row_fields["lidar_acquisition_year"])
        self.assertEqual(2020, row_fields["lidar_acquisition_start_year"])
        self.assertEqual(2020, row_fields["lidar_acquisition_end_year"])
        self.assertEqual("true", row_fields["lidar_acquisition_verified"])
        self.assertEqual(2024, row_fields["lidar_file_creation_year"])
        self.assertEqual("2024-03-14", row_fields["lidar_file_creation_date"])
        self.assertEqual(2022, row_fields["lidar_inferred_year_hint"])
        self.assertEqual(2020, row_fields["lidar_year"])
        self.assertTrue(set(row_fields) <= set(PATCH_FIELDS))

        summary = item.vintage.as_summary_mapping()
        self.assertEqual(2020, summary["acquisition"]["nominal_year"])
        self.assertEqual(2024, summary["file_metadata"]["creation_year"])
        self.assertEqual(2022, summary["inferred_year_hint"]["year"])

    def test_patch_fields_declare_every_provenance_column_once(self):
        self.assertEqual(len(PATCH_FIELDS), len(set(PATCH_FIELDS)))
        for name in (
            "lidar_acquisition_year",
            "lidar_acquisition_start_year",
            "lidar_acquisition_end_year",
            "lidar_acquisition_source",
            "lidar_acquisition_evidence",
            "lidar_acquisition_verified",
            "lidar_file_creation_year",
            "lidar_file_creation_date",
            "lidar_inferred_year_hint",
            "lidar_inferred_year_hint_source",
            "lidar_year",
            "lidar_year_source",
        ):
            self.assertIn(name, PATCH_FIELDS)


class SlidoTemporalFilterProtectionTests(unittest.TestCase):
    """Acceptance test 9: only the acquisition year reaches temporal filtering."""

    def metadata(self, acquisition):
        return TileMetadata(
            path=Path(A22_TILE_NAME),
            tile_id=f"R1:{A22_TILE_NAME}",
            region_id="R1",
            region_role="train_val",
            lidar_project=TILLAMOOK_PROJECT,
            vintage=LidarVintage(
                acquisition=acquisition,
                file_metadata=file_metadata_from_header(date(2024, 3, 14)),
                hint=infer_year_hint(TILLAMOOK_PROJECT, A22_TILE_NAME),
            ),
            source_crs=CRS.from_epsg(32610),
            metric_footprint=box(0, 0, 1, 1),
        )

    def capture(self, acquisition):
        calls = []

        def fake_rasterize(path, tile, **kwargs):
            calls.append(kwargs)
            return np.zeros((2, 2), dtype=np.uint8), []

        region = RegionInput(
            "R1", "train_val", Path("tiles"), Path("slido.geojson"),
            lidar_acquisition=acquisition,
        )
        rasterize_tile_mask(
            self.metadata(acquisition),
            region,
            tile(),
            positive_buffer_m=50.0,
            rasterize=fake_rasterize,
        )
        return calls[0]

    def test_rasterizer_receives_the_acquisition_year_only(self):
        kwargs = self.capture(registry_acquisition())

        self.assertEqual(2020, kwargs["lidar_year"])
        self.assertNotEqual(2024, kwargs["lidar_year"])
        self.assertNotEqual(2022, kwargs["lidar_year"])
        self.assertEqual(50.0, kwargs["positive_buffer_m"])

    def test_unknown_acquisition_passes_none_not_the_header_year(self):
        kwargs = self.capture(None)

        self.assertIsNone(kwargs["lidar_year"])

    def test_multi_year_range_without_nominal_fails_instead_of_guessing(self):
        acquisition = parse_acquisition(
            {
                "start_year": 2008,
                "end_year": 2009,
                "source": "USGS project/acquisition metadata",
                "evidence": "regions/example.md",
            },
            origin=REGISTRY_ORIGIN,
        )
        with self.assertRaisesRegex(AcquisitionMetadataError, "requires one authoritative"):
            self.capture(acquisition)


class AcquisitionCliResolutionTests(unittest.TestCase):
    def write_registry(self, directory, mutate=None):
        data = copy.deepcopy(load_registry())
        data.pop("_registry_path", None)
        if mutate is not None:
            mutate(data)
        path = Path(directory) / "regions.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def parse(self, argv):
        parser = _build_parser()
        return parser, parser.parse_args(argv)

    def test_legacy_build_accepts_explicit_acquisition_arguments(self):
        parser, args = self.parse(
            [
                "--lidar-acquisition-year", "2020",
                "--lidar-acquisition-source", "User-verified project metadata",
                "--lidar-acquisition-evidence", "regions/tillamook_lidar_acquisition.md",
                "--lidar-acquisition-verified",
            ]
        )
        regions = resolve_region_inputs(args, parser)

        self.assertEqual(1, len(regions))
        acquisition = regions[0].lidar_acquisition
        assert acquisition is not None
        self.assertEqual(2020, acquisition.nominal_year)
        self.assertEqual("User-verified project metadata", acquisition.source)
        self.assertTrue(acquisition.verified)
        self.assertEqual(CLI_ORIGIN, acquisition.origin)
        self.assertEqual([], missing_acquisition_regions(regions))
        self.assertEqual([], acquisition_scalar_errors(regions))

    def test_deprecated_lidar_year_alias_routes_to_acquisition(self):
        _, args = self.parse(
            ["--lidar-year", "2020", "--lidar-acquisition-source", "project report"]
        )
        acquisition = cli_acquisition(args)

        assert acquisition is not None
        self.assertEqual(2020, acquisition.nominal_year)

    def test_legacy_build_without_acquisition_is_unknown_and_gated(self):
        parser, args = self.parse([])
        regions = resolve_region_inputs(args, parser)

        self.assertIsNone(regions[0].lidar_acquisition)
        self.assertEqual(["legacy"], missing_acquisition_regions(regions))
        self.assertFalse(args.allow_unknown_lidar_acquisition)

    def test_acquisition_year_without_source_is_rejected(self):
        parser, args = self.parse(["--lidar-acquisition-year", "2020"])
        with self.assertRaises(SystemExit):
            resolve_region_inputs(args, parser)

    def test_multi_year_range_without_nominal_year_is_reported(self):
        parser, args = self.parse(
            [
                "--lidar-acquisition-start-year", "2008",
                "--lidar-acquisition-end-year", "2009",
                "--lidar-acquisition-source", "USGS project report",
            ]
        )
        regions = resolve_region_inputs(args, parser)
        errors = acquisition_scalar_errors(regions)

        self.assertEqual(1, len(errors))
        self.assertIn("2008-2009", errors[0])

    def test_registry_region_supplies_acquisition_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_registry(tmp)
            parser, args = self.parse(
                [
                    "--region", "R1",
                    "--registry", str(path),
                    "--allow-unpinned-rural-diagnostic",
                    "--cell-size", "1.5",
                ]
            )
            regions = resolve_region_inputs(args, parser)

            acquisition = regions[0].lidar_acquisition
            assert acquisition is not None
            self.assertEqual(2020, acquisition.nominal_year)
            self.assertEqual(REGISTRY_ORIGIN, acquisition.origin)
            self.assertEqual(TILLAMOOK_PROJECT, acquisition.lidar_project)
            self.assertEqual([], missing_acquisition_regions(regions))

    def test_registry_region_without_acquisition_metadata_stays_unknown(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_registry(
                tmp, lambda data: data["regions"][0].pop("lidar_acquisition")
            )
            parser, args = self.parse(
                [
                    "--region", "R1",
                    "--registry", str(path),
                    "--allow-unpinned-rural-diagnostic",
                    "--cell-size", "1.5",
                ]
            )
            regions = resolve_region_inputs(args, parser)

            self.assertIsNone(regions[0].lidar_acquisition)
            self.assertEqual(["R1"], missing_acquisition_regions(regions))

    def test_cli_override_agreeing_with_registry_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_registry(tmp)
            parser, args = self.parse(
                [
                    "--region", "R1",
                    "--registry", str(path),
                    "--allow-unpinned-rural-diagnostic",
                    "--cell-size", "1.5",
                    "--lidar-acquisition-year", "2020",
                    "--lidar-acquisition-source", "CLI restatement of project metadata",
                ]
            )
            regions = resolve_region_inputs(args, parser)

            acquisition = regions[0].lidar_acquisition
            assert acquisition is not None
            self.assertEqual(2020, acquisition.nominal_year)
            self.assertEqual(CLI_ORIGIN, acquisition.origin)

    def test_conflicting_cli_and_registry_years_fail_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self.write_registry(tmp)
            parser, args = self.parse(
                [
                    "--region", "R1",
                    "--registry", str(path),
                    "--allow-unpinned-rural-diagnostic",
                    "--cell-size", "1.5",
                    "--lidar-acquisition-year", "2021",
                    "--lidar-acquisition-source", "contradictory claim",
                ]
            )
            with self.assertRaises(SystemExit):
                resolve_region_inputs(args, parser)


class ManifestPropagationTests(unittest.TestCase):
    """Acceptance tests 7 and 8, run against a fully synthetic tile.

    No real LAZ, NPZ, or NAIP data is touched: the LAS header is stubbed and the
    ground DEM reader is replaced with an in-memory grid.
    """

    CELL = 1.0
    SIZE = 32
    ORIGIN_X = 450_000.0
    ORIGIN_Y = 5_030_000.0

    def synthetic_tile(self, *args, **kwargs):
        return TerrainTile(
            dem=np.tile(
                np.linspace(0.0, 20.0, self.SIZE, dtype=np.float32), (self.SIZE, 1)
            ),
            transform=Affine(
                self.CELL, 0, self.ORIGIN_X,
                0, -self.CELL, self.ORIGIN_Y + self.SIZE * self.CELL,
            ),
            crs=CRS.from_epsg(32610),
            valid_ground_mask=np.ones((self.SIZE, self.SIZE), dtype=bool),
            ground_point_count=self.SIZE * self.SIZE,
            ground_cell_fraction=1.0,
            source_path=Path(A22_TILE_NAME),
        )

    def slido_geojson(self, directory):
        from pyproj import Transformer

        to_wgs84 = Transformer.from_crs(
            CRS.from_epsg(32610), CRS.from_epsg(4326), always_xy=True
        )
        x0, y0 = self.ORIGIN_X + 4, self.ORIGIN_Y + 4
        x1, y1 = self.ORIGIN_X + 20, self.ORIGIN_Y + 20
        lon0, lat0 = to_wgs84.transform(x0, y0)
        lon1, lat1 = to_wgs84.transform(x1, y1)
        path = Path(directory) / "slido.geojson"
        path.write_text(
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": [
                        {
                            "type": "Feature",
                            "properties": {
                                "DESCRIPTION": "Landslide",
                                "CONFIDENCE": "High (>30)",
                                "UNIQUE_ID": "slide-1",
                            },
                            "geometry": mapping(box(lon0, lat0, lon1, lat1)),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return path

    def run_build(self, tmp, extra_args):
        import build_dataset

        laz_dir = Path(tmp) / "tiles"
        laz_dir.mkdir()
        (laz_dir / A22_TILE_NAME).write_bytes(b"")
        outdir = Path(tmp) / "dataset"
        argv = [
            "build_dataset.py",
            "--laz-dir", str(laz_dir),
            "--slido-geojson", str(self.slido_geojson(tmp)),
            "--outdir", str(outdir),
            "--cell-size", str(self.CELL),
            "--patch-size", "8",
            "--stride", "8",
            "--min-ground-cell-fraction", "0.0",
            "--min-patch-ground-fraction", "0.2",
            "--positive-buffer-m", "1",
            "--negative-buffer-m", "1",
            "--overwrite",
            *extra_args,
        ]
        with mock.patch.dict(sys.modules, {"laspy": stub_laspy(date(2024, 3, 14))}), \
                mock.patch.object(sys, "argv", argv), \
                mock.patch.object(
                    build_dataset, "read_laz_ground_dem", self.synthetic_tile
                ):
            status = build_dataset.main()
        return status, outdir

    def read_patches(self, outdir):
        import csv

        with (outdir / "patches.csv").open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            return list(reader.fieldnames or []), list(reader)

    def test_patches_csv_and_dataset_summary_carry_acquisition_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, outdir = self.run_build(
                tmp,
                [
                    "--lidar-acquisition-year", "2020",
                    "--lidar-acquisition-source", "User-verified project metadata",
                    "--lidar-acquisition-evidence",
                    "regions/tillamook_lidar_acquisition.md",
                    "--lidar-acquisition-verified",
                ],
            )
            self.assertEqual(0, status)

            header, rows = self.read_patches(outdir)
            self.assertEqual(list(PATCH_FIELDS), header)
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual("2020", row["lidar_acquisition_year"])
                self.assertEqual("2020", row["lidar_acquisition_start_year"])
                self.assertEqual("2020", row["lidar_acquisition_end_year"])
                self.assertEqual(
                    "User-verified project metadata", row["lidar_acquisition_source"]
                )
                self.assertEqual(
                    "regions/tillamook_lidar_acquisition.md",
                    row["lidar_acquisition_evidence"],
                )
                self.assertEqual("true", row["lidar_acquisition_verified"])
                self.assertEqual("2024", row["lidar_file_creation_year"])
                self.assertEqual("2024-03-14", row["lidar_file_creation_date"])
                self.assertEqual("2022", row["lidar_inferred_year_hint"])
                self.assertEqual("project_name", row["lidar_inferred_year_hint_source"])
                # Compatibility alias mirrors acquisition, not the header year.
                self.assertEqual("2020", row["lidar_year"])
                self.assertEqual(CLI_ORIGIN, row["lidar_year_source"])

            summary = json.loads(
                (outdir / "dataset_summary.json").read_text(encoding="utf-8")
            )
            vintage = summary["lidar_vintage_summary"]
            self.assertEqual(1, vintage["distinct_acquisition_vintage_count"])
            self.assertEqual(2020, vintage["distinct_acquisition_vintages"][0]["nominal_year"])
            self.assertEqual(0, vintage["unknown_acquisition_tiles"])
            self.assertEqual({"2024": 1}, vintage["distinct_file_creation_years"])
            self.assertEqual({"2022": 1}, vintage["inferred_year_hints"])
            self.assertEqual([], vintage["acquisition_conflict_failures"])
            self.assertFalse(vintage["unknown_acquisition_allowed"])

            tile_summary = summary["tile_summaries"][0]
            self.assertEqual(2020, tile_summary["slido_temporal_filter_year"])
            self.assertEqual(
                2020, tile_summary["lidar_vintage"]["acquisition"]["nominal_year"]
            )
            self.assertEqual(
                2024, tile_summary["lidar_vintage"]["file_metadata"]["creation_year"]
            )
            self.assertEqual(
                2022, tile_summary["lidar_vintage"]["inferred_year_hint"]["year"]
            )
            self.assertEqual(
                2020, summary["region_summaries"]["legacy"]["lidar_acquisition"]["nominal_year"]
            )

    def test_build_without_acquisition_metadata_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                self.run_build(tmp, [])

    def test_diagnostic_flag_keeps_acquisition_unknown_without_inventing_a_year(self):
        with tempfile.TemporaryDirectory() as tmp:
            status, outdir = self.run_build(tmp, ["--allow-unknown-lidar-acquisition"])
            self.assertEqual(0, status)

            _, rows = self.read_patches(outdir)
            self.assertTrue(rows)
            for row in rows:
                self.assertEqual("", row["lidar_acquisition_year"])
                self.assertEqual("", row["lidar_year"])
                self.assertEqual("unknown", row["lidar_year_source"])
                self.assertEqual("2024", row["lidar_file_creation_year"])
                self.assertEqual("2022", row["lidar_inferred_year_hint"])

            summary = json.loads(
                (outdir / "dataset_summary.json").read_text(encoding="utf-8")
            )
            vintage = summary["lidar_vintage_summary"]
            self.assertEqual(0, vintage["distinct_acquisition_vintage_count"])
            self.assertEqual(1, vintage["unknown_acquisition_tiles"])
            self.assertTrue(vintage["unknown_acquisition_allowed"])
            self.assertIsNone(summary["tile_summaries"][0]["slido_temporal_filter_year"])


class UnknownAcquisitionProductionSafetyTests(unittest.TestCase):
    """Acceptance test 10: unknown acquisition needs a named diagnostic flag."""

    def test_flag_is_documented_and_off_by_default(self):
        parser = _build_parser()
        self.assertFalse(parser.parse_args([]).allow_unknown_lidar_acquisition)
        self.assertTrue(
            parser.parse_args(
                ["--allow-unknown-lidar-acquisition"]
            ).allow_unknown_lidar_acquisition
        )

    def test_missing_acquisition_regions_lists_every_affected_region(self):
        regions = [
            RegionInput("R1", "train_val", Path("a"), Path("a.geojson"),
                        lidar_acquisition=registry_acquisition()),
            RegionInput("R2", "test_rural", Path("b"), Path("b.geojson")),
            RegionInput("R4", "train_val", Path("c"), Path("c.geojson")),
        ]
        self.assertEqual(["R2", "R4"], missing_acquisition_regions(regions))

    def test_region_summary_reports_region_acquisition(self):
        regions = [
            RegionInput("R1", "train_val", Path("a"), Path("a.geojson"),
                        lidar_acquisition=registry_acquisition()),
            RegionInput("R2", "test_rural", Path("b"), Path("b.geojson")),
        ]
        summaries = _region_summaries(regions, [], [], [])

        self.assertEqual(2020, summaries["R1"]["lidar_acquisition"]["nominal_year"])
        self.assertIsNone(summaries["R2"]["lidar_acquisition"])


if __name__ == "__main__":
    unittest.main()
