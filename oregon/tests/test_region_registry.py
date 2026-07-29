from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

OREGON_DIR = Path(__file__).resolve().parents[1]
if str(OREGON_DIR) not in sys.path:
    sys.path.insert(0, str(OREGON_DIR))

from region_registry import (
    load_registry,
    pin_region_decision,
    resolve_path,
    resolve_region,
    validate_registry,
)


class RegionRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = load_registry()

    def test_checked_in_registry_has_four_active_regions_and_budget(self) -> None:
        self.assertEqual([entry["id"] for entry in self.registry["regions"]], ["R1", "R2", "R3", "R4"])
        self.assertEqual(sum(entry["tile_budget"] for entry in self.registry["regions"]), 125)
        self.assertEqual(len([entry for entry in [*self.registry["regions"], *self.registry["comparison_candidates"]] if entry.get("include_in_candidate_report")]), 5)

    def test_oregon_city_resolves_to_existing_data(self) -> None:
        region = resolve_region("oregon_city", self.registry)
        self.assertEqual(region["id"], "R3")
        self.assertTrue(region["existing_data"])
        self.assertTrue(resolve_path(region, "slido_output").is_file())
        self.assertTrue((OREGON_DIR / region["lidar_dir"]).is_dir())

    def test_rejects_unknown_role(self) -> None:
        data = copy.deepcopy(self.registry)
        data["regions"][0]["role"] = "training"
        with self.assertRaisesRegex(ValueError, "role is unknown"):
            validate_registry(data)

    def test_rejects_overlapping_active_bboxes(self) -> None:
        data = copy.deepcopy(self.registry)
        data["regions"][1]["bbox"] = list(data["regions"][0]["bbox"])
        with self.assertRaisesRegex(ValueError, "bboxes overlap"):
            validate_registry(data)

    def test_rejects_missing_path(self) -> None:
        data = copy.deepcopy(self.registry)
        del data["regions"][0]["slido_output"]
        with self.assertRaisesRegex(ValueError, "path is required"):
            validate_registry(data)

    def test_rejects_empty_candidate_projects(self) -> None:
        data = copy.deepcopy(self.registry)
        data["regions"][0]["candidate_projects"] = []
        with self.assertRaisesRegex(ValueError, "non-empty"):
            validate_registry(data)

    def test_unpinned_entry_is_valid_but_partial_or_invalid_pins_are_rejected(self) -> None:
        data = copy.deepcopy(self.registry)
        validate_registry(data)
        data["regions"][0]["lidar_project"] = "not-a-candidate"
        with self.assertRaisesRegex(ValueError, "define lidar_project, cell_size, and selection_decision together"):
            validate_registry(data)

        entry = data["regions"][0]
        entry["cell_size"] = 0
        entry["selection_decision"] = {"reason": "measured probe comparison"}
        with self.assertRaisesRegex(ValueError, "one of candidate_projects"):
            validate_registry(data)
        entry["lidar_project"] = entry["candidate_projects"][0]
        with self.assertRaisesRegex(ValueError, "cell_size must be positive"):
            validate_registry(data)
        entry["cell_size"] = 1.5
        entry["selection_decision"] = {"reason": ""}
        with self.assertRaisesRegex(ValueError, "reason must be a non-empty string"):
            validate_registry(data)

    def test_pin_region_decision_validates_and_replaces_registry_atomically(self) -> None:
        data = copy.deepcopy(self.registry)
        data.pop("_registry_path", None)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "regions.json"
            original = json.dumps(data, indent=2) + "\n"
            path.write_text(original, encoding="utf-8")
            project = data["regions"][0]["candidate_projects"][1]
            pinned = pin_region_decision(
                path,
                "R1",
                lidar_project=project,
                cell_size=1.5,
                reason="best measured coverage and fragmentation tradeoff",
                decision_metadata={"probe_output": "probe.json"},
            )
            self.assertEqual(project, pinned["lidar_project"])
            self.assertEqual(1.5, pinned["cell_size"])
            self.assertEqual("probe.json", pinned["selection_decision"]["probe_output"])
            validate_registry(json.loads(path.read_text(encoding="utf-8")))

            stable = path.read_text(encoding="utf-8")
            with mock.patch("region_registry.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    pin_region_decision(
                        path,
                        "R1",
                        lidar_project=project,
                        cell_size=2.0,
                        reason="another measured decision",
                    )
            self.assertEqual(stable, path.read_text(encoding="utf-8"))
            self.assertEqual([], list(path.parent.glob(".regions.json.*.tmp")))

            with self.assertRaisesRegex(ValueError, "one of candidate_projects"):
                pin_region_decision(
                    path,
                    "R1",
                    lidar_project="not-a-candidate",
                    cell_size=1.0,
                    reason="invalid",
                )


if __name__ == "__main__":
    unittest.main()
