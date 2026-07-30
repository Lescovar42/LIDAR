#!/usr/bin/env python3
"""Apply the large-TNM-catalog recursion fix to oregon/select_tiles.py.

The original probe selector recursively skips one anchor at a time. With more
than roughly 1,000 anchors, Python reaches its recursion limit before it finds
co-located footprints. This hotfix builds complete candidate sets iteratively,
then backtracks only by selection depth, so recursion depth is bounded by the
requested probe count.
"""
from __future__ import annotations

import argparse
import py_compile
import shutil
from pathlib import Path

NEW_FUNCTION = r'''def select_probe_tiles(
    tiles: Sequence[Mapping[str, Any]],
    *,
    projects: Sequence[str],
    count: int,
    overlap_threshold: float = 0.8,
    max_total_gb: float | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Return exactly ``count`` pairwise co-located footprints per project.

    Candidate footprint sets are enumerated iteratively. Backtracking then
    recurses only once per selected set, rather than once per TNM anchor record.
    This keeps recursion depth bounded by ``count`` even for catalogs containing
    thousands of unmatched anchors.
    """
    if count < 0:
        raise ValueError("count must be non-negative")
    if max_total_gb is not None and max_total_gb < 0:
        raise ValueError("max_total_gb must be non-negative")
    if not 0 < overlap_threshold <= 1:
        raise ValueError("overlap_threshold must be in (0, 1]")

    unique_projects = list(dict.fromkeys(projects))
    if len(unique_projects) < 2:
        raise ValueError("probe selection requires at least two projects")

    grouped = {
        project: sorted(
            [dict(tile) for tile in tiles if project_matches(tile, project)],
            key=lambda item: tile_id(item).casefold(),
        )
        for project in unique_projects
    }
    budget_bytes = (
        math.floor(max_total_gb * 1_000_000_000)
        if max_total_gb is not None
        else 2**63 - 1
    )
    if max_total_gb is not None:
        unknown_sizes = [
            tile_id(tile)
            for project_tiles in grouped.values()
            for tile in project_tiles
            if tile_size(tile) <= 0
        ]
        if unknown_sizes:
            raise ValueError(
                "Cannot enforce probe byte budget; missing size for: "
                + ", ".join(unknown_sizes[:5])
            )
    if count == 0:
        return {project: [] for project in unique_projects}

    anchor_project = unique_projects[0]
    anchors = grouped[anchor_project]
    geometries = {
        (project, tile_id(tile)): tile_footprint(tile)
        for project, project_tiles in grouped.items()
        for tile in project_tiles
    }

    # Each entry is one complete, pairwise-compatible project set plus its bytes.
    # Enumeration is iterative, so catalogs with >1,000 unmatched anchors cannot
    # exhaust Python's recursion limit.
    candidate_sets: list[tuple[dict[str, dict[str, Any]], int]] = []
    for anchor in anchors:
        anchor_geometry = geometries[(anchor_project, tile_id(anchor))]
        options: list[list[dict[str, Any]]] = []

        for project in unique_projects[1:]:
            scored_matches: list[tuple[float, dict[str, Any]]] = []
            for tile in grouped[project]:
                iou = footprint_iou(
                    anchor_geometry,
                    geometries[(project, tile_id(tile))],
                )
                if iou >= overlap_threshold:
                    scored_matches.append((iou, tile))

            scored_matches.sort(
                key=lambda item: (-item[0], tile_id(item[1]).casefold())
            )
            matches = [tile for _, tile in scored_matches]
            if not matches:
                options = []
                break
            options.append(matches)

        for combination in product(*options) if options else ():
            selected = {anchor_project: anchor} | dict(
                zip(unique_projects[1:], combination)
            )
            selected_geometries = [
                geometries[(project, tile_id(selected[project]))]
                for project in unique_projects
            ]
            if any(
                footprint_iou(selected_geometries[left], selected_geometries[right])
                < overlap_threshold
                for left in range(len(selected_geometries))
                for right in range(left + 1, len(selected_geometries))
            ):
                continue

            set_bytes = sum(tile_size(selected[project]) for project in unique_projects)
            if set_bytes > budget_bytes:
                continue
            candidate_sets.append((selected, set_bytes))

    # Search over complete candidate sets. Recursion depth is at most ``count``.
    selected_sets: list[dict[str, dict[str, Any]]] = []
    used = {project: set() for project in unique_projects}

    def choose(start_index: int, used_bytes: int) -> bool:
        if len(selected_sets) == count:
            return True

        needed = count - len(selected_sets)
        if len(candidate_sets) - start_index < needed:
            return False

        for candidate_index in range(start_index, len(candidate_sets)):
            selected, set_bytes = candidate_sets[candidate_index]
            if used_bytes + set_bytes > budget_bytes:
                continue

            selected_ids = {
                project: tile_id(selected[project]) for project in unique_projects
            }
            if any(
                selected_ids[project] in used[project]
                for project in unique_projects
            ):
                continue

            selected_sets.append(selected)
            for project in unique_projects:
                used[project].add(selected_ids[project])

            if choose(candidate_index + 1, used_bytes + set_bytes):
                return True

            for project in unique_projects:
                used[project].remove(selected_ids[project])
            selected_sets.pop()

        return False

    if not choose(0, 0):
        complete_anchor_count = len(
            {
                tile_id(candidate[anchor_project])
                for candidate, _ in candidate_sets
            }
        )
        budget_note = (
            f" within {max_total_gb:g} GB"
            if max_total_gb is not None
            else ""
        )
        raise ValueError(
            f"Could not select {count} unique co-located sets at IoU "
            f">= {overlap_threshold:g}{budget_note}; "
            f"{complete_anchor_count} anchor footprints had at least one "
            "complete project match"
        )

    return {
        project: [selected[project] for selected in selected_sets]
        for project in unique_projects
    }
'''

REGRESSION_TEST = r'''import sys
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
'''


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root containing oregon/select_tiles.py",
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source = repo_root / "oregon" / "select_tiles.py"
    if not source.is_file():
        raise SystemExit(f"Missing {source}")

    text = source.read_text(encoding="utf-8")
    start_marker = "def select_probe_tiles("
    end_marker = "\ndef _load_json"
    start = text.find(start_marker)
    end = text.find(end_marker, start)
    if start < 0 or end < 0:
        raise SystemExit(
            "Could not locate select_probe_tiles() boundaries; no files changed."
        )

    current = text[start:end]
    already_fixed = "candidate_sets: list[tuple[dict[str, dict[str, Any]], int]]" in current
    if already_fixed:
        print(f"Already fixed: {source}")
    else:
        if "return search(anchor_index + 1, used_bytes)" not in current:
            raise SystemExit(
                "select_probe_tiles() does not match the expected recursive version; "
                "no files changed."
            )
        backup = source.with_suffix(source.suffix + ".bak_probe_recursion")
        if not backup.exists():
            shutil.copy2(source, backup)
            print(f"Backup: {backup}")
        updated = text[:start] + NEW_FUNCTION.rstrip() + text[end:]
        source.write_text(updated, encoding="utf-8")
        print(f"Updated: {source}")

    test_path = repo_root / "tests" / "test_select_tiles_probe_scaling.py"
    test_path.parent.mkdir(parents=True, exist_ok=True)
    test_path.write_text(REGRESSION_TEST, encoding="utf-8")
    print(f"Regression test: {test_path}")

    py_compile.compile(str(source), doraise=True)
    py_compile.compile(str(test_path), doraise=True)
    print("Compilation: OK")
    print()
    print("Next:")
    print(r"  python -m unittest .\tests\test_select_tiles.py .\tests\test_select_tiles_probe_scaling.py -v")
    print(r"  cd .\oregon")
    print(r"  .\run_tillamook_probe.ps1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
