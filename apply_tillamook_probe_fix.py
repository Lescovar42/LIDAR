#!/usr/bin/env python3
from __future__ import annotations
import argparse, py_compile, shutil
from pathlib import Path

NEW_FUNCTION = 'def select_probe_tiles(\n    tiles: Sequence[Mapping[str, Any]],\n    *,\n    projects: Sequence[str],\n    count: int,\n    overlap_threshold: float = 0.8,\n    overlap_metric: str = "iou",\n    max_total_gb: float | None = None,\n) -> dict[str, list[dict[str, Any]]]:\n    """Return exactly ``count`` co-located footprint sets per project.\n\n    ``overlap_metric="iou"`` preserves the original behavior. The explicit\n    ``"smaller"`` option uses intersection area divided by the smaller source\n    footprint. Every selected record is annotated with the common intersection\n    geometry so downstream diagnostics can compare identical ground extents.\n    """\n    if count < 0:\n        raise ValueError("count must be non-negative")\n    if max_total_gb is not None and max_total_gb < 0:\n        raise ValueError("max_total_gb must be non-negative")\n    if not 0 < overlap_threshold <= 1:\n        raise ValueError("overlap_threshold must be in (0, 1]")\n    if overlap_metric not in {"iou", "smaller"}:\n        raise ValueError("overlap_metric must be \'iou\' or \'smaller\'")\n\n    unique_projects = list(dict.fromkeys(projects))\n    if len(unique_projects) < 2:\n        raise ValueError("probe selection requires at least two projects")\n\n    grouped = {\n        project: sorted(\n            [dict(tile) for tile in tiles if project_matches(tile, project)],\n            key=lambda item: tile_id(item).casefold(),\n        )\n        for project in unique_projects\n    }\n    missing_projects = [project for project, values in grouped.items() if not values]\n    if missing_projects:\n        raise ValueError("No TNM records matched project(s): " + ", ".join(missing_projects))\n\n    budget_bytes = (\n        math.floor(max_total_gb * 1_000_000_000)\n        if max_total_gb is not None else 2**63 - 1\n    )\n    if max_total_gb is not None:\n        unknown_sizes = [\n            tile_id(tile)\n            for project_tiles in grouped.values()\n            for tile in project_tiles\n            if tile_size(tile) <= 0\n        ]\n        if unknown_sizes:\n            raise ValueError(\n                "Cannot enforce probe byte budget; missing size for: "\n                + ", ".join(unknown_sizes[:5])\n            )\n    if count == 0:\n        return {project: [] for project in unique_projects}\n\n    anchor_project = unique_projects[0]\n    anchors = grouped[anchor_project]\n    geometries = {\n        (project, tile_id(tile)): tile_footprint(tile)\n        for project, project_tiles in grouped.items()\n        for tile in project_tiles\n    }\n\n    def score(left: BaseGeometry, right: BaseGeometry) -> float:\n        return footprint_iou(left, right) if overlap_metric == "iou" else overlap_fraction(left, right)\n\n    candidate_sets: list[dict[str, Any]] = []\n    for anchor in anchors:\n        anchor_geometry = geometries[(anchor_project, tile_id(anchor))]\n        options: list[list[dict[str, Any]]] = []\n        for project in unique_projects[1:]:\n            scored_matches: list[tuple[float, float, str, dict[str, Any]]] = []\n            for tile in grouped[project]:\n                geometry = geometries[(project, tile_id(tile))]\n                match_score = score(anchor_geometry, geometry)\n                if match_score >= overlap_threshold:\n                    scored_matches.append((\n                        match_score,\n                        footprint_iou(anchor_geometry, geometry),\n                        tile_id(tile).casefold(),\n                        tile,\n                    ))\n            scored_matches.sort(key=lambda item: (-item[0], -item[1], item[2]))\n            matches = [tile for _, _, _, tile in scored_matches]\n            if not matches:\n                options = []\n                break\n            options.append(matches)\n\n        for combination in product(*options) if options else ():\n            selected = {anchor_project: anchor} | dict(zip(unique_projects[1:], combination))\n            selected_geometries = [\n                geometries[(project, tile_id(selected[project]))]\n                for project in unique_projects\n            ]\n            pair_scores: list[float] = []\n            pair_ious: list[float] = []\n            pair_smaller: list[float] = []\n            compatible = True\n            for left in range(len(selected_geometries)):\n                for right in range(left + 1, len(selected_geometries)):\n                    lg = selected_geometries[left]\n                    rg = selected_geometries[right]\n                    pair_score = score(lg, rg)\n                    if pair_score < overlap_threshold:\n                        compatible = False\n                        break\n                    pair_scores.append(pair_score)\n                    pair_ious.append(footprint_iou(lg, rg))\n                    pair_smaller.append(overlap_fraction(lg, rg))\n                if not compatible:\n                    break\n            if not compatible:\n                continue\n\n            common_geometry = selected_geometries[0]\n            for geometry in selected_geometries[1:]:\n                common_geometry = common_geometry.intersection(geometry)\n            if common_geometry.is_empty or common_geometry.area <= 0:\n                continue\n\n            set_bytes = sum(tile_size(selected[project]) for project in unique_projects)\n            if set_bytes > budget_bytes:\n                continue\n            candidate_sets.append({\n                "selected": selected,\n                "bytes": set_bytes,\n                "score": min(pair_scores),\n                "iou": min(pair_ious),\n                "smaller_overlap": min(pair_smaller),\n                "intersection": common_geometry,\n            })\n\n    selected_candidates: list[dict[str, Any]] = []\n    used = {project: set() for project in unique_projects}\n\n    def choose(start_index: int, used_bytes: int) -> bool:\n        if len(selected_candidates) == count:\n            return True\n        needed = count - len(selected_candidates)\n        if len(candidate_sets) - start_index < needed:\n            return False\n        for candidate_index in range(start_index, len(candidate_sets)):\n            candidate = candidate_sets[candidate_index]\n            if used_bytes + candidate["bytes"] > budget_bytes:\n                continue\n            selected = candidate["selected"]\n            selected_ids = {project: tile_id(selected[project]) for project in unique_projects}\n            if any(selected_ids[project] in used[project] for project in unique_projects):\n                continue\n            selected_candidates.append(candidate)\n            for project in unique_projects:\n                used[project].add(selected_ids[project])\n            if choose(candidate_index + 1, used_bytes + candidate["bytes"]):\n                return True\n            for project in unique_projects:\n                used[project].remove(selected_ids[project])\n            selected_candidates.pop()\n        return False\n\n    if not choose(0, 0):\n        complete_anchor_count = len({\n            tile_id(candidate["selected"][anchor_project]) for candidate in candidate_sets\n        })\n        budget_note = f" within {max_total_gb:g} GB" if max_total_gb is not None else ""\n        metric_label = "IoU" if overlap_metric == "iou" else "smaller-footprint overlap"\n        raise ValueError(\n            f"Could not select {count} unique co-located sets at {metric_label} "\n            f">= {overlap_threshold:g}{budget_note}; {complete_anchor_count} anchor "\n            "footprints had at least one complete project match"\n        )\n\n    output = {project: [] for project in unique_projects}\n    for pair_index, candidate in enumerate(selected_candidates, 1):\n        pair_id = f"probe_{pair_index:03d}"\n        intersection_mapping = mapping(candidate["intersection"])\n        for project in unique_projects:\n            annotated = dict(candidate["selected"][project])\n            annotated.update({\n                "_probe_pair_id": pair_id,\n                "_probe_overlap_metric": overlap_metric,\n                "_probe_overlap_threshold": float(overlap_threshold),\n                "_probe_pair_score": float(candidate["score"]),\n                "_probe_footprint_iou": float(candidate["iou"]),\n                "_probe_smaller_overlap": float(candidate["smaller_overlap"]),\n                "_probe_intersection_geometry": intersection_mapping,\n                "_probe_intersection_crs": "EPSG:4326",\n            })\n            output[project].append(annotated)\n    return output\n'

def replace_once(text: str, old: str, new: str, *, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} marker, found {count}. No files were written.")
    return text.replace(old, new, 1)

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    select_path = repo / "oregon" / "select_tiles.py"
    runner_path = repo / "oregon" / "run_tillamook_probe.ps1"
    source_dir = Path(__file__).resolve().parent
    matched_source = source_dir / "probe_matched_tiles.py"
    test_source = source_dir / "test_probe_overlap_fix.py"
    matched_target = repo / "oregon" / "diagnostics" / "probe_matched_tiles.py"
    test_target = repo / "tests" / "test_probe_overlap_fix.py"
    for path in (select_path, runner_path, matched_source, test_source):
        if not path.is_file():
            raise SystemExit(f"Missing required file: {path}")

    select_text = select_path.read_text(encoding="utf-8")
    start = select_text.index("def select_probe_tiles(")
    end = select_text.index("\ndef _load_json", start)
    select_text = select_text[:start] + NEW_FUNCTION.rstrip() + "\n\n" + select_text[end + 1:]
    if "from shapely.geometry import box, mapping, shape" not in select_text:
        select_text = replace_once(select_text, "from shapely.geometry import box, shape", "from shapely.geometry import box, mapping, shape", label="Shapely import")
    if "--probe-overlap-metric" not in select_text:
        marker = '    parser.add_argument("--output", type=Path, required=True)'
        addition = '''    parser.add_argument(
        "--probe-overlap-metric",
        choices=("iou", "smaller"),
        default="iou",
        help=("Probe matching metric. 'iou' preserves legacy behavior; "
              "'smaller' requires substantial coverage of the smaller footprint."),
    )
'''
        select_text = replace_once(select_text, marker, addition + marker, label="output argument")
    if "overlap_metric=args.probe_overlap_metric" not in select_text:
        old = '''            overlap_threshold=args.overlap_threshold if args.overlap_threshold is not None else 0.8,
            max_total_gb=args.max_total_gb,
'''
        new = '''            overlap_threshold=args.overlap_threshold if args.overlap_threshold is not None else 0.8,
            overlap_metric=args.probe_overlap_metric,
            max_total_gb=args.max_total_gb,
'''
        select_text = replace_once(select_text, old, new, label="probe CLI call")

    runner_text = runner_path.read_text(encoding="utf-8")
    if "$OverlapMetric" not in runner_text:
        runner_text = replace_once(runner_text, "    [double]$OverlapThreshold = 0.8,", '    [double]$OverlapThreshold = 0.8,\n    [ValidateSet("iou", "smaller")]\n    [string]$OverlapMetric = "smaller",', label="runner parameter")
    if "--probe-overlap-metric" not in runner_text:
        runner_text = replace_once(runner_text, "            --overlap-threshold $OverlapThreshold `\n            --max-total-gb $MaxTotalGB `", "            --overlap-threshold $OverlapThreshold `\n            --probe-overlap-metric $OverlapMetric `\n            --max-total-gb $MaxTotalGB `", label="runner selection arguments")
    if "probe_matched_tiles.py" not in runner_text:
        runner_text = replace_once(runner_text, '& $Python ".\\diagnostics\\probe_tiles.py" `', '& $Python ".\\diagnostics\\probe_matched_tiles.py" `', label="runner probe command")
        runner_text = replace_once(runner_text, '            --project "$LegacyProject=$LegacyDirectory" `\n            --cell-size 1.0 `', '            --project "$LegacyProject=$LegacyDirectory" `\n            --selection $Selection `\n            --cell-size 1.0 `', label="runner selection input")

    for path in (select_path, runner_path):
        shutil.copy2(path, path.with_name(path.name + ".bak_matched_probe"))
    select_path.write_text(select_text, encoding="utf-8")
    runner_path.write_text(runner_text, encoding="utf-8")
    matched_target.parent.mkdir(parents=True, exist_ok=True)
    test_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matched_source, matched_target)
    shutil.copy2(test_source, test_target)
    py_compile.compile(str(select_path), doraise=True)
    py_compile.compile(str(matched_target), doraise=True)
    py_compile.compile(str(test_target), doraise=True)
    print("Applied Tillamook matched-project probe fix.")
    print(f"Updated: {select_path}")
    print(f"Updated: {runner_path}")
    print(f"Added:   {matched_target}")
    print(f"Added:   {test_target}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
