#!/usr/bin/env python3
"""
Plan and freeze a leakage-resistant 64/16/20 Tillamook split from an existing
train-only candidate patch pool.

Phase 1 only:
- DOES NOT move/copy NPZ files.
- DOES NOT retrain/rebuild terrain features.
- DOES NOT modify the source patches.csv.
- Writes a split map + preview manifest for verification.

Leakage rules:
1. All patches from one LiDAR tile stay in one split.
2. Tiles whose retained patch footprints are < buffer_m apart are forced into
   the same atomic component.
3. Tiles sharing any landslide ID in the manifest are forced into the same
   atomic component (conservative: the builder stores all intersecting IDs).
4. Atomic components are assigned as whole units to train/validation/test.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from pyproj import CRS, Transformer
from shapely.geometry import box
from shapely.ops import transform as shapely_transform, unary_union

METRIC_CRS = CRS.from_epsg(5070)
SPLITS = ("train", "validation", "test")
FRACTIONS = {"train": 0.64, "validation": 0.16, "test": 0.20}


class DSU:
    def __init__(self, items: Iterable[str]):
        self.parent = {x: x for x in items}
        self.rank = {x: 0 for x in items}

    def find(self, x: str) -> str:
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


@dataclass
class TileInfo:
    tile_name: str
    footprint: object
    patch_count: int
    positive_patch_count: int
    negative_patch_count: int
    positive_interior_count: int
    positive_boundary_count: int
    hard_negative_count: int
    polygon_ids: set[str]


@dataclass
class Component:
    component_id: str
    tiles: list[str]
    tile_count: int
    patch_count: int
    positive_patch_count: int
    negative_patch_count: int
    positive_interior_count: int
    positive_boundary_count: int
    hard_negative_count: int
    polygon_ids: set[str]


def read_rows(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def parse_bool(value: str) -> bool:
    return str(value).strip().casefold() in {"1", "true", "yes", "y"}


def parse_ids(value: str) -> set[str]:
    return {x.strip() for x in str(value or "").split(";") if x.strip()}


def largest_remainder_targets(total: int) -> dict[str, int]:
    raw = {s: total * FRACTIONS[s] for s in SPLITS}
    out = {s: math.floor(raw[s]) for s in SPLITS}
    remaining = total - sum(out.values())
    order = sorted(SPLITS, key=lambda s: (raw[s] - out[s], FRACTIONS[s]), reverse=True)
    for s in order[:remaining]:
        out[s] += 1
    return out


def metric_patch_geometry(row: dict[str, str], transformer_cache: dict[str, Transformer]):
    crs_text = str(row["crs"]).strip()
    crs = CRS.from_user_input(crs_text)
    geom = box(
        float(row["x_min"]),
        float(row["y_min"]),
        float(row["x_max"]),
        float(row["y_max"]),
    )
    if crs != METRIC_CRS:
        key = crs.to_wkt()
        transformer = transformer_cache.get(key)
        if transformer is None:
            transformer = Transformer.from_crs(crs, METRIC_CRS, always_xy=True)
            transformer_cache[key] = transformer
        geom = shapely_transform(transformer.transform, geom)
    return geom


def build_tiles(rows: list[dict[str, str]]) -> dict[str, TileInfo]:
    grouped = defaultdict(list)
    for row in rows:
        tile = str(row.get("tile_name", "")).strip()
        if not tile:
            raise ValueError("Manifest contains a row with no tile_name")
        grouped[tile].append(row)

    transformer_cache: dict[str, Transformer] = {}
    result = {}

    for idx, (tile, items) in enumerate(sorted(grouped.items()), start=1):
        patch_geoms = [metric_patch_geometry(r, transformer_cache) for r in items]
        footprint = unary_union(patch_geoms)

        cats = Counter(str(r.get("category", "")).strip() for r in items)
        positive_count = sum(
            1 for r in items if float(r.get("positive_fraction", 0.0) or 0.0) > 0.0
        )
        negative_count = len(items) - positive_count
        hard_negative_count = sum(parse_bool(r.get("is_hard_negative", "")) for r in items)

        polygon_ids: set[str] = set()
        for r in items:
            polygon_ids.update(parse_ids(r.get("landslide_ids_in_tile", "")))

        result[tile] = TileInfo(
            tile_name=tile,
            footprint=footprint,
            patch_count=len(items),
            positive_patch_count=positive_count,
            negative_patch_count=negative_count,
            positive_interior_count=cats.get("positive_interior", 0),
            positive_boundary_count=cats.get("positive_boundary", 0),
            hard_negative_count=hard_negative_count,
            polygon_ids=polygon_ids,
        )

        if idx % 20 == 0 or idx == len(grouped):
            print(f"Aggregated tile footprints: {idx}/{len(grouped)}")

    return result


def make_components(tiles: dict[str, TileInfo], buffer_m: float):
    names = sorted(tiles)
    dsu = DSU(names)

    # Spatial leakage edges: if any retained patch footprint from tile A is
    # closer than buffer_m to any retained patch footprint from tile B, they
    # cannot be placed in different splits.
    spatial_edges = 0
    for i, a in enumerate(names):
        ga = tiles[a].footprint
        for b in names[i + 1:]:
            gb = tiles[b].footprint
            if ga.distance(gb) < buffer_m:
                dsu.union(a, b)
                spatial_edges += 1

    # Polygon leakage edges. The source manifest stores IDs intersecting each
    # tile. This is intentionally conservative.
    polygon_to_tiles = defaultdict(list)
    for name in names:
        for pid in tiles[name].polygon_ids:
            polygon_to_tiles[pid].append(name)

    polygon_edges = 0
    for pid, members in polygon_to_tiles.items():
        if len(members) > 1:
            first = members[0]
            for other in members[1:]:
                dsu.union(first, other)
                polygon_edges += 1

    groups = defaultdict(list)
    for name in names:
        groups[dsu.find(name)].append(name)

    components = []
    ordered_groups = sorted(
        groups.values(),
        key=lambda members: (-len(members), tuple(sorted(members))),
    )

    for n, members in enumerate(ordered_groups, start=1):
        members = sorted(members)
        polygon_ids = set().union(*(tiles[t].polygon_ids for t in members))
        components.append(
            Component(
                component_id=f"C{n:03d}",
                tiles=members,
                tile_count=len(members),
                patch_count=sum(tiles[t].patch_count for t in members),
                positive_patch_count=sum(tiles[t].positive_patch_count for t in members),
                negative_patch_count=sum(tiles[t].negative_patch_count for t in members),
                positive_interior_count=sum(tiles[t].positive_interior_count for t in members),
                positive_boundary_count=sum(tiles[t].positive_boundary_count for t in members),
                hard_negative_count=sum(tiles[t].hard_negative_count for t in members),
                polygon_ids=polygon_ids,
            )
        )

    return components, spatial_edges, polygon_edges


def totals_for_components(components: list[Component]):
    return {
        "tiles": sum(c.tile_count for c in components),
        "patches": sum(c.patch_count for c in components),
        "positive_patches": sum(c.positive_patch_count for c in components),
        "negative_patches": sum(c.negative_patch_count for c in components),
        "positive_interior": sum(c.positive_interior_count for c in components),
        "positive_boundary": sum(c.positive_boundary_count for c in components),
        "hard_negatives": sum(c.hard_negative_count for c in components),
        "polygons": sum(len(c.polygon_ids) for c in components),
    }


def summarize_assignment(components, assignment):
    out = {s: Counter() for s in SPLITS}
    polygons = {s: set() for s in SPLITS}
    for c in components:
        s = assignment[c.component_id]
        out[s]["tiles"] += c.tile_count
        out[s]["patches"] += c.patch_count
        out[s]["positive_patches"] += c.positive_patch_count
        out[s]["negative_patches"] += c.negative_patch_count
        out[s]["positive_interior"] += c.positive_interior_count
        out[s]["positive_boundary"] += c.positive_boundary_count
        out[s]["hard_negatives"] += c.hard_negative_count
        polygons[s].update(c.polygon_ids)
    for s in SPLITS:
        out[s]["polygons"] = len(polygons[s])
    return out, polygons


def score_assignment(components, assignment, total):
    summary, polygons = summarize_assignment(components, assignment)

    # Reject missing splits.
    if any(summary[s]["tiles"] == 0 for s in SPLITS):
        return math.inf

    # For model selection/test, require actual positive support if positives exist.
    if total["positive_patches"] > 0:
        if summary["validation"]["positive_patches"] == 0:
            return math.inf
        if summary["test"]["positive_patches"] == 0:
            return math.inf

    # Strongly prioritize tile ratio, then patch/positive/hard-negative balance.
    weights = {
        "tiles": 12.0,
        "patches": 3.0,
        "positive_patches": 4.0,
        "negative_patches": 1.0,
        "positive_interior": 2.0,
        "positive_boundary": 2.0,
        "hard_negatives": 2.0,
        "polygons": 3.0,
    }

    score = 0.0
    for metric, weight in weights.items():
        denom = max(1, total[metric])
        for s in SPLITS:
            target = FRACTIONS[s] * total[metric]
            diff = summary[s][metric] - target
            score += weight * (diff / denom) ** 2

    return score


def greedy_assignment(components, rng):
    assignment = {}
    current = {s: Counter() for s in SPLITS}
    total_tiles = sum(c.tile_count for c in components)

    ordered = list(components)
    rng.shuffle(ordered)
    # Largest components first, random tie-breaking already applied.
    ordered.sort(key=lambda c: c.tile_count, reverse=True)

    for c in ordered:
        choices = []
        for s in SPLITS:
            target_tiles = FRACTIONS[s] * total_tiles
            after = current[s]["tiles"] + c.tile_count
            # Penalize overfill, but allow it when atomic groups require it.
            tile_cost = ((after - target_tiles) / max(1.0, target_tiles)) ** 2
            patch_cost = (
                (current[s]["patches"] + c.patch_count)
                / max(1, sum(x.patch_count for x in components))
                - FRACTIONS[s]
            ) ** 2
            choices.append((tile_cost * 5.0 + patch_cost, rng.random(), s))
        _, _, chosen = min(choices)
        assignment[c.component_id] = chosen
        current[chosen]["tiles"] += c.tile_count
        current[chosen]["patches"] += c.patch_count

    return assignment


def improve_random_search(components, seed, trials):
    total = totals_for_components(components)
    rng = random.Random(seed)

    best_assignment = None
    best_score = math.inf

    for trial in range(trials):
        assignment = greedy_assignment(components, rng)

        # Random local mutations.
        for _ in range(max(50, len(components) * 4)):
            cid = rng.choice(components).component_id
            old = assignment[cid]
            new = rng.choice([s for s in SPLITS if s != old])
            assignment[cid] = new
            new_score = score_assignment(components, assignment, total)
            if new_score < best_score:
                best_score = new_score
                best_assignment = dict(assignment)
            else:
                # Mostly revert; tiny random acceptance helps escape local minima.
                if rng.random() > 0.01:
                    assignment[cid] = old

        score = score_assignment(components, assignment, total)
        if score < best_score:
            best_score = score
            best_assignment = dict(assignment)

        if (trial + 1) % 1000 == 0:
            print(f"Split search: {trial + 1}/{trials} best_score={best_score:.8f}")

    if best_assignment is None:
        raise RuntimeError("Could not construct a valid three-way split")
    return best_assignment, best_score


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("dataset_tillamook_expansion_100_train_binary_15m"),
    )
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument(
        "--outdir",
        type=Path,
        default=Path("phase1_tillamook_split_64_16_20"),
    )
    ap.add_argument("--buffer-m", type=float, default=500.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=10000)
    args = ap.parse_args()

    if args.buffer_m < 0:
        ap.error("--buffer-m must be non-negative")
    if args.trials <= 0:
        ap.error("--trials must be positive")

    dataset_dir = args.dataset_dir.resolve()
    manifest = (
        args.manifest.resolve()
        if args.manifest
        else (dataset_dir / "patches.csv").resolve()
    )
    outdir = args.outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    rows, fields = read_rows(manifest)
    if not rows:
        raise SystemExit("Manifest is empty")

    original_splits = sorted({str(r.get("split", "")).strip() for r in rows})
    if original_splits != ["train"]:
        raise SystemExit(
            f"Expected a train-only candidate pool; found splits={original_splits}"
        )

    print(f"Manifest rows: {len(rows)}")
    tiles = build_tiles(rows)
    print(f"Unique usable tiles: {len(tiles)}")

    if len(tiles) != 99:
        print(
            f"WARNING: expected 99 usable tiles from the current build, found {len(tiles)}. "
            "The planner will use the manifest as the source of truth."
        )

    components, spatial_edges, polygon_edges = make_components(tiles, args.buffer_m)
    component_sizes = sorted((c.tile_count for c in components), reverse=True)

    print(f"Atomic components: {len(components)}")
    print(f"Largest component: {component_sizes[0]} tile(s)")
    print(f"Spatial union edges (<{args.buffer_m:g} m): {spatial_edges}")
    print(f"Polygon union edges: {polygon_edges}")

    # If the constraints collapse the whole dataset into too few atomic groups,
    # do not silently make a bad split.
    if len(components) < 3:
        raise SystemExit(
            "Fewer than 3 leakage-safe components exist. A 3-way split cannot be "
            "made without dropping guard-band tiles. Stop and use a guard-band planner."
        )

    total_tiles = len(tiles)
    nominal = largest_remainder_targets(total_tiles)
    print(f"Nominal tile targets: {nominal}")

    assignment, best_score = improve_random_search(
        components, seed=args.seed, trials=args.trials
    )
    summary, polygon_sets = summarize_assignment(components, assignment)

    # Verify polygon ID disjointness.
    polygon_overlap = {}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            overlap = sorted(polygon_sets[a] & polygon_sets[b])
            polygon_overlap[f"{a}__{b}"] = overlap
    if any(polygon_overlap.values()):
        raise RuntimeError("Internal error: cross-split polygon overlap remains")

    # Build component map.
    component_rows = []
    tile_to_component = {}
    tile_to_split = {}
    for c in components:
        s = assignment[c.component_id]
        for t in c.tiles:
            tile_to_component[t] = c.component_id
            tile_to_split[t] = s
        component_rows.append(
            {
                "component_id": c.component_id,
                "split": s,
                "tile_count": c.tile_count,
                "patch_count": c.patch_count,
                "positive_patch_count": c.positive_patch_count,
                "negative_patch_count": c.negative_patch_count,
                "positive_interior_count": c.positive_interior_count,
                "positive_boundary_count": c.positive_boundary_count,
                "hard_negative_count": c.hard_negative_count,
                "unique_polygon_ids": len(c.polygon_ids),
                "tiles": ";".join(c.tiles),
            }
        )

    split_map_rows = []
    for t in sorted(tiles):
        info = tiles[t]
        split_map_rows.append(
            {
                "tile_name": t,
                "split": tile_to_split[t],
                "component_id": tile_to_component[t],
                "patch_count": info.patch_count,
                "positive_patch_count": info.positive_patch_count,
                "negative_patch_count": info.negative_patch_count,
                "positive_interior_count": info.positive_interior_count,
                "positive_boundary_count": info.positive_boundary_count,
                "hard_negative_count": info.hard_negative_count,
                "unique_polygon_ids": len(info.polygon_ids),
            }
        )

    # Preview manifest: same files and rows, split field changed only.
    preview_rows = []
    for r in rows:
        new = dict(r)
        new["split"] = tile_to_split[r["tile_name"]]
        preview_rows.append(new)

    write_csv(
        outdir / "split_map_64_16_20.csv",
        split_map_rows,
        [
            "tile_name",
            "split",
            "component_id",
            "patch_count",
            "positive_patch_count",
            "negative_patch_count",
            "positive_interior_count",
            "positive_boundary_count",
            "hard_negative_count",
            "unique_polygon_ids",
        ],
    )
    write_csv(
        outdir / "component_map.csv",
        component_rows,
        [
            "component_id",
            "split",
            "tile_count",
            "patch_count",
            "positive_patch_count",
            "negative_patch_count",
            "positive_interior_count",
            "positive_boundary_count",
            "hard_negative_count",
            "unique_polygon_ids",
            "tiles",
        ],
    )
    write_csv(outdir / "preview_manifest.csv", preview_rows, fields)

    total = totals_for_components(components)
    report = {
        "source_manifest": str(manifest),
        "seed": args.seed,
        "buffer_m": args.buffer_m,
        "target_fractions": FRACTIONS,
        "nominal_tile_targets": nominal,
        "usable_tiles": total_tiles,
        "patches": len(rows),
        "atomic_components": len(components),
        "largest_component_tiles": component_sizes[0],
        "spatial_union_edges": spatial_edges,
        "polygon_union_edges": polygon_edges,
        "optimization_score": best_score,
        "split_summary": {s: dict(summary[s]) for s in SPLITS},
        "cross_split_polygon_overlap_count": {
            k: len(v) for k, v in polygon_overlap.items()
        },
        "ground_truth_policy": "strict_binary_0_1",
        "phase": "Phase 1 split planning only; source NPZ files unchanged",
    }
    (outdir / "split_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    md = []
    md.append("# Tillamook 64/16/20 Phase 1 split")
    md.append("")
    md.append(f"- Source tiles: **{total_tiles}**")
    md.append(f"- Source patches: **{len(rows)}**")
    md.append(f"- Spatial guard: **{args.buffer_m:g} m**")
    md.append(f"- Atomic leakage-safe components: **{len(components)}**")
    md.append(f"- Largest component: **{component_sizes[0]} tile(s)**")
    md.append(f"- Nominal tile target: `{nominal}`")
    md.append("")
    md.append("| Split | Tiles | Tile % | Patches | Positive patches | Negative patches | Polygons | Hard negatives |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for s in SPLITS:
        q = summary[s]
        md.append(
            f"| {s} | {q['tiles']} | {100*q['tiles']/total_tiles:.2f}% | "
            f"{q['patches']} | {q['positive_patches']} | {q['negative_patches']} | "
            f"{q['polygons']} | {q['hard_negatives']} |"
        )
    md.append("")
    md.append("## Leakage constraints")
    md.append("")
    md.append("- A LiDAR tile appears in exactly one split.")
    md.append(f"- Any two tiles with retained patch footprints < {args.buffer_m:g} m apart are assigned to the same split.")
    md.append("- Any two tiles sharing a manifest landslide ID are assigned to the same split.")
    md.append("- Cross-split polygon overlap after assignment: **0**.")
    md.append("")
    md.append("Run `verify_splits.py` on `preview_manifest.csv` before freezing this split.")
    (outdir / "split_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    print()
    print("PHASE 1 SPLIT PLANNED")
    for s in SPLITS:
        q = summary[s]
        print(
            f"{s:10s}: tiles={q['tiles']:2d} ({100*q['tiles']/total_tiles:5.2f}%) "
            f"patches={q['patches']:4d} positive={q['positive_patches']:4d} "
            f"polygons={q['polygons']:3d}"
        )
    print("Cross-split polygon overlap: 0")
    print(f"Preview manifest: {outdir / 'preview_manifest.csv'}")
    print(f"Split map:        {outdir / 'split_map_64_16_20.csv'}")
    print()
    print("NEXT: run verify_splits.py on preview_manifest.csv. Do NOT train yet.")


if __name__ == "__main__":
    main()
