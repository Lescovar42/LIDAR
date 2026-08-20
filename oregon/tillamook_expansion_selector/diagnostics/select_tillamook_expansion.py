#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from pyproj import Transformer
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely.strtree import STRtree

PROJECT_DEFAULT = "USGS_LPC_OR_WesternWildfires_A22"
METRIC_CRS_DEFAULT = "EPSG:32610"  # UTM 10N; appropriate for Tillamook
POSITIVE_CONF = {"high", "moderate"}
CONF_FIELDS = ("confidence_class", "CONFIDENCE", "confidence", "Confidence")
KEY_FIELDS = ("REF_ID_COD", "ref_id_cod", "SLIDO_REF_ID", "OBJECTID", "FID", "id")


def first_value(obj: Mapping[str, Any], names: Sequence[str], default: Any = None) -> Any:
    for name in names:
        if name in obj and obj[name] not in (None, ""):
            return obj[name]
    return default


def normalize_confidence(value: Any) -> str:
    text = str(value or "").strip().casefold()
    if text.startswith("high"):
        return "high"
    if text.startswith("moderate"):
        return "moderate"
    if text.startswith("low"):
        return "low"
    return "unknown"


def laz_basename(value: Any) -> str:
    text = str(value or "").strip().replace("\\", "/")
    name = text.rsplit("/", 1)[-1]
    if name.endswith(".aria2"):
        name = name[:-6]
    if not name.lower().endswith(".laz"):
        m = re.search(r"(USGS_LPC_[^/\\]+?\.laz)", text, flags=re.I)
        if m:
            name = m.group(1)
    return name


def tile_name(tile: Mapping[str, Any]) -> str:
    for field in ("downloadURL", "downloadLazURL", "title"):
        value = tile.get(field)
        if value:
            name = laz_basename(value)
            if name.lower().endswith(".laz"):
                return name
    raise ValueError(f"Cannot determine LAZ filename for tile: {tile.get('title', tile)}")


def tile_size(tile: Mapping[str, Any]) -> int:
    try:
        return max(0, int(tile.get("sizeInBytes") or tile.get("size_bytes") or 0))
    except (TypeError, ValueError):
        return 0


def tile_url(tile: Mapping[str, Any]) -> str:
    return str(tile.get("downloadURL") or tile.get("downloadLazURL") or "")


def tile_footprint_wgs84(tile: Mapping[str, Any]) -> BaseGeometry:
    geometry = tile.get("geometry") or tile.get("footprint")
    if isinstance(geometry, Mapping):
        geom = shape(geometry)
    else:
        bbox = tile.get("boundingBox") or tile.get("bbox")
        if isinstance(bbox, Mapping):
            xmin = first_value(bbox, ("minX", "west", "xmin"))
            ymin = first_value(bbox, ("minY", "south", "ymin"))
            xmax = first_value(bbox, ("maxX", "east", "xmax"))
            ymax = first_value(bbox, ("maxY", "north", "ymax"))
            if None in (xmin, ymin, xmax, ymax):
                raise ValueError(f"Incomplete bbox for {tile_name(tile)}")
            geom = box(float(xmin), float(ymin), float(xmax), float(ymax))
        elif isinstance(bbox, Sequence) and not isinstance(bbox, (str, bytes)) and len(bbox) == 4:
            geom = box(*map(float, bbox))
        else:
            raise ValueError(f"Missing footprint for {tile_name(tile)}")
    if geom.is_empty or geom.area <= 0:
        raise ValueError(f"Invalid footprint for {tile_name(tile)}")
    minx, miny, maxx, maxy = geom.bounds
    if not (-180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90):
        raise ValueError(f"Expected WGS84 tile footprint for {tile_name(tile)}; got bounds {geom.bounds}")
    return geom


def project_matches(tile: Mapping[str, Any], project: str) -> bool:
    haystack = " ".join(str(tile.get(k, "")) for k in ("title", "downloadURL", "downloadLazURL", "moreInfo"))
    return project.casefold() in haystack.casefold()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_tiles(path: Path, project: str) -> list[dict[str, Any]]:
    raw = load_json(path)
    items = raw.get("items", raw.get("features", [])) if isinstance(raw, Mapping) else raw
    out: dict[str, dict[str, Any]] = {}
    for source in items:
        if not isinstance(source, Mapping) or not project_matches(source, project):
            continue
        item = dict(source)
        try:
            name = tile_name(item)
            tile_footprint_wgs84(item)
        except ValueError:
            continue
        # Keep deterministic first record unless a later duplicate has a known size and first does not.
        if name not in out or (tile_size(out[name]) <= 0 < tile_size(item)):
            out[name] = item
    return [out[name] for name in sorted(out)]


def feature_key(feature: Mapping[str, Any], index: int) -> str:
    props = feature.get("properties") or {}
    for key in KEY_FIELDS:
        value = props.get(key)
        if value not in (None, ""):
            return str(value)
    if feature.get("id") not in (None, ""):
        return str(feature["id"])
    return f"feature_{index:05d}"


@dataclass(frozen=True)
class PolygonRecord:
    key: str
    confidence: str
    geometry_wgs84: BaseGeometry
    geometry_metric: BaseGeometry


def load_polygons(path: Path, to_metric: Transformer) -> list[PolygonRecord]:
    raw = load_json(path)
    features = raw.get("features", raw) if isinstance(raw, Mapping) else raw
    records: list[PolygonRecord] = []
    for i, feature in enumerate(features):
        if not isinstance(feature, Mapping) or not feature.get("geometry"):
            continue
        geom = shape(feature["geometry"])
        if geom.is_empty:
            continue
        minx, miny, maxx, maxy = geom.bounds
        if not (-180 <= minx <= 180 and -180 <= maxx <= 180 and -90 <= miny <= 90 and -90 <= maxy <= 90):
            raise ValueError(
                f"Expected WGS84 SLIDO GeoJSON. Feature {feature_key(feature, i)} has bounds {geom.bounds}."
            )
        props = feature.get("properties") or {}
        conf = normalize_confidence(first_value(props, CONF_FIELDS))
        records.append(
            PolygonRecord(
                key=feature_key(feature, i),
                confidence=conf,
                geometry_wgs84=geom,
                geometry_metric=transform(to_metric.transform, geom),
            )
        )
    return records


def load_split_tiles(manifest: Path) -> tuple[set[str], set[str]]:
    train: set[str] = set()
    validation: set[str] = set()
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or "split" not in reader.fieldnames or "tile_name" not in reader.fieldnames:
            raise ValueError("Frozen manifest must contain split and tile_name columns")
        for row in reader:
            name = laz_basename(row.get("tile_name"))
            split = str(row.get("split") or "").strip().casefold()
            if split == "train":
                train.add(name)
            elif split in {"validation", "val"}:
                validation.add(name)
    return train, validation


def load_downloaded(path: Path | None) -> set[str]:
    if not path or not path.exists():
        return set()
    result: set[str] = set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = laz_basename(row.get("Name") or row.get("name") or row.get("tile_name") or row.get("FullName"))
            if name.lower().endswith(".laz"):
                result.add(name)
    return result


def load_exclusions(path: Path | None) -> dict[str, str]:
    if not path or not path.exists():
        return {}
    result: dict[str, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = laz_basename(row.get("tile_name") or row.get("Name") or row.get("name"))
            if name:
                result[name] = str(row.get("error") or row.get("reason") or "explicit_exclusion")
    return result


def load_probe_metrics(path: Path | None, cell_size: float = 2.0) -> dict[str, dict[str, Any]]:
    if not path or not path.exists():
        return {}
    raw = load_json(path)
    output: dict[str, dict[str, Any]] = {}
    for row in raw.get("tiles", []):
        if abs(float(row.get("cell_size", -1)) - cell_size) > 1e-9:
            continue
        name = laz_basename(row.get("tile"))
        patches = row.get("patches") or {}
        output[name] = {
            "accepted": int(patches.get("accepted") or 0),
            "positive": int(patches.get("positive_total") or 0),
            "negative": int(patches.get("negative") or 0),
            "ignore_only": int(patches.get("ignore_only") or 0),
            "ground_fraction": float(row.get("ground_cell_fraction") or 0.0),
            "retained_polygon_count": len(row.get("retained_polygon_keys") or []),
        }
    return output


def query_indices(tree: STRtree, geom: BaseGeometry, geometries: Sequence[BaseGeometry]) -> list[int]:
    found = tree.query(geom)
    if len(found) == 0:
        return []
    first = found[0]
    if isinstance(first, (int,)) or type(first).__module__.startswith("numpy"):
        return [int(x) for x in found]
    index_by_id = {id(g): i for i, g in enumerate(geometries)}
    return [index_by_id[id(g)] for g in found if id(g) in index_by_id]


@dataclass
class Candidate:
    name: str
    tile: dict[str, Any]
    footprint_wgs84: BaseGeometry
    footprint_metric: BaseGeometry
    centroid_metric: BaseGeometry
    downloaded: bool = False
    probe: dict[str, Any] | None = None
    category: str = ""
    positive_keys: set[str] = field(default_factory=set)
    new_positive_keys: set[str] = field(default_factory=set)
    high_count: int = 0
    moderate_count: int = 0
    positive_intersection_m2: float = 0.0
    new_positive_intersection_m2: float = 0.0
    min_existing_train_km: float = 0.0
    score_static: float = 0.0
    selection_rank: int | None = None
    selection_reason: str = ""


def safe_min_distance_km(geom: BaseGeometry, others: Sequence[BaseGeometry]) -> float:
    if not others:
        return 999.0
    return min(float(geom.distance(other)) for other in others) / 1000.0


def build_candidates(
    tiles: Sequence[dict[str, Any]],
    polygons: Sequence[PolygonRecord],
    train_names: set[str],
    validation_names: set[str],
    downloaded_names: set[str],
    exclusions: Mapping[str, str],
    probes: Mapping[str, dict[str, Any]],
    to_metric: Transformer,
    validation_buffer_m: float,
) -> tuple[list[Candidate], dict[str, str], set[str]]:
    by_name = {tile_name(tile): tile for tile in tiles}
    missing_split_tiles = sorted((train_names | validation_names) - set(by_name))
    if missing_split_tiles:
        raise ValueError(
            "Frozen manifest tile(s) not found in TNM candidate file: " + ", ".join(missing_split_tiles[:8])
        )

    tile_metric = {
        name: transform(to_metric.transform, tile_footprint_wgs84(tile)) for name, tile in by_name.items()
    }
    validation_geoms = [tile_metric[name] for name in validation_names]
    validation_guard = [geom.buffer(validation_buffer_m) for geom in validation_geoms]
    train_geoms = [tile_metric[name] for name in train_names]

    poly_geoms = [p.geometry_metric for p in polygons]
    tree = STRtree(poly_geoms)

    # Anything intersecting existing training tiles is already represented at tile scale.
    covered_train_positive: set[str] = set()
    for train_geom in train_geoms:
        for idx in query_indices(tree, train_geom, poly_geoms):
            p = polygons[idx]
            if p.confidence in POSITIVE_CONF and train_geom.intersects(p.geometry_metric):
                covered_train_positive.add(p.key)

    rejected: dict[str, str] = {}
    candidates: list[Candidate] = []
    for name, tile in by_name.items():
        if name in train_names:
            rejected[name] = "already_in_frozen_train"
            continue
        if name in validation_names:
            rejected[name] = "frozen_validation_tile"
            continue
        if name in exclusions:
            rejected[name] = f"explicit_exclusion:{exclusions[name]}"
            continue

        footprint_wgs84 = tile_footprint_wgs84(tile)
        footprint_metric = tile_metric[name]
        if any(footprint_metric.intersects(guard) for guard in validation_guard):
            rejected[name] = f"within_{validation_buffer_m:g}m_of_frozen_validation"
            continue

        positive_keys: set[str] = set()
        high = moderate = 0
        pos_area = 0.0
        new_area = 0.0
        any_deposit = False
        for idx in query_indices(tree, footprint_metric, poly_geoms):
            p = polygons[idx]
            if not footprint_metric.intersects(p.geometry_metric):
                continue
            inter_area = float(footprint_metric.intersection(p.geometry_metric).area)
            if inter_area <= 0:
                continue
            any_deposit = True
            if p.confidence in POSITIVE_CONF:
                positive_keys.add(p.key)
                pos_area += inter_area
                if p.key not in covered_train_positive:
                    new_area += inter_area
                if p.confidence == "high":
                    high += 1
                elif p.confidence == "moderate":
                    moderate += 1

        if positive_keys:
            category = "positive_diversity"
        elif not any_deposit:
            category = "hard_negative"
        else:
            rejected[name] = "low_or_unknown_slido_overlap_only"
            continue

        new_keys = positive_keys - covered_train_positive
        cand = Candidate(
            name=name,
            tile=tile,
            footprint_wgs84=footprint_wgs84,
            footprint_metric=footprint_metric,
            centroid_metric=footprint_metric.centroid,
            downloaded=name in downloaded_names,
            probe=probes.get(name),
            category=category,
            positive_keys=positive_keys,
            new_positive_keys=new_keys,
            high_count=high,
            moderate_count=moderate,
            positive_intersection_m2=pos_area,
            new_positive_intersection_m2=new_area,
            min_existing_train_km=safe_min_distance_km(footprint_metric, train_geoms),
        )
        candidates.append(cand)
    return candidates, rejected, covered_train_positive


def static_score(c: Candidate) -> float:
    # Score uses only pre-download measurable information.
    score = 0.0
    if c.category == "positive_diversity":
        score += 12.0 * len(c.new_positive_keys)
        score += 2.0 * len(c.positive_keys)
        score += 2.5 * c.high_count + 1.0 * c.moderate_count
        score += min(8.0, math.log1p(c.new_positive_intersection_m2 / 10000.0))
    else:
        score += 4.0
    score += min(8.0, c.min_existing_train_km)
    if c.downloaded:
        score += 6.0  # reuse complete local files when they are otherwise eligible
    if c.probe:
        accepted = int(c.probe.get("accepted", 0))
        score += min(8.0, accepted / 12.5)
        if c.category == "positive_diversity":
            score += min(5.0, int(c.probe.get("retained_polygon_count", 0)))
        else:
            score += min(5.0, int(c.probe.get("negative", 0)) / 20.0)
    return score


def greedy_select(
    candidates: Sequence[Candidate],
    target_tiles: int,
    hard_negative_fraction: float,
    max_total_gb: float | None,
) -> list[Candidate]:
    if target_tiles <= 0:
        return []
    if not 0 <= hard_negative_fraction <= 1:
        raise ValueError("hard_negative_fraction must be between 0 and 1")

    for c in candidates:
        c.score_static = static_score(c)

    budget = math.floor(max_total_gb * 1_000_000_000) if max_total_gb is not None else 2**63 - 1
    hard_target = round(target_tiles * hard_negative_fraction)
    pos_target = target_tiles - hard_target
    pools = {
        "positive_diversity": [c for c in candidates if c.category == "positive_diversity"],
        "hard_negative": [c for c in candidates if c.category == "hard_negative"],
    }

    selected: list[Candidate] = []
    selected_names: set[str] = set()
    selected_centroids: list[BaseGeometry] = []
    represented_new_keys: set[str] = set()

    def choose_one(pool_name: str) -> bool:
        nonlocal budget
        choices: list[tuple[float, str, Candidate, float, int]] = []
        for c in pools[pool_name]:
            if c.name in selected_names:
                continue
            size = tile_size(c.tile)
            if size > budget:
                continue
            novelty_km = safe_min_distance_km(c.centroid_metric, selected_centroids)
            additional_new = len(c.new_positive_keys - represented_new_keys)
            dynamic = c.score_static + min(10.0, novelty_km)
            if pool_name == "positive_diversity":
                dynamic += 10.0 * additional_new
                if additional_new == 0:
                    dynamic -= 8.0
            if c.downloaded:
                dynamic += 1.5
            choices.append((dynamic, c.name.casefold(), c, novelty_km, additional_new))
        if not choices:
            return False
        choices.sort(key=lambda x: (-x[0], x[1]))
        _, _, chosen, novelty_km, additional_new = choices[0]
        selected.append(chosen)
        selected_names.add(chosen.name)
        selected_centroids.append(chosen.centroid_metric)
        represented_new_keys.update(chosen.new_positive_keys)
        budget -= tile_size(chosen.tile)
        chosen.selection_rank = len(selected)
        chosen.selection_reason = (
            f"{chosen.category}; additional_new_polygons={additional_new}; "
            f"selected_spatial_novelty_km={novelty_km:.2f}; "
            f"downloaded={'yes' if chosen.downloaded else 'no'}"
        )
        return True

    for _ in range(pos_target):
        if not choose_one("positive_diversity"):
            break
    for _ in range(hard_target):
        if not choose_one("hard_negative"):
            break

    # Backfill deterministically if one pool cannot meet its target.
    while len(selected) < target_tiles:
        possible = []
        for pool_name in ("positive_diversity", "hard_negative"):
            before = len(selected)
            if choose_one(pool_name):
                possible.append(selected[-1])
                if len(selected) > before:
                    break
        if not possible:
            break
    return selected


def candidate_row(c: Candidate) -> dict[str, Any]:
    probe = c.probe or {}
    bb = c.footprint_wgs84.bounds
    return {
        "selection_rank": c.selection_rank or "",
        "tile_name": c.name,
        "selection_category": c.category,
        "already_downloaded": c.downloaded,
        "size_bytes": tile_size(c.tile),
        "size_gb": round(tile_size(c.tile) / 1e9, 6),
        "download_url": tile_url(c.tile),
        "new_positive_polygon_count": len(c.new_positive_keys),
        "positive_polygon_count": len(c.positive_keys),
        "high_confidence_polygon_count": c.high_count,
        "moderate_confidence_polygon_count": c.moderate_count,
        "positive_intersection_m2": round(c.positive_intersection_m2, 3),
        "new_positive_intersection_m2": round(c.new_positive_intersection_m2, 3),
        "min_distance_to_existing_train_km": round(c.min_existing_train_km, 3),
        "probe_accepted_patches_2m": probe.get("accepted", ""),
        "probe_positive_patches_2m": probe.get("positive", ""),
        "probe_negative_patches_2m": probe.get("negative", ""),
        "probe_retained_polygon_count_2m": probe.get("retained_polygon_count", ""),
        "static_score": round(c.score_static, 4),
        "selection_reason": c.selection_reason,
        "bbox_west": bb[0],
        "bbox_south": bb[1],
        "bbox_east": bb[2],
        "bbox_north": bb[3],
    }


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_outputs(
    outdir: Path,
    selected: Sequence[Candidate],
    candidates: Sequence[Candidate],
    rejected: Mapping[str, str],
    covered_train_positive: set[str],
    train_names: set[str],
    validation_names: set[str],
    downloaded_names: set[str],
    target_tiles: int,
    hard_negative_fraction: float,
    validation_buffer_m: float,
) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    selected_rows = [candidate_row(c) for c in selected]
    candidate_rows = [candidate_row(c) for c in sorted(candidates, key=lambda x: (-x.score_static, x.name))]
    write_csv(outdir / "proposed_tillamook_expansion.csv", selected_rows)
    write_csv(outdir / "candidate_scores.csv", candidate_rows)
    write_csv(outdir / "excluded_tiles.csv", [{"tile_name": k, "reason": v} for k, v in sorted(rejected.items())])

    selected_payload = []
    for c in selected:
        item = dict(c.tile)
        item["_selection"] = candidate_row(c)
        selected_payload.append(item)
    (outdir / "proposed_tillamook_expansion.json").write_text(
        json.dumps(selected_payload, indent=2), encoding="utf-8"
    )

    missing_urls = [tile_url(c.tile) for c in selected if not c.downloaded and tile_url(c.tile)]
    (outdir / "download_urls.txt").write_text("\n".join(missing_urls) + ("\n" if missing_urls else ""), encoding="utf-8")

    selected_pos = [c for c in selected if c.category == "positive_diversity"]
    selected_neg = [c for c in selected if c.category == "hard_negative"]
    selected_downloaded = [c for c in selected if c.downloaded]
    unique_new_polygons = set().union(*(c.new_positive_keys for c in selected_pos)) if selected_pos else set()
    total_bytes = sum(tile_size(c.tile) for c in selected)
    missing_count = len(selected) - len(selected_downloaded)

    summary = f"""# Tillamook expansion selection summary

## Guardrails enforced

- Frozen train tiles excluded: **{len(train_names)}**.
- Frozen validation tiles excluded: **{len(validation_names)}**.
- Candidate footprints within **{validation_buffer_m:g} m** of frozen validation were excluded.
- Only the Tillamook source-region A22 project is considered.
- No Buxton/Vernonia or Oregon City data are read.
- Existing training-covered positive polygons are treated as already represented at tile scale.
- Low/unknown-only SLIDO overlap tiles are not silently treated as clean negatives.

## Proposed selection

- Requested target: **{target_tiles} tiles**.
- Selected: **{len(selected)} tiles**.
- Positive-diversity tiles: **{len(selected_pos)}**.
- Hard-negative tiles: **{len(selected_neg)}**.
- Hard-negative target fraction: **{hard_negative_fraction:.0%}**.
- Already downloaded and reusable: **{len(selected_downloaded)}**.
- Still requiring download: **{missing_count}**.
- Planned storage: **{total_bytes / 1e9:.2f} GB**.
- Unique new high/moderate SLIDO polygons represented by selected positive tiles: **{len(unique_new_polygons)}**.
- Positive polygons already represented by the frozen training tiles: **{len(covered_train_positive)}**.

## Interpretation

This is a **pre-download selection**, not a semantic terrain classification. The selector can measure SLIDO novelty, spatial novelty, known probe quality, split safety, file reuse, and clean no-inventory hard-negative status. It cannot honestly identify unseen TNM tiles as ridge, drainage, road cut, forestry disturbance, or rough terrain before terrain derivatives/NAIP are available.

After download/preprocessing, prioritize semantic review of hard-negative patches in the measured error-review order: rough/dissected natural terrain, steep non-landslide slopes, ridge/convex terrain, and drainage/valley sides. Roads/cuts and forest-management disturbance require NAIP/context confirmation.

## Files

- `proposed_tillamook_expansion.csv`: reviewable planned manifest.
- `proposed_tillamook_expansion.json`: full TNM records with selection annotations.
- `candidate_scores.csv`: all eligible candidates and scores.
- `excluded_tiles.csv`: frozen split, validation-buffer, explicit, and label-quality exclusions.
- `download_urls.txt`: only selected files not already present locally.
"""
    (outdir / "selection_summary.md").write_text(summary, encoding="utf-8")

    provenance = {
        "target_tiles": target_tiles,
        "hard_negative_fraction": hard_negative_fraction,
        "validation_buffer_m": validation_buffer_m,
        "selected_count": len(selected),
        "selected_positive_diversity": len(selected_pos),
        "selected_hard_negative": len(selected_neg),
        "selected_already_downloaded": len(selected_downloaded),
        "downloaded_input_unique": len(downloaded_names),
        "new_positive_polygon_count": len(unique_new_polygons),
    }
    (outdir / "selection_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Select a split-safe Tillamook A22 expansion emphasizing new polygons, spatial diversity, and hard negatives."
    )
    parser.add_argument("--tiles", type=Path, required=True, help="Tillamook TNM discovery JSON")
    parser.add_argument("--polygons", type=Path, required=True, help="SLIDO Tillamook GeoJSON in EPSG:4326")
    parser.add_argument("--frozen-manifest", type=Path, required=True, help="Current patches_boundary_aware.csv")
    parser.add_argument("--downloaded-csv", type=Path, help="actual_100tile_attempt.csv")
    parser.add_argument("--exclusions-csv", type=Path, help="Known tile failures/exclusions (e.g. failed_tiles.csv)")
    parser.add_argument("--probe-metrics", type=Path, help="Optional 2 m probe metrics, preferably ground threshold 0.20")
    parser.add_argument("--project", default=PROJECT_DEFAULT)
    parser.add_argument("--target-tiles", type=int, default=100)
    parser.add_argument("--hard-negative-fraction", type=float, default=0.40)
    parser.add_argument("--validation-buffer-m", type=float, default=500.0)
    parser.add_argument("--max-total-gb", type=float)
    parser.add_argument("--metric-crs", default=METRIC_CRS_DEFAULT)
    parser.add_argument("--outdir", type=Path, default=Path("selection_tillamook_expansion_100"))
    args = parser.parse_args()

    if args.target_tiles < 1:
        parser.error("--target-tiles must be >= 1")
    if not 0 <= args.hard_negative_fraction <= 1:
        parser.error("--hard-negative-fraction must be between 0 and 1")
    if args.validation_buffer_m < 0:
        parser.error("--validation-buffer-m must be >= 0")

    to_metric = Transformer.from_crs("EPSG:4326", args.metric_crs, always_xy=True)
    tiles = load_tiles(args.tiles, args.project)
    polygons = load_polygons(args.polygons, to_metric)
    train_names, validation_names = load_split_tiles(args.frozen_manifest)
    downloaded_names = load_downloaded(args.downloaded_csv)
    exclusions = load_exclusions(args.exclusions_csv)
    probes = load_probe_metrics(args.probe_metrics)

    print(f"A22 TNM candidates: {len(tiles)}")
    print(f"SLIDO features loaded: {len(polygons)}")
    print(f"Frozen tiles: train={len(train_names)}, validation={len(validation_names)}")
    print(f"Downloaded unique LAZ names supplied: {len(downloaded_names)}")
    print(f"Explicit exclusions: {len(exclusions)}")
    print(f"Probe metrics at 2 m: {len(probes)}")

    candidates, rejected, covered_train_positive = build_candidates(
        tiles,
        polygons,
        train_names,
        validation_names,
        downloaded_names,
        exclusions,
        probes,
        to_metric,
        args.validation_buffer_m,
    )
    selected = greedy_select(candidates, args.target_tiles, args.hard_negative_fraction, args.max_total_gb)
    write_outputs(
        args.outdir,
        selected,
        candidates,
        rejected,
        covered_train_positive,
        train_names,
        validation_names,
        downloaded_names,
        args.target_tiles,
        args.hard_negative_fraction,
        args.validation_buffer_m,
    )
    print(
        f"Selected {len(selected)} tiles: "
        f"{sum(c.category == 'positive_diversity' for c in selected)} positive-diversity, "
        f"{sum(c.category == 'hard_negative' for c in selected)} hard-negative"
    )
    print(f"Already downloaded: {sum(c.downloaded for c in selected)}")
    print(f"Wrote: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
