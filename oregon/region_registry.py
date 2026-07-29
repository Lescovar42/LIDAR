#!/usr/bin/env python3
"""Load, validate, and resolve Oregon rural-pipeline regions."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REGISTRY_PATH = Path(__file__).with_name("regions.json")
VALID_ROLES = {"train_val", "test_rural", "test_urban_ood"}
_REQUIRED_PATHS = ("slido_output", "tnm_records", "naip_records")


def _validate_bbox(value: Any, label: str) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) != 4 or any(
        not isinstance(item, (int, float)) for item in value
    ):
        raise ValueError(f"{label}.bbox must contain four numbers")
    xmin, ymin, xmax, ymax = (float(item) for item in value)
    if not (-180 <= xmin < xmax <= 180 and -90 <= ymin < ymax <= 90):
        raise ValueError(f"{label}.bbox is not a valid WGS84 bounding box")
    return xmin, ymin, xmax, ymax


def _bboxes_overlap(a: tuple[float, ...], b: tuple[float, ...]) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def validate_registry(data: Mapping[str, Any]) -> None:
    """Validate registry structure and split-safety invariants."""
    regions = data.get("regions")
    candidates = data.get("comparison_candidates", [])
    if not isinstance(regions, list) or not regions:
        raise ValueError("registry must define a non-empty regions list")
    if not isinstance(candidates, list):
        raise ValueError("comparison_candidates must be a list")

    seen_ids: set[str] = set()
    active_bboxes: list[tuple[str, tuple[float, float, float, float]]] = []
    for entry in [*regions, *candidates]:
        if not isinstance(entry, Mapping):
            raise ValueError("each registry entry must be an object")
        label = str(entry.get("id") or "<missing id>")
        for field in ("id", "slug", "name", "role"):
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError(f"{label}.{field} must be a non-empty string")
        if entry["id"] in seen_ids:
            raise ValueError(f"duplicate region id: {entry['id']}")
        seen_ids.add(entry["id"])
        if entry["role"] not in VALID_ROLES:
            raise ValueError(f"{label}.role is unknown: {entry['role']}")
        bbox = _validate_bbox(entry.get("bbox"), label)
        for field in ("tile_budget", "storage_budget_gb", "slido_max_features"):
            value = entry.get(field)
            if not isinstance(value, (int, float)) or value < 0:
                raise ValueError(f"{label}.{field} must be zero or positive")
        if entry["slido_max_features"] <= 0:
            raise ValueError(f"{label}.slido_max_features must be positive")
        for field in _REQUIRED_PATHS:
            if not isinstance(entry.get(field), str) or not entry[field].strip():
                raise ValueError(f"{label}.{field} path is required")
        projects = entry.get("candidate_projects")
        if not isinstance(projects, list) or not projects or any(
            not isinstance(project, str) or not project.strip() for project in projects
        ):
            raise ValueError(f"{label}.candidate_projects must be a non-empty string list")

        pin_fields = ("lidar_project", "cell_size", "selection_decision")
        present_pin_fields = [field for field in pin_fields if field in entry]
        if present_pin_fields and len(present_pin_fields) != len(pin_fields):
            raise ValueError(f"{label} pin must define lidar_project, cell_size, and selection_decision together")
        if present_pin_fields:
            project = entry["lidar_project"]
            if not isinstance(project, str) or not project.strip() or project not in projects:
                raise ValueError(f"{label}.lidar_project must be one of candidate_projects")
            cell_size = entry["cell_size"]
            if (
                isinstance(cell_size, bool)
                or not isinstance(cell_size, (int, float))
                or not math.isfinite(float(cell_size))
                or cell_size <= 0
            ):
                raise ValueError(f"{label}.cell_size must be positive")
            decision = entry["selection_decision"]
            if not isinstance(decision, Mapping) or not decision:
                raise ValueError(f"{label}.selection_decision must be a non-empty object")
            reason = decision.get("reason")
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"{label}.selection_decision.reason must be a non-empty string")
        if entry in regions:
            active_bboxes.append((label, bbox))

    for index, (left_id, left_bbox) in enumerate(active_bboxes):
        for right_id, right_bbox in active_bboxes[index + 1 :]:
            if _bboxes_overlap(left_bbox, right_bbox):
                raise ValueError(f"active region bboxes overlap: {left_id} and {right_id}")


def load_registry(path: str | Path = REGISTRY_PATH) -> dict[str, Any]:
    registry_path = Path(path)
    with registry_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    validate_registry(data)
    data["_registry_path"] = str(registry_path.resolve())
    return data


def resolve_region(
    value: str,
    registry: Mapping[str, Any] | None = None,
    *,
    include_comparison: bool = False,
) -> dict[str, Any]:
    """Resolve a region by id, slug, or name, case-insensitively."""
    registry = registry or load_registry()
    entries = list(registry["regions"])
    if include_comparison:
        entries.extend(registry.get("comparison_candidates", []))
    wanted = value.strip().casefold()
    for entry in entries:
        if wanted in {str(entry[key]).casefold() for key in ("id", "slug", "name")}:
            return dict(entry)
    choices = ", ".join(entry["id"] for entry in entries)
    raise KeyError(f"unknown region {value!r}; choose one of: {choices}")


def pin_region_decision(
    registry_path: str | Path,
    region_name: str,
    *,
    lidar_project: str,
    cell_size: float,
    reason: str,
    decision_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Atomically persist a measured LiDAR choice for an active region."""
    path = Path(registry_path)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    validate_registry(data)
    region = resolve_region(region_name, data)

    project = lidar_project.strip() if isinstance(lidar_project, str) else ""
    if project not in region["candidate_projects"]:
        raise ValueError(f"{region['id']}.lidar_project must be one of candidate_projects")
    if (
        isinstance(cell_size, bool)
        or not isinstance(cell_size, (int, float))
        or not math.isfinite(float(cell_size))
        or cell_size <= 0
    ):
        raise ValueError("cell_size must be positive")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    if decision_metadata is not None and not isinstance(decision_metadata, Mapping):
        raise ValueError("decision_metadata must be an object")

    decision = dict(decision_metadata or {})
    decision["reason"] = reason.strip()
    decision.setdefault(
        "decided_at_utc",
        datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    )
    for entry in data["regions"]:
        if entry["id"] == region["id"]:
            entry["lidar_project"] = project
            entry["cell_size"] = float(cell_size)
            entry["selection_decision"] = decision
            break
    validate_registry(data)

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_name = handle.name
            json.dump(data, handle, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
    return resolve_region(region["id"], data)


def resolve_path(region: Mapping[str, Any], field: str, registry_path: str | Path = REGISTRY_PATH) -> Path:
    value = Path(str(region[field]))
    if value.is_absolute():
        return value
    return Path(registry_path).resolve().parent / value


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate and resolve an Oregon region registry entry.")
    parser.add_argument("--registry", type=Path, default=REGISTRY_PATH)
    parser.add_argument("--region", required=True, help="Region id, slug, or name.")
    parser.add_argument("--include-comparison", action="store_true")
    args = parser.parse_args()
    try:
        region = resolve_region(
            args.region,
            load_registry(args.registry),
            include_comparison=args.include_comparison,
        )
    except (KeyError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(region, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
