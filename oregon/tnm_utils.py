"""Shared helpers for interpreting TNM product records."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any, Mapping, Sequence
from urllib.parse import parse_qs, unquote, urlparse

_EXPLICIT_PROJECT_FIELDS = ("project", "projectName", "project_name", "collectionName")
_URL_FIELDS = ("downloadURL", "downloadLazURL")
_TILE_SUFFIX = re.compile(r"(?:[_\s-])(?:[we]\d+[ns]\d+|\d+)$", re.IGNORECASE)


def _sources(record: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    properties = record.get("properties")
    return (record, properties) if isinstance(properties, Mapping) else (record,)


def first_value(
    record: Mapping[str, Any], fields: Sequence[str], default: Any = None
) -> Any:
    """Return the first non-empty top-level or GeoJSON-property value."""
    for source in _sources(record):
        for field in fields:
            value = source.get(field)
            if value not in (None, ""):
                return value
    return default


def _project_from_download_url(value: Any) -> str | None:
    if not value:
        return None
    parsed = urlparse(str(value))
    path = unquote(parsed.path)
    match = re.search(r"/Elevation/LPC/Projects/([^/]+)", path, re.IGNORECASE)
    if match:
        project = match.group(1)
        return project if project.upper().startswith("USGS_LPC_") else f"USGS_LPC_{project}"

    stem = PurePosixPath(path).stem
    if stem.upper().startswith("USGS_LPC_"):
        return _TILE_SUFFIX.sub("", stem)
    return None


def _project_from_vendor_url(value: Any) -> str | None:
    if not value:
        return None
    parsed = urlparse(str(value))
    prefix = parse_qs(parsed.query).get("prefix", [""])[0]
    match = re.search(r"(?:^|/)Elevation/(?:metadata|LPC/Projects)/([^/]+)", prefix)
    if not match:
        return None
    project = match.group(1)
    return project if project.upper().startswith("USGS_LPC_") else f"USGS_LPC_{project}"


def _project_from_title(value: Any) -> str | None:
    title = str(value or "").strip()
    if not title:
        return None
    prefix = "USGS Lidar Point Cloud "
    if title.casefold().startswith(prefix.casefold()):
        project = _TILE_SUFFIX.sub("", title[len(prefix) :].strip())
        return f"USGS_LPC_{project.replace(' ', '_')}" if project else None
    return _TILE_SUFFIX.sub("", title).strip() or None


def canonical_project(record: Mapping[str, Any]) -> str:
    """Derive stable TNM project identity without using per-product ``sourceId``.

    Current TNM LPC snapshots assign a different ScienceBase sourceId to every
    tile. Stable collection identity instead appears in explicit project fields,
    the LPC project path, vendor metadata prefix, and tile title.
    """
    explicit = first_value(record, _EXPLICIT_PROJECT_FIELDS)
    if explicit:
        return str(explicit).strip()

    for field in _URL_FIELDS:
        project = _project_from_download_url(first_value(record, (field,)))
        if project:
            return project

    project = _project_from_vendor_url(first_value(record, ("vendorMetaUrl",)))
    if project:
        return project

    return _project_from_title(first_value(record, ("title", "name"))) or "unknown"


def project_matches(record_or_project: Mapping[str, Any] | str, candidate: str) -> bool:
    """Match a registry candidate to a canonical project at token boundaries.

    Registry entries may intentionally omit a TNM delivery suffix such as
    ``_A19`` while records retain it. The boundary check avoids accidental
    substring matches between unrelated project names.
    """
    actual = (
        canonical_project(record_or_project)
        if isinstance(record_or_project, Mapping)
        else str(record_or_project)
    ).strip().casefold()
    wanted = str(candidate).strip().casefold()
    return bool(wanted) and (actual == wanted or actual.startswith(wanted + "_"))
