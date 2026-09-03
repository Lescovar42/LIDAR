#!/usr/bin/env python3
"""Auditable LiDAR temporal provenance.

This module keeps three different temporal concepts strictly separated so that
no heuristic can silently become scientific evidence:

``LidarAcquisition``
    Authoritative airborne acquisition metadata. This is the only value allowed
    to drive SLIDO temporal filtering, NAIP nearest-year selection, and any
    displayed LiDAR/NAIP vintage gap. It must arrive from an explicit,
    attributable source (registry project metadata or an explicit CLI override).

``LidarFileMetadata``
    LAS/LAZ header ``creation_date``. This describes when the *file* was
    written, processed, or repackaged. It is preserved for auditing and never
    promoted to acquisition metadata.

``InferredYearHint``
    A four-digit year or TNM-style ``A22`` token parsed out of a project or
    tile name. Retained as a non-authoritative diagnostic hint only.

Nothing in this module ever infers acquisition from a header date, a filename,
a project token, the current year, or a NAIP year.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

# Airborne LiDAR acquisition predates 1990 only in research prototypes; a lower
# bound this loose still rejects obviously wrong values such as 1900 or 0.
MIN_ACQUISITION_YEAR = 1990
# LAS header dates are validated more loosely because bad writers emit odd dates
# and the value is preserved for auditing rather than used for reasoning.
MIN_FILE_CREATION_YEAR = 1900

UNKNOWN_YEAR_SOURCE = "unknown"
CLI_ORIGIN = "cli_acquisition_override"
REGISTRY_ORIGIN = "registry_verified_project_metadata"

ACQUISITION_KEYS = frozenset(
    {
        "start_year",
        "end_year",
        "nominal_year",
        "source",
        "evidence",
        "verified",
        "lidar_project",
    }
)

_FULL_YEAR_PATTERN = re.compile(r"(?<!\d)((?:19|20)\d{2})(?!\d)")
_SHORT_YEAR_PATTERN = re.compile(r"(?:^|[_-])A(\d{2})(?:[_-]|$)", re.I)

_TRUE_TOKENS = frozenset({"true", "1", "yes", "y", "t"})
_FALSE_TOKENS = frozenset({"false", "0", "no", "n", "f", ""})


class AcquisitionMetadataError(ValueError):
    """Invalid, partial, or contradictory acquisition metadata."""


class AcquisitionConflictError(AcquisitionMetadataError):
    """Two or more authoritative acquisition sources disagree."""


def max_acquisition_year() -> int:
    """Upper bound for a plausible acquisition year (next calendar year)."""
    return datetime.now(timezone.utc).year + 1


def _coerce_year(value: Any, *, label: str, minimum: int, maximum: int | None = None) -> int:
    maximum = max_acquisition_year() if maximum is None else maximum
    if isinstance(value, bool):
        raise AcquisitionMetadataError(f"{label} must be a four-digit year, not a boolean")
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise AcquisitionMetadataError(f"{label} must be a four-digit year, not an empty value")
        try:
            number = int(text, 10)
        except ValueError as exc:
            raise AcquisitionMetadataError(f"{label} is not a four-digit year: {value!r}") from exc
    elif isinstance(value, int):
        number = value
    elif isinstance(value, float):
        if not float(value).is_integer():
            raise AcquisitionMetadataError(f"{label} is not a whole year: {value!r}")
        number = int(value)
    else:
        raise AcquisitionMetadataError(f"{label} is not a four-digit year: {value!r}")
    if not minimum <= number <= maximum:
        raise AcquisitionMetadataError(
            f"{label} {number} is outside the allowed range {minimum}-{maximum}"
        )
    return number


def _require_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AcquisitionMetadataError(f"{label} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class LidarAcquisition:
    """Authoritative airborne acquisition vintage for one LiDAR project."""

    start_year: int
    end_year: int
    source: str
    nominal_year: int | None = None
    evidence: str = ""
    verified: bool = False
    origin: str = "unspecified"
    lidar_project: str = ""

    def __post_init__(self) -> None:
        maximum = max_acquisition_year()
        for name in ("start_year", "end_year"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise AcquisitionMetadataError(
                    f"lidar_acquisition.{name} must be an integer year, received {value!r}"
                )
            _coerce_year(
                value,
                label=f"lidar_acquisition.{name}",
                minimum=MIN_ACQUISITION_YEAR,
                maximum=maximum,
            )
        if self.start_year > self.end_year:
            raise AcquisitionMetadataError(
                "lidar_acquisition.start_year must not exceed lidar_acquisition.end_year "
                f"({self.start_year} > {self.end_year})"
            )
        if self.nominal_year is not None:
            _coerce_year(
                self.nominal_year,
                label="lidar_acquisition.nominal_year",
                minimum=MIN_ACQUISITION_YEAR,
                maximum=maximum,
            )
            if not self.start_year <= self.nominal_year <= self.end_year:
                raise AcquisitionMetadataError(
                    f"lidar_acquisition.nominal_year {self.nominal_year} falls outside the "
                    f"acquisition range {self.start_year}-{self.end_year}"
                )
        _require_text(self.source, label="lidar_acquisition.source")

    @property
    def is_multi_year(self) -> bool:
        return self.start_year != self.end_year

    @property
    def range_text(self) -> str:
        return (
            f"{self.start_year}-{self.end_year}"
            if self.is_multi_year
            else str(self.start_year)
        )

    def require_nominal_year(self, context: str) -> int:
        """Return the explicit nominal year, or fail when a scalar is required."""
        if self.nominal_year is None:
            raise AcquisitionMetadataError(
                f"{context} requires one authoritative LiDAR acquisition year, but "
                f"{self.origin} supplies the multi-year range {self.range_text} with no "
                "explicit nominal year. Set lidar_acquisition.nominal_year in the registry "
                "or pass --lidar-acquisition-year; the midpoint, start, end, and latest "
                "year are never chosen automatically."
            )
        return self.nominal_year

    def comparison_key(self) -> tuple[int, int, int | None]:
        """Year identity used for conflict detection (text fields excluded)."""
        return (self.start_year, self.end_year, self.nominal_year)

    def describe(self) -> str:
        nominal = self.nominal_year if self.nominal_year is not None else "unspecified"
        return (
            f"start={self.start_year} end={self.end_year} nominal={nominal} "
            f"source={self.source!r} verified={str(self.verified).lower()}"
        )

    def as_registry_mapping(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "start_year": self.start_year,
            "end_year": self.end_year,
        }
        if self.nominal_year is not None:
            data["nominal_year"] = self.nominal_year
        data["source"] = self.source
        if self.evidence:
            data["evidence"] = self.evidence
        data["verified"] = self.verified
        if self.lidar_project:
            data["lidar_project"] = self.lidar_project
        return data

    def as_summary_mapping(self) -> dict[str, Any]:
        return {
            "start_year": self.start_year,
            "end_year": self.end_year,
            "nominal_year": self.nominal_year,
            "source": self.source,
            "evidence": self.evidence,
            "verified": self.verified,
            "origin": self.origin,
            "lidar_project": self.lidar_project,
        }


def parse_acquisition(
    data: Mapping[str, Any] | None,
    *,
    origin: str,
    label: str = "lidar_acquisition",
    require_evidence: bool = True,
    allowed_projects: Sequence[str] | None = None,
    expected_project: str | None = None,
) -> LidarAcquisition | None:
    """Validate an explicit acquisition mapping, or return ``None`` when absent.

    Partial or contradictory metadata always raises; nothing is inferred.
    """
    if data is None:
        return None
    if not isinstance(data, Mapping):
        raise AcquisitionMetadataError(f"{label} must be an object")
    if not data:
        raise AcquisitionMetadataError(
            f"{label} is present but empty; remove it or supply complete metadata"
        )
    unknown = sorted(set(data) - ACQUISITION_KEYS)
    if unknown:
        raise AcquisitionMetadataError(f"{label} has unsupported field(s): {unknown}")

    maximum = max_acquisition_year()
    has_start = "start_year" in data
    has_end = "end_year" in data
    has_nominal = "nominal_year" in data
    if has_start != has_end:
        raise AcquisitionMetadataError(
            f"{label} must define start_year and end_year together, not one of them"
        )
    if not has_start and not has_nominal:
        raise AcquisitionMetadataError(
            f"{label} must define nominal_year, or start_year and end_year"
        )

    nominal = (
        _coerce_year(
            data["nominal_year"],
            label=f"{label}.nominal_year",
            minimum=MIN_ACQUISITION_YEAR,
            maximum=maximum,
        )
        if has_nominal
        else None
    )
    if has_start:
        start = _coerce_year(
            data["start_year"],
            label=f"{label}.start_year",
            minimum=MIN_ACQUISITION_YEAR,
            maximum=maximum,
        )
        end = _coerce_year(
            data["end_year"],
            label=f"{label}.end_year",
            minimum=MIN_ACQUISITION_YEAR,
            maximum=maximum,
        )
    else:
        start = end = nominal  # type: ignore[assignment]

    source = _require_text(data.get("source"), label=f"{label}.source")
    evidence_value = data.get("evidence", "")
    if require_evidence:
        evidence = _require_text(evidence_value, label=f"{label}.evidence")
    else:
        if evidence_value not in (None, "") and not isinstance(evidence_value, str):
            raise AcquisitionMetadataError(f"{label}.evidence must be a string")
        evidence = str(evidence_value or "").strip()

    verified_value = data.get("verified", False)
    if not isinstance(verified_value, bool):
        raise AcquisitionMetadataError(f"{label}.verified must be true or false")

    project_value = data.get("lidar_project", "")
    if project_value not in (None, "") and not isinstance(project_value, str):
        raise AcquisitionMetadataError(f"{label}.lidar_project must be a string")
    project = str(project_value or "").strip()
    if allowed_projects is not None:
        if not project:
            raise AcquisitionMetadataError(
                f"{label}.lidar_project is required so acquisition metadata is tied to "
                "one LiDAR project"
            )
        if project not in allowed_projects:
            raise AcquisitionMetadataError(
                f"{label}.lidar_project {project!r} must be one of candidate_projects"
            )
    if expected_project and project and project != expected_project:
        raise AcquisitionMetadataError(
            f"{label}.lidar_project {project!r} does not match the selected LiDAR project "
            f"{expected_project!r}; acquisition metadata must not be reused across projects"
        )

    return LidarAcquisition(
        start_year=start,
        end_year=end,
        nominal_year=nominal,
        source=source,
        evidence=evidence,
        verified=verified_value,
        origin=origin,
        lidar_project=project,
    )


def acquisition_from_cli(
    *,
    year: Any = None,
    start_year: Any = None,
    end_year: Any = None,
    source: Any = None,
    evidence: Any = None,
    verified: bool = False,
    lidar_project: str = "",
    origin: str = CLI_ORIGIN,
) -> LidarAcquisition | None:
    """Build acquisition metadata from explicit CLI arguments.

    Returns ``None`` only when no acquisition argument was supplied at all. Any
    partial combination raises so a half-specified override cannot slip through.
    """
    supplied = {
        "--lidar-acquisition-year": year,
        "--lidar-acquisition-start-year": start_year,
        "--lidar-acquisition-end-year": end_year,
        "--lidar-acquisition-source": (source or None),
        "--lidar-acquisition-evidence": (evidence or None),
    }
    if all(value is None for value in supplied.values()) and not verified:
        return None
    if source is None or not str(source).strip():
        provided = sorted(name for name, value in supplied.items() if value is not None)
        raise AcquisitionMetadataError(
            "--lidar-acquisition-source is required whenever acquisition metadata is "
            f"supplied (received {provided or ['--lidar-acquisition-verified']}); the LAS "
            "header date, project token, and filename are never used instead"
        )
    if year is None and (start_year is None or end_year is None):
        if start_year is None and end_year is None:
            raise AcquisitionMetadataError(
                "--lidar-acquisition-year, or both --lidar-acquisition-start-year and "
                "--lidar-acquisition-end-year, are required"
            )
        raise AcquisitionMetadataError(
            "--lidar-acquisition-start-year and --lidar-acquisition-end-year must be "
            "supplied together"
        )

    data: dict[str, Any] = {"source": source}
    if year is not None:
        data["nominal_year"] = year
    if start_year is not None and end_year is not None:
        data["start_year"] = start_year
        data["end_year"] = end_year
    elif year is not None:
        data["start_year"] = year
        data["end_year"] = year
    if evidence:
        data["evidence"] = evidence
    data["verified"] = bool(verified)
    if lidar_project:
        data["lidar_project"] = lidar_project
    return parse_acquisition(
        data,
        origin=origin,
        label="lidar_acquisition (CLI)",
        require_evidence=False,
    )


def resolve_acquisition(
    candidates: Iterable[tuple[str, LidarAcquisition | None]],
    *,
    context: str = "",
) -> LidarAcquisition | None:
    """Return the single authoritative acquisition record, or fail on conflict.

    ``candidates`` must be ordered by precedence (highest first). Sources that
    agree on the year identity are collapsed; disagreement is always an error so
    no source is silently preferred.
    """
    present = [(name, value) for name, value in candidates if value is not None]
    if not present:
        return None
    if len({value.comparison_key() for _, value in present}) > 1:
        detail = "; ".join(f"{name} -> {value.describe()}" for name, value in present)
        where = f" for {context}" if context else ""
        raise AcquisitionConflictError(
            f"Conflicting authoritative LiDAR acquisition metadata{where}: {detail}. "
            "Resolve the disagreement explicitly; no source is chosen automatically."
        )
    return present[0][1]


@dataclass(frozen=True)
class LidarFileMetadata:
    """LAS/LAZ header file metadata. Never authoritative for acquisition."""

    creation_year: int | None = None
    creation_date: str = ""

    @property
    def is_known(self) -> bool:
        return self.creation_year is not None or bool(self.creation_date)


def file_metadata_from_header(value: Any) -> LidarFileMetadata:
    """Preserve the LAS header creation date without interpreting it."""
    if value is None:
        return LidarFileMetadata()
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        year = value.year
        return LidarFileMetadata(
            creation_year=(
                year if MIN_FILE_CREATION_YEAR <= year <= max_acquisition_year() else None
            ),
            creation_date=value.isoformat(),
        )
    text = str(value).strip()
    if not text:
        return LidarFileMetadata()
    match = _FULL_YEAR_PATTERN.search(text)
    year = int(match.group(1)) if match else None
    if year is not None and not MIN_FILE_CREATION_YEAR <= year <= max_acquisition_year():
        year = None
    return LidarFileMetadata(creation_year=year, creation_date=text)


def parse_year_hint(text: str | None) -> int | None:
    """Parse a non-authoritative four-digit year or TNM-style ``A22`` token."""
    full_years = [int(value) for value in _FULL_YEAR_PATTERN.findall(text or "")]
    if full_years:
        return max(full_years)
    short_years = [int(value) for value in _SHORT_YEAR_PATTERN.findall(text or "")]
    if short_years:
        value = max(short_years)
        return 2000 + value if value <= 79 else 1900 + value
    return None


@dataclass(frozen=True)
class InferredYearHint:
    """Diagnostic-only year parsed from a project or tile name."""

    year: int | None = None
    source: str = "none"

    @property
    def is_known(self) -> bool:
        return self.year is not None


def infer_year_hint(project_name: str = "", tile_name: str = "") -> InferredYearHint:
    project_year = parse_year_hint(project_name)
    if project_year is not None:
        return InferredYearHint(year=project_year, source="project_name")
    tile_year = parse_year_hint(tile_name)
    if tile_year is not None:
        return InferredYearHint(year=tile_year, source="tile_name")
    return InferredYearHint()


@dataclass(frozen=True)
class LidarVintage:
    """Full temporal provenance for one LiDAR tile."""

    acquisition: LidarAcquisition | None = None
    file_metadata: LidarFileMetadata = field(default_factory=LidarFileMetadata)
    hint: InferredYearHint = field(default_factory=InferredYearHint)

    @property
    def acquisition_is_known(self) -> bool:
        return self.acquisition is not None

    @property
    def acquisition_nominal_year(self) -> int | None:
        return None if self.acquisition is None else self.acquisition.nominal_year

    def temporal_filter_year(self, context: str = "SLIDO temporal filtering") -> int | None:
        """The only year allowed to drive temporal filtering or NAIP selection."""
        if self.acquisition is None:
            return None
        return self.acquisition.require_nominal_year(context)

    @property
    def legacy_lidar_year(self) -> int | None:
        """Compatibility alias for the historical scalar ``lidar_year`` column."""
        return self.acquisition_nominal_year

    @property
    def legacy_lidar_year_source(self) -> str:
        return UNKNOWN_YEAR_SOURCE if self.acquisition is None else self.acquisition.origin

    def as_row_fields(self) -> dict[str, Any]:
        """Flat, CSV-safe provenance columns."""
        acquisition = self.acquisition
        return {
            "lidar_year": "" if self.legacy_lidar_year is None else self.legacy_lidar_year,
            "lidar_year_source": self.legacy_lidar_year_source,
            "lidar_acquisition_year": (
                "" if acquisition is None or acquisition.nominal_year is None
                else acquisition.nominal_year
            ),
            "lidar_acquisition_start_year": "" if acquisition is None else acquisition.start_year,
            "lidar_acquisition_end_year": "" if acquisition is None else acquisition.end_year,
            "lidar_acquisition_source": "" if acquisition is None else acquisition.source,
            "lidar_acquisition_evidence": "" if acquisition is None else acquisition.evidence,
            "lidar_acquisition_verified": (
                "" if acquisition is None else str(acquisition.verified).lower()
            ),
            "lidar_file_creation_year": (
                "" if self.file_metadata.creation_year is None
                else self.file_metadata.creation_year
            ),
            "lidar_file_creation_date": self.file_metadata.creation_date,
            "lidar_inferred_year_hint": "" if self.hint.year is None else self.hint.year,
            "lidar_inferred_year_hint_source": self.hint.source,
        }

    def as_summary_mapping(self) -> dict[str, Any]:
        """Typed provenance block for JSON summaries."""
        return {
            "acquisition": (
                None if self.acquisition is None else self.acquisition.as_summary_mapping()
            ),
            "file_metadata": {
                "creation_year": self.file_metadata.creation_year,
                "creation_date": self.file_metadata.creation_date,
            },
            "inferred_year_hint": {
                "year": self.hint.year,
                "source": self.hint.source,
            },
        }


VINTAGE_ROW_FIELDS: tuple[str, ...] = (
    "lidar_year",
    "lidar_year_source",
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
)

ACQUISITION_ROW_FIELDS: tuple[str, ...] = (
    "lidar_acquisition_year",
    "lidar_acquisition_start_year",
    "lidar_acquisition_end_year",
    "lidar_acquisition_source",
    "lidar_acquisition_evidence",
    "lidar_acquisition_verified",
)


def parse_bool_field(value: Any) -> bool | None:
    """Parse a CSV boolean; ``None`` when blank or unrecognized."""
    if value is None:
        return None
    text = str(value).strip().casefold()
    if not text:
        return None
    if text in _TRUE_TOKENS:
        return True
    if text in _FALSE_TOKENS:
        return False
    return None


def parse_year_field(
    value: Any,
    *,
    label: str = "year",
    minimum: int = MIN_FILE_CREATION_YEAR,
    maximum: int | None = None,
    strict: bool = True,
) -> int | None:
    """Read an optional year from CSV text.

    Blank is ``None``. A non-blank but malformed value raises when ``strict``
    so a corrupt manifest cannot quietly become "unknown".
    """
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return _coerce_year(
            text,
            label=label,
            minimum=minimum,
            maximum=max_acquisition_year() if maximum is None else maximum,
        )
    except AcquisitionMetadataError:
        if strict:
            raise
        return None


def row_has_acquisition_columns(row: Mapping[str, Any]) -> bool:
    """True when a manifest row uses the post-fix acquisition schema."""
    return any(name in row for name in ACQUISITION_ROW_FIELDS)


def acquisition_year_from_row(
    row: Mapping[str, Any],
    *,
    trust_legacy_lidar_year: bool = False,
    label: str = "row",
) -> int | None:
    """Read the authoritative acquisition year from a manifest row.

    A legacy row that only carries ``lidar_year`` is treated as *unknown*
    acquisition unless the caller explicitly opts in, because that historical
    column could hold a LAS file-creation year or a filename hint.
    """
    if row_has_acquisition_columns(row):
        return parse_year_field(
            row.get("lidar_acquisition_year"),
            label=f"{label}.lidar_acquisition_year",
            minimum=MIN_ACQUISITION_YEAR,
        )
    if trust_legacy_lidar_year:
        return parse_year_field(
            row.get("lidar_year"),
            label=f"{label}.lidar_year",
            minimum=MIN_ACQUISITION_YEAR,
        )
    return None


def acquisition_provenance_from_row(
    row: Mapping[str, Any],
    *,
    trust_legacy_lidar_year: bool = False,
    label: str = "row",
) -> dict[str, Any]:
    """Copy authoritative acquisition provenance out of a patch manifest row.

    Used to (re)build downstream manifests from the *current* patch manifest so
    stale cached artifacts can never reintroduce an old vintage.
    """
    year = acquisition_year_from_row(
        row, trust_legacy_lidar_year=trust_legacy_lidar_year, label=label
    )
    legacy_only = not row_has_acquisition_columns(row)
    start = parse_year_field(
        row.get("lidar_acquisition_start_year"),
        label=f"{label}.lidar_acquisition_start_year",
        minimum=MIN_ACQUISITION_YEAR,
    )
    end = parse_year_field(
        row.get("lidar_acquisition_end_year"),
        label=f"{label}.lidar_acquisition_end_year",
        minimum=MIN_ACQUISITION_YEAR,
    )
    verified = parse_bool_field(row.get("lidar_acquisition_verified"))
    source = str(row.get("lidar_acquisition_source", "") or "").strip()
    if legacy_only and year is not None:
        source = source or "legacy_manifest_lidar_year (explicitly trusted)"
    creation_year = parse_year_field(
        row.get("lidar_file_creation_year"),
        label=f"{label}.lidar_file_creation_year",
        strict=False,
    )
    hint_year = parse_year_field(
        row.get("lidar_inferred_year_hint"),
        label=f"{label}.lidar_inferred_year_hint",
        strict=False,
    )
    return {
        "lidar_acquisition_year": "" if year is None else year,
        "lidar_acquisition_start_year": "" if start is None else start,
        "lidar_acquisition_end_year": "" if end is None else end,
        "lidar_acquisition_source": source,
        "lidar_acquisition_evidence": str(
            row.get("lidar_acquisition_evidence", "") or ""
        ).strip(),
        "lidar_acquisition_verified": "" if verified is None else str(verified).lower(),
        "lidar_year": "" if year is None else year,
        "lidar_year_source": str(row.get("lidar_year_source", "") or "").strip()
        or (UNKNOWN_YEAR_SOURCE if year is None else source or UNKNOWN_YEAR_SOURCE),
        "lidar_file_creation_year": "" if creation_year is None else creation_year,
        "lidar_file_creation_date": str(row.get("lidar_file_creation_date", "") or "").strip(),
        "lidar_inferred_year_hint": "" if hint_year is None else hint_year,
        "lidar_inferred_year_hint_source": str(
            row.get("lidar_inferred_year_hint_source", "") or ""
        ).strip(),
    }


def summarize_vintages(vintages: Sequence[LidarVintage]) -> dict[str, Any]:
    """Aggregate distinct acquisition vintages, file years, and hints."""
    acquisition_groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    file_years: dict[str, int] = {}
    hints: dict[str, int] = {}
    unknown = 0
    for vintage in vintages:
        acquisition = vintage.acquisition
        if acquisition is None:
            unknown += 1
        else:
            key = (
                acquisition.start_year,
                acquisition.end_year,
                acquisition.nominal_year,
                acquisition.source,
                acquisition.evidence,
                acquisition.verified,
                acquisition.origin,
                acquisition.lidar_project,
            )
            group = acquisition_groups.setdefault(
                key, {**acquisition.as_summary_mapping(), "tile_count": 0}
            )
            group["tile_count"] += 1
        creation = vintage.file_metadata.creation_year
        creation_key = "unknown" if creation is None else str(creation)
        file_years[creation_key] = file_years.get(creation_key, 0) + 1
        hint_key = "unknown" if vintage.hint.year is None else str(vintage.hint.year)
        hints[hint_key] = hints.get(hint_key, 0) + 1
    return {
        "distinct_acquisition_vintages": [
            acquisition_groups[key] for key in sorted(acquisition_groups, key=repr)
        ],
        "distinct_acquisition_vintage_count": len(acquisition_groups),
        "unknown_acquisition_tiles": unknown,
        "distinct_file_creation_years": dict(sorted(file_years.items())),
        "inferred_year_hints": dict(sorted(hints.items())),
    }
