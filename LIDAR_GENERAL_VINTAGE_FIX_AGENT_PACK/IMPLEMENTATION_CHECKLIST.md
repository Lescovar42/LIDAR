# Implementation Checklist

## Before coding

- [ ] Record current `git rev-parse HEAD`.
- [ ] Record `git status --short`.
- [ ] Inspect recent user commits.
- [ ] Read all relevant existing tests.
- [ ] Identify every reader/writer of `lidar_year` and `lidar_year_source`.
- [ ] Identify every call path into SLIDO temporal filtering.
- [ ] Identify cached NAIP provenance behavior.

## Data model

- [ ] Separate acquisition metadata from file creation metadata.
- [ ] Preserve non-authoritative project/filename hints separately.
- [ ] Define conflict behavior.
- [ ] Support single-year and multi-year surveys.
- [ ] Define compatibility behavior for legacy `lidar_year`.

## Registry and CLI

- [ ] Add validated registry acquisition metadata.
- [ ] Add explicit legacy/single-region CLI override.
- [ ] Require source/evidence where appropriate.
- [ ] Never infer acquisition from `creation_date`.
- [ ] Never infer authoritative acquisition from `A22`.

## Dataset output

- [ ] Patch rows contain acquisition provenance.
- [ ] Patch rows contain file creation provenance.
- [ ] Tile summaries contain both.
- [ ] Dataset summary exposes unknown/conflict counts.
- [ ] SLIDO filtering receives only authoritative acquisition year.

## NAIP

- [ ] Nearest-year selection uses authoritative acquisition.
- [ ] Unknown acquisition produces unavailable gap.
- [ ] Cached imagery can be reused.
- [ ] Cached imagery does not preserve stale LiDAR vintage in a new manifest.
- [ ] NAIP manifest includes acquisition source and file creation metadata.

## Viewer

- [ ] Display says “LiDAR acquisition,” not ambiguous “LiDAR.”
- [ ] File creation shown separately.
- [ ] Unknown acquisition does not produce “within 2 years.”
- [ ] Existing mask positive/ignore fix preserved.

## Tests

- [ ] Header 2024 / acquisition 2020.
- [ ] A22 without explicit provenance remains unknown.
- [ ] CLI override.
- [ ] Registry metadata.
- [ ] Multi-year range.
- [ ] Conflict detection.
- [ ] Manifest propagation.
- [ ] SLIDO year argument.
- [ ] NAIP cache refresh.
- [ ] Viewer known/unknown display.
- [ ] Full test suite.
- [ ] Compile checks.

## Final report

- [ ] Files changed.
- [ ] Tests and results.
- [ ] Exact migration/rebuild commands.
- [ ] Any unresolved evidence.
- [ ] No unrelated design changes.
