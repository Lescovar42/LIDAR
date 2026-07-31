# Tillamook (R1) LiDAR acquisition provenance

This file is the cited `evidence` value for
`regions.json -> R1.lidar_acquisition`. It exists so the acquisition vintage
used for SLIDO temporal filtering, NAIP nearest-year selection, and the QC
viewer's vintage gap is traceable to a written decision rather than to a LAS
header date or a project-name token.

## Record

| field | value |
| --- | --- |
| region | `R1` / `tillamook` / Tillamook Coast Range |
| LiDAR project | `OR_WesternWildfires_A22` |
| acquisition start year | 2020 |
| acquisition end year | 2020 |
| nominal acquisition year | 2020 |
| source | User-supplied project decision for the Tillamook subset of the USGS 3DEP `OR_WesternWildfires_A22` collection |
| verified | `true` (attested by the project owner for the current diagnostic rebuild) |

## What this record does and does not assert

It asserts that the project owner has determined the Tillamook tiles selected
from `USGS_LPC_OR_WesternWildfires_A22` were flown in 2020, and that 2020 is the
year to use wherever a single authoritative acquisition year is required.

It does not assert any of the following, and none of them were used to derive
the year:

* the LAS/LAZ header `creation_date` (observed as 2024 on these files, which
  reflects file writing/repackaging, not the flight);
* the `A22` project token, which parses to a non-authoritative 2022 hint;
* a four-digit year in a tile filename;
* the NAIP year;
* the current year.

## Outstanding evidence

A machine-readable USGS citation is still missing. To upgrade this record from
an owner attestation to an externally traceable one, attach one of:

* the USGS 3DEP project metadata record (work-unit / project report) for
  `OR_WesternWildfires_A22` stating the Tillamook-area flight dates;
* the LAS `OGC WKT` / `VLR` acquisition-date entries, if the delivery includes
  them;
* the delivery/collection report supplied with the tiles.

Once such a record is available, update `source` and `evidence` in
`regions.json` (or re-run `region_registry.set_region_acquisition`) to point at
it. The code does not treat this file as authoritative in itself; it treats the
registry entry as authoritative and this file as its audit trail.
