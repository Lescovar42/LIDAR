# NAIP QC v1.1 fix

This patch fixes a crash caused by USGS ImageServer records whose `Year` field
is null or malformed.

The downloader now uses one `parse_year()` helper everywhere it:

- discovers available years;
- filters records for the selected year;
- writes available-year provenance.

Replace `oregon/fetch_naip_qc.py`, then rerun the same command. Existing
successfully cached files can remain; use `--overwrite` when rebuilding all
patches.
