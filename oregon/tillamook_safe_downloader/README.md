# Tillamook 100-tile expansion — safe parallel downloader

Purpose: download only the missing files from `selection_tillamook_expansion_100/proposed_tillamook_expansion.csv` without changing the selection or touching frozen external-test regions.

## Safety properties
- Detects exact-size complete LAZ files already anywhere under the repository and does not re-download them.
- Uses the selector's authoritative `size_bytes` as the post-download byte check.
- Downloads into a dedicated expansion directory.
- aria2 `.aria2` control files are retained for resume.
- Automatic multi-pass retries target only files still incomplete.
- Produces `downloaded_ok.csv`, `partial_downloads.csv`, `failed_downloads.csv`, logs, and JSON/Markdown summaries.
- Does not modify `download_tiles.py` or selection manifests.

## Default speed profile
`4` concurrent files × `4` HTTP connections per file. This is deliberately faster than the repository's serial Requests downloader but less aggressive than opening dozens of concurrent files. aria2's official manual distinguishes `--max-concurrent-downloads` (parallel items) from `--split` / `--max-connection-per-server` (connections within an item).

If Rockyweb is stable and your network is not saturated, try `-ConcurrentFiles 6 -ConnectionsPerFile 4`. If you see repeated 429/503/timeouts, lower to `3 × 2` rather than increasing it.
