from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = ["tile_name", "expected_bytes", "actual_bytes", "path", "status", "download_url"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def verify(manifest_csv: Path, repo_root: Path, outdir: Path, workdir: Path) -> dict:
    manifest = load_manifest(manifest_csv)
    wanted = {r["tile_name"]: r for r in manifest}

    # Prefer target directory, then any matching complete file elsewhere in repo.
    repo_matches: dict[str, list[Path]] = {n: [] for n in wanted}
    for p in repo_root.rglob("*.laz"):
        if p.name in repo_matches:
            repo_matches[p.name].append(p)

    ok, partial, failed = [], [], []
    for name, r in wanted.items():
        expected = int(r["expected_bytes"])
        url = r["download_url"]
        target = outdir / name
        candidates = ([target] if target.exists() else []) + [p for p in repo_matches[name] if p != target]
        complete = None
        best_partial = None
        for p in candidates:
            try:
                size = p.stat().st_size
            except OSError:
                continue
            if size == expected:
                complete = (p, size); break
            if best_partial is None or size > best_partial[1]:
                best_partial = (p, size)

        if complete:
            p, size = complete
            ok.append({"tile_name": name, "expected_bytes": expected, "actual_bytes": size, "path": str(p), "status": "OK", "download_url": url})
        elif best_partial or (outdir / (name + ".aria2")).exists():
            p, size = best_partial if best_partial else (target, 0)
            partial.append({"tile_name": name, "expected_bytes": expected, "actual_bytes": size, "path": str(p), "status": "PARTIAL", "download_url": url})
        else:
            failed.append({"tile_name": name, "expected_bytes": expected, "actual_bytes": 0, "path": str(target), "status": "MISSING", "download_url": url})

    write_csv(workdir / "downloaded_ok.csv", ok)
    write_csv(workdir / "partial_downloads.csv", partial)
    write_csv(workdir / "failed_downloads.csv", failed)
    summary = {
        "selected": len(manifest),
        "ok": len(ok),
        "partial": len(partial),
        "failed_or_missing": len(failed),
        "complete": len(ok) == len(manifest),
        "verified_bytes": sum(int(x["actual_bytes"]) for x in ok),
    }
    (workdir / "download_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (workdir / "download_summary.md").write_text(
        "# Tillamook expansion download verification\n\n"
        f"- Selected: **{summary['selected']}**\n"
        f"- Complete and exact-size verified: **{summary['ok']}**\n"
        f"- Partial: **{summary['partial']}**\n"
        f"- Missing/failed: **{summary['failed_or_missing']}**\n"
        f"- Complete: **{summary['complete']}**\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest-csv", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    args = ap.parse_args()
    s = verify(args.manifest_csv.resolve(), args.repo_root.resolve(), args.outdir.resolve(), args.workdir.resolve())
    return 0 if s["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
