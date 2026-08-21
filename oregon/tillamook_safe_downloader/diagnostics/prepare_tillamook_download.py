from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import urlparse, unquote

REQUIRED = {"tile_name", "download_url", "size_bytes"}
SKIP_DIR_NAMES = {".git", ".venv", "__pycache__", "node_modules"}


def url_filename(url: str) -> str:
    return Path(unquote(urlparse(url).path)).name


def index_existing(repo_root: Path, wanted_names: set[str], exclude: Path | None = None) -> dict[str, list[Path]]:
    found: dict[str, list[Path]] = {n: [] for n in wanted_names}
    exclude_resolved = exclude.resolve() if exclude else None
    for path in repo_root.rglob("*.laz"):
        try:
            if any(part in SKIP_DIR_NAMES for part in path.parts):
                continue
            if exclude_resolved is not None:
                try:
                    path.resolve().relative_to(exclude_resolved)
                    continue
                except ValueError:
                    pass
            if path.name in found:
                found[path.name].append(path)
        except OSError:
            continue
    return found


def load_selection(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError("selection CSV is empty")
    missing = REQUIRED - set(rows[0])
    if missing:
        raise ValueError(f"selection CSV missing required columns: {sorted(missing)}")
    return rows


def prepare(selection_csv: Path, repo_root: Path, outdir: Path, workdir: Path) -> dict:
    rows = load_selection(selection_csv)
    outdir.mkdir(parents=True, exist_ok=True)
    workdir.mkdir(parents=True, exist_ok=True)

    names = {str(r["tile_name"]).strip() or url_filename(r["download_url"]) for r in rows}
    existing = index_existing(repo_root, names, exclude=outdir)

    status_rows: list[dict[str, object]] = []
    missing_urls: list[str] = []
    missing_bytes = 0
    selected_bytes = 0

    for r in rows:
        url = str(r["download_url"]).strip()
        name = str(r["tile_name"]).strip() or url_filename(url)
        expected = int(float(r["size_bytes"]))
        selected_bytes += expected
        if not url:
            status = "NO_URL"
            location = ""
        else:
            complete_paths = []
            wrong_paths = []
            for p in existing.get(name, []):
                try:
                    size = p.stat().st_size
                except OSError:
                    continue
                (complete_paths if size == expected else wrong_paths).append(p)

            target = outdir / name
            target_size = target.stat().st_size if target.exists() else None
            if target_size == expected:
                status = "COMPLETE_TARGET"
                location = str(target)
            elif complete_paths:
                status = "COMPLETE_ELSEWHERE"
                location = str(complete_paths[0])
            else:
                status = "MISSING"
                location = str(target)
                missing_urls.append(url)
                missing_bytes += expected
                if target.exists() and target_size not in (None, expected):
                    status = "PARTIAL_TARGET"
                elif wrong_paths:
                    status = "WRONG_SIZE_ELSEWHERE"

        status_rows.append({
            "tile_name": name,
            "expected_bytes": expected,
            "status": status,
            "location": location,
            "download_url": url,
        })

    input_path = workdir / "aria2_input.txt"
    input_path.write_text("\n".join(missing_urls) + ("\n" if missing_urls else ""), encoding="utf-8")

    status_path = workdir / "preflight_status.csv"
    with status_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(status_rows[0]))
        w.writeheader(); w.writerows(status_rows)

    manifest_path = workdir / "expected_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fields = ["tile_name", "expected_bytes", "download_url"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for sr in status_rows:
            w.writerow({k: sr[k] for k in fields})

    summary = {
        "selected_tiles": len(rows),
        "already_complete": sum(s["status"] in {"COMPLETE_TARGET", "COMPLETE_ELSEWHERE"} for s in status_rows),
        "to_download": len(missing_urls),
        "selected_bytes": selected_bytes,
        "missing_bytes": missing_bytes,
        "input_file": str(input_path),
        "outdir": str(outdir),
    }
    (workdir / "preflight_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection-csv", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--workdir", type=Path, required=True)
    args = ap.parse_args()
    prepare(args.selection_csv.resolve(), args.repo_root.resolve(), args.outdir.resolve(), args.workdir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
