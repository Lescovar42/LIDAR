import csv
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "diagnostics"))
from prepare_tillamook_download import prepare
from verify_tillamook_download import verify


class DownloaderToolsTests(unittest.TestCase):
    def _selection(self, p: Path, rows):
        with p.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["tile_name", "download_url", "size_bytes"])
            w.writeheader(); w.writerows(rows)

    def test_prepare_excludes_complete_elsewhere(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); out = root / "out"; work = root / "work"; existing = root / "old"; existing.mkdir()
            (existing / "a.laz").write_bytes(b"1234")
            sel = root / "sel.csv"
            self._selection(sel, [
                {"tile_name":"a.laz","download_url":"https://x/a.laz","size_bytes":4},
                {"tile_name":"b.laz","download_url":"https://x/b.laz","size_bytes":5},
            ])
            s = prepare(sel, root, out, work)
            self.assertEqual(s["already_complete"], 1)
            self.assertEqual(s["to_download"], 1)
            self.assertEqual((work/"aria2_input.txt").read_text().strip(), "https://x/b.laz")

    def test_verify_exact_size_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); out = root / "out"; work = root / "work"; out.mkdir(); work.mkdir()
            manifest = work / "expected_manifest.csv"
            with manifest.open("w", newline="", encoding="utf-8") as f:
                w=csv.DictWriter(f, fieldnames=["tile_name","expected_bytes","download_url"]); w.writeheader()
                w.writerow({"tile_name":"a.laz","expected_bytes":4,"download_url":"https://x/a.laz"})
                w.writerow({"tile_name":"b.laz","expected_bytes":5,"download_url":"https://x/b.laz"})
            (out/"a.laz").write_bytes(b"1234")
            (out/"b.laz").write_bytes(b"123")
            (out/"b.laz.aria2").write_bytes(b"ctl")
            s=verify(manifest, root, out, work)
            self.assertEqual(s["ok"],1); self.assertEqual(s["partial"],1); self.assertFalse(s["complete"])

if __name__ == "__main__": unittest.main()
