import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "oregon"))
from download_tiles import download_tiles


class Response:
    headers = {}
    def __init__(self, chunks=None, error=None):
        self.chunks = chunks or []
        self.error = error
    def raise_for_status(self):
        return None
    def iter_content(self, chunk_size=8192):
        for chunk in self.chunks:
            yield chunk
        if self.error:
            raise self.error


class Session:
    def __init__(self, response):
        self.response = response
        self.calls = []
    def get(self, url, **kwargs):
        self.calls.append(url)
        return self.response


class DownloadTests(unittest.TestCase):
    def write_subset(self, root, tiles):
        path = root / "tiles.json"
        path.write_text(json.dumps(tiles), encoding="utf-8")
        return path

    def test_budget_stop_avoids_transport_call(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            subset = self.write_subset(root, [{"title":"large", "downloadURL":"https://x/large.laz", "sizeInBytes":2000}])
            session = Session(Response([b"x"]))
            summary = download_tiles(subset, root / "out", max_total_gb=0.000001, session=session)
            self.assertEqual([], session.calls)
            self.assertEqual(1, summary["budget_skipped"])

    def test_size_match_resume_is_logged_and_skipped(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            out = root / "out"
            out.mkdir()
            (out / "same.laz").write_bytes(b"abc")
            subset = self.write_subset(root, [{"title":"same", "downloadURL":"https://x/same.laz", "sizeInBytes":3}])
            session = Session(Response())
            summary = download_tiles(subset, out, session=session)
            self.assertEqual(1, summary["skipped"])
            self.assertEqual([], session.calls)
            with (out / "download_log.csv").open(newline="", encoding="utf-8") as handle:
                self.assertEqual("skipped_existing", next(csv.DictReader(handle))["status"])

    def test_partial_file_is_removed_after_transport_failure(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            out = root / "out"
            subset = self.write_subset(root, [{"title":"bad", "downloadURL":"https://x/bad.laz", "sizeInBytes":10}])
            summary = download_tiles(subset, out, session=Session(Response([b"abc"], RuntimeError("broken"))))
            self.assertEqual(1, summary["failed"])
            self.assertFalse((out / "bad.laz").exists())
            self.assertFalse((out / "bad.laz.part").exists())

    def test_mismatched_final_is_removed_when_retry_fails(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            out = root / "out"
            out.mkdir()
            (out / "bad.laz").write_bytes(b"stale")
            subset = self.write_subset(root, [{"title":"bad", "downloadURL":"https://x/bad.laz", "sizeInBytes":5000}])
            download_tiles(subset, out, session=Session(Response(error=RuntimeError("broken"))))
            self.assertFalse((out / "bad.laz").exists())

    def test_region_cannot_escape_output_root(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            subset = self.write_subset(root, [{"title":"x", "downloadURL":"https://x/x.laz", "sizeInBytes":1}])
            with self.assertRaisesRegex(ValueError, "safe directory"):
                download_tiles(subset, root / "out", region="../escape", session=Session(Response([b"x"])))

    def test_region_uses_region_subdirectory(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            subset = self.write_subset(root, [{"title":"x", "downloadURL":"https://x/x.laz", "sizeInBytes":1}])
            download_tiles(subset, root / "out", region="r1", session=Session(Response([b"x"])))
            self.assertTrue((root / "out" / "r1" / "x.laz").exists())


if __name__ == "__main__":
    unittest.main()
