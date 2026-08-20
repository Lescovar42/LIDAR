from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("patcher", ROOT / "apply_pos_weight_override.py")
assert spec and spec.loader
patcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(patcher)


class PatcherTests(unittest.TestCase):
    def test_all_expected_current_snippets_patch_once(self) -> None:
        source = "\n\n".join(old for old, _ in patcher.REPLACEMENTS)
        patched = patcher.patch_text(source)
        self.assertIn("def resolve_pos_weight(", patched)
        self.assertIn('"--pos-weight"', patched)
        self.assertIn('"run_config.json"', patched)
        self.assertIn('"pos_weight_used": pos_weight', patched)

    def test_refuses_double_patch(self) -> None:
        source = "\n\n".join(old for old, _ in patcher.REPLACEMENTS)
        patched = patcher.patch_text(source)
        with self.assertRaises(RuntimeError):
            patcher.patch_text(patched)


if __name__ == "__main__":
    unittest.main()
