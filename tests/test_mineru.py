from __future__ import annotations

import tempfile
import unittest
import zipfile
from pathlib import Path

from paper_rag.ingest.mineru import MinerUError, safe_extract_zip


class MinerUZipTests(unittest.TestCase):
    def test_safe_extract_rejects_path_traversal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output_dir = root / "output"
            output_dir.mkdir()
            zip_path = root / "malicious.zip"
            with zipfile.ZipFile(zip_path, "w") as archive:
                archive.writestr("../escape.txt", "bad")

            with zipfile.ZipFile(zip_path) as archive:
                with self.assertRaises(MinerUError):
                    safe_extract_zip(archive, output_dir)

            self.assertFalse((root / "escape.txt").exists())


if __name__ == "__main__":
    unittest.main()
