from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper_rag.config import Settings
from paper_rag.retrieval.data.aliases import find_alias_matches, load_paper_annotation_aliases


class AliasTests(unittest.TestCase):
    def test_alias_match_expands_to_canonical_and_aliases(self) -> None:
        matches = find_alias_matches([
            {
                "canonical": "Deep Residual Learning for Image Recognition",
                "aliases": ["ResNet"],
            },
        ], "which papers cite resnet")
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].alias, "ResNet")
        self.assertEqual(matches[0].canonical, "Deep Residual Learning for Image Recognition")

    def test_alias_match_requires_all_alias_tokens(self) -> None:
        matches = find_alias_matches([
            {
                "canonical": "Supervised Contrastive Learning",
                "aliases": ["SupCon loss"],
            },
        ], "supcon")
        self.assertEqual(matches, [])

    def test_loads_aliases_from_paper_annotations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            (data / "paper_annotations.json").write_text(
                json.dumps({
                    "hash1": {
                        "title": "Deep Residual Learning for Image Recognition",
                        "aliases": ["ResNet"],
                        "tags": ["CNN"],
                    },
                    "hash2": {
                        "title": "Paper Without Aliases",
                        "aliases": [],
                        "tags": [],
                    },
                }),
                encoding="utf-8",
            )

            entries = load_paper_annotation_aliases(Settings.load(root))

        self.assertEqual(entries, [{
            "canonical": "Deep Residual Learning for Image Recognition",
            "aliases": ["ResNet"],
        }])


if __name__ == "__main__":
    unittest.main()
