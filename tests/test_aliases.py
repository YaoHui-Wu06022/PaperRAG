from __future__ import annotations

import unittest

from paper_rag.retrieval.data.aliases import find_alias_matches


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


if __name__ == "__main__":
    unittest.main()
