from __future__ import annotations

import unittest
from urllib.parse import parse_qs

from paper_rag.ingest.metadata_sources.dblp import build_query, select_exact_match


def hit(title: str, year: int, venue: str, authors: list[str] | None = None) -> dict:
    return {
        "info": {
            "title": title,
            "year": str(year),
            "venue": venue,
            "authors": {
                "author": [{"text": author} for author in (authors or ["A. Author"])]
            },
        }
    }


def result(*hits: dict) -> dict:
    return {"result": {"hits": {"hit": list(hits)}}}


class DblpSelectionTests(unittest.TestCase):
    def test_prefers_earliest_non_corr_for_exact_title(self) -> None:
        match = select_exact_match(
            "ImageNet Classification with Deep Convolutional Neural Networks",
            result(
                hit("ImageNet classification with deep convolutional neural networks.", 2017, "Commun. ACM"),
                hit("ImageNet Classification with Deep Convolutional Neural Networks.", 2012, "NIPS"),
            ),
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.year, 2012)
        self.assertEqual(match.venue, "NIPS")

    def test_prefers_non_corr_over_corr(self) -> None:
        match = select_exact_match(
            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
            result(
                hit("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.", 2018, "CoRR"),
                hit("BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.", 2019, "NAACL-HLT"),
            ),
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.year, 2019)
        self.assertEqual(match.venue, "NAACL-HLT")

    def test_accepts_corr_when_only_exact_candidate(self) -> None:
        match = select_exact_match("Some Paper", result(hit("Some Paper.", 2024, "CoRR")))
        self.assertIsNotNone(match)
        self.assertEqual(match.venue, "CoRR")

    def test_rejects_non_exact_title(self) -> None:
        match = select_exact_match("Attention Is All You Need", result(hit("Attention Is All You Need But More.", 2024, "CoRR")))
        self.assertIsNone(match)

    def test_build_query_uses_candidate_limit(self) -> None:
        params = parse_qs(build_query("Long Short-Term Memory", 20))
        self.assertEqual(params["h"], ["20"])
        self.assertEqual(params["format"], ["json"])


if __name__ == "__main__":
    unittest.main()
