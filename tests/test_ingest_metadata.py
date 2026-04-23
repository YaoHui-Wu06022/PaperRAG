from __future__ import annotations

import unittest

from paper_rag.dataprocess.ingest import lookup_metadata


class MissingClient:
    def __init__(self) -> None:
        self.called = False

    def lookup_exact_title(self, *args, **kwargs):
        self.called = True
        return None


class ErrorClient:
    def __init__(self, message: str) -> None:
        self.message = message
        self.called = False

    def lookup_exact_title(self, *args, **kwargs):
        self.called = True
        raise RuntimeError(self.message)


class Match:
    def __init__(self, title: str, authors: list[str], year: int, venue: str) -> None:
        self.title = title
        self.authors = authors
        self.year = year
        self.venue = venue


class MatchClient:
    def __init__(self, match: Match) -> None:
        self.match = match
        self.called = False

    def lookup_exact_title(self, *args, **kwargs):
        self.called = True
        return self.match


class IngestMetadataLookupTests(unittest.TestCase):
    def test_falls_through_semantic_scholar_error_to_arxiv(self) -> None:
        arxiv = MatchClient(Match("Some Paper", ["A. Author"], 2024, "ArXiv"))
        messages: list[str] = []

        match = lookup_metadata(
            "Some Paper",
            MissingClient(),
            ErrorClient("HTTP 429"),
            arxiv,
            report=messages.append,
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.source, "ArXiv")
        self.assertTrue(arxiv.called)
        self.assertTrue(any("Semantic Scholar lookup failed" in message for message in messages))

    def test_uses_semantic_scholar_before_arxiv(self) -> None:
        semantic = MatchClient(Match("Long Short-Term Memory", ["Sepp Hochreiter"], 1997, "Neural Computation"))
        arxiv = MissingClient()

        match = lookup_metadata("Long Short-Term Memory", MissingClient(), semantic, arxiv)

        self.assertIsNotNone(match)
        self.assertEqual(match.source, "Semantic Scholar")
        self.assertFalse(arxiv.called)


if __name__ == "__main__":
    unittest.main()
