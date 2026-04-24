from __future__ import annotations

import unittest

from paper_rag.dataprocess.ingest import lookup_metadata


class MissingClient:
    def __init__(self) -> None:
        self.called = False
        self.calls: list[dict] = []

    def lookup_exact_title(self, *args, **kwargs):
        self.called = True
        self.calls.append(kwargs)
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
        self.preprint_year = year
        self.venue = venue


class MatchClient:
    def __init__(self, match: Match) -> None:
        self.match = match
        self.called = False
        self.calls: list[dict] = []

    def lookup_exact_title(self, *args, **kwargs):
        self.called = True
        self.calls.append(kwargs)
        return self.match


class IngestMetadataLookupTests(unittest.TestCase):
    def test_uses_arxiv_before_formal_sources(self) -> None:
        arxiv = MatchClient(Match("Some Paper", ["A. Author"], 2024, "ArXiv"))
        messages: list[str] = []

        match = lookup_metadata(
            "Some Paper",
            MatchClient(Match("Some Paper", ["Formal Author"], 2025, "ICML")),
            MissingClient(),
            arxiv,
            report=messages.append,
        )

        self.assertIsNotNone(match)
        self.assertEqual(match.source, "ArXiv+DBLP")
        self.assertEqual(match.year, {"preprint_year": 2024, "publish_year": 2025})
        self.assertEqual(match.venue, "ICML")
        self.assertEqual(match.authors, ["Formal Author"])
        self.assertTrue(arxiv.called)
        self.assertTrue(any("ArXiv matched preprint year" in message for message in messages))

    def test_skips_corr_and_uses_semantic_scholar_formal_metadata(self) -> None:
        semantic = MatchClient(Match("Long Short-Term Memory", ["Sepp Hochreiter"], 1997, "Neural Computation"))
        arxiv = MissingClient()

        match = lookup_metadata("Long Short-Term Memory", MatchClient(Match("Long Short-Term Memory", ["A"], 1997, "CoRR")), semantic, arxiv)

        self.assertIsNotNone(match)
        self.assertEqual(match.source, "Semantic Scholar")
        self.assertTrue(arxiv.called)
        self.assertEqual(match.year, {"preprint_year": None, "publish_year": 1997})

    def test_arxiv_only_match_has_no_venue(self) -> None:
        match = lookup_metadata("Some Paper", MissingClient(), MissingClient(), MatchClient(Match("Some Paper", ["A"], 2024, "ArXiv")))
        self.assertIsNotNone(match)
        self.assertEqual(match.source, "ArXiv")
        self.assertEqual(match.year, {"preprint_year": 2024, "publish_year": None})
        self.assertIsNone(match.venue)

    def test_uses_source_delay_as_retry_delay(self) -> None:
        dblp = MissingClient()
        semantic = MissingClient()
        arxiv = MissingClient()

        lookup_metadata(
            "Some Paper",
            dblp,
            semantic,
            arxiv,
            dblp_retry_delay_seconds=1.5,
            semantic_scholar_retry_delay_seconds=5.5,
            arxiv_retry_delay_seconds=3.5,
        )

        self.assertEqual(arxiv.calls[0]["retry_delay_seconds"], 3.5)
        self.assertEqual(dblp.calls[0]["retry_delay_seconds"], 1.5)
        self.assertEqual(semantic.calls[0]["retry_delay_seconds"], 5.5)


if __name__ == "__main__":
    unittest.main()
