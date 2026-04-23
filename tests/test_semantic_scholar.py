from __future__ import annotations

import unittest

from paper_rag.dataprocess.metadata.semantic_scholar import select_exact_match


def paper(
    title: str,
    year: int | None = 1997,
    venue: str | None = "Neural Computation",
    publication_venue: str | None = None,
    authors: list[str] | None = None,
) -> dict:
    data: dict = {
        "title": title,
        "year": year,
        "venue": venue,
        "authors": [{"name": author} for author in (authors or ["Sepp Hochreiter", "J. Schmidhuber"])],
    }
    if publication_venue:
        data["publicationVenue"] = {"name": publication_venue}
    return data


def result(*papers: dict) -> dict:
    return {"data": list(papers)}


class SemanticScholarSelectionTests(unittest.TestCase):
    def test_accepts_lstm_exact_title(self) -> None:
        match = select_exact_match(
            "LONG SHORT-TERM MEMORY",
            result(paper("Long Short-Term Memory")),
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.year, 1997)
        self.assertEqual(match.venue, "Neural Computation")
        self.assertEqual(match.authors, ["Sepp Hochreiter", "J. Schmidhuber"])

    def test_rejects_near_title(self) -> None:
        match = select_exact_match(
            "LONG SHORT-TERM MEMORY",
            result(paper("xLSTM: Extended Long Short-Term Memory", year=2024, venue="NeurIPS")),
        )
        self.assertIsNone(match)

    def test_uses_publication_venue_name_when_venue_missing(self) -> None:
        match = select_exact_match(
            "LONG SHORT-TERM MEMORY",
            result(paper("Long Short-Term Memory", venue="", publication_venue="Neural Computation")),
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.venue, "Neural Computation")

    def test_rejects_missing_required_metadata(self) -> None:
        match = select_exact_match(
            "LONG SHORT-TERM MEMORY",
            result(paper("Long Short-Term Memory", year=None, venue="Neural Computation")),
        )
        self.assertIsNone(match)


if __name__ == "__main__":
    unittest.main()
