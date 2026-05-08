from __future__ import annotations

import unittest

from paper_rag.ingest.metadata_sources.arxiv import select_exact_match


def feed(*entries: str) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        + "".join(entries)
        + "</feed>"
    )


def entry(title: str, published: str = "2017-06-12T17:57:34Z", authors: list[str] | None = None) -> str:
    author_xml = "".join(f"<author><name>{author}</name></author>" for author in (authors or ["A. Author"]))
    return f"<entry><title>{title}</title><published>{published}</published>{author_xml}</entry>"


class ArxivSelectionTests(unittest.TestCase):
    def test_accepts_normalized_exact_title(self) -> None:
        match = select_exact_match(
            "Attention Is All You Need",
            feed(entry("Attention Is All You Need", authors=["Ashish Vaswani", "Noam Shazeer"])),
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.preprint_year, 2017)
        self.assertEqual(match.venue, "ArXiv")
        self.assertEqual(match.authors[:2], ["Ashish Vaswani", "Noam Shazeer"])

    def test_rejects_non_exact_title(self) -> None:
        match = select_exact_match(
            "Attention Is All You Need",
            feed(entry("Attention Is All You Need for Images")),
        )
        self.assertIsNone(match)

    def test_uses_updated_year_when_published_missing(self) -> None:
        match = select_exact_match(
            "Some Paper",
            feed("<entry><title>Some Paper</title><updated>2024-01-02T00:00:00Z</updated></entry>"),
        )
        self.assertIsNotNone(match)
        self.assertEqual(match.preprint_year, 2024)


if __name__ == "__main__":
    unittest.main()
