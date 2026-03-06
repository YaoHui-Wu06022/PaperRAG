from __future__ import annotations

from datetime import datetime
import unittest

from langchain_core.documents import Document

from retrieval.metadata_filter import apply_query_metadata_filter, parse_query_metadata_filters


class MetadataFilterTest(unittest.TestCase):
    def test_parse_year_range(self) -> None:
        filters = parse_query_metadata_filters("author: He 2020-2023")
        self.assertEqual(filters.years, ("2020", "2021", "2022", "2023"))

    def test_parse_recent_years(self) -> None:
        current_year = datetime.now().year
        expected = tuple(str(year) for year in range(current_year - 4, current_year + 1))
        filters = parse_query_metadata_filters("last 5 years transformer survey")
        self.assertEqual(filters.years, expected)

    def test_parse_mixed_field_constraints(self) -> None:
        filters = parse_query_metadata_filters("author: He venue: CVPR 2016")
        self.assertEqual(filters.author_terms, ("He",))
        self.assertEqual(filters.venue_terms, ("CVPR",))
        self.assertEqual(filters.years, ("2016",))

    def test_apply_filter_builds_doc_id_expr_for_milvus(self) -> None:
        docs = [
            Document(
                page_content="Residual learning for image recognition.",
                metadata={
                    "doc_id": "doc_he_2015",
                    "source": "resnet.pdf",
                    "title": "Deep Residual Learning for Image Recognition",
                    "paper_title": "Deep Residual Learning for Image Recognition",
                    "authors": "Kaiming He",
                    "paper_authors": "Kaiming He",
                    "year": "2015",
                    "paper_year": "2015",
                    "paper_venue": "CVPR",
                },
            ),
            Document(
                page_content="Attention is all you need.",
                metadata={
                    "doc_id": "doc_vaswani_2017",
                    "source": "transformer.pdf",
                    "title": "Attention is All You Need",
                    "paper_title": "Attention is All You Need",
                    "authors": "Ashish Vaswani",
                    "paper_authors": "Ashish Vaswani",
                    "year": "2017",
                    "paper_year": "2017",
                    "paper_venue": "NeurIPS",
                },
            ),
        ]

        result = apply_query_metadata_filter("author: He 2015", docs)

        self.assertTrue(result.applied)
        self.assertEqual(len(result.docs), 1)
        self.assertEqual(result.allowed_doc_ids, {"doc_he_2015"})
        self.assertEqual(result.milvus_expr, 'doc_id in ["doc_he_2015"]')


if __name__ == "__main__":
    unittest.main()
