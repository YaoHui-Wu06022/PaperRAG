from __future__ import annotations

from pathlib import Path
import unittest

from langchain_core.documents import Document

from services.paper_representation import build_paper_assets


class PaperRepresentationTest(unittest.TestCase):
    def test_builds_paper_docs_catalog_and_citation_rows(self) -> None:
        parent_docs = [
            Document(
                page_content="This paper studies residual learning for deep networks and veri\u001bcation.",
                metadata={
                    "doc_id": "paper-1",
                    "source": "resnet.pdf",
                    "section_path": "Abstract",
                    "page": 1,
                    "page_end": 1,
                    "paper_title": "Deep Residual Learning",
                    "paper_year": "2016",
                    "paper_authors": "Kaiming He, Xiangyu Zhang",
                    "paper_venue": "CVPR",
                    "paper_keywords": "residual learning, cnn",
                    "paper_language": "en",
                },
            ),
            Document(
                page_content="We introduce residual blocks and evaluate on ImageNet.",
                metadata={
                    "doc_id": "paper-1",
                    "source": "resnet.pdf",
                    "section_path": "Method",
                    "page": 2,
                    "page_end": 2,
                    "paper_title": "Deep Residual Learning",
                    "paper_year": "2016",
                    "paper_authors": "Kaiming He, Xiangyu Zhang",
                    "paper_venue": "CVPR",
                    "paper_keywords": "residual learning, cnn",
                    "paper_language": "en",
                },
            ),
        ]
        block_rows = [
            {
                "doc_id": "paper-1",
                "source": "resnet.pdf",
                "section_path": "Abstract",
                "page": 1,
                "text": "This paper studies residual learning for deep networks.",
                "paper_title": "Deep Residual Learning",
                "paper_year": "2016",
                "paper_authors": "Kaiming He, Xiangyu Zhang",
                "paper_venue": "CVPR",
                "paper_keywords": "residual learning, cnn",
                "paper_language": "en",
            },
            {
                "doc_id": "paper-1",
                "source": "resnet.pdf",
                "section_path": "Method",
                "page": 2,
                "text": "We introduce residual blocks and evaluate on ImageNet.",
                "paper_title": "Deep Residual Learning",
                "paper_year": "2016",
                "paper_authors": "Kaiming He, Xiangyu Zhang",
                "paper_venue": "CVPR",
                "paper_keywords": "residual learning, cnn",
                "paper_language": "en",
            },
        ]
        reference_docs = [
            Document(
                page_content="A. Author, B. Writer. 2014. Going deeper with convolutions.",
                metadata={"doc_id": "paper-1"},
            )
        ]

        assets = build_paper_assets(
            parent_docs,
            block_rows,
            reference_docs,
            source_root=Path("data/pdf"),
            paper_summary_max_chars=1200,
            section_summary_max_chars=300,
        )

        self.assertEqual(len(assets.paper_docs), 1)
        self.assertGreaterEqual(len(assets.section_docs), 1)
        self.assertEqual(len(assets.catalog_rows), 1)
        self.assertEqual(len(assets.citation_rows), 1)
        self.assertIn("Deep Residual Learning", assets.paper_docs[0].page_content)
        self.assertEqual(assets.catalog_rows[0]["venue"], "CVPR")
        self.assertEqual(assets.citation_rows[0]["source_doc_id"], "paper-1")

    def test_strips_front_matter_and_noise_sections(self) -> None:
        parent_docs = [
            Document(
                page_content=(
                    "Article in Press Rapid culture-free diagnosis of clinical pathogens "
                    "via integrated microfluidic-Raman microspectroscopy Yuetao Li, Jiabao Xu"
                ),
                metadata={
                    "doc_id": "paper-2",
                    "source": "rapid.pdf",
                    "section_path": "正文",
                    "page": 1,
                    "page_end": 1,
                    "paper_title": "Rapid culture-free diagnosis of clinical pathogens via integrated microfluidic-Raman microspectroscopy",
                    "paper_year": "2025",
                    "paper_authors": "Yuetao Li, Jiabao Xu",
                    "paper_venue": "NATURE",
                    "paper_keywords": "",
                    "paper_language": "en",
                },
            ),
            Document(
                page_content="ARTICLE IN PRESS Abstract: Antimicrobial resistance is a critical global health challenge.",
                metadata={
                    "doc_id": "paper-2",
                    "source": "rapid.pdf",
                    "section_path": "ARTICLE IN PRESS",
                    "page": 2,
                    "page_end": 2,
                    "paper_title": "Rapid culture-free diagnosis of clinical pathogens via integrated microfluidic-Raman microspectroscopy",
                    "paper_year": "2025",
                    "paper_authors": "Yuetao Li, Jiabao Xu",
                    "paper_venue": "NATURE",
                    "paper_keywords": "",
                    "paper_language": "en",
                },
            ),
            Document(
                page_content="1 James Watt School of Engineering, University of Glasgow",
                metadata={
                    "doc_id": "paper-2",
                    "source": "rapid.pdf",
                    "section_path": "1 James Watt School of Engineering, University of Glasgow",
                    "page": 2,
                    "page_end": 2,
                    "paper_title": "Rapid culture-free diagnosis of clinical pathogens via integrated microfluidic-Raman microspectroscopy",
                    "paper_year": "2025",
                    "paper_authors": "Yuetao Li, Jiabao Xu",
                    "paper_venue": "NATURE",
                    "paper_keywords": "",
                    "paper_language": "en",
                },
            ),
        ]
        block_rows = [
            {
                "doc_id": "paper-2",
                "source": "rapid.pdf",
                "section_path": "ARTICLE IN PRESS",
                "page": 2,
                "text": "ARTICLE IN PRESS Abstract: Antimicrobial resistance is a critical global health challenge.",
                "paper_title": "Rapid culture-free diagnosis of clinical pathogens via integrated microfluidic-Raman microspectroscopy",
                "paper_year": "2025",
                "paper_authors": "Yuetao Li, Jiabao Xu",
                "paper_venue": "NATURE",
                "paper_keywords": "",
                "paper_language": "en",
            }
        ]

        assets = build_paper_assets(
            parent_docs,
            block_rows,
            [],
            source_root=Path("data/pdf"),
            paper_summary_max_chars=1200,
            section_summary_max_chars=300,
        )

        self.assertEqual(len(assets.paper_docs), 1)
        paper_meta = assets.paper_docs[0].metadata
        self.assertTrue(str(paper_meta["paper_abstract"]).startswith("Antimicrobial resistance"))
        self.assertNotIn("Article in Press", str(paper_meta["paper_summary"]))
        self.assertNotIn(
            "1 James Watt School of Engineering",
            paper_meta["section_names"],
        )

    def test_extracts_abstract_from_heading_followed_by_body(self) -> None:
        parent_docs = [
            Document(
                page_content="Thanks to the recent developments of Convolutional Neural Networks, the performance of face verification methods has increased rapidly.",
                metadata={
                    "doc_id": "paper-3",
                    "source": "normface.pdf",
                    "section_path": "ABSTRACT",
                    "page": 2,
                    "page_end": 2,
                    "paper_title": "NormFace: L2 Hypersphere Embedding for Face Verification",
                    "paper_year": "2017",
                    "paper_authors": "XIANG XIANG ; JIAN CHENG ; ALAN LODDON YUILLE ; FENG WANG",
                    "paper_venue": "ACM",
                    "paper_keywords": "",
                    "paper_language": "en",
                },
            ),
            Document(
                page_content="In recent years, Convolutional neural networks achieve state-of-the-art performance.",
                metadata={
                    "doc_id": "paper-3",
                    "source": "normface.pdf",
                    "section_path": "1 INTRODUCTION",
                    "page": 2,
                    "page_end": 2,
                    "paper_title": "NormFace: L2 Hypersphere Embedding for Face Verification",
                    "paper_year": "2017",
                    "paper_authors": "XIANG XIANG ; JIAN CHENG ; ALAN LODDON YUILLE ; FENG WANG",
                    "paper_venue": "ACM",
                    "paper_keywords": "",
                    "paper_language": "en",
                },
            ),
        ]
        block_rows = [
            {
                "doc_id": "paper-3",
                "source": "normface.pdf",
                "section_path": "ABSTRACT",
                "page": 2,
                "text": "ABSTRACT",
                "paper_title": "NormFace: L2 Hypersphere Embedding for Face Verification",
                "paper_year": "2017",
                "paper_authors": "XIANG XIANG ; JIAN CHENG ; ALAN LODDON YUILLE ; FENG WANG",
                "paper_venue": "ACM",
                "paper_keywords": "",
                "paper_language": "en",
            },
            {
                "doc_id": "paper-3",
                "source": "normface.pdf",
                "section_path": "ABSTRACT",
                "page": 2,
                "text": (
                    "Thanks to the recent developments of Convolutional Neural Networks, "
                    "the performance of face veri\u001bcation methods has increased rapidly and "
                    "the e\u001dect of normalization is analyzed."
                ),
                "paper_title": "NormFace: L2 Hypersphere Embedding for Face Verification",
                "paper_year": "2017",
                "paper_authors": "XIANG XIANG ; JIAN CHENG ; ALAN LODDON YUILLE ; FENG WANG",
                "paper_venue": "ACM",
                "paper_keywords": "",
                "paper_language": "en",
            },
            {
                "doc_id": "paper-3",
                "source": "normface.pdf",
                "section_path": "1 INTRODUCTION",
                "page": 2,
                "text": "1 INTRODUCTION",
                "paper_title": "NormFace: L2 Hypersphere Embedding for Face Verification",
                "paper_year": "2017",
                "paper_authors": "XIANG XIANG ; JIAN CHENG ; ALAN LODDON YUILLE ; FENG WANG",
                "paper_venue": "ACM",
                "paper_keywords": "",
                "paper_language": "en",
            },
        ]

        assets = build_paper_assets(
            parent_docs,
            block_rows,
            [],
            source_root=Path("data/pdf"),
            paper_summary_max_chars=1200,
            section_summary_max_chars=300,
        )

        self.assertEqual(len(assets.paper_docs), 1)
        self.assertTrue(
            str(assets.paper_docs[0].metadata["paper_abstract"]).startswith(
                "Thanks to the recent developments"
            )
        )
        self.assertIn("verification", str(assets.paper_docs[0].metadata["paper_abstract"]))
        self.assertIn("effect", str(assets.paper_docs[0].metadata["paper_abstract"]))


if __name__ == "__main__":
    unittest.main()
