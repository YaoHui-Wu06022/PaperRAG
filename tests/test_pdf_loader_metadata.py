from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ingestion.pdf_loader import (
    _extract_paper_metadata,
    _mineru_job_dir,
    load_pdf_pages,
)


class PdfLoaderMetadataTest(unittest.TestCase):
    def test_skips_generic_article_in_press_header(self) -> None:
        first_page_text = "\n".join(
            [
                "Article in Press",
                "Rapid culture-free diagnosis of clinical pathogens via integrated microfluidic-Raman microspectroscopy",
                "Yuchen Li, Ming Zhang, Lei Wang",
                "Available online 7 January 2025",
                "Abstract",
                "We present a rapid diagnosis pipeline.",
            ]
        )

        meta = _extract_paper_metadata(
            Path("Li - 2025 - Rapid culture-free diagnosis of clinical pathogens.pdf"),
            first_page_text,
        )

        self.assertEqual(
            meta["paper_title"],
            "Rapid culture-free diagnosis of clinical pathogens via integrated microfluidic-Raman microspectroscopy",
        )
        self.assertEqual(meta["paper_year"], "2025")
        self.assertIn("Yuchen Li", meta["paper_authors"])

    def test_skips_pdf_download_banner_and_uses_true_year(self) -> None:
        first_page_text = "\n".join(
            [
                "PDF Download 3123266.3123359.pdf 02 March 2026 Total Citations: 497 Total Downloads: 3493",
                "XIANG XIANG, Johns Hopkins University, Baltimore, MD, United States",
                "JIAN CHENG, University of Electronic Science and Technology of China, Chengdu, Sichuan, China",
                "ALAN LODDON YUILLE, Johns Hopkins University, Baltimore, MD, United States",
                "FENG WANG, University of Electronic Science and Technology of China, Chengdu, Sichuan, China",
                "NormFace: L2 Hypersphere Embedding for Face Verification",
                "Proceedings of the ACM International Conference on Multimedia 2017",
                "Abstract",
                "We study hypersphere embedding for face verification.",
            ]
        )

        meta = _extract_paper_metadata(
            Path("Wang - 2017 - NormFace L2 Hypersphere Embedding for Face Verification.pdf"),
            first_page_text,
        )

        self.assertEqual(
            meta["paper_title"],
            "NormFace: L2 Hypersphere Embedding for Face Verification",
        )
        self.assertEqual(meta["paper_year"], "2017")
        self.assertEqual(meta["paper_venue"], "ACM")
        self.assertEqual(
            meta["paper_authors"],
            "XIANG XIANG ; JIAN CHENG ; ALAN LODDON YUILLE ; FENG WANG",
        )

    def test_uses_author_line_not_nature_header(self) -> None:
        first_page_text = "\n".join(
            [
                "Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning",
                "NATURE COMMUNICATIONS | (2019) 10:4927 | https://doi.org/10.1038/s41467-019-12898-9 | www.nature.com/naturecommunications",
                "Chi-Sing Ho 1,2,12*, Neal Jean3,4, Catherine A. Hogan5,6, Jennifer Dionne2*",
                "Abstract",
                "We generate an extensive dataset of bacterial Raman spectra.",
            ]
        )

        meta = _extract_paper_metadata(
            Path(
                "Ho - 2019 - Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning.pdf"
            ),
            first_page_text,
        )

        self.assertEqual(
            meta["paper_title"],
            "Rapid identification of pathogenic bacteria using Raman spectroscopy and deep learning",
        )
        self.assertEqual(meta["paper_year"], "2019")
        self.assertEqual(meta["paper_venue"], "NATURE")
        self.assertTrue(meta["paper_authors"].startswith("Chi-Sing Ho"))
        self.assertNotIn("NATURE COMMUNICATIONS", meta["paper_authors"])
        self.assertNotIn("\\", meta["paper_authors"])

    def test_prefers_long_author_list_over_plain_venue_line(self) -> None:
        first_page_text = "\n".join(
            [
                "Article in Press",
                "Rapid culture-free diagnosis of clinical pathogens via integrated microfluidic-Raman microspectroscopy",
                (
                    "Yuetao Li, Jiabao Xu, Xiaofei Yi, Xiaobo Li, Yanjun Luo, Andrew Glidle, "
                    "Phil Summersgill, Simon Allen, Tim Ryan, Xiaochen Liu, Wei Yu, Xiaobing Chu, "
                    "Shiyu Chen, Qian Zhang, Xiaogang Xu, Xiaoting Hua, Qiwen Yang, Julien Reboud, "
                    "Yunsong Yu, Wei E. Huang, Jonathan M. Cooper & Huabing Yin"
                ),
                "Nature Communications",
                "Received: 14 December 2024 Accepted: 19 November 2025",
                "Abstract",
                "We present a rapid diagnosis pipeline.",
            ]
        )

        meta = _extract_paper_metadata(
            Path(
                "Li - 2025 - Rapid culture-free diagnosis of clinical pathogens via integrated microfluidic-Raman microspectroscopy.pdf"
            ),
            first_page_text,
        )

        self.assertEqual(
            meta["paper_title"],
            "Rapid culture-free diagnosis of clinical pathogens via integrated microfluidic-Raman microspectroscopy",
        )
        self.assertEqual(meta["paper_year"], "2025")
        self.assertIn("Yuetao Li", meta["paper_authors"])
        self.assertIn("Huabing Yin", meta["paper_authors"])
        self.assertNotEqual(meta["paper_authors"], "Nature Communications")

    def test_normalizes_ligature_artifacts_in_title(self) -> None:
        first_page_text = "\n".join(
            [
                "Article in Press",
                "Rapid culture-free diagnosis of clinical pathogens via integrated micro\u0080uidic-Raman microspectroscopy",
                "Yuetao Li, Jiabao Xu, Huabing Yin",
                "Abstract",
                "We present a rapid diagnosis pipeline.",
            ]
        )

        meta = _extract_paper_metadata(
            Path(
                "Li - 2025 - Rapid culture-free diagnosis of clinical pathogens via integrated microfluidic-Raman microspectroscopy.pdf"
            ),
            first_page_text,
        )

        self.assertIn("microfluidic", meta["paper_title"])

    def test_skips_handling_editor_line_for_title_and_year(self) -> None:
        first_page_text = "\n".join(
            [
                "Intelligent identification of foodborne pathogenic bacteria by self-transfer deep learning and ensemble prediction based on single-cell Raman spectrum",
                "Daixi Li a,b,* , Yuqi Zhu a , Aamir Mehmood c , Yangtai Liu a , Xiaojie Qin a , Qingli Dong a",
                "A R T I C L E I N F O",
                "Handling Editor: J. Wang",
                "A B S T R A C T",
                "Foodborne pathogenic infections pose a significant threat to human health.",
                "Talanta 285 (2025) 127268",
            ]
        )

        meta = _extract_paper_metadata(
            Path(
                "Intelligent identification of foodborne pathogenic bacteria by self-transferdeep learning and ensemble prediction based on single-cellRaman spectrum(1).pdf"
            ),
            first_page_text,
        )

        self.assertEqual(
            meta["paper_title"],
            "Intelligent identification of foodborne pathogenic bacteria by self-transfer deep learning and ensemble prediction based on single-cell Raman spectrum",
        )
        self.assertEqual(meta["paper_year"], "2025")

    def test_load_pdf_pages_uses_only_first_page_for_metadata(self) -> None:
        payload = [
            {
                "page_idx": 0,
                "type": "text",
                "text": "Deep Residual Learning for Image Recognition",
            },
            {
                "page_idx": 0,
                "type": "text",
                "text": "Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun",
            },
            {
                "page_idx": 0,
                "type": "text",
                "text": "Abstract Deeper neural networks are more difficult to train.",
            },
            {
                "page_idx": 1,
                "type": "text",
                "text": "Proceedings of the 54th Annual Meeting of the ACL 2016.",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            pdf_path = base / "He - 2015 - Deep Residual Learning for Image Recognition.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")

            output_root = base / "mineru_output"
            job_dir = _mineru_job_dir(pdf_path, output_root)
            job_dir.mkdir(parents=True, exist_ok=True)
            content_list_path = job_dir / "sample_content_list.json"
            content_list_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

            result = load_pdf_pages(
                pdf_path,
                mineru_output_dir=output_root,
            )

        self.assertGreater(len(result.documents), 0)
        metadata = result.documents[0].metadata
        self.assertEqual(
            metadata.get("paper_title"),
            "Deep Residual Learning for Image Recognition",
        )
        self.assertEqual(
            metadata.get("paper_authors"),
            "Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun",
        )
        self.assertEqual(metadata.get("paper_venue"), "")


if __name__ == "__main__":
    unittest.main()
