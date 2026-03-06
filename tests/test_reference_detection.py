from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ingestion.pdf_loader import _build_chunk_input_documents, _parse_mineru_content_list
from services.local_cache_store import build_reference_purity_summary, split_reference_docs


class ReferenceDetectionTest(unittest.TestCase):
    def test_reference_detection_exits_after_reference_section(self) -> None:
        payload = [
            {"page_idx": 0, "type": "text", "text": "1. Introduction"},
            {
                "page_idx": 0,
                "type": "text",
                "text": "This paper introduces a new model for image recognition.",
            },
            {"page_idx": 4, "type": "text", "text": "References"},
            {
                "page_idx": 4,
                "type": "text",
                "text": (
                    "[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual "
                    "learning for image recognition. CVPR 2016."
                ),
            },
            {"page_idx": 5, "type": "text", "text": "A. Appendix"},
            {
                "page_idx": 5,
                "type": "text",
                "text": "Additional ablations are reported here with further analysis.",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            base = Path(temp_dir)
            pdf_path = base / "sample.pdf"
            pdf_path.write_bytes(b"%PDF-1.4\n")
            content_list_path = base / "content_list.json"
            content_list_path.write_text(
                json.dumps(payload, ensure_ascii=False),
                encoding="utf-8",
            )

            blocks, total_pages = _parse_mineru_content_list(
                content_list_path,
                pdf_path=pdf_path,
                paper_meta={
                    "paper_title": "Sample Paper",
                    "paper_year": "2024",
                    "paper_authors": "A. Author",
                },
            )

            block_by_text = {str(block["text"]): block for block in blocks}
            self.assertFalse(block_by_text["1. Introduction"]["is_reference"])
            self.assertTrue(block_by_text["References"]["is_reference"])
            self.assertTrue(
                block_by_text[
                    "[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. CVPR 2016."
                ]["is_reference"]
            )
            self.assertFalse(block_by_text["A. Appendix"]["is_reference"])
            self.assertFalse(
                block_by_text[
                    "Additional ablations are reported here with further analysis."
                ]["is_reference"]
            )

            docs = _build_chunk_input_documents(
                blocks,
                total_pages=total_pages,
                paper_meta={
                    "paper_title": "Sample Paper",
                    "paper_year": "2024",
                    "paper_authors": "A. Author",
                },
                min_chars=1,
            )
            _, reference_docs = split_reference_docs(docs)
            purity_summary = build_reference_purity_summary(reference_docs)

            self.assertGreater(len(reference_docs), 0)
            self.assertEqual(purity_summary["suspicious_reference_chunks"], 0)


if __name__ == "__main__":
    unittest.main()
