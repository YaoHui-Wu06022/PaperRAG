from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path

from paper_rag.ingest.extract import extract_paper_data, group_chunk_blocks
from paper_rag.utils import normalize_text


ROOT = Path(__file__).resolve().parents[1]
MINERU = ROOT / "data" / "mineru_output"


class ExtractPaperDataTests(unittest.TestCase):
    def assert_no_acknowledgement_section_or_block(self, data: dict, blocks: list[dict]) -> None:
        self.assertFalse(
            any("acknowledg" in normalize_text(section["title"]) for section in data["toc"]["sections"]),
        )
        self.assertFalse(
            any("acknowledg" in normalize_text(block["text"]) for block in blocks),
        )

    def extract_sample(
        self,
        folder: str,
        *,
        chunk_target_chars: int = 1400,
        chunk_overlap_chars: int = 200,
    ) -> tuple[Path, dict, list[dict], list[dict], list[dict]]:
        sample_dir = MINERU / folder
        if not sample_dir.exists():
            parts = folder.split("_-_")
            title_slug = parts[2].split("__", 1)[0] if len(parts) >= 3 else folder.split("__", 1)[0]
            title_slug = re.sub(r"^(?:19|20)\d{2}_", "", title_slug)
            title_norm = normalize_text(title_slug)
            matches = [
                path for path in sorted(MINERU.iterdir())
                if path.is_dir() and title_norm[:24] in normalize_text(path.name)
            ]
            sample_dir = next((path for path in matches if (path / "content_list_v2.json").exists()), sample_dir)
        with tempfile.TemporaryDirectory() as tmp:
            result = extract_paper_data(
                sample_dir,
                Path(tmp) / "paper",
                {"pdf_path": "sample.pdf"},
                chunk_target_chars=chunk_target_chars,
                chunk_overlap_chars=chunk_overlap_chars,
            )
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            toc = json.loads(result.toc_path.read_text(encoding="utf-8"))
            blocks = [
                json.loads(line)
                for line in result.blocks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            chunks = [
                json.loads(line)
                for line in result.chunks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            refs = [
                json.loads(line)
                for line in result.references_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(result.chunk_count, len(chunks))
            return result.paper_data_dir, {"metadata": metadata, "toc": toc}, blocks, refs, chunks

    def test_efficientnet_title_falls_back_to_page_header(self) -> None:
        _, data, blocks, refs, chunks = self.extract_sample(
            "Tan_Le_-_2019_-_EfficientNet_Rethinking_Model_Scaling_for_Convolutional_Neural_Networks__41ad4255609cda21"
        )
        self.assertEqual(
            data["metadata"]["title"],
            "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks",
        )
        self.assertTrue(any(block["region"] == "abstract" for block in blocks))
        self.assertTrue(any(block["region"] == "appendix" for block in blocks))
        self.assertTrue(any(chunk["region"] == "appendix" for chunk in chunks))
        self.assertGreater(len(refs), 40)

    def test_wen_paragraph_abstract_is_detected(self) -> None:
        _, data, blocks, refs, chunks = self.extract_sample(
            "Wen_-_2016_-_A_Discriminative_Feature_Learning_Approach_for_Deep_Face_Recognition__0fbc50d1e12bea79"
        )
        self.assertEqual(
            data["metadata"]["title"],
            "A Discriminative Feature Learning Approach for Deep Face Recognition",
        )
        self.assertFalse(blocks[0]["text"].startswith("Abstract."))
        self.assertTrue(blocks[0]["text"].startswith("Convolutional neural networks"))
        self.assertFalse(any(block["text"].startswith("Keywords:") for block in blocks))
        self.assert_no_acknowledgement_section_or_block(data, blocks)
        self.assertGreater(len(refs), 30)

    def test_bert_post_reference_appendix_is_not_reference(self) -> None:
        _, data, blocks, refs, chunks = self.extract_sample(
            "Devlin_-_2019_-_BERT_Pre-training_of_Deep_Bidirectional_Transformers_for_Language_Understanding__a7b7a17d0cf9953e"
        )
        self.assertEqual(
            data["metadata"]["title"],
            "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
        )
        self.assertEqual(len(refs), 48)
        self.assertTrue(any(block["region"] == "appendix" for block in blocks))
        self.assertTrue(any("Task #2: Next Sentence Prediction" in block["text"] for block in blocks))
        self.assertTrue(any(chunk["region"] == "appendix" for chunk in chunks))
        self.assertTrue(any("Task #2: Next Sentence Prediction" in chunk["text"] for chunk in chunks))

    def test_appendix_titles_split_chunk_groups_except_list_like_titles(self) -> None:
        blocks = [
            {"order": 1, "region": "body", "section_id": "sec_1", "type": "paragraph", "text": "body"},
            {"order": 2, "region": "appendix", "section_id": "sec_appendix", "type": "title", "text": "Appendix"},
            {"order": 3, "region": "appendix", "section_id": "sec_appendix", "type": "title", "text": "A Details"},
            {"order": 4, "region": "appendix", "section_id": "sec_appendix", "type": "paragraph", "text": "details"},
            {"order": 5, "region": "appendix", "section_id": "sec_appendix", "type": "title", "text": "• Batch size"},
            {"order": 6, "region": "appendix", "section_id": "sec_appendix", "type": "paragraph", "text": "batch"},
            {"order": 7, "region": "appendix", "section_id": "sec_appendix", "type": "title", "text": "B Results"},
            {"order": 8, "region": "appendix", "section_id": "sec_appendix", "type": "paragraph", "text": "results"},
        ]

        group_texts = [[block["text"] for block in group] for group in group_chunk_blocks(blocks)]

        self.assertEqual(
            group_texts,
            [
                ["body"],
                ["Appendix", "A Details", "details", "• Batch size", "batch"],
                ["B Results", "results"],
            ],
        )

    def test_attention_toc_has_numbered_tree(self) -> None:
        _, data, blocks, refs, chunks = self.extract_sample(
            "Vaswani_-_2017_-_Attention_is_All_you_Need__8656df5ece8f482c"
        )
        section_ids = {section["section_id"] for section in data["toc"]["sections"]}
        self.assertIn("sec_abstract", section_ids)
        self.assertIn("sec_3", section_ids)
        self.assertIn("sec_3_2", section_ids)
        self.assertIn("sec_3_2_1", section_ids)
        self.assertTrue(all(block["region"] in {"abstract", "body"} for block in blocks))
        self.assertTrue(all("section_path" in block for block in blocks))
        self.assertIn("start_block_index", data["toc"]["sections"][0])
        self.assertIn("end_block_index", data["toc"]["sections"][0])
        self.assertEqual(len(refs), 32)
        self.assertNotIn("bbox", refs[0])
        self.assertNotIn("label", refs[0])

    def test_inception_unnumbered_toc_is_flat(self) -> None:
        _, data, blocks, refs, chunks = self.extract_sample(
            "2017_Inception_v4_Inception_ResNet_and_the_Impact_of_Residual_Connections_on_Learning"
        )
        body_sections = [section for section in data["toc"]["sections"] if section["region"] == "body"]
        self.assertGreater(len(body_sections), 5)
        self.assertTrue(all(section["parent_id"] is None for section in body_sections))
        self.assertTrue(any(section["title"] == "Introduction" for section in body_sections))

    def test_lstm_media_blocks_keep_structured_fields(self) -> None:
        _, data, blocks, refs, chunks = self.extract_sample("1997_Long_Short_Term_Memory")
        self.assert_no_acknowledgement_section_or_block(data, blocks)
        image = next(block for block in blocks if block["type"] == "image")
        table = next(block for block in blocks if block["type"] == "table")
        self.assertIn("source_path", image)
        self.assertIn("caption", image)
        self.assertNotIn("image_footnote", image)
        self.assertIn("source_path", table)
        self.assertIn("caption", table)
        self.assertIn("html", table)
        self.assertIn("Columns:", table["text"])
        self.assertIn("Row 1:", table["text"])
        self.assertIn("method =", table["text"])
        self.assertNotIn("table_footnote", table)

    def test_normface_numbered_acknowledgement_is_filtered(self) -> None:
        _, data, blocks, refs, chunks = self.extract_sample(
            "2017_NormFace_L2_Hypersphere_Embedding_for_Face_Verification"
        )
        self.assert_no_acknowledgement_section_or_block(data, blocks)
        self.assertGreater(len(refs), 30)

    def test_attention_chunks_have_schema_and_prefix(self) -> None:
        paper_dir, data, blocks, refs, chunks = self.extract_sample(
            "Vaswani_-_2017_-_Attention_is_All_you_Need__8656df5ece8f482c"
        )
        self.assertTrue(chunks)
        first = chunks[0]
        for key in {
            "chunk_id",
            "paper_id",
            "chunk_index",
            "region",
            "section_id",
            "section_path",
            "pages",
            "block_ids",
            "text",
            "embedding_text",
            "char_count",
        }:
            self.assertIn(key, first)
        self.assertEqual(first["paper_id"], paper_dir.name)
        self.assertTrue(first["chunk_id"].startswith(f"{paper_dir.name}::chunk_"))
        self.assertTrue(first["chunk_id"].endswith("chunk_0000"))
        self.assertEqual(first["chunk_index"], 0)
        self.assertTrue(first["embedding_text"].startswith("Paper: Attention Is All You Need\nSection:"))
        self.assertFalse(first["text"].startswith("Paper:"))
        self.assertTrue(all(chunk["region"] in {"abstract", "body"} for chunk in chunks))
        self.assertTrue(all(isinstance(page, int) for chunk in chunks for page in chunk["pages"]))
        self.assertTrue(all(chunk["block_ids"] for chunk in chunks))

    def test_long_section_splits_with_embedding_overlap(self) -> None:
        _, data, blocks, refs, chunks = self.extract_sample(
            "Vaswani_-_2017_-_Attention_is_All_you_Need__8656df5ece8f482c",
            chunk_target_chars=600,
            chunk_overlap_chars=80,
        )
        by_section: dict[str, list[dict]] = {}
        for chunk in chunks:
            by_section.setdefault(chunk["section_id"], []).append(chunk)
        split_sections = [items for items in by_section.values() if len(items) > 1]
        self.assertTrue(split_sections)
        first, second = split_sections[0][0], split_sections[0][1]
        tail = first["text"][-80:].strip()
        self.assertIn(tail, second["embedding_text"])
        self.assertNotIn(tail, second["text"])

    def test_table_text_is_available_in_chunks(self) -> None:
        _, data, blocks, refs, chunks = self.extract_sample("1997_Long_Short_Term_Memory")
        table_chunk = next(chunk for chunk in chunks if "Columns:" in chunk["text"] and "Row 1:" in chunk["text"])
        self.assertIn("method =", table_chunk["text"])

    def test_oversize_block_is_not_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mineru_dir = Path(tmp) / "mineru"
            mineru_dir.mkdir()
            long_text = "A" * 2000
            content = [
                [
                    {"type": "title", "content": {"title_content": [{"type": "text", "content": "Sample Paper"}]}},
                    {"type": "title", "content": {"title_content": [{"type": "text", "content": "Abstract"}]}},
                    {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "A short abstract."}]}},
                    {"type": "title", "content": {"title_content": [{"type": "text", "content": "1 Introduction"}]}},
                    {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": long_text}]}},
                ]
            ]
            (mineru_dir / "content_list_v2.json").write_text(json.dumps(content), encoding="utf-8")
            result = extract_paper_data(
                mineru_dir,
                Path(tmp) / "paper",
                {"pdf_path": "sample.pdf"},
                chunk_target_chars=500,
                chunk_overlap_chars=100,
            )
            chunks = [
                json.loads(line)
                for line in result.chunks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            long_chunk = next(chunk for chunk in chunks if long_text in chunk["text"])
            self.assertEqual(long_chunk["text"], long_text)
            self.assertEqual(long_chunk["char_count"], len(long_text))

    def test_page_dict_content_shape_is_supported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mineru_dir = Path(tmp) / "mineru"
            mineru_dir.mkdir()
            content = [
                {
                    "content": [
                        {"type": "title", "content": {"title_content": [{"type": "text", "content": "Sample Paper"}]}},
                        {"type": "title", "content": {"title_content": [{"type": "text", "content": "Abstract"}]}},
                        {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "A short abstract."}]}},
                        {"type": "title", "content": {"title_content": [{"type": "text", "content": "1 Introduction"}]}},
                        {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "Body text."}]}},
                    ]
                }
            ]
            (mineru_dir / "content_list_v2.json").write_text(json.dumps(content), encoding="utf-8")
            result = extract_paper_data(mineru_dir, Path(tmp) / "paper", {"pdf_path": "sample.pdf"})
            metadata = json.loads(result.metadata_path.read_text(encoding="utf-8"))
            blocks = [
                json.loads(line)
                for line in result.blocks_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(metadata["title"], "Sample Paper")
            self.assertEqual(blocks[0]["region"], "abstract")
            self.assertEqual(blocks[-1]["section_path"], ["1 Introduction"])


if __name__ == "__main__":
    unittest.main()
