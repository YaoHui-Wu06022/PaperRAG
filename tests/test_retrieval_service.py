from __future__ import annotations

import unittest

from langchain_core.documents import Document

from config import load_config
from retrieval.query_router import QueryRoute
from services.retrieval_service import run_retrieval_flow


class RetrievalServiceTest(unittest.TestCase):
    def test_metadata_alias_query_seeds_paper_docs_from_title_hint(self) -> None:
        config = load_config()
        paper_doc = Document(
            page_content=(
                "标题: Deep Residual Learning for Image Recognition\n"
                "作者: Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun\n"
                "摘要: Residual learning framework."
            ),
            metadata={
                "doc_id": "resnet_doc",
                "title": "Deep Residual Learning for Image Recognition",
                "paper_title": "Deep Residual Learning for Image Recognition",
                "authors": "Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun",
                "paper_authors": "Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun",
                "year": "2015",
                "paper_year": "2015",
                "source": "He et al. - ResNet.pdf",
            },
        )
        chunk_doc = Document(
            page_content=(
                "Deep Residual Learning for Image Recognition by "
                "Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun."
            ),
            metadata={
                "doc_id": "resnet_doc",
                "title": "Deep Residual Learning for Image Recognition",
                "paper_title": "Deep Residual Learning for Image Recognition",
                "authors": "Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun",
                "paper_authors": "Kaiming He Xiangyu Zhang Shaoqing Ren Jian Sun",
                "page": 1,
                "section_path": "正文",
                "source": "He et al. - ResNet.pdf",
            },
        )
        route = QueryRoute(
            route_type="metadata",
            retrieval_scope="main",
            retrieval_mode="bm25",
            use_parent_context=False,
            prefer_paper_context=True,
            paper_top_k=5,
            chunk_top_k=5,
            final_top_k=3,
            prompt_mode="metadata",
            reason="test",
        )

        result = run_retrieval_flow(
            config,
            "谁提出的ResNet?",
            None,
            chunk_corpus=[chunk_doc],
            chunk_corpus_key=None,
            paper_vector_store=None,
            paper_corpus=[paper_doc],
            paper_corpus_key=None,
            section_corpus=[],
            parent_map=None,
            apply_metadata_filters=False,
            retrieval_mode_override=None,
            use_parent_context=False,
            route=route,
        )

        self.assertEqual(result.route_type, "metadata")
        self.assertTrue(result.paper_docs)
        self.assertEqual(result.paper_docs[0].metadata.get("doc_id"), "resnet_doc")
        self.assertTrue(result.generation_docs)
        self.assertEqual(result.generation_docs[0].metadata.get("doc_id"), "resnet_doc")


if __name__ == "__main__":
    unittest.main()
