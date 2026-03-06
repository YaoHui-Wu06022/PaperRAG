from __future__ import annotations

from dataclasses import replace
import unittest

from config import load_config
from services import local_cache_store
from services.retrieval_service import run_retrieval_flow


class ReferenceRetrievalPathTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = replace(load_config(), use_reranker=False)

    def test_reference_bm25_flow_returns_reference_docs(self) -> None:
        reference_docs, reference_key = local_cache_store.load_reference_chunk_corpus(
            self.config
        )
        self.assertGreater(len(reference_docs), 0)

        result = run_retrieval_flow(
            self.config,
            "dropout",
            None,
            chunk_corpus=reference_docs,
            chunk_corpus_key=reference_key,
            parent_map=None,
            apply_metadata_filters=False,
            retrieval_mode_override="bm25",
            use_parent_context=False,
        )

        self.assertGreater(len(result.evidence_docs), 0)
        self.assertTrue(
            all(bool(doc.metadata.get("is_reference")) for doc in result.evidence_docs)
        )
        joined = "\n".join(doc.page_content.lower() for doc in result.evidence_docs)
        self.assertIn("dropout", joined)


if __name__ == "__main__":
    unittest.main()
