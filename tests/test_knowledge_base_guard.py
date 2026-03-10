from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from config import load_config
from retrieval.query_router import QueryRoute
from services.knowledge_base_guard import (
    build_readiness_error_message,
    check_query_readiness,
)


class KnowledgeBaseGuardTest(unittest.TestCase):
    @patch("services.knowledge_base_guard.vector_index_exists", return_value=False)
    def test_query_readiness_reports_missing_paper_assets(self, _: object) -> None:
        config = load_config()
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            test_config = replace(
                config,
                local_cache_dir=temp_path,
                data_pdf_dir=temp_path / "pdf",
                milvus_papers_collection="rag_pdf_papers",
            )
            route = QueryRoute(
                route_type="survey",
                retrieval_scope="main",
                retrieval_mode="hybrid",
                use_parent_context=False,
                prefer_paper_context=True,
                paper_top_k=8,
                chunk_top_k=30,
                final_top_k=6,
                prompt_mode="survey",
                reason="test",
            )
            health = check_query_readiness(test_config, route=route)
            self.assertFalse(health.ok)
            self.assertIn("missing_paper_corpus", health.reasons)
            self.assertIn("missing_remote_collection:rag_pdf_papers", health.reasons)
            message = build_readiness_error_message(health, route=route)
            self.assertIn("python main.py ingest", message)


if __name__ == "__main__":
    unittest.main()
