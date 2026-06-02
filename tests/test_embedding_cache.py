from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper_rag.config import Settings
from paper_rag.corpus.chunks import load_chunk_documents
from paper_rag.retrieval.dense.cache import EmbeddingCache
from paper_rag.retrieval.dense.service import build_query_embedder, run_index
from paper_rag.retrieval.sparse.bm25 import BM25CorpusIndex


class EmbeddingCacheTests(unittest.TestCase):
    def test_embedding_cache_appends_new_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "embedding_cache.jsonl"
            cache = EmbeddingCache(path)
            cache.set("a", [1.0])
            cache.save()
            first = path.read_text(encoding="utf-8")

            cache.set("b", [2.0])
            cache.save()
            second = path.read_text(encoding="utf-8")
            reloaded = EmbeddingCache(path)

        self.assertTrue(second.startswith(first))
        self.assertEqual(len(second.splitlines()), 2)
        self.assertEqual(reloaded.get("a"), [1.0])
        self.assertEqual(reloaded.get("b"), [2.0])

    def test_query_embedder_uses_query_cache_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text("QUERY_EMBEDDING_CACHE_PATH=data/index/query-cache.jsonl\n", encoding="utf-8")
            settings = Settings.load(root)
            embedder = build_query_embedder(settings)

        self.assertEqual(embedder.cache.path, root / "data" / "index" / "query-cache.jsonl")

    def test_run_index_writes_bm25_derived_index(self) -> None:
        with index_fixture() as settings:
            summary = run_index(settings, reporter=lambda message: None, embedder=FakeEmbedder(), store=FakeStore())
            loaded = BM25CorpusIndex.load(settings.bm25_index_path, load_chunk_documents(settings.paper_data_dir))
            self.assertEqual(summary.chunk_count, 1)
            self.assertTrue(settings.bm25_index_path.exists())
            self.assertIsNotNone(loaded)


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]


class FakeStore:
    def recreate_collection(self) -> None:
        pass

    def insert_chunk_documents(self, chunk_documents, vectors) -> int:
        _ = vectors
        return len(chunk_documents)


class index_fixture:
    def __enter__(self) -> Settings:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        paper = root / "data" / "paper_data" / "Paper"
        paper.mkdir(parents=True)
        (paper / "metadata.json").write_text(json.dumps({"title": "Paper"}, ensure_ascii=False), encoding="utf-8")
        row = {
            "chunk_id": "Paper::chunk_0000",
            "paper_id": "Paper",
            "chunk_index": 0,
            "region": "body",
            "section_id": "s1",
            "section_path": ["Intro"],
            "pages": [1],
            "block_ids": ["b1"],
            "text": "residual learning",
            "embedding_text": "Paper\n\nresidual learning",
        }
        (paper / "chunks.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
        return Settings.load(root)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
