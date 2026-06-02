from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from paper_rag.corpus.chunks import ChunkDocument
from paper_rag.retrieval.sparse.bm25 import BM25CorpusIndex, BM25Document, normalize_bm25_token


class BM25Tests(unittest.TestCase):
    def test_tokenize_normalizes_dash_variants_and_filters_stopwords(self) -> None:
        tokens = normalize_bm25_token("How does intra_class and long\u2013short-term memory work?")
        self.assertNotIn("how", tokens)
        self.assertNotIn("and", tokens)
        self.assertIn("intra-class", tokens)
        self.assertIn("long-short-term", tokens)
        self.assertIn("memory", tokens)

    def test_bm25_ignores_stopword_only_query(self) -> None:
        index = BM25CorpusIndex([
            BM25Document("doc1", "center loss improves compactness", {}),
        ])
        self.assertEqual(index.search("the and of", top_k=5), [])

    def test_bm25_scope_limits_results(self) -> None:
        index = BM25CorpusIndex([
            BM25Document("doc1", "residual learning image classification", {}),
            BM25Document("doc2", "language model pretraining corpus", {}),
        ])

        hits = index.search_many(["residual learning"], top_k=5, allowed_chunk_ids=["doc2"])

        self.assertEqual(hits, [])

    def test_bm25_index_round_trips_and_detects_stale_chunks(self) -> None:
        chunks = [chunk("c1", "residual learning image classification")]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bm25_chunks.json"
            BM25CorpusIndex.from_chunks(chunks).save(path)

            loaded = BM25CorpusIndex.load(path, chunks)
            stale = BM25CorpusIndex.load(path, [chunk("c1", "changed text")])

        self.assertIsNotNone(loaded)
        self.assertEqual([hit.doc_id for hit in loaded.search("residual", top_k=5)], ["c1"])
        self.assertIsNone(stale)


def chunk(chunk_id: str, text: str) -> ChunkDocument:
    return ChunkDocument(
        chunk_id=chunk_id,
        paper_id="paper",
        chunk_index=0,
        region="body",
        section_id="s1",
        title="Paper",
        section_path=["Section"],
        pages=[1],
        block_ids=["b1"],
        text=text,
        embedding_text=text,
    )


if __name__ == "__main__":
    unittest.main()
