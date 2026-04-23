from __future__ import annotations

import unittest

from paper_rag.retrieval.sparse.bm25 import BM25Document, BM25Index, tokenize


class BM25Tests(unittest.TestCase):
    def test_tokenize_normalizes_dash_variants_and_filters_stopwords(self) -> None:
        tokens = tokenize("How does intra_class and long\u2013short-term memory work?")
        self.assertNotIn("how", tokens)
        self.assertNotIn("and", tokens)
        self.assertIn("intra-class", tokens)
        self.assertIn("long-short-term", tokens)
        self.assertIn("memory", tokens)

    def test_bm25_ignores_stopword_only_query(self) -> None:
        index = BM25Index([
            BM25Document("doc1", "center loss improves compactness", {}),
        ])
        self.assertEqual(index.search("the and of", top_k=5), [])


if __name__ == "__main__":
    unittest.main()
