import unittest

from retrieval.query_rewrite import build_query_variants


class QueryRewriteTest(unittest.TestCase):
    def test_disabled_returns_original_query(self) -> None:
        query = "What datasets are used for evaluation in RAG?"
        self.assertEqual(build_query_variants(query, enabled=False), [query])

    def test_english_query_gets_academic_expansion(self) -> None:
        query = "What datasets and metrics are used for evaluation in RAG?"
        variants = build_query_variants(query, max_variants=5)

        self.assertEqual(variants[0], query)
        self.assertTrue(any("retrieval augmented generation" in item.lower() for item in variants))
        self.assertTrue(
            any("benchmark" in item.lower() and "metrics" in item.lower() for item in variants)
        )

    def test_chinese_query_gets_cross_lingual_terms(self) -> None:
        query = "请问多模态检索模型的评测指标是什么？"
        variants = build_query_variants(query, max_variants=5)

        self.assertEqual(variants[0], query)
        self.assertTrue(any(not item.startswith("请问") for item in variants[1:]))
        self.assertTrue(
            any("multimodal" in item.lower() and "retrieval" in item.lower() for item in variants)
        )


if __name__ == "__main__":
    unittest.main()
