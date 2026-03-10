from __future__ import annotations

import unittest

from config import load_config
from retrieval.query_router import route_query


class QueryRouterTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_config()

    def test_routes_survey_query(self) -> None:
        route = route_query(self.config, "深度学习的发展路径是什么？")
        self.assertEqual(route.route_type, "survey")
        self.assertTrue(route.prefer_paper_context)
        self.assertEqual(route.prompt_mode, "survey")

    def test_routes_metadata_query(self) -> None:
        route = route_query(self.config, "author: He venue: CVPR 2016")
        self.assertEqual(route.route_type, "metadata")
        self.assertEqual(route.retrieval_mode, "bm25")

    def test_routes_model_origin_query_as_metadata(self) -> None:
        route = route_query(self.config, "谁提出的ResNet?")
        self.assertEqual(route.route_type, "metadata")
        self.assertTrue(route.prefer_paper_context)

    def test_routes_reference_query(self) -> None:
        route = route_query(self.config, "这篇论文引用了哪些工作？")
        self.assertEqual(route.route_type, "references")
        self.assertEqual(route.retrieval_scope, "references")


if __name__ == "__main__":
    unittest.main()
