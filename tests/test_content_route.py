from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from paper_rag.config import Settings
from paper_rag.retrieval.data.aliases_match import AliasMatch
from paper_rag.retrieval.domains.common.errors import PlanParseError
from paper_rag.retrieval.domains.content.planner import plan_body
from paper_rag.retrieval.domains.content.retrieval_query import build_content_retrieval_query
from paper_rag.retrieval.domains.content.router import build_content_decision
from paper_rag.retrieval.domains.content.schema import validate_content_parse
from paper_rag.retrieval.route import RouteDecision


RESNET_TITLE = "Deep Residual Learning for Image Recognition"
BERT_TITLE = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
VIT_TITLE = "An Image is Worth 16x16 Words"


class ContentSchemaTests(unittest.TestCase):
    def test_accepts_current_content_shape(self) -> None:
        payload = validate_content_parse({
            "intent": "lookup",
            "paper_semantic": "",
            "filters": [{"field": "paper", "op": "=", "value": "ResNet", "negated": False}],
            "paper_groups": [],
            "group_mode": "single",
            "content_objects": ["模型结构"],
            "compare_objects": [],
        })

        self.assertEqual(payload["intent"], "lookup")
        self.assertEqual(payload["content_objects"], ["模型结构"])
        self.assertEqual(payload["filters"][0]["field"], "paper")

    def test_rejects_old_content_shape(self) -> None:
        with self.assertRaises(PlanParseError):
            validate_content_parse({
                "intent": "fact",
                "anchors": ["ResNet"],
                "objects": ["模型结构"],
                "compare_objects": [],
                "filters": [],
            })

    def test_rejects_invalid_compare_objects(self) -> None:
        with self.assertRaises(PlanParseError):
            validate_content_parse({
                "intent": "lookup",
                "paper_semantic": "",
                "filters": [],
                "paper_groups": [],
                "group_mode": "single",
                "content_objects": ["模型结构"],
                "compare_objects": ["ResNet", "Transformer"],
            })

    def test_rejects_and_mode_outside_exists(self) -> None:
        with self.assertRaises(PlanParseError):
            validate_content_parse({
                "intent": "list",
                "paper_semantic": "",
                "filters": [],
                "paper_groups": [{"semantic": "", "filters": [{"field": "paper", "op": "=", "value": "ResNet", "negated": False}]}],
                "group_mode": "and",
                "content_objects": ["数据集"],
                "compare_objects": [],
            })


class ContentRouterTests(unittest.TestCase):
    def test_resolves_aliases_and_year_boundaries(self) -> None:
        with content_fixture() as settings:
            parser = StaticContentParser({
                "intent": "lookup",
                "paper_semantic": "",
                "filters": [{"field": "paper", "op": "=", "value": "ResNet", "negated": False}],
                "paper_groups": [
                    {
                        "semantic": "",
                        "filters": [{"field": "year", "op": "interval", "value": ["ResNet", "inf"], "negated": False}],
                    }
                ],
                "group_mode": "per",
                "content_objects": ["模型结构"],
                "compare_objects": [],
            })
            route = build_content_decision(
                settings,
                RouteDecision(route="content", query="ResNet 之后的论文分别是什么模型结构？", parse_status="ok"),
                [],
                plan_parser=parser,
            )

        self.assertEqual(route.filters[0]["value"], RESNET_TITLE)
        self.assertEqual(route.paper_groups[0]["filters"][0]["value"], [2017, "inf"])
        self.assertEqual(route.parser_result["content_objects"], ["模型结构"])


class ContentPlannerTests(unittest.TestCase):
    def test_plan_body_uses_scope_records_to_filter_chunks(self) -> None:
        with content_fixture() as settings:
            route = RouteDecision(
                route="content",
                intent="lookup",
                query="ResNet 的 ImageNet top error 是多少？",
                parser_result={"content_objects": ["ImageNet top error"], "compare_objects": []},
                filters=[{"field": "paper", "op": "=", "value": RESNET_TITLE, "negated": False}],
                group_mode="single",
                parse_status="ok",
            )
            evidence = plan_body(settings, route, [], embedder=EmptyEmbedder(), store=EmptyStore())

        self.assertEqual(evidence["status"], "ok")
        self.assertNotIn("resolved", evidence)
        self.assertNotIn("warnings", evidence)
        self.assertNotIn("scope_records", evidence)
        self.assertNotIn("scope_records", evidence["results"])
        self.assertNotIn("context_units", evidence["results"])
        self.assertEqual([unit["title"] for unit in evidence["results"]["contexts"]], [RESNET_TITLE])
        self.assertNotIn("expanded_blocks", evidence["results"]["contexts"][0])
        self.assertNotIn("sources", evidence["results"]["contexts"][0])
        self.assertEqual(evidence["plan"]["content_objects"], ["ImageNet top error"])

    def test_group_results_show_each_content_scope(self) -> None:
        with content_fixture() as settings:
            route = RouteDecision(
                route="content",
                intent="list",
                query="ResNet 和 BERT 分别用了哪些数据集？",
                parser_result={"content_objects": ["数据集"], "compare_objects": []},
                paper_groups=[
                    {"semantic": "", "filters": [{"field": "paper", "op": "=", "value": RESNET_TITLE, "negated": False}]},
                    {"semantic": "", "filters": [{"field": "paper", "op": "=", "value": BERT_TITLE, "negated": False}]},
                ],
                group_mode="per",
                parse_status="ok",
            )
            evidence = plan_body(settings, route, [], embedder=EmptyEmbedder(), store=EmptyStore())

        self.assertEqual([group["count"] for group in evidence["results"]["groups"]], [1, 1])
        self.assertEqual(
            {title for group in evidence["results"]["groups"] for title in group["papers"]},
            {RESNET_TITLE, BERT_TITLE},
        )

    def test_debug_mode_includes_retrieval_query(self) -> None:
        with content_fixture() as settings:
            embedder = CapturingEmbedder()
            route = RouteDecision(
                route="content",
                intent="lookup",
                query="BERT 的预训练数据是什么？",
                parser_result={"content_objects": ["预训练数据"], "compare_objects": []},
                filters=[{"field": "paper", "op": "=", "value": BERT_TITLE, "negated": False}],
                group_mode="single",
                parse_status="ok",
            )
            evidence = plan_body(settings, route, [], embedder=embedder, store=EmptyStore(), debug=True)

        self.assertIn("debug", evidence)
        self.assertIn("retrieval_query", evidence["debug"])
        self.assertIn("dense_query", evidence["debug"]["retrieval_query"])
        self.assertIn("bm25_queries", evidence["debug"]["retrieval_query"])
        self.assertIn("retrieval_query", evidence["plan"])
        self.assertIn("dense_query", evidence["plan"]["retrieval_query"])
        self.assertIn("bm25_queries", evidence["plan"]["retrieval_query"])
        self.assertIn("scope_records", evidence["debug"])
        self.assertIn("context_units", evidence["debug"])
        self.assertEqual(embedder.texts, [evidence["debug"]["retrieval_query"]["dense_query"]])

    def test_content_retrieval_query_omits_scope_from_dense_query(self) -> None:
        with content_fixture() as settings:
            route = RouteDecision(
                route="content",
                intent="lookup",
                query="ResNet 的 ImageNet top error 是多少？",
                paper_semantic="目标检测论文",
                parser_result={"content_objects": ["ImageNet top error"], "compare_objects": []},
                filters=[
                    {"field": "paper", "op": "=", "value": RESNET_TITLE, "negated": False},
                    {"field": "venue", "op": "=", "value": "CVPR", "negated": False},
                ],
                group_mode="single",
                parse_status="ok",
            )
            evidence = plan_body(settings, route, [], embedder=EmptyEmbedder(), store=EmptyStore(), debug=True)

        retrieval_query = evidence["debug"]["retrieval_query"]
        self.assertIn("ImageNet top error", retrieval_query["dense_query"])
        self.assertNotIn("CVPR", retrieval_query["dense_query"])
        self.assertNotIn(RESNET_TITLE, retrieval_query["dense_query"])
        self.assertIn("ImageNet top error", retrieval_query["bm25_queries"])

    def test_bm25_queries_omit_structured_paper_mentions(self) -> None:
        with content_fixture() as settings:
            route = RouteDecision(
                route="content",
                intent="lookup",
                query="VIT的模型结构是什么",
                parser_result={"content_objects": ["模型结构"], "compare_objects": []},
                filters=[{"field": "paper", "op": "=", "value": "An Image is Worth 16x16 Words", "negated": False}],
                alias_matches=[AliasMatch("VIT", "An Image is Worth 16x16 Words")],
                group_mode="single",
                parse_status="ok",
            )
            retrieval_query = build_content_retrieval_query(settings, route, [], translator=None)

        self.assertIn("模型结构", retrieval_query["bm25_queries"])
        self.assertNotIn("VIT", retrieval_query["bm25_queries"])

    def test_bm25_queries_keep_non_scope_compare_objects(self) -> None:
        with content_fixture() as settings:
            route = RouteDecision(
                route="content",
                intent="compare",
                query="ResNet 里的 BasicBlock 和 Bottleneck 有什么区别？",
                parse_status="ok",
                parser_result={"content_objects": ["模型结构"], "compare_objects": ["BasicBlock", "Bottleneck"]},
                filters=[{"field": "paper", "op": "=", "value": RESNET_TITLE, "negated": False}],
                resolved_papers=[{"title": RESNET_TITLE, "matched_alias": "ResNet"}],
            )
            retrieval_query = build_content_retrieval_query(settings, route, [], translator=None)

        self.assertIn("模型结构", retrieval_query["bm25_queries"])
        self.assertIn("BasicBlock", retrieval_query["bm25_queries"])
        self.assertIn("Bottleneck", retrieval_query["bm25_queries"])
        self.assertNotIn("ResNet", retrieval_query["bm25_queries"])

    def test_bm25_queries_omit_scope_compare_objects(self) -> None:
        with content_fixture() as settings:
            route = RouteDecision(
                route="content",
                intent="compare",
                query="ResNet和VIT的模型结构有什么区别？",
                parse_status="ok",
                parser_result={"content_objects": ["模型结构"], "compare_objects": ["ResNet", "VIT"]},
                paper_groups=[
                    {"semantic": "", "filters": [{"field": "paper", "op": "=", "value": RESNET_TITLE, "negated": False}]},
                    {"semantic": "", "filters": [{"field": "paper", "op": "=", "value": VIT_TITLE, "negated": False}]},
                ],
                alias_matches=[],
                resolved_papers=[
                    {"title": RESNET_TITLE, "matched_alias": "ResNet"},
                    {"title": VIT_TITLE, "matched_alias": "VIT"},
                ],
            )
            retrieval_query = build_content_retrieval_query(settings, route, [], translator=None)

        self.assertEqual(retrieval_query["bm25_queries"], ["模型结构"])

    def test_bm25_queries_include_translation_candidates(self) -> None:
        with content_fixture(env_text=translation_env()) as settings:
            route = RouteDecision(
                route="content",
                intent="lookup",
                query="BERT 的预训练数据是什么？",
                parser_result={"content_objects": ["预训练数据"], "compare_objects": []},
                filters=[{"field": "paper", "op": "=", "value": BERT_TITLE, "negated": False}],
                group_mode="single",
                parse_status="ok",
            )
            evidence = plan_body(
                settings,
                route,
                [],
                embedder=EmptyEmbedder(),
                store=EmptyStore(),
                translator=FakeTranslator(),
                debug=True,
            )

        bm25_queries = evidence["debug"]["retrieval_query"]["bm25_queries"]
        self.assertIn("预训练数据", bm25_queries)
        self.assertIn("BooksCorpus Wikipedia datasets", bm25_queries)
        self.assertIn("pretraining corpus", bm25_queries)
        self.assertEqual([unit["title"] for unit in evidence["results"]["contexts"]], [BERT_TITLE])


class StaticContentParser:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def parse_content(self, query: str) -> dict[str, Any]:
        _ = query
        return validate_content_parse(self.payload)


class EmptyEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        _ = texts
        return [[0.0]]


class CapturingEmbedder:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return [[0.0] for _ in texts]


class EmptyStore:
    def search(self, query_vector: list[float], top_k: int) -> list[Any]:
        _ = query_vector, top_k
        return []


class FakeTranslator:
    def translate(self, text: str, provider: str, settings: Settings) -> str | list[str] | None:
        _ = text, settings
        if provider == "tencent":
            return "BooksCorpus Wikipedia datasets"
        if provider == "aliyun":
            return "pretraining corpus"
        return None


class content_fixture:
    def __init__(self, *, env_text: str = "") -> None:
        self.env_text = env_text

    def __enter__(self) -> Settings:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        if self.env_text:
            (root / ".env").write_text(self.env_text, encoding="utf-8")
        data = root / "data"
        data.mkdir()
        write_manifest(data / "manifest.jsonl")
        write_annotations(data / "paper_annotations.json")
        write_venue_aliases(data / "venue_aliases.json")
        write_chunks(data / "paper_data")
        return Settings.load(root)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.tmp.cleanup()


def translation_env() -> str:
    return "\n".join([
        "TENCENT_TRANSLATE_SECRET_ID=test-id",
        "TENCENT_TRANSLATE_SECRET_KEY=test-key",
        "ALIYUN_TRANSLATE_ACCESS_KEY_ID=test-id",
        "ALIYUN_TRANSLATE_ACCESS_KEY_SECRET=test-key",
        "PLAN_BM25_TRANSLATE_PROVIDERS=tencent,aliyun",
    ])


def write_manifest(path: Path) -> None:
    records = [
        {
            "file_hash": "resnet",
            "status": "active",
            "title": RESNET_TITLE,
            "author": ["Kaiming He"],
            "year": {"preprint_year": 2015, "publish_year": 2016},
            "venue": "CVPR",
            "paper_data_path": str(path.parent / "paper_data" / "ResNet"),
            "pdf_path": str(path.parent / "pdf" / "resnet.pdf"),
        },
        {
            "file_hash": "bert",
            "status": "active",
            "title": BERT_TITLE,
            "author": ["Jacob Devlin"],
            "year": {"preprint_year": 2018, "publish_year": 2019},
            "venue": "NAACL",
            "paper_data_path": str(path.parent / "paper_data" / "BERT"),
            "pdf_path": str(path.parent / "pdf" / "bert.pdf"),
        },
    ]
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")


def write_annotations(path: Path) -> None:
    path.write_text(json.dumps({
        "resnet": {"title": RESNET_TITLE, "aliases": ["ResNet"], "tags": {"zh": ["残差网络"], "en": ["residual network"]}},
        "bert": {"title": BERT_TITLE, "aliases": ["BERT"], "tags": {"zh": ["预训练"], "en": ["language model"]}},
    }, ensure_ascii=False), encoding="utf-8")


def write_venue_aliases(path: Path) -> None:
    path.write_text("[]", encoding="utf-8")


def write_chunks(paper_data_dir: Path) -> None:
    write_paper_chunks(
        paper_data_dir / "ResNet",
        RESNET_TITLE,
        "ResNet reports ImageNet top error and model structure details.",
    )
    write_paper_chunks(
        paper_data_dir / "BERT",
        BERT_TITLE,
        "BERT uses BooksCorpus and Wikipedia datasets for pre-training.",
    )


def write_paper_chunks(directory: Path, title: str, text: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "metadata.json").write_text(json.dumps({"title": title}, ensure_ascii=False), encoding="utf-8")
    row = {
        "chunk_id": f"{directory.name}-c1",
        "paper_id": directory.name,
        "chunk_index": 0,
        "region": "body",
        "section_id": "s1",
        "section_path": ["Experiments"],
        "pages": [1],
        "block_ids": ["b1"],
        "text": text,
        "embedding_text": text,
    }
    (directory / "chunks.jsonl").write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")
    block = {
        "block_id": "b1",
        "order": 1,
        "region": "body",
        "type": "paragraph",
        "text": text,
        "page": 1,
        "section_id": "s1",
        "section_path": ["Experiments"],
    }
    (directory / "blocks.jsonl").write_text(json.dumps(block, ensure_ascii=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
