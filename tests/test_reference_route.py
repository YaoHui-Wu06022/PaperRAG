from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper_rag.config import Settings
from paper_rag.retrieval.domains.common.errors import PlanParseError
from paper_rag.retrieval.domains.reference.planner import plan_reference
from paper_rag.retrieval.domains.reference.router import build_reference_decision
from paper_rag.retrieval.domains.reference.schema import validate_reference_parse
from paper_rag.retrieval.route import RouteDecision


RESNET_TITLE = "Deep Residual Learning for Image Recognition"
VIT_TITLE = "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"
ATTN_TITLE = "Attention as Activation"
ARXIV_2013_TITLE = "Old ArXiv Vision Paper"
SUPCON_TITLE = "Supervised Contrastive Learning"


class ReferenceSchemaTests(unittest.TestCase):
    def test_accepts_current_reference_shape(self) -> None:
        payload = validate_reference_parse({
            "intent": "list",
            "return_side": "source",
            "source_semantic": "",
            "source_filters": [{"field": "title", "op": "contains", "value": "attention", "negated": False}],
            "source_groups": [],
            "source_mode": "single",
            "object_semantic": "",
            "object_filters": [{"field": "paper", "op": "=", "value": "VIT", "negated": False}],
            "object_groups": [],
            "object_mode": "single",
        })

        self.assertEqual(payload["return_side"], "source")
        self.assertEqual(payload["object_filters"][0]["field"], "paper")

    def test_rejects_old_reference_shape(self) -> None:
        with self.assertRaises(PlanParseError):
            validate_reference_parse({
                "intent": "list",
                "direction": "cited_by",
                "anchors": ["ResNet"],
                "anchor_mode": "per",
                "filters": [],
            })

    def test_rejects_exists_with_return_side(self) -> None:
        with self.assertRaises(PlanParseError):
            validate_reference_parse({
                "intent": "exists",
                "return_side": "source",
                "source_semantic": "",
                "source_filters": [],
                "source_groups": [],
                "source_mode": "single",
                "object_semantic": "",
                "object_filters": [],
                "object_groups": [],
                "object_mode": "single",
            })


class ReferenceRouterTests(unittest.TestCase):
    def test_resolves_aliases_and_year_boundaries_on_both_sides(self) -> None:
        with reference_fixture() as settings:
            parser = StaticReferenceParser({
                "intent": "list",
                "return_side": "source",
                "source_semantic": "",
                "source_filters": [{"field": "year", "op": "interval", "value": ["ResNet", "inf"], "negated": False}],
                "source_groups": [],
                "source_mode": "single",
                "object_semantic": "",
                "object_filters": [{"field": "paper", "op": "=", "value": "ResNet", "negated": False}],
                "object_groups": [],
                "object_mode": "single",
            })
            route = build_reference_decision(
                settings,
                RouteDecision(route="reference", original_query="ResNet之后，有哪些论文引用ResNet？", parse_status="ok"),
                [],
                plan_parser=parser,
            )

        self.assertEqual(route.object_filters[0]["value"], RESNET_TITLE)
        self.assertEqual(route.source_filters[0]["value"], [2017, "inf"])


class ReferencePlannerTests(unittest.TestCase):
    def test_returns_source_side_papers(self) -> None:
        with reference_fixture() as settings:
            route = RouteDecision(
                route="reference",
                intent="list",
                return_side="source",
                source_filters=[],
                object_filters=[{"field": "paper", "op": "=", "value": RESNET_TITLE, "negated": False}],
            )
            evidence = plan_reference(settings, route, [])

        titles = {paper["title"] for paper in evidence["answer_papers"]}
        self.assertEqual(titles, {ATTN_TITLE, SUPCON_TITLE})

    def test_returns_object_side_papers(self) -> None:
        with reference_fixture() as settings:
            route = RouteDecision(
                route="reference",
                intent="list",
                return_side="object",
                source_filters=[{"field": "paper", "op": "=", "value": ATTN_TITLE, "negated": False}],
                object_filters=[],
            )
            evidence = plan_reference(settings, route, [])

        self.assertEqual(
            {paper["title"] for paper in evidence["answer_papers"]},
            {VIT_TITLE, RESNET_TITLE},
        )

    def test_source_filters_limit_citing_side(self) -> None:
        with reference_fixture() as settings:
            route = RouteDecision(
                route="reference",
                intent="list",
                return_side="source",
                source_filters=[
                    {"field": "year", "op": "interval", "value": ["-inf", 2014], "negated": False},
                    {"field": "venue", "op": "=", "value": "ArXiv", "negated": False},
                ],
                object_filters=[{"field": "paper", "op": "=", "value": "Long Short-Term Memory", "negated": False}],
            )
            evidence = plan_reference(settings, route, [])

        self.assertEqual([paper["title"] for paper in evidence["answer_papers"]], [ARXIV_2013_TITLE])

    def test_count_counts_unique_answer_papers(self) -> None:
        with reference_fixture() as settings:
            route = RouteDecision(
                route="reference",
                intent="count",
                return_side="source",
                source_filters=[],
                object_filters=[{"field": "paper", "op": "=", "value": RESNET_TITLE, "negated": False}],
            )
            evidence = plan_reference(settings, route, [])

        self.assertEqual(evidence["count"], 2)

    def test_exists_checks_edges_between_scopes(self) -> None:
        with reference_fixture() as settings:
            route = RouteDecision(
                route="reference",
                intent="exists",
                return_side=None,
                source_filters=[{"field": "paper", "op": "=", "value": SUPCON_TITLE, "negated": False}],
                object_filters=[{"field": "paper", "op": "=", "value": RESNET_TITLE, "negated": False}],
            )
            evidence = plan_reference(settings, route, [])

        self.assertTrue(evidence["exists"])

    def test_graph_missing_returns_warning_status(self) -> None:
        with reference_fixture(write_graph=False) as settings:
            route = RouteDecision(
                route="reference",
                intent="list",
                return_side="source",
                object_filters=[{"field": "paper", "op": "=", "value": RESNET_TITLE, "negated": False}],
            )
            warnings: list[str] = []
            evidence = plan_reference(settings, route, warnings)

        self.assertEqual(evidence["parse_status"], "graph_missing")
        self.assertTrue(warnings)


class StaticReferenceParser:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def parse_reference(self, query: str) -> dict[str, object]:
        _ = query
        return self.payload


class reference_fixture:
    def __init__(self, *, write_graph: bool = True) -> None:
        self.write_graph = write_graph

    def __enter__(self) -> Settings:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        data = root / "data"
        data.mkdir()
        write_manifest(data / "manifest.jsonl")
        write_venue_aliases(data / "venue_aliases.json")
        write_annotations(data / "paper_annotations.json")
        if self.write_graph:
            write_citation_graph(data / "paper_data" / "citation_graph.json")
        return Settings.load(root)

    def __exit__(self, exc_type, exc, tb) -> None:
        self.tmp.cleanup()


def write_manifest(path: Path) -> None:
    records = [
        {
            "file_hash": "resnet",
            "status": "active",
            "title": RESNET_TITLE,
            "author": ["Kaiming He"],
            "year": {"preprint_year": 2015, "publish_year": 2016},
            "venue": "2016 IEEE Conference on Computer Vision and Pattern Recognition",
            "paper_data_path": str(path.parent / "paper_data" / "ResNet"),
            "pdf_path": str(path.parent / "pdf" / "resnet.pdf"),
        },
        {
            "file_hash": "vit",
            "status": "active",
            "title": VIT_TITLE,
            "author": ["Alexey Dosovitskiy"],
            "year": {"preprint_year": 2020, "publish_year": 2021},
            "venue": "ICLR",
            "paper_data_path": str(path.parent / "paper_data" / "VIT"),
            "pdf_path": str(path.parent / "pdf" / "vit.pdf"),
        },
        {
            "file_hash": "attn",
            "status": "active",
            "title": ATTN_TITLE,
            "author": ["Ada Vision"],
            "year": {"preprint_year": 2021, "publish_year": 2021},
            "venue": "Neural Information Processing Systems",
            "paper_data_path": str(path.parent / "paper_data" / "ATTN"),
            "pdf_path": str(path.parent / "pdf" / "attn.pdf"),
        },
        {
            "file_hash": "supcon",
            "status": "active",
            "title": SUPCON_TITLE,
            "author": ["Prannay Khosla"],
            "year": {"preprint_year": 2020, "publish_year": 2020},
            "venue": "Neural Information Processing Systems",
            "paper_data_path": str(path.parent / "paper_data" / "SupCon"),
            "pdf_path": str(path.parent / "pdf" / "supcon.pdf"),
        },
        {
            "file_hash": "lstm",
            "status": "active",
            "title": "Long Short-Term Memory",
            "author": ["Sepp Hochreiter"],
            "year": {"preprint_year": None, "publish_year": 1997},
            "venue": "Neural Computation",
            "paper_data_path": str(path.parent / "paper_data" / "LSTM"),
            "pdf_path": str(path.parent / "pdf" / "lstm.pdf"),
        },
        {
            "file_hash": "old",
            "status": "active",
            "title": ARXIV_2013_TITLE,
            "author": ["Early Visioner"],
            "year": {"preprint_year": 2013, "publish_year": None},
            "venue": None,
            "paper_data_path": str(path.parent / "paper_data" / "OldArxiv"),
            "pdf_path": str(path.parent / "pdf" / "old.pdf"),
        },
    ]
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")


def write_venue_aliases(path: Path) -> None:
    path.write_text(json.dumps([
        {
            "canonical": "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
            "display": "CVPR",
            "aliases": ["CVPR", "IEEE Conference on Computer Vision and Pattern Recognition"],
        },
        {
            "canonical": "Conference on Neural Information Processing Systems",
            "display": "NeurIPS",
            "aliases": ["NeurIPS", "Neural Information Processing Systems", "NIPS"],
        },
        {
            "canonical": "arXiv",
            "display": "ArXiv",
            "aliases": ["ArXiv", "arXiv"],
        },
    ], ensure_ascii=False), encoding="utf-8")


def write_annotations(path: Path) -> None:
    path.write_text(json.dumps({
        "resnet": {
            "title": RESNET_TITLE,
            "aliases": ["ResNet"],
            "tags": {"zh": ["残差网络"], "en": ["residual network"]},
        },
        "vit": {
            "title": VIT_TITLE,
            "aliases": ["VIT", "ViT"],
            "tags": {"zh": ["视觉 Transformer"], "en": ["vision transformer"]},
        },
        "lstm": {
            "title": "Long Short-Term Memory",
            "aliases": ["LSTM"],
            "tags": {"zh": ["循环网络"], "en": ["recurrent network"]},
        },
    }, ensure_ascii=False), encoding="utf-8")


def write_citation_graph(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": 1,
        "nodes": [
            {"paper_id": "ResNet", "title": RESNET_TITLE, "author": ["Kaiming He"], "year": {"preprint_year": 2015, "publish_year": 2016}, "venue": "CVPR"},
            {"paper_id": "VIT", "title": VIT_TITLE, "author": ["Alexey Dosovitskiy"], "year": {"preprint_year": 2020, "publish_year": 2021}, "venue": "ICLR"},
            {"paper_id": "ATTN", "title": ATTN_TITLE, "author": ["Ada Vision"], "year": {"preprint_year": 2021, "publish_year": 2021}, "venue": "NeurIPS"},
            {"paper_id": "SupCon", "title": SUPCON_TITLE, "author": ["Prannay Khosla"], "year": {"preprint_year": 2020, "publish_year": 2020}, "venue": "NeurIPS"},
            {"paper_id": "LSTM", "title": "Long Short-Term Memory", "author": ["Sepp Hochreiter"], "year": {"preprint_year": None, "publish_year": 1997}, "venue": "Neural Computation"},
            {"paper_id": "OldArxiv", "title": ARXIV_2013_TITLE, "author": ["Early Visioner"], "year": {"preprint_year": 2013, "publish_year": None}, "venue": None},
        ],
        "edges": [
            {"source_paper_id": "ATTN", "target_paper_id": "VIT", "ref_index": 1, "raw_text": "ViT", "page": 3, "source_block_id": "b3", "match_type": "canonical_title"},
            {"source_paper_id": "ATTN", "target_paper_id": "ResNet", "ref_index": 2, "raw_text": "ResNet", "page": 3, "source_block_id": "b3", "match_type": "canonical_title"},
            {"source_paper_id": "SupCon", "target_paper_id": "ResNet", "ref_index": 4, "raw_text": "ResNet", "page": 5, "source_block_id": "b5", "match_type": "canonical_title"},
            {"source_paper_id": "OldArxiv", "target_paper_id": "LSTM", "ref_index": 7, "raw_text": "LSTM", "page": 2, "source_block_id": "b2", "match_type": "canonical_title"},
        ],
    }, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
