from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper_rag.config import Settings
from paper_rag.retrieval.domains.common.errors import PlanParseError
from paper_rag.retrieval.data.citation_scope import citation_scope_paper_ids
from paper_rag.retrieval.domains.common.paper_resolver import resolve_parser_papers
from paper_rag.retrieval.domains.metadata.planner import plan_metadata
from paper_rag.retrieval.domains.metadata.schema import validate_metadata_parse
from paper_rag.retrieval.route import RouteDecision


RESNET_TITLE = "Deep Residual Learning for Image Recognition"
BERT_TITLE = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
SUPCON_TITLE = "Supervised Contrastive Learning"


class MetadataSchemaTests(unittest.TestCase):
    def test_accepts_current_metadata_shape(self) -> None:
        payload = validate_metadata_parse({
            "intent": "lookup",
            "return_fields": ["author", "year"],
            "paper_semantic": "",
            "filters": [{"field": "paper", "op": "=", "value": "ResNet", "negated": False}],
            "paper_groups": [],
            "group_mode": "single",
        })

        self.assertEqual(payload["intent"], "lookup")
        self.assertEqual(payload["return_fields"], ["author", "year"])
        self.assertEqual(payload["filters"][0]["field"], "paper")

    def test_rejects_old_metadata_shape(self) -> None:
        with self.assertRaises(PlanParseError):
            validate_metadata_parse({
                "intent": "lookup",
                "return_field": "author",
                "anchors": ["ResNet"],
                "filters": [],
            })

    def test_rejects_invalid_return_field(self) -> None:
        with self.assertRaises(PlanParseError):
            validate_metadata_parse({
                "intent": "lookup",
                "return_fields": ["abstract"],
                "paper_semantic": "",
                "filters": [],
                "paper_groups": [],
                "group_mode": "single",
            })

    def test_rejects_and_mode_outside_exists(self) -> None:
        with self.assertRaises(PlanParseError):
            validate_metadata_parse({
                "intent": "list",
                "return_fields": ["title"],
                "paper_semantic": "",
                "filters": [],
                "paper_groups": [{"semantic": "", "filters": [{"field": "venue", "op": "=", "value": "CVPR", "negated": False}]}],
                "group_mode": "and",
            })


class MetadataResolverTests(unittest.TestCase):
    def test_resolves_paper_aliases_in_filters_and_interval_bounds(self) -> None:
        with metadata_fixture() as settings:
            resolved = resolve_parser_papers(settings, {
                "filters": [{"field": "paper", "op": "=", "value": "ResNet", "negated": False}],
                "paper_groups": [
                    {
                        "semantic": "",
                        "filters": [{"field": "year", "op": "interval", "value": ["ResNet", "inf"], "negated": False}],
                    }
                ],
            })

        self.assertEqual(resolved["filters"][0]["value"], RESNET_TITLE)
        self.assertEqual(resolved["paper_groups"][0]["filters"][0]["value"][0], RESNET_TITLE)
        self.assertEqual(resolved["alias_matches"][0].alias, "ResNet")
        self.assertEqual(resolved["resolved_papers"][0]["title"], RESNET_TITLE)


class CitationScopeTests(unittest.TestCase):
    def test_follow_and_prior_use_local_citation_graph(self) -> None:
        with metadata_fixture() as settings:
            self.assertEqual(citation_scope_paper_ids(settings, [RESNET_TITLE], "follow"), {"SupCon"})
            self.assertEqual(citation_scope_paper_ids(settings, [SUPCON_TITLE], "prior"), {"ResNet"})

    def test_missing_graph_or_unknown_target_returns_empty_scope(self) -> None:
        with metadata_fixture(write_graph=False) as settings:
            self.assertEqual(citation_scope_paper_ids(settings, [RESNET_TITLE], "follow"), set())
        with metadata_fixture() as settings:
            self.assertEqual(citation_scope_paper_ids(settings, ["Unknown Paper"], "follow"), set())


class MetadataPlannerTests(unittest.TestCase):
    def test_lookup_returns_requested_values(self) -> None:
        with metadata_fixture() as settings:
            route = RouteDecision(
                route="metadata",
                intent="lookup",
                return_fields=["author", "year"],
                filters=[{"field": "paper", "op": "=", "value": RESNET_TITLE, "negated": False}],
            )
            evidence = plan_metadata(settings, route, [])

        self.assertEqual(evidence["records"][0]["title"], RESNET_TITLE)
        self.assertEqual(evidence["records"][0]["values"]["author"], ["Kaiming He"])
        self.assertEqual(evidence["records"][0]["values"]["year"]["preprint_year"], 2015)

    def test_count_uses_manifest_filters(self) -> None:
        with metadata_fixture() as settings:
            route = RouteDecision(
                route="metadata",
                intent="count",
                filters=[{"field": "venue", "op": "=", "value": "CVPR", "negated": False}],
            )
            evidence = plan_metadata(settings, route, [])

        self.assertEqual(evidence["count"], 1)
        self.assertEqual(evidence["records"][0]["title"], RESNET_TITLE)
        self.assertEqual(evidence["records"][0]["venue"], "CVPR")

    def test_venue_aliases_match_canonical_aliases_and_display_values(self) -> None:
        with metadata_fixture() as settings:
            counts = []
            for venue in ["NeurIPS", "NIPS", "Neural Information Processing Systems"]:
                route = RouteDecision(
                    route="metadata",
                    intent="count",
                    filters=[{"field": "venue", "op": "=", "value": venue, "negated": False}],
                )
                evidence = plan_metadata(settings, route, [])
                counts.append(evidence["count"])
                self.assertEqual(evidence["records"][0]["venue"], "NeurIPS")

        self.assertEqual(counts, [1, 1, 1])

    def test_venue_in_and_negation_use_alias_expansion(self) -> None:
        with metadata_fixture() as settings:
            route = RouteDecision(
                route="metadata",
                intent="list",
                return_fields=["title"],
                filters=[{"field": "venue", "op": "in", "value": ["CVPR", "NIPS"], "negated": False}],
            )
            evidence = plan_metadata(settings, route, [])
            titles = {record["title"] for record in evidence["records"]}
            self.assertEqual(titles, {RESNET_TITLE, SUPCON_TITLE})

            route = RouteDecision(
                route="metadata",
                intent="list",
                return_fields=["title"],
                filters=[{"field": "venue", "op": "=", "value": "NIPS", "negated": True}],
            )
            evidence = plan_metadata(settings, route, [])
            titles = {record["title"] for record in evidence["records"]}
            self.assertEqual(titles, {RESNET_TITLE, BERT_TITLE})

    def test_follow_prior_filters_limit_metadata_scope(self) -> None:
        with metadata_fixture() as settings:
            follow_route = RouteDecision(
                route="metadata",
                intent="list",
                return_fields=["title"],
                filters=[{"field": "paper", "op": "follow", "value": RESNET_TITLE, "negated": False}],
            )
            follow_evidence = plan_metadata(settings, follow_route, [])
            self.assertEqual([record["title"] for record in follow_evidence["records"]], [SUPCON_TITLE])

            prior_route = RouteDecision(
                route="metadata",
                intent="list",
                return_fields=["title"],
                filters=[{"field": "paper", "op": "prior", "value": SUPCON_TITLE, "negated": False}],
            )
            prior_evidence = plan_metadata(settings, prior_route, [])
            self.assertEqual([record["title"] for record in prior_evidence["records"]], [RESNET_TITLE])

    def test_paper_semantic_uses_annotation_tags_for_candidate_recall(self) -> None:
        with metadata_fixture() as settings:
            route = RouteDecision(
                route="metadata",
                intent="list",
                return_fields=["title"],
                paper_semantic="残差连接",
            )
            evidence = plan_metadata(settings, route, [])
            self.assertEqual([record["title"] for record in evidence["records"]], [RESNET_TITLE])

            route = RouteDecision(
                route="metadata",
                intent="list",
                return_fields=["title"],
                paper_semantic="CNN",
            )
            evidence = plan_metadata(settings, route, [])
            self.assertEqual({record["title"] for record in evidence["records"]}, {RESNET_TITLE, SUPCON_TITLE})

    def test_title_semantic_recall_is_still_available(self) -> None:
        with metadata_fixture() as settings:
            route = RouteDecision(
                route="metadata",
                intent="list",
                return_fields=["title"],
                paper_semantic=RESNET_TITLE,
            )
            evidence = plan_metadata(settings, route, [])
            self.assertEqual([record["title"] for record in evidence["records"]], [RESNET_TITLE])

    def test_metadata_filters_narrow_semantic_tag_candidates(self) -> None:
        with metadata_fixture() as settings:
            route = RouteDecision(
                route="metadata",
                intent="list",
                return_fields=["title"],
                paper_semantic="CNN",
                filters=[{"field": "venue", "op": "=", "value": "CVPR", "negated": False}],
            )
            evidence = plan_metadata(settings, route, [])
            self.assertEqual([record["title"] for record in evidence["records"]], [RESNET_TITLE])

    def test_exists_and_requires_every_group_to_match(self) -> None:
        with metadata_fixture() as settings:
            route = RouteDecision(
                route="metadata",
                intent="exists",
                filters=[{"field": "venue", "op": "=", "value": "CVPR", "negated": False}],
                paper_groups=[
                    {"semantic": "", "filters": [{"field": "paper", "op": "=", "value": RESNET_TITLE, "negated": False}]},
                    {"semantic": "", "filters": [{"field": "paper", "op": "=", "value": BERT_TITLE, "negated": False}]},
                ],
                group_mode="and",
            )
            evidence = plan_metadata(settings, route, [])

        self.assertFalse(evidence["exists"])
        self.assertEqual(evidence["group_results"][0]["count"], 1)
        self.assertEqual(evidence["group_results"][1]["count"], 0)

    def test_per_mode_keeps_group_results_separate(self) -> None:
        with metadata_fixture() as settings:
            route = RouteDecision(
                route="metadata",
                intent="list",
                return_fields=["title"],
                paper_groups=[
                    {"semantic": "", "filters": [{"field": "year", "op": "=", "value": 2016, "negated": False}]},
                    {"semantic": "", "filters": [{"field": "year", "op": "=", "value": 2020, "negated": False}]},
                ],
                group_mode="per",
            )
            evidence = plan_metadata(settings, route, [])

        self.assertEqual([group["count"] for group in evidence["group_results"]], [1, 1])
        self.assertEqual(evidence["group_results"][0]["records"][0]["title"], RESNET_TITLE)
        self.assertEqual(evidence["group_results"][1]["records"][0]["title"], SUPCON_TITLE)

    def test_year_filters_use_preprint_year_inside_arxiv_scope(self) -> None:
        with metadata_fixture() as settings:
            route = RouteDecision(
                route="metadata",
                intent="list",
                return_fields=["title"],
                filters=[
                    {"field": "venue", "op": "=", "value": "ArXiv", "negated": False},
                    {"field": "year", "op": "=", "value": 2018, "negated": False},
                ],
            )
            arxiv_evidence = plan_metadata(settings, route, [])

            route = RouteDecision(
                route="metadata",
                intent="list",
                return_fields=["title"],
                filters=[{"field": "year", "op": "=", "value": 2018, "negated": False}],
            )
            published_evidence = plan_metadata(settings, route, [])

        self.assertEqual([record["title"] for record in arxiv_evidence["records"]], [BERT_TITLE])
        self.assertEqual(published_evidence["records"], [])


class metadata_fixture:
    def __init__(self, *, write_graph: bool = True) -> None:
        self.write_graph = write_graph

    def __enter__(self) -> Settings:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        data = root / "data"
        data.mkdir()
        write_manifest(data / "manifest.jsonl")
        write_venue_aliases(data / "venue_aliases.json")
        if self.write_graph:
            write_citation_graph(data / "paper_data" / "citation_graph.json")
        (data / "paper_annotations.json").write_text(json.dumps({
            "resnet": {
                "title": RESNET_TITLE,
                "aliases": ["ResNet"],
                "tags": {
                    "zh": ["卷积神经网络", "残差连接", "图像分类"],
                    "en": ["CNN", "residual connection", "image classification"],
                },
            },
            "bert": {
                "title": BERT_TITLE,
                "aliases": ["BERT"],
                "tags": {
                    "zh": ["预训练", "语言理解"],
                    "en": ["transformer", "language understanding"],
                },
            },
            "supcon": {
                "title": SUPCON_TITLE,
                "aliases": ["SupCon"],
                "tags": {
                    "zh": ["对比学习"],
                    "en": ["CNN", "contrastive learning"],
                },
            },
        }, ensure_ascii=False), encoding="utf-8")
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
            "file_hash": "bert",
            "status": "active",
            "title": BERT_TITLE,
            "author": ["Jacob Devlin"],
            "year": {"preprint_year": 2018, "publish_year": 2019},
            "venue": "NAACL",
            "paper_data_path": str(path.parent / "paper_data" / "BERT"),
            "pdf_path": str(path.parent / "pdf" / "bert.pdf"),
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
    ]
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")


def write_venue_aliases(path: Path) -> None:
    path.write_text(json.dumps([
        {
            "canonical": "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
            "display": "CVPR",
            "aliases": [
                "CVPR",
                "IEEE Conference on Computer Vision and Pattern Recognition",
                "Computer Vision and Pattern Recognition",
            ],
        },
        {
            "canonical": "Conference on Neural Information Processing Systems",
            "display": "NeurIPS",
            "aliases": [
                "NeurIPS",
                "Neural Information Processing Systems",
                "NIPS",
                "Advances in Neural Information Processing Systems",
            ],
        },
    ], ensure_ascii=False), encoding="utf-8")


def write_citation_graph(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "version": 1,
        "nodes": [
            {"paper_id": "ResNet", "title": RESNET_TITLE},
            {"paper_id": "BERT", "title": BERT_TITLE},
            {"paper_id": "SupCon", "title": SUPCON_TITLE},
        ],
        "edges": [
            {"source_paper_id": "SupCon", "target_paper_id": "ResNet"},
        ],
    }, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    unittest.main()
