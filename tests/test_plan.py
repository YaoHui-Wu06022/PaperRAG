from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from paper_rag.config import Settings
from paper_rag.answer import run_ask
from paper_rag.retrieval.dense.milvus_store import SearchResult
from paper_rag.retrieval.domains.metadata.router import build_metadata_decision
from paper_rag.retrieval.planner import prepare_query, run_plan
from paper_rag.retrieval.domains.common.errors import PlanParseError
from paper_rag.retrieval.domains.metadata.schema import validate_metadata_parse
from paper_rag.retrieval.domains.reference.router import build_reference_decision
from paper_rag.retrieval.domains.reference.schema import validate_reference_parse
from paper_rag.retrieval.top_router import route_query


class StaticMetadataParser:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[str] = []

    def parse_metadata(self, query: str) -> dict:
        self.calls.append(query)
        payload = dict(self.payload)
        payload.setdefault("anchors", [])
        payload.setdefault("filters", [])
        return validate_metadata_parse(payload, query)


class StaticReferenceParser:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[str] = []

    def parse_reference(self, query: str) -> dict:
        self.calls.append(query)
        payload = dict(self.payload)
        payload.setdefault("anchors", [])
        payload.setdefault("anchor_mode", "per")
        payload.setdefault("filters", [])
        return validate_reference_parse(payload, query)


def metadata_lookup(field: str, title: str) -> StaticMetadataParser:
    return StaticMetadataParser({
        "intent": "lookup",
        "return_field": field,
        "filters": [{"field": "title", "op": "contains", "value": title, "negated": False}],
    })


def metadata_list(filters: list[dict]) -> StaticMetadataParser:
    return StaticMetadataParser({
        "intent": "list",
        "return_field": None,
        "filters": filters,
    })


def reference_payload(
    direction: str | None,
    anchors: list[str],
    *,
    intent: str = "list",
    anchor_mode: str = "per",
    filters: list[dict] | None = None,
) -> StaticReferenceParser:
    return StaticReferenceParser({
        "intent": intent,
        "direction": direction,
        "anchors": anchors,
        "anchor_mode": anchor_mode,
        "filters": filters or [],
    })


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return [[0.1, 0.2]]


class FakeStore:
    def __init__(self) -> None:
        self.top_k: int | None = None

    def search(self, query_vector: list[float], top_k: int) -> list[SearchResult]:
        self.top_k = top_k
        return [
            SearchResult(
                score=0.91,
                chunk_id="Center_Loss_abc::chunk_0000",
                paper_id="Center_Loss_abc",
                chunk_index=0,
                title="A Discriminative Feature Learning Approach for Deep Face Recognition",
                section_path_text="Abstract",
                pages_text="1",
                text="Dense text",
            ),
            SearchResult(
                score=0.71,
                chunk_id="Center_Loss_abc::chunk_0001",
                paper_id="Center_Loss_abc",
                chunk_index=1,
                title="A Discriminative Feature Learning Approach for Deep Face Recognition",
                section_path_text="1 Introduction",
                pages_text="2",
                text="Dense text",
            ),
        ]


class PlanTests(unittest.TestCase):
    def test_router_uses_explicit_routes_before_body_default(self) -> None:
        self.assertEqual(route_query("这篇论文是哪一年发表的").route, "metadata")
        self.assertEqual(route_query("这篇论文是哪一年发表的").intent, None)
        self.assertEqual(route_query("这篇论文引用了哪些论文").route, "reference")
        self.assertEqual(route_query("哪些论文引用了ResNet").route, "reference")
        self.assertEqual(route_query("BERT的参考文献有哪些").route, "reference")
        self.assertEqual(route_query("how does center loss improve compactness").route, "content")

    def test_rule_router_keeps_reference_as_top_level_entry_only(self) -> None:
        self.assertEqual(route_query("哪些论文引用了ResNet").route, "reference")
        self.assertEqual(route_query("哪些2019年的论文引用了ResNet").route, "reference")
        self.assertEqual(route_query("ResNet引用了哪些论文").route, "reference")
        self.assertEqual(route_query("ResNet的参考文献有哪些").route, "reference")
        self.assertEqual(route_query("BERT被哪些论文引用").route, "reference")
        self.assertEqual(route_query("有多少论文引用了BERT").route, "reference")
        self.assertEqual(route_query("哪些论文把BERT作为参考文献").route, "reference")
        self.assertEqual(route_query("ResNet的引文关系").route, "reference")

    def test_router_uses_token_boundaries(self) -> None:
        self.assertEqual(route_query("recite the result").route, "content")
        self.assertEqual(route_query("authorization method").route, "content")
        self.assertEqual(route_query("BERT是哪一年发表的").route, "metadata")

    def test_metadata_router_field_routes(self) -> None:
        self.assertEqual(route_query("BERT是谁写的").route, "metadata")
        self.assertEqual(route_query("BERT的作者是谁").route, "metadata")
        self.assertEqual(route_query("BERT是哪一年发表的").route, "metadata")
        self.assertEqual(route_query("BERT发表在哪个期刊").route, "metadata")
        self.assertEqual(route_query("BERT发表在哪个会议").route, "metadata")
        self.assertEqual(route_query("BERT的标题是什么").route, "metadata")

    def test_metadata_router_paper_list_filters(self) -> None:
        by_year = route_query("哪些论文在2015-2019发表")
        self.assertEqual(by_year.route, "metadata")
        self.assertEqual(by_year.intent, None)
        self.assertEqual(by_year.filters, [])
        by_author = route_query("哪些论文是Kaiming He写的")
        self.assertEqual(by_author.route, "metadata")
        self.assertEqual(by_author.intent, None)
        self.assertEqual(by_author.filters, [])

    def test_reference_route_keeps_original_query(self) -> None:
        query = "哪些论文引用了RESNET"
        decision = route_query(query)
        self.assertEqual(decision.route, "reference")
        self.assertEqual(decision.query, query)

    def test_reference_parser_schema_normalizes_valid_payload(self) -> None:
        payload = validate_reference_parse({
            "intent": "count",
            "direction": "cited_by",
            "anchors": ["ResNet"],
            "anchor_mode": "and",
            "filters": [{"field": "year", "op": "interval", "value": [2015, "inf"], "negated": False}],
        })
        self.assertEqual(payload["intent"], "count")
        self.assertEqual(payload["direction"], "cited_by")
        self.assertEqual(payload["anchor_mode"], "and")
        self.assertEqual(payload["filters"][0]["value"], [2015, "inf"])

    def test_reference_parser_schema_normalizes_missing_or_null_filters(self) -> None:
        missing_filters = validate_reference_parse({
            "intent": "list",
            "direction": "cites",
            "anchors": ["ResNet"],
            "anchor_mode": "per",
        })
        null_filters = validate_reference_parse({
            "intent": "list",
            "direction": "cites",
            "anchors": ["ResNet"],
            "anchor_mode": "per",
            "filters": None,
        })
        self.assertEqual(missing_filters["filters"], [])
        self.assertEqual(null_filters["filters"], [])

    def test_reference_parser_rejects_null_intent(self) -> None:
        with self.assertRaises(PlanParseError):
            validate_reference_parse({
                "intent": None,
                "direction": "cites",
                "anchors": ["ResNet"],
                "anchor_mode": "per",
                "filters": [],
            })

    def test_reference_parser_ignores_unknown_fields(self) -> None:
        payload = validate_reference_parse({
            "intent": "list",
            "direction": "cites",
            "anchors": ["ResNet"],
            "anchor_mode": "per",
            "filters": [],
            "raw_query": "ResNet的参考文献",
            "router": "reference",
            "extra": "ignored",
        })
        self.assertEqual(payload["intent"], "list")
        self.assertEqual(payload["anchors"], ["ResNet"])

    def test_metadata_anchor_interval_updates_parser_result_filters(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            decision = build_metadata_decision(
                settings,
                route_query("ResNet以后还有哪些论文"),
                "ResNet以后还有哪些论文",
                warnings=[],
                plan_parser=StaticMetadataParser({
                    "intent": "list",
                    "return_field": None,
                    "anchors": ["ResNet"],
                    "filters": [{"field": "year", "op": "interval", "value": ["ResNet", "inf"], "negated": False}],
                }),
            )
            self.assertEqual(decision.filters[0]["value"], [2016, "inf"])
            self.assertEqual(decision.parser_result["filters"][0]["value"], [2016, "inf"])

    def test_reference_anchor_interval_updates_parser_result_filters(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            decision = build_reference_decision(
                settings,
                route_query("2015到2020年之间引用了ResNet的论文有哪些"),
                "2015到2020年之间引用了ResNet的论文有哪些",
                warnings=[],
                plan_parser=StaticReferenceParser({
                    "intent": "list",
                    "direction": "cited_by",
                    "anchors": ["ResNet"],
                    "anchor_mode": "per",
                    "filters": [{"field": "year", "op": "interval", "value": ["ResNet", "inf"], "negated": False}],
                }),
            )
            self.assertEqual(decision.filters[0]["value"], [2016, "inf"])
            self.assertEqual(decision.parser_result["filters"][0]["value"], [2016, "inf"])

    def test_metadata_parser_schema_allows_filter_only_queries(self) -> None:
        payload = validate_metadata_parse({
            "intent": "list",
            "return_field": None,
            "filters": [
                {"field": "year", "op": "interval", "value": [2015, 2020], "negated": False},
                {"field": "venue", "op": "contains", "value": "CVPR", "negated": True},
            ],
        })
        self.assertEqual(payload["anchors"], [])
        self.assertEqual(payload["filters"][0]["value"], [2015, 2020])
        self.assertTrue(payload["filters"][1]["negated"])

    def test_metadata_parser_schema_normalizes_missing_or_null_filters(self) -> None:
        missing_filters = validate_metadata_parse({
            "intent": "lookup",
            "return_field": "year",
            "anchors": ["BERT"],
        })
        null_filters = validate_metadata_parse({
            "intent": "lookup",
            "return_field": "year",
            "anchors": ["BERT"],
            "filters": None,
        })
        self.assertEqual(missing_filters["filters"], [])
        self.assertEqual(null_filters["filters"], [])

    def test_metadata_parser_schema_ignores_blank_anchors(self) -> None:
        payload = validate_metadata_parse({
            "intent": "list",
            "return_field": None,
            "anchors": [""],
            "filters": [{"field": "year", "op": "interval", "value": [2015, 2020], "negated": False}],
        })
        self.assertEqual(payload["anchors"], [])
        self.assertEqual(payload["filters"][0]["value"], [2015, 2020])

    def test_metadata_parser_ignores_unknown_fields(self) -> None:
        payload = validate_metadata_parse({
            "router": "metadata",
            "raw_query": "BERT是谁写的",
            "intent": "lookup",
            "return_field": "year",
            "anchors": ["BERT"],
            "filters": [],
            "extra": "ignored",
        })
        self.assertEqual(payload["return_field"], "year")
        self.assertEqual(payload["anchors"], ["BERT"])

    def test_reference_parser_schema_rejects_non_list_filters(self) -> None:
        with self.assertRaises(PlanParseError):
            validate_reference_parse({
                "intent": "list",
                "direction": "cites",
                "anchors": ["ResNet"],
                "anchor_mode": "per",
                "filters": {"field": "year", "op": "=", "value": 2019, "negated": False},
            })

    def test_reference_parser_schema_rejects_invalid_anchor_field(self) -> None:
        with self.assertRaises(PlanParseError):
            validate_reference_parse({
                "intent": "list",
                "direction": "cites",
                "anchors": [{"field": "author", "value": "Kaiming He"}],
                "anchor_mode": "per",
                "filters": [],
            })

    def test_prepare_query_keeps_original_query_without_translation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings.load(Path(tmp))
            prepared = prepare_query(settings, "中心损失如何提升类内紧凑性")
            self.assertEqual(prepared.original_query, "中心损失如何提升类内紧凑性")
            self.assertEqual(prepared.warnings, [])
            self.assertIsNone(prepared.error)

    def test_metadata_route_reads_manifest_without_retrieval(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "Center Loss 的作者是谁",
                plan_parser=metadata_lookup("author", "Center loss"),
            )
            self.assertEqual(pack["route"], "metadata")
            self.assertEqual(pack["intent"], "lookup")
            self.assertNotIn("sub_route", pack)
            self.assertNotIn("scope", pack["evidence"])
            self.assertNotIn("expanded_query", pack["evidence"])
            self.assertNotIn("query", pack["evidence"])
            self.assertNotIn("parser_result", pack["evidence"])
            self.assertNotIn("target_papers", pack["evidence"])
            self.assertNotIn("top_route", pack["evidence"])
            self.assertNotIn("anchors", pack["evidence"])
            self.assertEqual(pack["evidence"]["intent"], "lookup")
            self.assertEqual(pack["evidence"]["return_field"], "author")
            records = pack["evidence"]["records"]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["year"], {"preprint_year": None, "publish_year": 2016})
            self.assertEqual(records[0]["venue"], "ECCV")
            self.assertEqual(records[0]["value"], ["Yandong Wen", "Kaipeng Zhang"])
            self.assertIn("pdf_path", records[0])
            self.assertNotIn("file_hash", records[0])
            self.assertNotIn("paper_data_path", records[0])

    def test_chinese_metadata_route_uses_original_query_for_routing(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "Center Loss 是哪一年发表的",
                plan_parser=metadata_lookup("year", "Center loss"),
            )
            self.assertEqual(pack["route"], "metadata")
            self.assertEqual(pack["intent"], "lookup")
            self.assertNotIn("parser_result", pack["evidence"])
            self.assertNotIn("language", pack)
            self.assertNotIn("retrieval_query", pack)

    def test_metadata_paper_list_filters_manifest_records(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            by_year = run_plan(
                settings,
                "哪些论文在2015-2019发表",
                plan_parser=metadata_list([
                    {"field": "year", "op": "interval", "value": [2015, 2019], "negated": False},
                ]),
            )
            self.assertEqual(by_year["route"], "metadata")
            self.assertEqual(by_year["intent"], "list")
            self.assertEqual(by_year["evidence"]["filters"], [{"field": "year", "op": "interval", "value": [2015, 2019], "negated": False}])
            self.assertEqual(len(by_year["evidence"]["records"]), 2)
            by_author = run_plan(
                settings,
                "哪些论文是Kaiming He写的",
                plan_parser=metadata_list([
                    {"field": "author", "op": "=", "value": "Kaiming He", "negated": False},
                ]),
            )
            self.assertEqual(len(by_author["evidence"]["records"]), 1)
            self.assertEqual(by_author["evidence"]["records"][0]["title"], "Deep Residual Learning for Image Recognition")
            short_author = run_plan(
                settings,
                "哪些论文是He写的",
                plan_parser=metadata_list([
                    {"field": "author", "op": "=", "value": "He", "negated": False},
                ]),
            )
            self.assertEqual(short_author["evidence"]["records"], [])

    def test_metadata_parser_grouped_count_and_negated_filters(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            grouped = run_plan(
                settings,
                "Center Loss 和 ResNet 的作者分别是谁",
                plan_parser=StaticMetadataParser({
                    "intent": "lookup",
                    "return_field": "author",
                    "filters": [{"field": "title", "op": "in", "value": ["Center Loss", "ResNet"], "negated": False}],
                }),
            )
            self.assertEqual(grouped["intent"], "lookup")
            self.assertNotIn("groups", grouped["evidence"])
            self.assertEqual(len(grouped["evidence"]["records"]), 2)
            count = run_plan(
                settings,
                "2016年发表了多少篇论文",
                plan_parser=StaticMetadataParser({
                    "intent": "count",
                    "return_field": None,
                    "filters": [{"field": "year", "op": "=", "value": 2016, "negated": False}],
                }),
            )
            self.assertEqual(count["intent"], "count")
            self.assertEqual(count["evidence"]["count"], 1)
            not_cvpr = run_plan(
                settings,
                "哪些论文不是发表在CVPR",
                plan_parser=StaticMetadataParser({
                    "intent": "list",
                    "return_field": None,
                    "filters": [{"field": "venue", "op": "contains", "value": "CVPR", "negated": True}],
                }),
            )
            self.assertEqual([record["venue"] for record in not_cvpr["evidence"]["records"]], ["ECCV"])

    def test_metadata_venue_filter_uses_aliases(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "哪些论文发表在CVPR",
                plan_parser=metadata_list([
                    {"field": "venue", "op": "contains", "value": "CVPR", "negated": False},
                ]),
            )
            titles = [record["title"] for record in pack["evidence"]["records"]]
            self.assertEqual(titles, ["Deep Residual Learning for Image Recognition"])

    def test_ask_displays_canonical_venue_alias(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            result = run_ask(
                settings,
                "ResNet发表在哪个会议",
                plan_parser=metadata_lookup("venue", "ResNet"),
            )
            self.assertIn("发表在 CVPR", result.answer)

    def test_metadata_parser_failure_returns_parse_failed_evidence(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "Center Loss 是谁写的",
                plan_parser=StaticMetadataParser({
                    "intent": "made_up",
                    "return_field": "author",
                }),
            )
            self.assertEqual(pack["route"], "metadata")
            self.assertIsNone(pack["intent"])
            self.assertEqual(pack["evidence"]["parse_status"], "parse_failed")
            self.assertEqual(pack["evidence"]["records"], [])
            self.assertNotIn("parser_result", pack["evidence"])
            self.assertIn("metadata_parse_failed", pack["warnings"][0])

    def test_metadata_parser_accepts_string_null_return_field(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "哪些论文发表在arXiv",
                plan_parser=StaticMetadataParser({
                    "intent": "list",
                    "return_field": "null",
                    "filters": [{"field": "venue", "op": "contains", "value": "arXiv", "negated": False}],
                }),
            )
            self.assertEqual(pack["route"], "metadata")
            self.assertEqual(pack["intent"], "list")
            self.assertEqual(pack["evidence"]["parse_status"], "ok")

    def test_metadata_parser_rejects_legacy_target_field(self) -> None:
        with self.assertRaises(PlanParseError):
            validate_metadata_parse({
                "intent": "lookup",
                "target_field": "author",
                "filters": [],
            })

    def test_metadata_parser_rejects_unknown_intent(self) -> None:
        with self.assertRaises(PlanParseError):
            validate_metadata_parse({
                "intent": "unknown",
                "return_field": None,
                "filters": [],
            })

    def test_metadata_parser_accepts_interval_numeric_list(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "哪些论文在2015到2020年发表",
                plan_parser=StaticMetadataParser({
                    "intent": "list",
                    "return_field": None,
                    "filters": [{"field": "year", "op": "interval", "value": [2015, 2020], "negated": False}],
                }),
            )
            self.assertEqual(pack["route"], "metadata")
            self.assertEqual(pack["evidence"]["filters"][0]["value"], [2015, 2020])

    def test_metadata_parser_accepts_open_ended_interval_upper_bound(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "哪些论文在2015年以后发表",
                plan_parser=StaticMetadataParser({
                    "intent": "list",
                    "return_field": None,
                    "filters": [{"field": "year", "op": "interval", "value": [2015, "inf"], "negated": False}],
                }),
            )
            self.assertEqual(pack["route"], "metadata")
            self.assertEqual(pack["evidence"]["filters"][0]["value"], [2015, "inf"])
            self.assertEqual(len(pack["evidence"]["records"]), 2)

    def test_metadata_parser_accepts_open_ended_interval_lower_bound(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "哪些论文在2019年以前发表",
                plan_parser=StaticMetadataParser({
                    "intent": "list",
                    "return_field": None,
                    "filters": [{"field": "year", "op": "interval", "value": ["-inf", 2019], "negated": False}],
                }),
            )
            self.assertEqual(pack["route"], "metadata")
            self.assertEqual(pack["evidence"]["filters"][0]["value"], ["-inf", 2019])
            self.assertEqual(len(pack["evidence"]["records"]), 2)

    def test_chinese_metadata_paper_list_uses_original_query(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            by_year = run_plan(
                settings,
                "哪些论文在2015-2019发表",
                plan_parser=metadata_list([
                    {"field": "year", "op": "interval", "value": [2015, 2019], "negated": False},
                ]),
            )
            self.assertEqual(by_year["route"], "metadata")
            self.assertEqual(by_year["intent"], "list")
            self.assertEqual(by_year["evidence"]["filters"], [{"field": "year", "op": "interval", "value": [2015, 2019], "negated": False}])
            full_author = run_plan(
                settings,
                "哪些论文是Kaiming He写的",
                plan_parser=metadata_list([
                    {"field": "author", "op": "=", "value": "Kaiming He", "negated": False},
                ]),
            )
            self.assertEqual(len(full_author["evidence"]["records"]), 1)
            self.assertEqual(full_author["evidence"]["records"][0]["title"], "Deep Residual Learning for Image Recognition")
            short_author = run_plan(
                settings,
                "哪些论文是He写的",
                plan_parser=metadata_list([
                    {"field": "author", "op": "=", "value": "He", "negated": False},
                ]),
            )
            self.assertEqual(short_author["evidence"]["records"], [])

    def test_metadata_relative_year_query_uses_anchor_effective_year(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "Resnet以后发表的论文有哪些",
                plan_parser=StaticMetadataParser({
                    "intent": "list",
                    "return_field": None,
                    "anchors": ["ResNet"],
                    "filters": [{"field": "year", "op": "interval", "value": ["ResNet", "inf"], "negated": False}],
                }),
            )
            self.assertEqual(pack["route"], "metadata")
            self.assertEqual(pack["evidence"]["filters"], [
                {"field": "year", "op": "interval", "value": [2016, "inf"], "negated": False},
            ])
            titles = [record["title"] for record in pack["evidence"]["records"]]
            self.assertIn("A Discriminative Feature Learning Approach for Deep Face Recognition", titles)
            self.assertNotIn("Deep Residual Learning for Image Recognition", titles)

    def test_metadata_between_anchors_resolves_interval_bounds(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "ResNet和Center Loss之间发表的论文有哪些",
                plan_parser=StaticMetadataParser({
                    "intent": "list",
                    "return_field": None,
                    "anchors": ["Center Loss", "ResNet"],
                    "filters": [{"field": "year", "op": "interval", "value": ["Center Loss", "ResNet"], "negated": False}],
                }),
            )
            self.assertEqual(pack["route"], "metadata")
            self.assertEqual(pack["evidence"]["filters"], [
                {"field": "year", "op": "interval", "value": [2015, 2016], "negated": False},
            ])
            titles = [record["title"] for record in pack["evidence"]["records"]]
            self.assertIn("Deep Residual Learning for Image Recognition", titles)
            self.assertIn("A Discriminative Feature Learning Approach for Deep Face Recognition", titles)

    def test_metadata_multiple_anchor_interval_uses_min_max_years(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "ResNet和Center Loss之间发表的论文有哪些",
                plan_parser=StaticMetadataParser({
                    "intent": "list",
                    "return_field": None,
                    "anchors": ["ResNet", "Center Loss"],
                    "filters": [{"field": "year", "op": "interval", "value": ["ResNet", "Center Loss"], "negated": False}],
                }),
            )
            self.assertEqual(pack["evidence"]["filters"], [
                {"field": "year", "op": "interval", "value": [2015, 2016], "negated": False},
            ])

    def test_metadata_year_intervals_are_merged(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "ResNet之后2019年以前发表的论文有哪些",
                plan_parser=StaticMetadataParser({
                    "intent": "list",
                    "return_field": None,
                    "anchors": ["ResNet"],
                    "filters": [
                        {"field": "year", "op": "interval", "value": ["ResNet", "inf"], "negated": False},
                        {"field": "year", "op": "interval", "value": ["-inf", 2019], "negated": False},
                    ],
                }),
            )
            self.assertEqual(pack["evidence"]["filters"], [
                {"field": "year", "op": "interval", "value": [2016, 2019], "negated": False},
            ])

    def test_ask_metadata_lookup_formats_answer(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            result = run_ask(
                settings,
                "Center Loss 是谁写的",
                plan_parser=metadata_lookup("author", "Center loss"),
            )
            self.assertEqual(result.route, "metadata")
            self.assertIn("作者是", result.answer)
            self.assertIn("Yandong Wen", result.answer)
            self.assertIn("Kaipeng Zhang", result.answer)
            self.assertIn("plan(metadata)", result.provenance[0])

    def test_ask_metadata_list_formats_answer_in_chinese(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            result = run_ask(
                settings,
                "哪些论文在2016发表",
                plan_parser=metadata_list([
                    {"field": "year", "op": "=", "value": 2016, "negated": False},
                ]),
            )
            self.assertEqual(result.route, "metadata")
            self.assertIn("共找到 1 篇论文", result.answer)
            self.assertIn("A Discriminative Feature Learning Approach for Deep Face Recognition", result.answer)

    def test_ask_reference_route_formats_answer(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            result = run_ask(
                settings,
                "哪些论文引用了ResNet",
                plan_parser=reference_payload("cited_by", ["ResNet"]),
            )
            self.assertEqual(result.route, "reference")
            self.assertIn("共找到 1 条引用证据", result.answer)
            self.assertIn("Deep Face Recognition", result.answer)

    def test_ask_parse_failed_reports_failure(self) -> None:
        class BadParser:
            def parse_metadata(self, query: str) -> dict:
                raise PlanParseError("invalid parser payload")

        with sample_project() as root:
            settings = Settings.load(Path(root))
            result = run_ask(
                settings,
                "Center Loss 是谁写的",
                plan_parser=BadParser(),
            )
            self.assertEqual(result.route, "metadata")
            self.assertIn("无法解析", result.answer)

    def test_chinese_reference_with_year_constraint_stays_reference(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "哪些在2019发表的论文引用了resnet",
                plan_parser=reference_payload(
                    "cited_by",
                    ["ResNet"],
                    filters=[{"field": "year", "op": "=", "value": 2016, "negated": False}],
                ),
            )
            self.assertNotIn("retrieval_query", pack)
            self.assertEqual(pack["route"], "reference")
            self.assertEqual(pack["intent"], "list")
            self.assertNotIn("return_field", pack)
            self.assertEqual(pack["direction"], "cited_by")
            self.assertEqual(pack["anchors"], ["ResNet"])
            self.assertEqual(pack["anchor_mode"], "per")
            self.assertEqual(pack["filters"], [{"field": "year", "op": "=", "value": 2016, "negated": False}])
            self.assertEqual(pack["evidence"]["parse_status"], "ok")
            self.assertEqual(len(pack["evidence"]["citing_papers"]), 1)
            self.assertEqual(pack["evidence"]["reference_items"], [])

    def test_reference_incoming_returns_local_citing_papers(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "哪些论文引用了ResNet",
                plan_parser=reference_payload("cited_by", ["ResNet"]),
            )
            self.assertEqual(pack["route"], "reference")
            self.assertEqual(pack["intent"], "list")
            self.assertNotIn("return_field", pack)
            self.assertEqual(pack["direction"], "cited_by")
            self.assertEqual(pack["anchors"], ["ResNet"])
            self.assertEqual(pack["anchor_mode"], "per")
            self.assertEqual(pack["filters"], [])
            self.assertNotIn("scope", pack["evidence"])
            self.assertNotIn("expanded_query", pack["evidence"])
            self.assertNotIn("query", pack["evidence"])
            self.assertNotIn("parser_result", pack["evidence"])
            self.assertNotIn("target_papers", pack["evidence"])
            self.assertNotIn("top_route", pack["evidence"])
            self.assertEqual(pack["evidence"]["parse_status"], "ok")
            self.assertEqual(len(pack["evidence"]["citing_papers"]), 1)
            self.assertEqual(pack["evidence"]["reference_items"], [])
            self.assertNotIn("references", pack["evidence"])
            self.assertEqual(pack["evidence"]["anchors"], ["ResNet"])
            anchor_target = pack["evidence"]["anchor_results"][0]["resolved_papers"][0]
            self.assertEqual(anchor_target["title"], "Deep Residual Learning for Image Recognition")
            self.assertEqual(anchor_target["matched_alias"], "ResNet")
            self.assertNotIn("file_hash", anchor_target)
            self.assertNotIn("pdf_path", anchor_target)
            self.assertEqual(pack["evidence"]["citing_papers"][0]["anchor_mention"], "ResNet")
            self.assertEqual(pack["evidence"]["anchor_results"][0]["anchor_mention"], "ResNet")
            citing_paper = pack["evidence"]["citing_papers"][0]["citing_paper"]
            self.assertEqual(citing_paper["title"], "A Discriminative Feature Learning Approach for Deep Face Recognition")
            self.assertNotIn("file_hash", citing_paper)
            self.assertNotIn("paper_data_path", citing_paper)
            self.assertNotIn("matched_alias", citing_paper)
            self.assertNotIn("direction", pack["evidence"]["citing_papers"][0])
            self.assertNotIn("reference", pack["evidence"]["citing_papers"][0])
            self.assertNotIn("anchor_terms", pack["evidence"]["citing_papers"][0])
            self.assertNotIn("citing_papers", pack["evidence"]["anchor_results"][0])
            self.assertNotIn("reference_items", pack["evidence"]["anchor_results"][0])

    def test_reference_outgoing_reads_anchor_references_and_filters_raw_text(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "ResNet引用的论文里哪些和ImageNet有关",
                plan_parser=reference_payload(
                    "cites",
                    ["ResNet"],
                    filters=[{"field": "title", "op": "contains", "value": "ImageNet", "negated": False}],
                ),
            )
            self.assertEqual(pack["route"], "reference")
            self.assertNotIn("return_field", pack)
            self.assertEqual(pack["direction"], "cites")
            self.assertEqual(pack["anchors"], ["ResNet"])
            self.assertEqual(pack["anchor_mode"], "per")
            self.assertEqual(pack["filters"], [{"field": "title", "op": "contains", "value": "ImageNet", "negated": False}])
            self.assertEqual(pack["evidence"]["direction"], "cites")
            self.assertNotIn("query", pack["evidence"])
            self.assertNotIn("parser_result", pack["evidence"])
            self.assertNotIn("target_papers", pack["evidence"])
            self.assertNotIn("top_route", pack["evidence"])
            self.assertEqual(len(pack["evidence"]["reference_items"]), 1)
            self.assertEqual(pack["evidence"]["citing_papers"], [])
            self.assertNotIn("references", pack["evidence"])
            self.assertIn("ImageNet", pack["evidence"]["reference_items"][0]["reference"]["raw_text"])
            self.assertEqual(pack["evidence"]["reference_items"][0]["anchor_mention"], "ResNet")
            anchor_paper = pack["evidence"]["reference_items"][0]["anchor_paper"]
            self.assertEqual(anchor_paper["title"], "Deep Residual Learning for Image Recognition")
            self.assertNotIn("paper_id", anchor_paper)
            self.assertEqual(anchor_paper["matched_alias"], "ResNet")
            self.assertNotIn("direction", pack["evidence"]["reference_items"][0])
            self.assertNotIn("citing_papers", pack["evidence"]["anchor_results"][0])
            self.assertNotIn("reference_items", pack["evidence"]["anchor_results"][0])

    def test_reference_and_mode_uses_all_anchors_for_intersection(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "哪些论文同时引用了ResNet和EfficientNet",
                plan_parser=reference_payload("cited_by", ["ResNet", "EfficientNet"], anchor_mode="and"),
            )
            self.assertEqual(pack["route"], "reference")
            self.assertEqual(pack["evidence"]["anchor_mode"], "and")
            self.assertNotIn("query", pack["evidence"])
            self.assertEqual(pack["evidence"]["citing_papers"], [])

    def test_reference_unknown_direction_does_not_retrieve(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "ResNet的引用关系",
                plan_parser=reference_payload(None, ["ResNet"]),
            )
            self.assertEqual(pack["route"], "reference")
            self.assertEqual(pack["evidence"]["parse_status"], "unknown_direction")
            self.assertNotIn("parser_result", pack["evidence"])
            self.assertEqual(pack["evidence"]["reference_items"], [])
            self.assertEqual(pack["evidence"]["citing_papers"], [])

    def test_chinese_reference_route_uses_original_query_for_routing(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "Center Loss 引用了哪些工作",
                plan_parser=reference_payload("cites", ["Center Loss"]),
            )
            self.assertEqual(pack["route"], "reference")
            self.assertEqual(pack["intent"], "list")
            self.assertEqual(pack["evidence"]["parse_status"], "ok")
            self.assertGreaterEqual(len(pack["evidence"]["reference_items"]), 2)

    def test_body_route_fuses_dense_bm25_and_expands_context(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            embedder = FakeEmbedder()
            store = FakeStore()
            pack = run_plan(
                settings,
                "center loss intra class compactness",
                embedder=embedder,
                store=store,
            )
            self.assertEqual(pack["route"], "content")
            self.assertEqual(pack["intent"], None)
            self.assertNotIn("scope", pack["evidence"])
            self.assertNotIn("expanded_query", pack["evidence"])
            self.assertNotIn("top_route", pack["evidence"])
            self.assertNotIn("query", pack["evidence"])
            self.assertEqual(store.top_k, 20)
            self.assertEqual(embedder.calls, [["center loss intra class compactness"]])
            units = pack["evidence"]["context_units"]
            self.assertGreaterEqual(len(units), 1)
            self.assertIn("dense", units[0]["sources"])
            self.assertIn("bm25", units[0]["sources"])
            block_ids = [block["block_id"] for block in units[0]["expanded_blocks"]]
            self.assertIn("b1", block_ids)
            self.assertIn("b2", block_ids)

def sample_project():
    temp = tempfile.TemporaryDirectory()
    root = Path(temp.name)
    data = root / "data"
    paper = data / "paper_data" / "Center_Loss_abc"
    resnet_paper = data / "paper_data" / "ResNet_def"
    paper.mkdir(parents=True)
    resnet_paper.mkdir(parents=True)
    (root / ".env").write_text(
        "\n".join([
            "PLAN_DENSE_TOP_K=20",
            "PLAN_BM25_TOP_K=20",
            "PLAN_FINAL_TOP_K=8",
            "PLAN_BLOCK_WINDOW=1",
        ]),
        encoding="utf-8",
    )
    (data / "paper_aliases.json").write_text(
        json.dumps([
            {
                "canonical": "Long Short-Term Memory",
                "aliases": ["LSTM"],
            },
            {
                "canonical": "A Discriminative Feature Learning Approach for Deep Face Recognition",
                "aliases": ["Center loss"],
            },
            {
                "canonical": "Deep Residual Learning for Image Recognition",
                "aliases": ["ResNet"],
            },
        ]),
        encoding="utf-8",
    )
    (data / "venue_aliases.json").write_text(
        json.dumps([
            {
                "canonical": "CVPR",
                "aliases": [
                    "IEEE/CVF Conference on Computer Vision and Pattern Recognition",
                    "Computer Vision and Pattern Recognition",
                ],
            },
        ]),
        encoding="utf-8",
    )
    (data / "manifest.jsonl").write_text(
        "\n".join([
            json.dumps({
                "file_hash": "abc",
                "status": "active",
                "pdf_path": str(data / "pdf" / "Center_Loss.pdf"),
                "title": "A Discriminative Feature Learning Approach for Deep Face Recognition",
                "author": ["Yandong Wen", "Kaipeng Zhang"],
                "year": 2016,
                "venue": "ECCV",
                "mineru_output_path": None,
                "archived_mineru_output_path": None,
                "paper_data_path": str(paper),
                "message": None,
            }, ensure_ascii=False),
            json.dumps({
                "file_hash": "def",
                "status": "active",
                "pdf_path": str(data / "pdf" / "ResNet.pdf"),
                "title": "Deep Residual Learning for Image Recognition",
                "author": ["Kaiming He", "Xiangyu Zhang", "Shaoqing Ren", "Jian Sun"],
                "year": {"preprint_year": 2015, "publish_year": 2016},
                "venue": "2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition",
                "mineru_output_path": None,
                "archived_mineru_output_path": None,
                "paper_data_path": str(data / "paper_data" / "ResNet_def"),
                "message": None,
            }, ensure_ascii=False),
        ]) + "\n",
        encoding="utf-8",
    )
    (paper / "metadata.json").write_text(
        json.dumps({
            "title": "A Discriminative Feature Learning Approach for Deep Face Recognition",
            "year": 2016,
            "venue": "ECCV",
        }),
        encoding="utf-8",
    )
    (paper / "chunks.jsonl").write_text(
        "\n".join([
            json.dumps({
                "chunk_id": "Center_Loss_abc::chunk_0000",
                "paper_id": "Center_Loss_abc",
                "chunk_index": 0,
                "region": "abstract",
                "section_id": "sec_abstract",
                "section_path": ["Abstract"],
                "pages": [1],
                "block_ids": ["b1"],
                "text": "Center loss improves intra class compactness for deep face recognition.",
                "embedding_text": "Paper: A Discriminative Feature Learning Approach for Deep Face Recognition\nSection: Abstract\n\nCenter loss improves intra class compactness.",
            }),
            json.dumps({
                "chunk_id": "Center_Loss_abc::chunk_0001",
                "paper_id": "Center_Loss_abc",
                "chunk_index": 1,
                "region": "body",
                "section_id": "sec_1",
                "section_path": ["1 Introduction"],
                "pages": [2],
                "block_ids": ["b3"],
                "text": "Softmax loss learns separable features while center loss adds compactness.",
                "embedding_text": "Paper: A Discriminative Feature Learning Approach for Deep Face Recognition\nSection: 1 Introduction\n\nSoftmax loss learns separable features.",
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    (paper / "blocks.jsonl").write_text(
        "\n".join([
            json.dumps({
                "block_id": "b1",
                "order": 0,
                "region": "abstract",
                "type": "paragraph",
                "text": "Center loss improves intra class compactness.",
                "page": 1,
                "section_id": "sec_abstract",
                "section_path": ["Abstract"],
            }),
            json.dumps({
                "block_id": "b2",
                "order": 1,
                "region": "abstract",
                "type": "paragraph",
                "text": "It is optimized jointly with softmax loss.",
                "page": 1,
                "section_id": "sec_abstract",
                "section_path": ["Abstract"],
            }),
            json.dumps({
                "block_id": "b3",
                "order": 2,
                "region": "body",
                "type": "paragraph",
                "text": "Softmax loss learns separable features.",
                "page": 2,
                "section_id": "sec_1",
                "section_path": ["1 Introduction"],
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    (paper / "references.jsonl").write_text(
        "\n".join([
            json.dumps({
                "ref_index": 1,
                "raw_text": "Y. LeCun et al. Gradient-based learning applied to document recognition.",
                "page": 9,
                "source_block_id": "b9",
            }),
            json.dumps({
                "ref_index": 2,
                "raw_text": "S. Hochreiter and J. Schmidhuber. Long Short-Term Memory (LSTM). Neural Computation, 1997.",
                "page": 9,
                "source_block_id": "b9",
            }),
            json.dumps({
                "ref_index": 3,
                "raw_text": "Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. CVPR, 2016.",
                "page": 9,
                "source_block_id": "b9",
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    (resnet_paper / "references.jsonl").write_text(
        "\n".join([
            json.dumps({
                "ref_index": 1,
                "raw_text": "Jia Deng et al. ImageNet: A large-scale hierarchical image database. CVPR, 2009.",
                "page": 9,
                "source_block_id": "r9",
            }),
            json.dumps({
                "ref_index": 2,
                "raw_text": "Karen Simonyan and Andrew Zisserman. Very Deep Convolutional Networks for Large-Scale Image Recognition. ICLR, 2015.",
                "page": 9,
                "source_block_id": "r9",
            }),
            json.dumps({
                "ref_index": 3,
                "raw_text": "Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun. Deep Residual Learning for Image Recognition. CVPR, 2016.",
                "page": 9,
                "source_block_id": "r9",
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    return temp


if __name__ == "__main__":
    unittest.main()

