from __future__ import annotations

import json
import hashlib
import urllib.parse
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_rag.config import Settings
from paper_rag.retrieval.answer import run_ask
from paper_rag.retrieval.dense.milvus_store import SearchResult
from paper_rag.retrieval.plan.planner import prepare_query, run_plan
from paper_rag.retrieval.plan.domains.metadata.schema import PlanParseError
from paper_rag.retrieval.plan.domains.reference.schema import validate_reference_parse
from paper_rag.retrieval.plan.top_router import route_query
from paper_rag.retrieval.plan.translation import BaiduTranslator, TranslationError, TranslationResult


class FakeResponse:
    def __init__(self, payload: dict):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")


class FakeTranslator:
    def translate_to_english(self, text: str) -> TranslationResult:
        return TranslationResult("center loss improves intra class compactness", "fake")


class MetadataTranslator:
    def translate_to_english(self, text: str) -> TranslationResult:
        return TranslationResult("which year was Center Loss published", "fake")


class ReferenceTranslator:
    def translate_to_english(self, text: str) -> TranslationResult:
        return TranslationResult("References of Center Loss", "fake")


class YearRangeTranslator:
    def translate_to_english(self, text: str) -> TranslationResult:
        return TranslationResult("Which papers were published in 2015-2019", "fake")


class YearListTranslator:
    def translate_to_english(self, text: str) -> TranslationResult:
        return TranslationResult("Which papers were published in 2016", "fake")


class FullAuthorTranslator:
    def translate_to_english(self, text: str) -> TranslationResult:
        return TranslationResult("Which papers were written by Kaiming he", "fake")


class ShortAuthorTranslator:
    def translate_to_english(self, text: str) -> TranslationResult:
        return TranslationResult("Which papers were written by he", "fake")


class ReferenceYearTranslator:
    def translate_to_english(self, text: str) -> TranslationResult:
        return TranslationResult("Which papers published in 2019 cite RESNET", "fake")


class StaticMetadataParser:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[str] = []

    def parse_metadata(self, query: str) -> dict:
        self.calls.append(query)
        payload = dict(self.payload)
        payload.setdefault("router", "metadata")
        payload.setdefault("filters", [])
        payload.setdefault("raw_query", query)
        return payload


class StaticReferenceParser:
    def __init__(self, payload: dict):
        self.payload = payload
        self.calls: list[str] = []

    def parse_reference(self, query: str) -> dict:
        self.calls.append(query)
        payload = dict(self.payload)
        payload.setdefault("router", "reference")
        payload.setdefault("anchor", [])
        payload.setdefault("anchor_mode", "per")
        payload.setdefault("filters", [])
        payload.setdefault("raw_query", query)
        return payload


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
    intent: str | None = "list",
    anchor_mode: str = "per",
    filters: list[dict] | None = None,
) -> StaticReferenceParser:
    return StaticReferenceParser({
        "intent": intent,
        "direction": direction,
        "anchor": [{"field": "title", "value": anchor} for anchor in anchors],
        "anchor_mode": anchor_mode,
        "filters": filters or [],
    })


class FailingTranslator:
    def translate_to_english(self, text: str) -> TranslationResult:
        raise TranslationError("rate limited")


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
        self.assertEqual(route_query("which year was this paper published").route, "metadata")
        self.assertEqual(route_query("which year was this paper published").intent, None)
        self.assertEqual(route_query("which references did this paper cite").route, "reference")
        self.assertEqual(route_query("which papers cite ResNet").route, "reference")
        self.assertEqual(route_query("papers citing BERT").route, "reference")
        self.assertEqual(route_query("how does center loss improve compactness").route, "content")

    def test_rule_router_keeps_reference_as_top_level_entry_only(self) -> None:
        self.assertEqual(route_query("Which papers cited ResNet").route, "reference")
        self.assertEqual(route_query("Which papers cite ResNet").route, "reference")
        self.assertEqual(route_query("Which papers published in 2019 cite ResNet").route, "reference")
        self.assertEqual(route_query("Papers citing ResNet").route, "reference")
        self.assertEqual(route_query("Which papers are cited by ResNet").route, "reference")
        self.assertEqual(route_query("Which papers are referenced by ResNet").route, "reference")
        self.assertEqual(route_query("References of ResNet").route, "reference")
        self.assertEqual(route_query("Bibliography of ResNet").route, "reference")

    def test_router_uses_token_boundaries(self) -> None:
        self.assertEqual(route_query("recite the result").route, "content")
        self.assertEqual(route_query("authorization method").route, "content")
        self.assertEqual(route_query("publication year of BERT").route, "metadata")

    def test_metadata_router_field_routes(self) -> None:
        self.assertEqual(route_query("Who wrote BERT").route, "metadata")
        self.assertEqual(route_query("Who are the authors of BERT").route, "metadata")
        self.assertEqual(route_query("When was BERT published").route, "metadata")
        self.assertEqual(route_query("publication year of BERT").route, "metadata")
        self.assertEqual(route_query("Which journal published BERT").route, "metadata")
        self.assertEqual(route_query("Which conference published BERT").route, "metadata")
        self.assertEqual(route_query("What is the title of BERT").route, "metadata")

    def test_metadata_router_paper_list_filters(self) -> None:
        by_year = route_query("Which papers were published in 2015-2019")
        self.assertEqual(by_year.route, "metadata")
        self.assertEqual(by_year.intent, None)
        self.assertEqual(by_year.filters, [])
        by_author = route_query("Which papers are written by Kaiming He")
        self.assertEqual(by_author.route, "metadata")
        self.assertEqual(by_author.intent, None)
        self.assertEqual(by_author.filters, [])

    def test_reference_route_keeps_original_query(self) -> None:
        query = "Which papers cited RESNET"
        decision = route_query(query)
        self.assertEqual(decision.route, "reference")
        self.assertEqual(decision.target_query, query)

    def test_reference_parser_schema_normalizes_valid_payload(self) -> None:
        payload = validate_reference_parse({
            "router": "reference",
            "intent": "count",
            "direction": "cited_by",
            "anchor": [{"field": "title", "value": "ResNet"}],
            "anchor_mode": "and",
            "filters": [{"field": "year", "op": "interval", "value": [2015, "inf"], "negated": False}],
            "raw_query": "Which papers cited ResNet after 2015?",
        })
        self.assertEqual(payload["intent"], "count")
        self.assertEqual(payload["direction"], "cited_by")
        self.assertEqual(payload["anchor_mode"], "and")
        self.assertEqual(payload["filters"][0]["value"], [2015, "inf"])

    def test_reference_parser_schema_rejects_invalid_anchor_field(self) -> None:
        with self.assertRaises(PlanParseError):
            validate_reference_parse({
                "router": "reference",
                "intent": "list",
                "direction": "cite",
                "anchor": [{"field": "author", "value": "Kaiming He"}],
                "anchor_mode": "per",
                "filters": [],
                "raw_query": "References by Kaiming He",
            })

    def test_chinese_query_uses_translator_and_english_query_skips_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings.load(Path(tmp))
            translated = prepare_query(settings, "中心损失如何提升类内紧凑性", translator=FakeTranslator())
            english = prepare_query(settings, "center loss compactness", translator=FailingTranslator())
            self.assertEqual(translated.retrieval_query, "center loss improves intra class compactness")
            self.assertEqual(translated.translation_provider, "fake")
            self.assertEqual(english.retrieval_query, "center loss compactness")
            self.assertEqual(english.warnings, [])

    def test_translation_failure_warns_and_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings.load(Path(tmp))
            prepared = prepare_query(settings, "中心损失", translator=FailingTranslator())
            self.assertEqual(prepared.retrieval_query, "")
            self.assertEqual(prepared.error, "translation_failed")
            self.assertIn("translation_failed", prepared.warnings[0])

    def test_translation_failure_returns_error_pack_without_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            settings = Settings.load(Path(tmp))
            pack = run_plan(settings, "中心损失", translator=FailingTranslator())
            self.assertEqual(pack["route"], "error")
            self.assertEqual(pack["router_reason"], "translation_failed")
            self.assertEqual(pack["evidence"], {})
            self.assertNotIn("language", pack)

    def test_baidu_domain_translation_uses_academic_payload_and_sign(self) -> None:
        captured = {}

        def fake_urlopen(request, timeout):
            captured["url"] = request.full_url
            captured["payload"] = urllib.parse.parse_qs(request.data.decode("utf-8"))
            return FakeResponse({"trans_result": [{"dst": "center loss"}]})

        translator = BaiduTranslator(
            app_id="appid",
            secret_key="secret",
            endpoint="https://example.com/fieldtranslate",
            domain="academic",
        )
        with patch("random.randint", return_value=12345), patch("urllib.request.urlopen", fake_urlopen):
            result = translator.translate_to_english("中心损失")
        expected_sign = hashlib.md5("appid中心损失12345academicsecret".encode("utf-8")).hexdigest()
        self.assertEqual(result.text, "center loss")
        self.assertEqual(captured["url"], "https://example.com/fieldtranslate")
        self.assertEqual(captured["payload"]["domain"], ["academic"])
        self.assertEqual(captured["payload"]["sign"], [expected_sign])

    def test_metadata_route_reads_manifest_without_retrieval(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "Center Loss authors",
                plan_parser=metadata_lookup("author", "Center loss"),
            )
            self.assertEqual(pack["route"], "metadata")
            self.assertEqual(pack["intent"], "lookup")
            self.assertNotIn("sub_route", pack)
            self.assertNotIn("scope", pack["evidence"])
            self.assertNotIn("expanded_query", pack["evidence"])
            self.assertEqual(pack["evidence"]["intent"], "lookup")
            self.assertEqual(pack["evidence"]["return_field"], "author")
            records = pack["evidence"]["records"]
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["year"], 2016)
            self.assertEqual(records[0]["venue"], "ECCV")
            self.assertEqual(records[0]["value"], ["Yandong Wen", "Kaipeng Zhang"])

    def test_chinese_metadata_route_uses_translated_query_for_routing(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "Center Loss 是哪一年发表的",
                translator=MetadataTranslator(),
                plan_parser=metadata_lookup("year", "Center loss"),
            )
            self.assertEqual(pack["route"], "metadata")
            self.assertEqual(pack["intent"], "lookup")
            self.assertEqual(pack["retrieval_query"], "which year was Center Loss published")
            self.assertNotIn("language", pack)

    def test_metadata_paper_list_filters_manifest_records(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            by_year = run_plan(
                settings,
                "Which papers were published in 2015-2019",
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
                "Which papers are written by Kaiming He",
                plan_parser=metadata_list([
                    {"field": "author", "op": "=", "value": "Kaiming He", "negated": False},
                ]),
            )
            self.assertEqual(len(by_author["evidence"]["records"]), 1)
            self.assertEqual(by_author["evidence"]["records"][0]["title"], "Deep Residual Learning for Image Recognition")
            short_author = run_plan(
                settings,
                "Which papers are written by He",
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
                "Who are the authors of Center Loss and ResNet respectively",
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
                "How many papers were published in 2016",
                plan_parser=StaticMetadataParser({
                    "intent": "count",
                    "return_field": None,
                    "filters": [{"field": "year", "op": "=", "value": 2016, "negated": False}],
                }),
            )
            self.assertEqual(count["intent"], "count")
            self.assertEqual(count["evidence"]["count"], 2)
            not_cvpr = run_plan(
                settings,
                "Which papers were not published at CVPR",
                plan_parser=StaticMetadataParser({
                    "intent": "list",
                    "return_field": None,
                    "filters": [{"field": "venue", "op": "contains", "value": "CVPR", "negated": True}],
                }),
            )
            self.assertEqual([record["venue"] for record in not_cvpr["evidence"]["records"]], ["ECCV"])

    def test_metadata_parser_failure_returns_parse_failed_evidence(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "Who wrote Center Loss",
                plan_parser=StaticMetadataParser({
                    "intent": "made_up",
                    "return_field": "author",
                }),
            )
            self.assertEqual(pack["route"], "metadata")
            self.assertEqual(pack["intent"], "unknown")
            self.assertEqual(pack["evidence"]["parse_status"], "parse_failed")
            self.assertEqual(pack["evidence"]["records"], [])
            self.assertIn("metadata_parse_failed", pack["warnings"][0])

    def test_metadata_parser_accepts_string_null_return_field(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "What papers were published on arXiv?",
                plan_parser=StaticMetadataParser({
                    "intent": "list",
                    "return_field": "null",
                    "filters": [{"field": "venue", "op": "contains", "value": "arXiv", "negated": False}],
                }),
            )
            self.assertEqual(pack["route"], "metadata")
            self.assertEqual(pack["intent"], "list")
            self.assertEqual(pack["evidence"]["parse_status"], "ok")

    def test_metadata_parser_accepts_interval_numeric_list(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "Which papers were published between 2015 and 2020?",
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
                "Which papers were published after 2015?",
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
                "Which papers were published before 2019?",
                plan_parser=StaticMetadataParser({
                    "intent": "list",
                    "return_field": None,
                    "filters": [{"field": "year", "op": "interval", "value": ["-inf", 2019], "negated": False}],
                }),
            )
            self.assertEqual(pack["route"], "metadata")
            self.assertEqual(pack["evidence"]["filters"][0]["value"], ["-inf", 2019])
            self.assertEqual(len(pack["evidence"]["records"]), 2)

    def test_chinese_metadata_paper_list_uses_translated_query(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            by_year = run_plan(
                settings,
                "哪些论文在2015-2019发表",
                translator=YearRangeTranslator(),
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
                translator=FullAuthorTranslator(),
                plan_parser=metadata_list([
                    {"field": "author", "op": "=", "value": "Kaiming He", "negated": False},
                ]),
            )
            self.assertEqual(len(full_author["evidence"]["records"]), 1)
            self.assertEqual(full_author["evidence"]["records"][0]["title"], "Deep Residual Learning for Image Recognition")
            short_author = run_plan(
                settings,
                "哪些论文是He写的",
                translator=ShortAuthorTranslator(),
                plan_parser=metadata_list([
                    {"field": "author", "op": "=", "value": "He", "negated": False},
                ]),
            )
            self.assertEqual(short_author["evidence"]["records"], [])

    def test_ask_metadata_lookup_formats_answer(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            result = run_ask(
                settings,
                "Who wrote Center Loss",
                plan_parser=metadata_lookup("author", "Center loss"),
            )
            self.assertEqual(result.route, "metadata")
            self.assertIn("was written by", result.answer)
            self.assertIn("Yandong Wen", result.answer)
            self.assertIn("Kaipeng Zhang", result.answer)
            self.assertIn("plan(metadata)", result.provenance[0])

    def test_ask_metadata_list_formats_answer_in_chinese(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            result = run_ask(
                settings,
                "哪些论文在2016发表",
                translator=YearListTranslator(),
                plan_parser=metadata_list([
                    {"field": "year", "op": "=", "value": 2016, "negated": False},
                ]),
            )
            self.assertEqual(result.route, "metadata")
            self.assertIn("共找到 2 篇论文", result.answer)
            self.assertIn("Deep Residual Learning for Image Recognition", result.answer)
            self.assertIn("A Discriminative Feature Learning Approach for Deep Face Recognition", result.answer)

    def test_ask_reference_route_formats_answer(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            result = run_ask(
                settings,
                "Which papers cited ResNet",
                plan_parser=reference_payload("cited_by", ["ResNet"]),
            )
            self.assertEqual(result.route, "reference")
            self.assertIn("reference match", result.answer.lower())
            self.assertIn("Deep Face Recognition", result.answer)

    def test_ask_parse_failed_reports_failure(self) -> None:
        class BadParser:
            def parse_metadata(self, query: str) -> dict:
                return {
                    "router": "metadata",
                    "intent": "bogus",
                    "return_field": None,
                    "filters": [],
                    "raw_query": query,
                }

        with sample_project() as root:
            settings = Settings.load(Path(root))
            result = run_ask(
                settings,
                "Who wrote Center Loss",
                plan_parser=BadParser(),
            )
            self.assertEqual(result.route, "metadata")
            self.assertIn("could not be parsed", result.answer)

    def test_chinese_reference_with_year_constraint_stays_reference(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "哪些在2019发表的论文引用了resnet",
                translator=ReferenceYearTranslator(),
                plan_parser=reference_payload(
                    "cited_by",
                    ["ResNet"],
                    filters=[{"field": "year", "op": "=", "value": 2016, "negated": False}],
                ),
            )
            self.assertEqual(pack["retrieval_query"], "Which papers published in 2019 cite RESNET")
            self.assertEqual(pack["route"], "reference")
            self.assertEqual(pack["intent"], "list")
            self.assertEqual(pack["evidence"]["parse_status"], "ok")
            self.assertEqual(len(pack["evidence"]["references"]), 1)

    def test_reference_cited_by_returns_local_citing_papers(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "Which papers cited ResNet",
                plan_parser=reference_payload("cited_by", ["ResNet"]),
            )
            self.assertEqual(pack["route"], "reference")
            self.assertEqual(pack["intent"], "list")
            self.assertEqual(pack["evidence"]["query"], "Which papers cited ResNet")
            self.assertNotIn("scope", pack["evidence"])
            self.assertNotIn("expanded_query", pack["evidence"])
            self.assertEqual(pack["evidence"]["parse_status"], "ok")
            self.assertEqual(len(pack["evidence"]["references"]), 1)
            self.assertEqual(pack["evidence"]["references"][0]["citing_paper"]["title"], "A Discriminative Feature Learning Approach for Deep Face Recognition")

    def test_reference_cite_reads_anchor_references_and_filters_raw_text(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "References of ResNet about ImageNet",
                plan_parser=reference_payload(
                    "cite",
                    ["ResNet"],
                    filters=[{"field": "title", "op": "contains", "value": "ImageNet", "negated": False}],
                ),
            )
            self.assertEqual(pack["route"], "reference")
            self.assertEqual(pack["evidence"]["direction"], "cite")
            self.assertEqual(len(pack["evidence"]["references"]), 1)
            self.assertIn("ImageNet", pack["evidence"]["references"][0]["reference"]["raw_text"])

    def test_reference_unknown_direction_does_not_retrieve(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "Reference graph around ResNet",
                plan_parser=reference_payload(None, ["ResNet"]),
            )
            self.assertEqual(pack["route"], "reference")
            self.assertEqual(pack["evidence"]["parse_status"], "unknown_direction")
            self.assertEqual(pack["evidence"]["references"], [])

    def test_chinese_reference_route_uses_translated_query_for_routing(self) -> None:
        with sample_project() as root:
            settings = Settings.load(Path(root))
            pack = run_plan(
                settings,
                "Center Loss 引用了哪些工作",
                translator=ReferenceTranslator(),
                plan_parser=reference_payload("cite", ["Center Loss"]),
            )
            self.assertEqual(pack["route"], "reference")
            self.assertEqual(pack["intent"], "list")
            self.assertEqual(pack["evidence"]["parse_status"], "ok")
            self.assertGreaterEqual(len(pack["evidence"]["references"]), 2)

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
                "year": 2016,
                "venue": "CVPR",
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
        ]) + "\n",
        encoding="utf-8",
    )
    return temp


if __name__ == "__main__":
    unittest.main()

