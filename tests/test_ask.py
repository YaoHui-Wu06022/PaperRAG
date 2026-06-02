from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from paper_rag.answer import run_ask
from paper_rag.answer.llm import AnswerComposerClient, AnswerError, build_answer_user_prompt
from paper_rag.cli.main import build_parser
from paper_rag.config import Settings


class AskCliTests(unittest.TestCase):
    def test_ask_subcommand_is_registered(self) -> None:
        args = build_parser().parse_args(["ask", "ResNet", "的结构是什么", "--debug", "--json"])

        self.assertEqual(args.command, "ask")
        self.assertEqual(args.query, ["ResNet", "的结构是什么"])
        self.assertTrue(args.debug)
        self.assertTrue(args.json)
        self.assertTrue(callable(args.handler))


class RunAskTests(unittest.TestCase):
    def test_run_ask_composes_metadata_locally(self) -> None:
        with temp_settings() as settings:
            payload = run_ask(
                settings,
                "BERT 是谁写的",
                planner=static_metadata_planner,
                answer_client=FailingAnswerClient(),
            )

        self.assertEqual(payload["query"], "BERT 是谁写的")
        self.assertEqual(payload["answer_mode"], "local")
        self.assertIn("作者", payload["answer"])
        self.assertIn("Devlin", payload["answer"])
        self.assertIn("BERT 的作者是", payload["answer"])
        self.assertNotIn("BERT：", payload["answer"])
        self.assertNotIn("BERT: Pre-training", payload["answer"])
        self.assertEqual(payload["evidence"]["route"], "metadata")

    def test_run_ask_numbers_metadata_list_results(self) -> None:
        with temp_settings() as settings:
            payload = run_ask(
                settings,
                "发表在 CVPR 上的论文有哪些",
                planner=static_metadata_list_planner,
                answer_client=FailingAnswerClient(),
            )

        self.assertIn("[1] Deep Residual Learning for Image Recognition", payload["answer"])
        self.assertIn("[2] Squeeze-and-Excitation Networks", payload["answer"])

    def test_run_ask_shows_metadata_group_lookup_values(self) -> None:
        with temp_settings() as settings:
            payload = run_ask(
                settings,
                "ResNet 和 Transformer 分别是哪一年发表的？",
                planner=static_metadata_group_lookup_planner,
                answer_client=FailingAnswerClient(),
            )

        self.assertIn("ResNet：", payload["answer"])
        self.assertIn("正式发表 2016", payload["answer"])
        self.assertIn("Transformer：", payload["answer"])
        self.assertIn("正式发表 2017", payload["answer"])
        self.assertNotIn("ResNet 的年份是", payload["answer"])
        self.assertNotIn("1 篇", payload["answer"])
        self.assertNotIn("分组结果", payload["answer"])
        self.assertNotIn("[1]", payload["answer"])

    def test_run_ask_outputs_count_with_item_evidence(self) -> None:
        with temp_settings() as settings:
            payload = run_ask(
                settings,
                "发表在 CVPR 上的论文有多少篇",
                planner=static_metadata_count_planner,
                answer_client=FailingAnswerClient(),
            )

        self.assertIn("共找到 2 篇", payload["answer"])
        self.assertIn("[1] Deep Residual Learning for Image Recognition", payload["answer"])
        self.assertIn("[2] Squeeze-and-Excitation Networks", payload["answer"])

    def test_run_ask_reference_and_groups_use_aggregate_answer(self) -> None:
        with temp_settings() as settings:
            payload = run_ask(
                settings,
                "哪些论文引用了 Transformer 和 ResNet？",
                planner=static_reference_and_group_planner,
                answer_client=FailingAnswerClient(),
            )

        self.assertIn("[1] Attention as Activation", payload["answer"])
        self.assertNotIn("paper=Attention is All you Need", payload["answer"])
        self.assertNotIn("paper=Deep Residual Learning", payload["answer"])

    def test_run_ask_shows_actual_metadata_when_exists_is_false(self) -> None:
        with temp_settings() as settings:
            payload = run_ask(
                settings,
                "BERT 是 CVPR 的论文吗",
                planner=static_metadata_exists_false_planner,
                answer_client=FailingAnswerClient(),
            )

        self.assertIn("不是", payload["answer"])
        self.assertIn("BERT 的发表 venue", payload["answer"])
        self.assertIn("NAACL", payload["answer"])

    def test_run_ask_uses_llm_for_content_contexts(self) -> None:
        client = StaticAnswerClient("VIT 的模型结构包含 patch embedding 和 Transformer encoder。")
        with temp_settings() as settings:
            payload = run_ask(
                settings,
                "VIT 的模型结构是什么",
                planner=static_content_planner,
                answer_client=client,
            )

        self.assertEqual(payload["answer_mode"], "llm")
        self.assertEqual(payload["answer"], "VIT 的模型结构包含 patch embedding 和 Transformer encoder。")
        self.assertEqual(client.evidence["route"], "content")

    def test_run_ask_debug_includes_plan_and_answer_timing(self) -> None:
        with temp_settings() as settings:
            payload = run_ask(
                settings,
                "BERT 是谁写的",
                debug=True,
                planner=static_metadata_planner,
                answer_client=FailingAnswerClient(),
            )

        timings = payload["evidence"]["debug"]["timings_ms"]
        self.assertIn("plan", timings)
        self.assertIn("answer", timings)

    def test_answer_prompt_contains_compact_evidence(self) -> None:
        prompt = build_answer_user_prompt({
            "query": "BERT 是谁写的",
            "route": "metadata",
            "results": {"items": [{"title": "BERT", "values": {"author": ["Devlin"]}}]},
        })

        self.assertIn("BERT 是谁写的", prompt)
        self.assertIn("Devlin", prompt)

    def test_answer_settings_fall_back_to_plan_parser_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / ".env").write_text(
                "\n".join([
                    "PLAN_PARSER_BASE_URL=https://example.test/v1",
                    "PLAN_PARSER_API_KEY=parser-key",
                    "PLAN_PARSER_MODEL=parser-model",
                    "ANSWER_BASE_URL=",
                    "ANSWER_API_KEY=",
                    "ANSWER_MODEL=",
                ]),
                encoding="utf-8",
            )
            settings = Settings.load(root)

        self.assertEqual(settings.answer_base_url, "https://example.test/v1")
        self.assertEqual(settings.answer_api_key, "parser-key")
        self.assertEqual(settings.answer_model, "parser-model")

    def test_answer_client_retries_without_enable_thinking_when_unsupported(self) -> None:
        client = EnableThinkingFallbackClient()

        answer = client.complete_answer({"query": "VIT 的结构是什么", "route": "content", "results": {"contexts": []}})

        self.assertEqual(answer, "fallback ok")
        self.assertIn("enable_thinking", client.payloads[0])
        self.assertNotIn("enable_thinking", client.payloads[1])


class StaticAnswerClient:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def complete_answer(self, evidence: dict) -> str:
        self.evidence = evidence
        return self.answer


class FailingAnswerClient:
    def complete_answer(self, evidence: dict) -> str:
        _ = evidence
        raise AssertionError("metadata/reference ask should not call answer LLM")


class EnableThinkingFallbackClient(AnswerComposerClient):
    def __init__(self) -> None:
        super().__init__(
            base_url="https://example.test/v1",
            api_key="test-key",
            model="test-model",
        )
        object.__setattr__(self, "payloads", [])

    def chat_completion(self, payload: dict) -> dict:
        self.payloads.append(dict(payload))
        if len(self.payloads) == 1:
            raise AnswerError("unsupported parameter: enable_thinking")
        return {"choices": [{"message": {"content": "fallback ok"}}]}


def static_metadata_planner(settings: Settings, query: str, *, debug: bool = False) -> dict:
    _ = settings, debug
    return {
        "query": query,
        "route": "metadata",
        "status": "ok",
        "intent": "lookup",
        "resolved": {
            "aliases": [
                {
                    "alias": "BERT",
                    "canonical": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                }
            ]
        },
        "results": {
            "items": [
                {
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                    "values": {"author": ["Jacob Devlin", "Ming-Wei Chang"]},
                }
            ]
        },
    }


def static_content_planner(settings: Settings, query: str, *, debug: bool = False) -> dict:
    _ = settings, debug
    return {
        "query": query,
        "route": "content",
        "status": "ok",
        "intent": "lookup",
        "results": {
            "contexts": [
                {
                    "chunk_id": "chunk-1",
                    "title": "An Image is Worth 16x16 Words",
                    "text": "The model uses patch embeddings and a Transformer encoder.",
                }
            ]
        },
    }


def static_metadata_list_planner(settings: Settings, query: str, *, debug: bool = False) -> dict:
    _ = settings, query, debug
    return {
        "query": "发表在 CVPR 上的论文有哪些",
        "route": "metadata",
        "status": "ok",
        "intent": "list",
        "results": {
            "items": [
                {"title": "Deep Residual Learning for Image Recognition"},
                {"title": "Squeeze-and-Excitation Networks"},
            ]
        },
    }


def static_metadata_count_planner(settings: Settings, query: str, *, debug: bool = False) -> dict:
    _ = settings, query, debug
    return {
        "query": "发表在 CVPR 上的论文有多少篇",
        "route": "metadata",
        "status": "ok",
        "intent": "count",
        "results": {
            "count": 2,
            "items": [
                {"title": "Deep Residual Learning for Image Recognition"},
                {"title": "Squeeze-and-Excitation Networks"},
            ],
        },
    }


def static_metadata_exists_false_planner(settings: Settings, query: str, *, debug: bool = False) -> dict:
    _ = settings, query, debug
    return {
        "query": "BERT 是 CVPR 的论文吗",
        "route": "metadata",
        "status": "ok",
        "intent": "exists",
        "resolved": {
            "aliases": [
                {
                    "alias": "BERT",
                    "canonical": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                }
            ]
        },
        "results": {
            "exists": False,
            "actual": [
                {
                    "title": "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding",
                    "values": {"venue": "NAACL"},
                }
            ],
        },
    }


def static_metadata_group_lookup_planner(settings: Settings, query: str, *, debug: bool = False) -> dict:
    _ = settings, query, debug
    return {
        "query": "ResNet 和 Transformer 分别是哪一年发表的？",
        "route": "metadata",
        "status": "ok",
        "intent": "lookup",
        "resolved": {
            "aliases": [
                {
                    "alias": "ResNet",
                    "canonical": "Deep Residual Learning for Image Recognition",
                },
                {
                    "alias": "Transformer",
                    "canonical": "Attention is All you Need",
                },
            ]
        },
        "results": {
            "groups": [
                {
                    "scope": ["paper=Deep Residual Learning for Image Recognition"],
                    "count": 1,
                    "exists": True,
                    "items": [
                        {
                            "title": "Deep Residual Learning for Image Recognition",
                            "values": {"year": {"preprint_year": 2015, "publish_year": 2016}},
                        }
                    ],
                },
                {
                    "scope": ["paper=Attention is All you Need"],
                    "count": 1,
                    "exists": True,
                    "items": [
                        {
                            "title": "Attention is All you Need",
                            "values": {"year": {"publish_year": 2017}},
                        }
                    ],
                },
            ]
        },
    }


def static_reference_and_group_planner(settings: Settings, query: str, *, debug: bool = False) -> dict:
    _ = settings, query, debug
    return {
        "query": "哪些论文引用了 Transformer 和 ResNet？",
        "route": "reference",
        "status": "ok",
        "intent": "list",
        "plan": {
            "return_side": "source",
            "object_mode": "and",
            "object_groups": [
                {"scope": ["paper=Attention is All you Need"]},
                {"scope": ["paper=Deep Residual Learning for Image Recognition"]},
            ],
        },
        "results": {
            "papers": ["Attention as Activation"],
            "groups": [
                {
                    "scope": ["paper=Attention is All you Need"],
                    "papers": ["Attention as Activation"],
                    "count": 1,
                    "exists": True,
                },
                {
                    "scope": ["paper=Deep Residual Learning for Image Recognition"],
                    "papers": ["Attention as Activation", "Supervised Contrastive Learning"],
                    "count": 2,
                    "exists": True,
                },
            ],
        },
    }


class temp_settings:
    def __enter__(self) -> Settings:
        self.tmp = tempfile.TemporaryDirectory()
        return Settings.load(Path(self.tmp.name))

    def __exit__(self, exc_type, exc, tb) -> None:
        self.tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
