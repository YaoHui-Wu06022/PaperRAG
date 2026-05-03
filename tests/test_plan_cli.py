from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from paper_rag.cli.main import build_parser
from paper_rag.config import Settings
from paper_rag.retrieval.plan import run_plan
from paper_rag.retrieval.route import RouteDecision


class PlanCliTests(unittest.TestCase):
    def test_plan_subcommand_is_registered(self) -> None:
        args = build_parser().parse_args(["plan", "BERT", "是谁写的", "--debug"])

        self.assertEqual(args.command, "plan")
        self.assertEqual(args.query, ["BERT", "是谁写的"])
        self.assertTrue(args.debug)
        self.assertTrue(callable(args.handler))


class RunPlanTests(unittest.TestCase):
    def test_run_plan_dispatches_metadata_route(self) -> None:
        with temp_settings() as settings:
            with (
                patch("paper_rag.retrieval.plan.build_metadata_decision") as build_decision,
                patch("paper_rag.retrieval.plan.plan_metadata") as planner,
            ):
                build_decision.return_value = RouteDecision(
                    route="metadata",
                    query="BERT 是谁写的",
                    intent="lookup",
                    parse_status="ok",
                )
                planner.return_value = {
                    "query": "BERT 是谁写的",
                    "route": "metadata",
                    "status": "ok",
                    "intent": "lookup",
                    "results": {"items": [{"title": "BERT"}]},
                }

                evidence = run_plan(settings, "BERT 是谁写的", top_parser=StaticTopParser("metadata"))

        self.assertEqual(evidence["route"], "metadata")
        self.assertEqual(evidence["results"]["items"][0]["title"], "BERT")
        build_decision.assert_called_once()
        planner.assert_called_once()

    def test_run_plan_returns_unclear_without_domain_parser(self) -> None:
        with temp_settings() as settings:
            evidence = run_plan(settings, "这个问题不明确", top_parser=StaticTopParser("unclear"))

        self.assertEqual(evidence["route"], "unclear")
        self.assertEqual(evidence["status"], "unclear")
        self.assertTrue(evidence["warnings"])


class StaticTopParser:
    def __init__(self, router: str) -> None:
        self.router = router

    def parse_top(self, query: str) -> dict[str, str]:
        _ = query
        return {"router": self.router}


class temp_settings:
    def __enter__(self) -> Settings:
        self.tmp = tempfile.TemporaryDirectory()
        return Settings.load(Path(self.tmp.name))

    def __exit__(self, exc_type, exc, tb) -> None:
        self.tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
