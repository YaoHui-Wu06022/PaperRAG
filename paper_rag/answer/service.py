"""ask 入口：先执行 retrieval plan，再按 route 组织回答。"""

from __future__ import annotations

from typing import Any

from paper_rag.answer.llm import AnswerClientProtocol, AnswerComposerClient, AnswerError
from paper_rag.answer.local import compose_answer_failure_answer, compose_local_answer, should_use_answer_llm
from paper_rag.config import Settings
from paper_rag.corpus.context import CorpusContext
from paper_rag.retrieval.plan import run_plan
from paper_rag.retrieval.timing import Timings, attach_timings


def run_ask(
    settings: Settings,
    query: str,
    *,
    debug: bool = False,
    planner=run_plan,
    answer_client: AnswerClientProtocol | None = None,
) -> dict[str, Any]:
    """执行 ask：先 plan；metadata/reference 本地回答，content 再调用 LLM。"""
    timings = Timings(debug)
    corpus = CorpusContext(settings)
    with timings.measure("plan"):
        if planner is run_plan:
            evidence = planner(settings, query, debug=debug, corpus=corpus, timings=timings)
        else:
            evidence = planner(settings, query, debug=debug)
    with timings.measure("answer"):
        if should_use_answer_llm(evidence):
            client = answer_client or AnswerComposerClient.from_settings(settings)
            try:
                answer = client.complete_answer(evidence)
                answer_mode = "llm"
            except (AnswerError, OSError, ValueError) as exc:
                evidence.setdefault("warnings", []).append(f"answer generation failed: {exc}")
                answer = compose_answer_failure_answer()
                answer_mode = "local"
        else:
            answer = compose_local_answer(evidence)
            answer_mode = "local"
    attach_timings(evidence, timings)
    payload: dict[str, Any] = {
        "query": query,
        "answer": answer,
        "answer_mode": answer_mode,
        "evidence": evidence,
    }
    warnings = evidence.get("warnings")
    if warnings:
        payload["warnings"] = warnings
    return payload
