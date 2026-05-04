"""ask 入口：先执行 retrieval plan，再按 route 组织回答。"""

from __future__ import annotations

from typing import Any

from .answer_llm import AnswerClientProtocol, AnswerComposerClient
from .answer_local import compose_local_answer, should_use_answer_llm
from .config import Settings
from .retrieval.plan import run_plan


def run_ask(
    settings: Settings,
    query: str,
    *,
    debug: bool = False,
    planner=run_plan,
    answer_client: AnswerClientProtocol | None = None,
) -> dict[str, Any]:
    """执行 ask：先 plan；metadata/reference 本地回答，content 再调用 LLM。"""
    evidence = planner(settings, query, debug=debug)
    if should_use_answer_llm(evidence):
        client = answer_client or AnswerComposerClient.from_settings(settings)
        answer = client.complete_answer(evidence)
        answer_mode = "llm"
    else:
        answer = compose_local_answer(evidence)
        answer_mode = "local"
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
