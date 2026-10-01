"""CLI 检索入口。

Paper_RAG 进程只返回检索证据，最终自然语言回答由连接的 Agent 负责生成。
"""

from __future__ import annotations

from typing import Any

from paper_rag.config import Settings
from paper_rag.corpus.context import CorpusContext
from paper_rag.retrieval.plan import run_plan


def run_ask(
    settings: Settings,
    query: str,
    *,
    debug: bool = False,
    planner=run_plan,
    corpus: CorpusContext | None = None,
) -> dict[str, Any]:
    """执行一次检索计划，供 CLI 和调试使用。"""
    corpus = corpus or CorpusContext(settings)
    if planner is run_plan:
        evidence = planner(settings, query, debug=debug, corpus=corpus)
    else:
        evidence = planner(settings, query, debug=debug)
    payload: dict[str, Any] = {
        "query": query,
        "answer_mode": "evidence",
        "evidence": evidence,
    }
    if evidence.get("warnings"):
        payload["warnings"] = evidence["warnings"]
    return payload
