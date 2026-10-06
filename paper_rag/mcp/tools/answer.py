"""Agent 生成答案的引用上下文校验 MCP 工具。"""

from __future__ import annotations

from typing import Any

from paper_rag.answering import ANSWER_CONTEXTS, validate_answer_payload
from paper_rag.mcp._app import mcp


def _response(status: str, data: dict[str, Any], warnings: list[str] | None = None) -> dict[str, Any]:
    return {
        "status": status,
        "data": data,
        "warnings": [item for item in (warnings or []) if item],
        "read_only": True,
    }


@mcp.tool(
    name="library_validate_answer",
    description="校验 Agent 基于 library_retrieve 生成的可溯源答案；只检查 context_id、source_id、claims 和内联引用，不调用答案生成模型。校验通过后 Agent 才能展示答案。",
)
def library_validate_answer(
    context_id: str,
    answer_status: str,
    answer: str,
    claims: list[dict[str, Any]],
    citations: list[str],
) -> dict[str, Any]:
    """确定性校验 Agent 答案并恢复服务端保存的真实来源信息。"""

    context = ANSWER_CONTEXTS.get(context_id)
    if context is None:
        return _response(
            "context_expired",
            {
                "context_id": context_id,
                "validation": {
                    "valid": False,
                    "errors": [
                        {
                            "code": "context_expired",
                            "message": "answer_context_id 不存在或已过期",
                        }
                    ],
                },
            },
        )

    errors = validate_answer_payload(
        context,
        answer_status=answer_status,
        answer=answer,
        claims=claims,
        citations=citations,
    )
    if errors:
        return _response(
            "invalid_answer",
            {
                "context_id": context.context_id,
                "answer_status": answer_status,
                "answer": answer,
                "claims": claims,
                "citations": citations,
                "validation": {"valid": False, "errors": errors},
            },
        )

    resolved = [
        {"citation_id": source_id, **context.citation_registry[source_id]}
        for source_id in citations
    ]
    return _response(
        "ok",
        {
            "context_id": context.context_id,
            "answer_status": answer_status,
            "answer": answer,
            "claims": claims,
            "citations": resolved,
            "validation": {"valid": True, "errors": []},
            "presentation": {
                "answer_type": "grounded_answer",
                "render_policy": "verbatim",
                "answer_text": answer,
            },
        },
    )


__all__ = ["library_validate_answer"]
