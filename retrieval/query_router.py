from __future__ import annotations

from dataclasses import dataclass
import re

from config import AppConfig


# router 负责决定一个问题应该走哪条检索策略。
# 这样宽泛的综述类问题，就不会和窄范围事实查询、
# 引用查询走同一条链路。
@dataclass(frozen=True)
class QueryRoute:
    route_type: str
    retrieval_scope: str
    retrieval_mode: str
    use_parent_context: bool
    prefer_paper_context: bool
    paper_top_k: int
    chunk_top_k: int
    final_top_k: int
    prompt_mode: str
    reason: str


def route_query(
    config: AppConfig,
    question: str,
    *,
    scope: str = "main",
) -> QueryRoute:
    """判断问题类型，并返回对应的检索计划。"""
    normalized_scope = str(scope or "main").strip().lower()
    if normalized_scope in {"references", "reference", "ref", "refs"}:
        return QueryRoute(
            route_type="references",
            retrieval_scope="references",
            retrieval_mode="bm25",
            use_parent_context=False,
            prefer_paper_context=False,
            paper_top_k=0,
            chunk_top_k=max(config.retriever_top_k, 12),
            final_top_k=config.final_top_k,
            prompt_mode="references",
            reason="explicit_reference_scope",
        )

    question_text = str(question or "").strip()
    if _is_reference_query(question_text):
        return QueryRoute(
            route_type="references",
            retrieval_scope="references",
            retrieval_mode="bm25",
            use_parent_context=False,
            prefer_paper_context=False,
            paper_top_k=0,
            chunk_top_k=max(config.retriever_top_k, 12),
            final_top_k=config.final_top_k,
            prompt_mode="references",
            reason="reference_query",
        )

    if _is_metadata_query(question_text):
        return QueryRoute(
            route_type="metadata",
            retrieval_scope="main",
            retrieval_mode="bm25",
            use_parent_context=False,
            prefer_paper_context=True,
            paper_top_k=max(6, config.final_top_k + 2),
            chunk_top_k=max(config.retriever_top_k, 16),
            final_top_k=max(config.final_top_k, 5),
            prompt_mode="metadata",
            reason="metadata_query",
        )

    if _is_survey_query(question_text):
        return QueryRoute(
            route_type="survey",
            retrieval_scope="main",
            retrieval_mode="hybrid",
            use_parent_context=False,
            prefer_paper_context=True,
            paper_top_k=max(8, config.final_top_k + 4),
            chunk_top_k=max(config.retriever_top_k, 36),
            final_top_k=max(config.final_top_k, 6),
            prompt_mode="survey",
            reason="survey_query",
        )

    if _is_comparison_query(question_text):
        return QueryRoute(
            route_type="comparison",
            retrieval_scope="main",
            retrieval_mode="hybrid",
            use_parent_context=True,
            prefer_paper_context=False,
            paper_top_k=max(8, config.final_top_k + 3),
            chunk_top_k=max(config.retriever_top_k, 40),
            final_top_k=max(config.final_top_k, 6),
            prompt_mode="comparison",
            reason="comparison_query",
        )

    return QueryRoute(
        route_type="factual",
        retrieval_scope="main",
        retrieval_mode=config.retrieval_mode,
        use_parent_context=True,
        prefer_paper_context=False,
        paper_top_k=max(6, config.final_top_k + 2),
        chunk_top_k=config.retriever_top_k,
        final_top_k=config.final_top_k,
        prompt_mode="factual",
        reason="default",
    )


def _is_reference_query(text: str) -> bool:
    low = text.lower()
    return any(
        token in low
        for token in (
            "参考文献",
            "引用了谁",
            "引用了哪些",
            "引文",
            "cite",
            "cited",
            "references",
            "reference list",
        )
    )


def _is_metadata_query(text: str) -> bool:
    low = text.lower()
    field_markers = (
        "author:",
        "title:",
        "venue:",
        "year:",
        "source:",
        "keyword:",
        "作者",
        "年份",
        "期刊",
        "会议",
        "题目",
        "标题",
        "出处",
    )
    question_markers = (
        "作者是谁",
        "哪一年",
        "发表于",
        "发表在",
        "题目是什么",
        "标题是什么",
        "venue",
        "metadata",
    )
    provenance_markers = (
        "谁提出",
        "谁发明",
        "由谁提出",
        "是谁提出",
        "哪篇论文提出",
        "首次提出",
        "who proposed",
        "proposed by",
        "who introduced",
        "introduced by",
        "which paper proposed",
    )
    return any(marker in low for marker in field_markers + question_markers + provenance_markers)


def _is_survey_query(text: str) -> bool:
    low = text.lower()
    markers = (
        "发展路径",
        "发展脉络",
        "发展历程",
        "研究现状",
        "综述",
        "总体情况",
        "有哪些方向",
        "有哪些方法",
        "主要方法",
        "演进",
        "趋势",
        "survey",
        "overview",
        "state of the art",
        "landscape",
        "roadmap",
        "history",
        "trend",
    )
    return any(marker in low for marker in markers)


def _is_comparison_query(text: str) -> bool:
    low = text.lower()
    has_hint = any(
        token in low
        for token in (
            "比较",
            "对比",
            "区别",
            "差异",
            "异同",
            "共同点",
            "compare",
            "comparison",
            "difference",
            "vs",
        )
    )
    has_connector = bool(
        re.search(r"(和|与|及|以及|跟|vs\.?|/|、|&|\band\b)", text, flags=re.IGNORECASE)
    )
    model_like = {
        item.lower()
        for item in re.findall(r"[A-Za-z][A-Za-z0-9._+-]{1,30}", text)
        if len(item) >= 3
    }
    return (has_hint and has_connector) or len(model_like) >= 2 and has_hint
