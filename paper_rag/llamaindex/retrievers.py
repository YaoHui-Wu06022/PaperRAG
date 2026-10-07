"""LlamaIndex Retriever 适配层和确定性的 RRF 混合检索。"""

from __future__ import annotations

import re
from typing import Any, Iterable

from llama_index.core.retrievers import BaseRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle

from paper_rag.catalog.service import search_chunks
from paper_rag.config import Settings
from paper_rag.llamaindex.nodes import chunk_row_to_node
from paper_rag.llamaindex.translation import LexicalQuery, prepare_lexical_query


class SQLiteLexicalRetriever(BaseRetriever):
    """把既有 SQLite FTS5 检索暴露为 LlamaIndex Retriever。"""

    def __init__(
        self,
        settings: Settings,
        *,
        paper_ids: Iterable[str] | None = None,
        regions: Iterable[str] | None = None,
        top_k: int = 50,
        prepared_query: LexicalQuery | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.paper_ids = list(paper_ids or ())
        self.regions = set(str(value) for value in (regions or ()) if str(value))
        self.top_k = max(1, int(top_k))
        self.prepared_query = prepared_query
        self.debug: dict[str, Any] = {}
        self.warnings: list[str] = []

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        prepared = self.prepared_query or prepare_lexical_query(query_bundle.query_str, self.settings)
        self.debug = prepared.debug()
        self.warnings = list(prepared.warnings)
        items = search_chunks(
            self.settings,
            prepared.query,
            self.paper_ids or None,
            self.top_k,
            tuple(self.regions),
            fts_query=prepared.fts_query,
        )
        result: list[NodeWithScore] = []
        for rank, item in enumerate(items, start=1):
            node = chunk_row_to_node(_item_to_row(item))
            node.metadata["lexical_rank"] = rank
            node.metadata["lexical_query"] = prepared.query
            result.append(NodeWithScore(node=node, score=1.0 / rank))
        return result


class HybridRetriever(BaseRetriever):
    """组合 lexical 和 semantic Retriever，并按 chunk_id 做 RRF。"""

    def __init__(
        self,
        settings: Settings,
        semantic_retriever: BaseRetriever | None = None,
        *,
        paper_ids: Iterable[str] | None = None,
        regions: Iterable[str] | None = None,
        mode: str = "hybrid",
        task: str = "fact",
        lexical_top_k: int = 50,
        semantic_top_k: int = 50,
        rrf_k: int = 60,
        lexical_query_override: LexicalQuery | None = None,
        lexical_fallback_query: LexicalQuery | None = None,
        preferred_paper_ids: Iterable[str] | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings
        self.semantic_retriever = semantic_retriever
        self.paper_ids = list(paper_ids or ())
        self.regions = set(str(value) for value in (regions or ()) if str(value))
        self.mode = mode.casefold()
        self.task = str(task).casefold()
        self.lexical_top_k = max(1, int(lexical_top_k))
        self.semantic_top_k = max(1, int(semantic_top_k))
        self.rrf_k = max(1, int(rrf_k))
        self.lexical_query_override = lexical_query_override
        self.lexical_fallback_query = lexical_fallback_query
        self.preferred_paper_ids = set(preferred_paper_ids or ())
        self.recall_debug: dict[str, Any] = {}
        self.warnings: list[str] = []
        self.lexical_debug: dict[str, Any] = {
            "lexical_query": "",
            "translation_used": False,
            "translation_provider": None,
            "translation_fallback": False,
            "stopwords_removed": [],
            "rewriter_used": False,
            "rewriter_fallback": False,
            "core_terms": [],
        }

    def _retrieve(self, query_bundle: QueryBundle) -> list[NodeWithScore]:
        self.warnings = []
        self.lexical_debug = {
            "lexical_query": "",
            "translation_used": False,
            "translation_provider": None,
            "translation_fallback": False,
            "stopwords_removed": [],
            "rewriter_used": False,
            "rewriter_fallback": False,
            "core_terms": [],
        }
        lexical: list[NodeWithScore] = []
        semantic: list[NodeWithScore] = []
        if self.mode in {"lexical", "hybrid"}:
            lexical_retriever = SQLiteLexicalRetriever(
                self.settings,
                paper_ids=self.paper_ids,
                regions=self.regions,
                top_k=self.lexical_top_k,
                prepared_query=self.lexical_query_override,
            )
            lexical = lexical_retriever.retrieve(query_bundle)
            self.lexical_debug = lexical_retriever.debug
            self.warnings.extend(lexical_retriever.warnings)
        if self.mode in {"semantic", "hybrid"} and self.semantic_retriever is not None:
            try:
                semantic = self.semantic_retriever.retrieve(query_bundle)
                semantic = [item for item in semantic if self._allowed(item)]
            except Exception as exc:
                self.warnings.append(f"semantic_unavailable:{type(exc).__name__}")
        if self.mode == "semantic" and not semantic:
            self.warnings.append("lexical_fallback")
            if not lexical:
                lexical_retriever = SQLiteLexicalRetriever(
                    self.settings,
                    paper_ids=self.paper_ids,
                    regions=self.regions,
                    top_k=self.lexical_top_k,
                    prepared_query=self.lexical_query_override,
                )
                lexical = lexical_retriever.retrieve(query_bundle)
                self.lexical_debug = lexical_retriever.debug
                self.warnings.extend(lexical_retriever.warnings)
        if not lexical and (self.mode != "semantic" or not semantic) and self.lexical_fallback_query and self.lexical_fallback_query.fts_query:
            # 包括语义失败后的词法分支，回退始终保持同一论文范围。
            fallback = SQLiteLexicalRetriever(self.settings, paper_ids=self.paper_ids, regions=self.regions, top_k=self.lexical_top_k, prepared_query=self.lexical_fallback_query)
            lexical = fallback.retrieve(query_bundle)
            self.warnings.extend(fallback.warnings)
            self.lexical_debug["entity_fallback_used"] = True
            self.lexical_debug["entity_fallback_query"] = self.lexical_fallback_query.fts_query
        merged: dict[str, dict[str, Any]] = {}
        for rank, item in enumerate(lexical, start=1):
            key = item.node.node_id
            entry = merged.setdefault(key, {"node": item.node, "lexical_rank": rank, "semantic_rank": None})
            entry["lexical_rank"] = min(entry["lexical_rank"], rank)
        for rank, item in enumerate(semantic, start=1):
            key = item.node.node_id
            entry = merged.setdefault(key, {"node": item.node, "lexical_rank": None, "semantic_rank": rank})
            entry["semantic_rank"] = rank
            entry["semantic_score"] = item.score
        ranked: list[NodeWithScore] = []
        section_counts: dict[tuple[str, str], int] = {}
        base_entries = sorted(merged.values(), key=lambda value: self._rrf_score(value), reverse=True)
        for entry in base_entries:
            score = 0.0
            if entry["lexical_rank"]:
                score += 1 / (self.rrf_k + entry["lexical_rank"])
            if entry["semantic_rank"]:
                score += 1 / (self.rrf_k + entry["semantic_rank"])
            metadata = entry["node"].metadata
            features = self._ranking_features(metadata, query_bundle.query_str, section_counts)
            entry["node"].metadata.update(
                {
                    "lexical_rank": entry["lexical_rank"],
                    "semantic_rank": entry["semantic_rank"],
                    "semantic_score": entry.get("semantic_score"),
                    "rrf_score": score,
                    "ranking_features": features,
                }
            )
            ranked.append(NodeWithScore(node=entry["node"], score=score))
        ranked.sort(key=lambda item: self._sort_key(item, query_bundle.query_str), reverse=True)
        # 普通问题优先给正文解释，只有明确要求媒体时才让媒体进入同级结果。
        if not _media_query_requested(query_bundle.query_str):
            ranked = [item for item in ranked if str(item.node.metadata.get("type") or "text").casefold() not in _MEDIA_TYPES] + [
                item for item in ranked if str(item.node.metadata.get("type") or "text").casefold() in _MEDIA_TYPES
            ]
        self.recall_debug = {"lexical_count": len(lexical), "semantic_count": len(semantic), "fused_count": len(merged), "lexical_query": self.lexical_debug.get("lexical_query"), "entity_fallback_used": self.lexical_debug.get("entity_fallback_used", False)}
        if self.lexical_debug.get("entity_fallback_used"):
            self.recall_debug["entity_fallback_query"] = self.lexical_debug["entity_fallback_query"]
        return ranked

    def _sort_key(self, item: NodeWithScore, query: str) -> tuple[float, ...]:
        """以 RRF 为主排序；显式表/图编号先满足硬约束。"""

        metadata = item.node.metadata
        features = metadata.get("ranking_features") or {}
        rrf_score = float(metadata.get("rrf_score") or 0.0)
        table_requested = _reference_number(query, _TABLE_REF_RE) is not None
        figure_requested = _reference_number(query, _FIGURE_REF_RE) is not None
        if table_requested or figure_requested:
            return (
                # 编号在各论文内重复，明确的对象标题仅作排序线索，不能过滤召回。
                float(metadata.get("paper_id") in self.preferred_paper_ids),
                float(bool(features.get("exact_table_caption_hit"))) if table_requested else float(bool(features.get("exact_figure_caption_hit"))),
                float(bool(features.get("exact_table_ref_hit"))) if table_requested else float(bool(features.get("exact_figure_ref_hit"))),
                rrf_score,
                float(bool(features.get("exact_entity_hit"))),
                float(bool(features.get("section_exact_hit"))),
                -float(features.get("duplicate_penalty") or 0.0),
            )
        return (
            rrf_score,
            float(bool(features.get("exact_entity_hit"))),
            float(bool(features.get("section_exact_hit"))),
            float(features.get("region_score") or 0.0),
            float(features.get("type_score") or 0.0),
            -float(features.get("duplicate_penalty") or 0.0),
        )

    def _rrf_score(self, entry: dict[str, Any]) -> float:
        score = 0.0
        if entry.get("lexical_rank"):
            score += 1 / (self.rrf_k + entry["lexical_rank"])
        if entry.get("semantic_rank"):
            score += 1 / (self.rrf_k + entry["semantic_rank"])
        return score

    def _ranking_features(
        self,
        metadata: dict[str, Any],
        query: str,
        section_counts: dict[tuple[str, str], int],
    ) -> dict[str, Any]:
        """计算不依赖额外模型的证据质量特征。"""

        text = _normalize_text(f"{metadata.get('content_text', '')} {metadata.get('section_label', '')}")
        phrases = _query_phrases(query)
        exact_entity_hit = any(_phrase_hit(text, phrase) for phrase in phrases)
        section_label = _normalize_text(str(metadata.get("section_label") or ""))
        section_exact_hit = any(_phrase_hit(section_label, phrase) for phrase in phrases) or _section_reference_hit(metadata, query)
        kind = str(metadata.get("type") or "text").casefold()
        type_priority = {"text": 3, "paragraph": 3, "table": 3, "list": 2, "list_item": 2, "formula": 3, "equation": 3, "image": 0, "figure": 0, "chart": 0}.get(kind, 1)
        region = str(metadata.get("region") or "").casefold()
        region_priority = self._region_priority(region)
        table_ref = _reference_number(query, _TABLE_REF_RE)
        figure_ref = _reference_number(query, _FIGURE_REF_RE)
        exact_table_ref_hit = bool(table_ref and _media_reference_hit(text, "table", table_ref))
        exact_figure_ref_hit = bool(figure_ref and _media_reference_hit(text, "figure", figure_ref))
        exact_table_caption_hit = exact_table_ref_hit and kind == "table"
        exact_figure_caption_hit = exact_figure_ref_hit and kind in {"image", "figure", "chart"}
        section_key = (str(metadata.get("paper_id") or ""), section_label)
        duplicate_penalty = 0.00035 if section_counts.get(section_key, 0) else 0.0
        section_counts[section_key] = section_counts.get(section_key, 0) + 1
        media_requested = _media_query_requested(query)
        if media_requested and kind in {"image", "figure", "chart", "table"}:
            type_priority += 2
        quality_bonus = (
            (0.0014 if exact_entity_hit else 0.0)
            + (0.0011 if section_exact_hit else 0.0)
            + (0.00025 * region_priority)
            + (0.00015 * type_priority)
        )
        return {
            "exact_entity_hit": exact_entity_hit,
            "section_exact_hit": section_exact_hit,
            "exact_table_ref_hit": exact_table_ref_hit,
            "exact_figure_ref_hit": exact_figure_ref_hit,
            "exact_table_caption_hit": exact_table_caption_hit,
            "exact_figure_caption_hit": exact_figure_caption_hit,
            "type_priority": kind,
            "type_score": type_priority,
            "region_priority": region,
            "region_score": region_priority,
            "duplicate_penalty": duplicate_penalty,
            "quality_bonus": quality_bonus,
        }

    def _region_priority(self, region: str) -> int:
        if self.task == "summary":
            return {"abstract": 3, "content": 2, "appendix": 0}.get(region, 1)
        if self.task == "comparison":
            return {"content": 3, "abstract": 2, "appendix": 0}.get(region, 1)
        return {"content": 3, "abstract": 2, "appendix": 0}.get(region, 1)

    def _allowed(self, item: NodeWithScore) -> bool:
        metadata = item.node.metadata
        if self.paper_ids and str(metadata.get("paper_id")) not in self.paper_ids:
            return False
        return not self.regions or str(metadata.get("region")) in self.regions


def _item_to_row(item: dict[str, Any]) -> tuple[Any, ...]:
    """将 Catalog 查询结果还原为 chunk_row_to_node 所需的列顺序。"""

    return (
        item.get("chunk_id"),
        item.get("paper_id"),
        item.get("canonical_id"),
        item.get("ordinal", 0),
        item.get("region", ""),
        item.get("chapter_number"),
        item.get("chapter_title"),
        item.get("section_path", []),
        item.get("section_label"),
        item.get("type", "text"),
        item.get("text", ""),
        item.get("retrieval_text", item.get("text", "")),
        item.get("page_start"),
        item.get("page_end"),
        item.get("source_blocks", []),
        item.get("asset_refs", []),
        item.get("content_hash", ""),
        item.get("retrieval_text_hash", ""),
    )


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[-_/]+", " ", str(value or "").casefold())).strip()


def _query_phrases(query: str) -> list[str]:
    """提取方法名和连续术语，避免普通疑问词触发精确命中。"""

    phrases: list[str] = []
    for match in re.findall(r"\b(?:[A-Z]{2,}[A-Za-z0-9.-]*|[A-Za-z]+[A-Z][A-Za-z0-9.-]*|[A-Za-z]+[0-9][A-Za-z0-9.-]*|[A-Za-z]+(?:-[A-Za-z]+)+)\b", str(query or "")):
        normalized = _normalize_text(match)
        if len(normalized) >= 3:
            phrases.append(normalized)
    return list(dict.fromkeys(phrases))


_TABLE_REF_RE = re.compile(r"(?:\btable\s*|\btab\.?\s*|表\s*)(\d+)", re.IGNORECASE)
_FIGURE_REF_RE = re.compile(r"(?:\bfigure\s*|\bfig\.?\s*|图\s*)(\d+)", re.IGNORECASE)
_SECTION_REF_RE = re.compile(r"(?:\bsection\s*|\bsec\.?\s*|第\s*)(\d+(?:\.\d+)*)", re.IGNORECASE)
_MEDIA_TYPES = {"image", "figure", "chart"}


def _reference_number(query: str, pattern: re.Pattern[str]) -> str | None:
    match = pattern.search(str(query or ""))
    return match.group(1) if match else None


def _media_reference_hit(text: str, media: str, number: str) -> bool:
    aliases = "table|tab" if media == "table" else "figure|fig"
    return bool(re.search(rf"\b(?:{aliases})\.?\s*{re.escape(number)}\b|(?:表|图)\s*{re.escape(number)}\b", text, re.IGNORECASE))


def _phrase_hit(text: str, phrase: str) -> bool:
    if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", text):
        return True
    words = phrase.split()
    singular = " ".join(words[:-1] + [words[-1].rstrip("s")])
    return bool(singular and re.search(rf"(?<![a-z0-9]){re.escape(singular)}(?![a-z0-9])", text))


def _media_query_requested(query: str) -> bool:
    return bool(re.search(r"(?:figure|fig\.?|image|table|tab\.?|formula|equation|图|表格|公式)", str(query or ""), re.IGNORECASE))


def _section_reference_hit(metadata: dict[str, Any], query: str) -> bool:
    match = _SECTION_REF_RE.search(str(query or ""))
    if not match:
        return False
    number = match.group(1)
    chapter_number = str(metadata.get("chapter_number") or "")
    section_label = str(metadata.get("section_label") or "")
    return chapter_number == number or bool(re.match(rf"^{re.escape(number)}(?:\s|\.|$)", section_label, re.IGNORECASE))


__all__ = ["HybridRetriever", "SQLiteLexicalRetriever"]
